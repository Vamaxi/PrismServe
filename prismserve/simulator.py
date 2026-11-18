"""Deterministic trace simulator. No model execution occurs in this module.

Requests reserve full declared context to completion, so a newly admitted request
cannot cause an OOM later simply by growing its KV. Prompt prefill is chunked;
the last prefill chunk emits output token one. Policy search wall time is measured
separately and is not added to analytical virtual time.
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field
import math
from statistics import mean
import time

from .types import ControlState, HardwareProfile, ModelSpec, PlannerConfig, RequestSpec, WorkItem
from .policy import Controller


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    point = (len(ordered) - 1) * q
    low = math.floor(point)
    high = math.ceil(point)
    return ordered[low] + (ordered[high] - ordered[low]) * (point - low)


@dataclass
class _Running:
    spec: RequestSpec
    prefill_remaining: int
    token_times: list[float] = field(default_factory=list)
    admitted_s: float = 0


def simulate(trace: list[RequestSpec], profile: HardwareProfile, model: ModelSpec,
             config: PlannerConfig, policy: str = "joint", max_batch_size: int = 32,
             prefill_chunk_size: int = 128, max_iterations: int = 100000) -> dict:
    if not trace:
        raise ValueError("trace must not be empty")
    if min(max_batch_size, prefill_chunk_size, max_iterations) <= 0:
        raise ValueError("simulation limits must be positive")
    if len({r.request_id for r in trace}) != len(trace):
        raise ValueError("duplicate request IDs")
    ordered = sorted(trace, key=lambda r: (r.arrival_s, r.request_id))
    controller = Controller(profile, model, config, policy=policy)
    pending: deque[RequestSpec] = deque()
    active: list[_Running] = []
    completed: list[_Running] = []
    rejected: list[dict] = []
    iterations: list[dict] = []
    state = ControlState()
    arrival_index = 0
    now = ordered[0].arrival_s
    start = now
    wall_start = time.perf_counter()
    searches_us = []
    changes = 0
    for iteration in range(max_iterations):
        while arrival_index < len(ordered) and ordered[arrival_index].arrival_s <= now + 1e-12:
            pending.append(ordered[arrival_index])
            arrival_index += 1
        if not active and not pending:
            if arrival_index == len(ordered):
                break
            now = max(now, ordered[arrival_index].arrival_s)
            continue
        extra = min(max_batch_size - len(active), len(pending))
        new_specs = list(pending)[:extra]
        decision = None
        candidates: list[_Running] = []
        while True:
            candidates = active + [_Running(r, r.prompt_tokens, admitted_s=now) for r in new_specs]
            prior_ids = {r.spec.request_id for r in active}
            work = [WorkItem(r.spec.request_id, r.spec.prompt_tokens + r.spec.output_tokens,
                             min(prefill_chunk_size, r.prefill_remaining),
                             r.spec.prompt_tokens + r.spec.output_tokens if r.spec.request_id in prior_ids else 0)
                    for r in candidates]
            decision = controller.choose(work, state, now)
            searches_us.append(decision.search_us)
            if decision.estimate.feasible or not new_specs:
                break
            new_specs.pop()
        assert decision is not None
        if not decision.estimate.feasible:
            if active:
                raise RuntimeError("previously admitted reserved contexts became infeasible: " + decision.reason)
            bad = pending.popleft()
            rejected.append({"request_id": bad.request_id, "arrival_s": bad.arrival_s,
                             "reason": decision.estimate.reason or decision.reason})
            continue
        if not candidates:
            # The single request does not fit; a choice on an empty batch does not admit it.
            bad = pending.popleft()
            rejected.append({"request_id": bad.request_id, "arrival_s": bad.arrival_s,
                             "reason": "no feasible plan for single request reservation"})
            continue
        for _ in new_specs:
            pending.popleft()
        active = candidates
        plan = decision.plan
        changed = (plan.quantized_layers != state.quantized_layers or
                   plan.cpu_request_ids != state.cpu_request_ids)
        if changed:
            changes += 1
        state = ControlState(plan.quantized_layers, plan.cpu_request_ids,
                             now if changed else state.last_change_s)
        duration_ms = decision.estimate.iteration_ms + decision.estimate.transition_ms
        if not math.isfinite(duration_ms) or duration_ms <= 0:
            raise RuntimeError("estimator returned a nonpositive or nonfinite iteration duration")
        finish = now + duration_ms / 1000
        emitted = 0
        for request in active:
            if request.prefill_remaining:
                request.prefill_remaining -= min(prefill_chunk_size, request.prefill_remaining)
                if not request.prefill_remaining:
                    request.token_times.append(finish)
                    emitted += 1
            else:
                request.token_times.append(finish)
                emitted += 1
        iterations.append({"iteration": len(iterations), "start_s": now, "end_s": finish,
                           "batch_size": len(active), "pending": len(pending),
                           "quantized_layers": list(plan.quantized_layers),
                           "cpu_request_ids": list(plan.cpu_request_ids),
                           "device_bytes": decision.estimate.device_bytes,
                           "host_bytes": decision.estimate.host_bytes,
                           "quality_proxy": decision.estimate.quality_risk,
                           "transition_ms": decision.estimate.transition_ms,
                           "iteration_ms": decision.estimate.iteration_ms,
                           "search_us": decision.search_us, "emitted_tokens": emitted,
                           "reason": decision.reason})
        done = [r for r in active if len(r.token_times) == r.spec.output_tokens]
        completed.extend(done)
        active = [r for r in active if len(r.token_times) < r.spec.output_tokens]
        live = {r.spec.request_id for r in active}
        state = ControlState(state.quantized_layers,
                             tuple(x for x in state.cpu_request_ids if x in live), state.last_change_s)
        now = finish
    else:
        raise RuntimeError(f"simulation exceeded {max_iterations} iterations; no partial results saved")
    request_rows = []
    all_gaps = []
    for request in sorted(completed, key=lambda r: r.spec.request_id):
        times = request.token_times
        ttft = (times[0] - request.spec.arrival_s) * 1000
        gaps = [(b - a) * 1000 for a, b in zip(times, times[1:])]
        all_gaps.extend(gaps)
        max_tbt = max(gaps, default=0)
        violates = ttft > config.ttft_slo_ms or max_tbt > config.tbt_slo_ms
        request_rows.append({**asdict(request.spec), "admitted_s": request.admitted_s,
                             "ttft_ms": ttft, "tbt_p95_ms": percentile(gaps, .95),
                             "max_tbt_ms": max_tbt, "latency_ms": (times[-1] - request.spec.arrival_s) * 1000,
                             "completed_s": times[-1], "slo_violated": violates,
                             "output_token_times_s": times})
    elapsed = max(now - start, 1e-12)
    n_good = sum(not r["slo_violated"] for r in request_rows)
    failed = len(rejected) + sum(r["slo_violated"] for r in request_rows)
    summary = {"mode": "analytical_simulation", "profile": profile.name, "policy": policy,
               "requests": len(trace), "completed": len(completed), "rejected": len(rejected),
               "ttft_p95_ms": percentile([r["ttft_ms"] for r in request_rows], .95),
               "tbt_p95_ms": percentile(all_gaps, .95),
               "slo_violation_rate": failed / len(trace),
               "request_throughput_per_s": len(completed) / elapsed,
               "output_tokens_per_s": sum(r.spec.output_tokens for r in completed) / elapsed,
               "goodput_requests_per_s": n_good / elapsed, "elapsed_virtual_s": elapsed,
               "plan_changes": changes, "iterations": len(iterations),
               "peak_device_bytes": max((i["device_bytes"] for i in iterations), default=0),
               "mean_quantized_layers": mean([len(i["quantized_layers"]) for i in iterations]) if iterations else 0,
               "mean_cpu_requests": mean([len(i["cpu_request_ids"]) for i in iterations]) if iterations else 0,
               "max_quality_proxy": max((i["quality_proxy"] for i in iterations), default=0),
               "planner_search_p95_us_measured": percentile(searches_us, .95),
               "simulation_wall_s_measured": time.perf_counter() - wall_start}
    return {"summary": summary, "requests": request_rows, "rejections": rejected,
            "iterations": iterations, "hardware": asdict(profile), "model": asdict(model),
            "planner": asdict(config), "assumptions": [
                "Hardware rates and iteration times are analytical, not executed LLM timings.",
                "Attention scan costs use full reserved context: conservative relative to current length.",
                "All contexts are reserved through declared max output; no eviction/recomputation.",
                "Ideal partial CPU/device overlap is modeled; the reference backend is synchronous.",
                "Quality risk is a policy surrogate, not a task accuracy guarantee.",
                "Policy search wall time is reported separately, not added to virtual time."]}
