"""Real continuous batching: one GPU owner, multiple live per-request KV caches.

Every timestamp is host monotonic wall time; no analytical service time advances
the worker clock. Dense projections batch all current tokens; attention is ragged
per request in the reference backend. Admission reserves maximum context.
"""
from __future__ import annotations

from collections import deque
from concurrent.futures import Future
from dataclasses import asdict, dataclass, field, replace
import threading
import time
from typing import Any

from .types import ControlState, HardwareProfile, PlannerConfig, WorkItem, Plan
from .simulator import percentile


@dataclass(frozen=True)
class ServingConfig:
    max_batch_size: int = 8
    prefill_chunk_size: int = 64
    prefill_token_budget: int = 128
    device_memory_budget_bytes: int = 0
    host_memory_budget_bytes: int = 0
    max_pending_requests: int = 256
    ttft_slo_ms: float = 2000
    tbt_slo_ms: float = 100
    quality_budget: float = .03
    cooldown_s: float = .25
    backend_margin_bytes: int = 64 * 2**20
    scheduler_strategy: str = 'efficient'
    decision_interval: int = 4
    reuse_decisions: bool = True
    conservative_gate: bool = True

    def __post_init__(self):
        for name in ("max_batch_size", "prefill_chunk_size", "prefill_token_budget", "max_pending_requests", "decision_interval"):
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.prefill_token_budget < self.max_batch_size:
            raise ValueError("prefill_token_budget must allow at least one token per active request")
        if min(self.device_memory_budget_bytes, self.host_memory_budget_bytes, self.backend_margin_bytes) < 0:
            raise ValueError("memory budgets must be nonnegative")
        if min(self.ttft_slo_ms, self.tbt_slo_ms) <= 0 or min(self.quality_budget, self.cooldown_s) < 0:
            raise ValueError("SLOs must be positive; quality/cooldown nonnegative")
        if self.scheduler_strategy not in ('efficient', 'legacy'):
            raise ValueError('scheduler strategy must be efficient/legacy and decision interval positive')


@dataclass
class _Request:
    request_id: str
    prompt_ids: list[int]
    max_tokens: int
    arrival: float
    stop_on_eos: bool
    future: Future
    forced_tokens: list[int] | None = None
    cache: Any = None
    cursor: int = 0
    output_ids: list[int] = field(default_factory=list)
    predicted_ids: list[int] = field(default_factory=list)
    times: list[float] = field(default_factory=list)
    nll: list[float] = field(default_factory=list)
    precision: list[list[int]] = field(default_factory=list)
    placements: list[str] = field(default_factory=list)
    offloaded: list[bool] = field(default_factory=list)
    admitted: float | None = None


class ContinuousEngine:
    def __init__(self, executor, calibration, profile: HardwareProfile,
                 config: ServingConfig | None = None, policy: str = "joint", sensitivity: tuple = ()):
        from .measured_policy import MeasuredController
        self.executor = executor
        self.calibration, self.profile, self.policy = calibration, profile, policy
        if profile.device_kind != executor.device.type:
            raise ValueError("timing calibration device differs from executor; calibrate on the selected device")
        self.config = config or ServingConfig()
        base_policy = policy
        if policy in ('full_v03', 'joint_v03'):
            self.config = replace(self.config, scheduler_strategy='legacy')
            base_policy = policy.removesuffix('_v03')
        if policy in ('joint_no_gate', 'joint_no_reuse'):
            base_policy = 'joint'
            self.config = replace(self.config, conservative_gate=policy != 'joint_no_gate',
                                  reuse_decisions=policy != 'joint_no_reuse')
        self._efficient = self.config.scheduler_strategy == 'efficient' and base_policy != 'joint_uncalibrated'
        self.model = executor.model_spec(sensitivity)
        self._fixed_buffer_bytes = sum(buffer.numel() * buffer.element_size()
                                       for name, buffer in executor.model.named_buffers()
                                       if "packed_data" not in name and "packed_scales" not in name)
        self.planner_config = PlannerConfig(quality_budget=self.config.quality_budget,
                                            ttft_slo_ms=self.config.ttft_slo_ms,
                                            tbt_slo_ms=self.config.tbt_slo_ms, memory_fraction=1,
                                            cooldown_s=self.config.cooldown_s,
                                            offload_fractions=(0, .5, 1), max_quantized_fraction=.75)
        sets = tuple(dict.fromkeys(s.quantized_layers for s in calibration.samples))
        if self._efficient:
            from .efficient_policy import EfficientController
            self.controller = EfficientController(profile, self.model, self.planner_config,
                                                  calibration, policy=base_policy, quantization_candidates=sets,
                                                  conservative_gate=self.config.conservative_gate)
        else:
            self.controller = MeasuredController(profile, self.model, self.planner_config,
                                                 calibration, policy=base_policy, quantization_candidates=sets)
        self._memory_snapshot = executor.memory_report()
        self._last_plan_key = None
        self._last_plan_iteration = -1000
        self._condition = threading.Condition()
        self._pending = deque()
        self._active: list[_Request] = []
        self._ids: set[str] = set()
        self._worker = None
        self._closing = False
        self._drain = True
        self._fatal = None
        self._failure_diagnostic = None
        self._state = ControlState()
        self._origin = time.perf_counter()
        self._completed: list[dict] = []
        self._rejected: list[dict] = []
        self._events: list[dict] = []
        self._max_active = 0
        self._peak_cache_device = 0
        self._peak_cache_host = 0

    def start(self):
        with self._condition:
            if self._closing:
                raise RuntimeError("continuous engine is closed")
            if self._worker is not None:
                return self
            self._worker = threading.Thread(target=self._run, name="prismserve-batch-worker", daemon=True)
            self._worker.start()
        return self

    def close(self, drain: bool = True):
        with self._condition:
            if self._worker is None:
                for request in self._pending:
                    request.future.cancel()
                self._pending.clear()
            self._closing, self._drain = True, drain
            self._condition.notify_all()
        if self._worker is not None and threading.current_thread() is not self._worker:
            self._worker.join(timeout=60)
            if self._worker.is_alive():
                raise RuntimeError("batch worker did not stop within60 seconds")

    def submit(self, prompt: str, max_tokens: int = 32, request_id: str | None = None,
               arrival_time: float | None = None, stop_on_eos: bool = True) -> Future:
        arrival = time.perf_counter() if arrival_time is None else arrival_time
        return self.submit_tokens(self.executor.tokenize(prompt), max_tokens, request_id, arrival,
                                  stop_on_eos=stop_on_eos)

    def submit_tokens(self, prompt_ids: list[int], max_tokens: int = 32, request_id: str | None = None,
                      arrival_time: float | None = None, stop_on_eos: bool = True,
                      forced_tokens: list[int] | None = None) -> Future:
        if not prompt_ids or any(not isinstance(i, int) or not 0 <= i < self.executor.config.vocab_size for i in prompt_ids):
            raise ValueError("prompt must contain valid integer tokens")
        if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        if len(prompt_ids) + max_tokens > self.executor.context_limit:
            raise ValueError("prompt plus output exceeds model context capacity")
        if forced_tokens is not None and (len(forced_tokens) != max_tokens or any(not 0 <= i < self.executor.config.vocab_size for i in forced_tokens)):
            raise ValueError("forced continuation must contain exactly max_tokens valid targets")
        future = Future()
        with self._condition:
            if self._closing or self._fatal is not None:
                raise RuntimeError("continuous engine is closed or failed")
            if len(self._pending) >= self.config.max_pending_requests:
                raise OverflowError("continuous engine pending queue is full")
            identifier = request_id or f"req-{len(self._ids):06d}"
            if identifier in self._ids:
                raise ValueError("duplicate request ID")
            self._ids.add(identifier)
            request = _Request(identifier, list(prompt_ids), max_tokens,
                               time.perf_counter() if arrival_time is None else arrival_time,
                               stop_on_eos, future, forced_tokens)
            self._pending.append(request)
            self._condition.notify_all()
        return future

    def metrics(self) -> dict:
        with self._condition:
            rows = list(self._completed)
            events = list(self._events)
            alive = bool(self._worker and self._worker.is_alive())
            status = "failed" if self._fatal else ("running" if alive else ("stopped" if self._closing else "created"))
            return {"policy": self.policy, "worker_running": bool(self._worker and self._worker.is_alive()),
                    "worker_status": status, "worker_error": self._fatal,
                    "pending": len(self._pending), "active": len(self._active), "completed_requests": len(rows),
                    "rejected_requests": len(self._rejected), "submitted_requests": len(self._ids),
                    "generated_tokens": sum(r["completion_tokens"] for r in rows), "max_active": self._max_active,
                    "max_execution_batch_size": max((r["batch_size"] for r in events), default=0),
                    "peak_live_device_cache_bytes": self._peak_cache_device,
                    "peak_live_host_cache_bytes": self._peak_cache_host,
                    "iterations": len(events), "fatal_error": self._fatal}

    def snapshot(self) -> dict:
        with self._condition:
            return {"requests": list(self._completed), "rejections": list(self._rejected),
                    "iterations": list(self._events), "metrics": self.metrics(), "config": asdict(self.config),
                    "failure_diagnostic": self._failure_diagnostic}

    def _chunks(self, candidates):
        prefill = [r for r in candidates if r.cursor < len(r.prompt_ids)]
        allocation = {r.request_id: 1 for r in prefill}
        remaining = self.config.prefill_token_budget - len(prefill)
        # Arrival order is FIFO; allocating one token first prevents starvation.
        for request in prefill:
            extra = min(remaining, self.config.prefill_chunk_size - 1,
                        len(request.prompt_ids) - request.cursor - 1)
            allocation[request.request_id] += extra
            remaining -= extra
        return allocation

    def _work(self, candidates, chunks):
        return [WorkItem(r.request_id, len(r.prompt_ids) + r.max_tokens,
                         chunks.get(r.request_id, 0), 0 if r.cache is None else int(r.cache[0][0].shape[2]))
                for r in candidates]

    @staticmethod
    def _structure(active, incoming):
        return (tuple((r.request_id, len(r.prompt_ids), r.max_tokens, r.cursor < len(r.prompt_ids)) for r in active),
                tuple(r.request_id for r in incoming))

    def _plan_efficient(self, active, incoming):
        controller = self.controller
        iteration = len(self._events)
        state = self._state
        checks_before = controller.resource_checks
        stats = {'plan_reused': False, 'admission_candidates': 0, 'gate_fallback': False}
        key = self._structure(active, incoming)
        if (active and self.config.reuse_decisions and key == self._last_plan_key
                and iteration - self._last_plan_iteration < self.config.decision_interval):
            chunks = self._chunks(active)
            decision = controller.evaluate_plan(self._work(active, chunks),
                                                Plan(state.quantized_layers, state.cpu_request_ids), state)
            if decision.estimate.feasible:
                stats.update(plan_reused=True, resource_checks=controller.resource_checks-checks_before)
                score = decision.estimate.iteration_ms / len(active)
                return (score, 0, active, chunks, decision), decision, stats
        self._last_plan_iteration = iteration
        full_options = []
        last = None
        candidates_by_count = {}
        for count in range(len(incoming), -1, -1):
            candidates = active + incoming[:count]
            if not candidates:
                continue
            chunks = self._chunks(candidates)
            work = self._work(candidates, chunks)
            candidates_by_count[count] = candidates, chunks, work
            last = controller.evaluate_plan(work, Plan(), state, 'full-precision admission reference')
            if last.estimate.feasible:
                predicted = last.estimate.iteration_ms + last.estimate.transition_ms / self.planner_config.transition_horizon
                score = predicted / len(candidates) + max(0, predicted-self.config.tbt_slo_ms)
                full_options.append((score, -count, candidates, chunks, last))
        options = list(full_options) if controller.policy != 'naive' else []
        if controller.policy != 'full':
            max_full = max((-row[1] for row in full_options), default=0)
            counts = {len(incoming), len(incoming)//2, max_full, min(len(incoming), max_full+1), 0}
            if not active:
                counts.add(1)
            for count in sorted(counts, reverse=True):
                if count not in candidates_by_count:
                    continue
                candidates, chunks, work = candidates_by_count[count]
                last = controller.choose(work, state, time.perf_counter()-self._origin)
                stats['admission_candidates'] += 1
                if last.estimate.feasible:
                    predicted = last.estimate.iteration_ms + last.estimate.transition_ms / self.planner_config.transition_horizon
                    score = predicted / len(candidates) + max(0, predicted-self.config.tbt_slo_ms)
                    options.append((score, -count, candidates, chunks, last))
        stats['resource_checks'] = controller.resource_checks-checks_before
        if not options:
            return None, last, stats
        best = min(options, key=lambda row: (row[0], row[1]))
        if (full_options and controller.policy != 'naive'
                and (best[4].plan.quantized_layers or best[4].plan.cpu_request_ids)):
            reference = min(full_options, key=lambda row: (row[0], row[1]))
            if not controller.supports_change(best[4], len(best[2]), reference[4], len(reference[2])):
                best = reference
                stats['gate_fallback'] = True
        return best, last, stats

    def _resource_profile(self):
        from .telemetry import sample_resources
        telemetry = sample_resources(include_process=not self._efficient)
        memory = self._memory_snapshot if self._efficient else self.executor.memory_report()
        primary_is_cpu = self.executor.device.type == "cpu"
        device = (telemetry["device_free_bytes"] + telemetry["device_allocated_bytes"]
                  if not primary_is_cpu else telemetry["host_available_bytes"] + memory["resident_model_bytes"])
        host = telemetry["host_available_bytes"] + memory["host_master_bytes"]
        # Available RAM excludes this worker's already-owned host KV. Add it
        # back because the planner's reservation includes those same bytes.
        if self._efficient:
            host += sum(int(request.cache[0][0].shape[2]) * self.model.kv_bytes_per_token
                        for request in self._active if request.cache is not None and request.cache[0][0].device.type == 'cpu')
        else:
            host += sum(k.numel() * k.element_size() + v.numel() * v.element_size()
                        for request in self._active if request.cache is not None
                        for k, v in request.cache if k.device.type == "cpu")
        if primary_is_cpu:
            host += memory["resident_model_bytes"]
        if self.config.device_memory_budget_bytes:
            device = min(device, self.config.device_memory_budget_bytes)
        if self.config.host_memory_budget_bytes:
            host = min(host, self.config.host_memory_budget_bytes)
        margin = self._fixed_buffer_bytes + self.config.backend_margin_bytes
        if primary_is_cpu:
            host = max(1, host - margin)
        else:
            host = max(1, host - self.config.backend_margin_bytes)
        return replace(self.profile, device_memory_bytes=max(1, device - margin),
                       host_memory_bytes=max(1, host), device_kind="cpu" if primary_is_cpu else self.profile.device_kind), telemetry

    def _run(self):
        import torch
        try:
            with torch.inference_mode():
                while True:
                    with self._condition:
                        self._active = [r for r in self._active if not r.future.cancelled()]
                        self._pending = deque(r for r in self._pending if not r.future.cancelled())
                        if self._closing and not self._drain:
                            for request in list(self._pending) + self._active:
                                request.future.cancel()
                            self._pending.clear()
                            self._active.clear()
                            break
                        if not self._active and not self._pending:
                            if self._closing:
                                break
                            self._condition.wait(timeout=.1)
                            continue
                    self._tick()
        except BaseException as error:
            with self._condition:
                self._fatal = f"{type(error).__name__}: {error}"
                # Preserve the exact last sampled capacities and current plan
                # before clearing failed requests. No work is done on healthy ticks.
                try:
                    active = list(self._active)
                    work = self._work(active, self._chunks(active))
                    plan = Plan(self._state.quantized_layers, self._state.cpu_request_ids)
                    estimate = self.controller._estimate(work, plan, self._state)
                    self._failure_diagnostic = {
                        'error': self._fatal, 'profile_at_failure': asdict(self.controller.profile),
                        'control_state': asdict(self._state), 'active_work': [asdict(w) for w in work],
                        'pending_request_ids': [r.request_id for r in self._pending],
                        'current_plan_estimate': asdict(estimate), 'iteration': len(self._events)}
                except BaseException as diagnostic_error:
                    self._failure_diagnostic = {'error': self._fatal, 'diagnostic_error': str(diagnostic_error)}
                for request in list(self._pending) + self._active:
                    if not request.future.done():
                        request.future.set_exception(RuntimeError(self._fatal))
                self._active.clear()
                self._pending.clear()
                self._condition.notify_all()

    def _tick(self):
        import torch
        from .batching import BatchInput, execute_batch
        started = time.perf_counter()
        resource, telemetry = self._resource_profile()
        telemetry_finished = time.perf_counter()
        # Preserve online correction state while replacing available capacities.
        self.controller.profile = resource
        if hasattr(self.controller, "estimator"):
            self.controller.estimator.profile = resource
        with self._condition:
            active = list(self._active)
            incoming = list(self._pending)[:self.config.max_batch_size - len(active)]
        options = []
        last_decision = None
        plan_stats = {'plan_reused': False, 'admission_candidates': 0, 'gate_fallback': False}
        # Joint batch admission and execution-plan comparison. Existing requests
        # continue every tick; only admission varies. Never evict a live cache.
        counts = range(len(incoming), -1, -1) if not self._efficient else ()
        if self._efficient:
            choice, last_decision, plan_stats = self._plan_efficient(active, incoming)
            if choice is not None:
                options.append(choice)
        for count in counts:
            candidates = active + incoming[:count]
            if not candidates:
                continue
            chunks = self._chunks(candidates)
            decision = self.controller.choose(self._work(candidates, chunks), self._state,
                                              time.perf_counter() - self._origin)
            last_decision = decision
            if decision.estimate.feasible:
                predicted = decision.estimate.iteration_ms + decision.estimate.transition_ms / self.planner_config.transition_horizon
                # NEO-inspired time per serviced request with an explicit TBT
                # penalty, using measured predictions instead of virtual timing.
                score = predicted / len(candidates) + max(0, predicted - self.config.tbt_slo_ms)
                options.append((score, -count, candidates, chunks, decision))
        if not options:
            if active:
                raise RuntimeError("live reserved work became infeasible: " + (last_decision.reason if last_decision else "unknown"))
            with self._condition:
                bad = self._pending.popleft()
                reason = "no memory/quality-feasible plan for one request"
                self._rejected.append({"request_id": bad.request_id, "reason": reason, "arrival_s": bad.arrival - self._origin})
                if not bad.future.done():
                    bad.future.set_exception(MemoryError(reason))
            return
        _, _, candidates, chunks, decision = min(options, key=lambda row: (row[0], row[1]))
        ids = {r.request_id for r in candidates}
        with self._condition:
            self._pending = deque(r for r in self._pending if r.request_id not in ids)
            for request in candidates:
                if request.admitted is None:
                    request.admitted = time.perf_counter()
            self._active = candidates
            self._max_active = max(self._max_active, len(candidates))
            if self._efficient:
                remaining = list(self._pending)[:self.config.max_batch_size-len(candidates)]
                self._last_plan_key = self._structure(candidates, remaining)
        decision_finished = time.perf_counter()
        if not self._efficient or decision.plan.quantized_layers != self._state.quantized_layers:
            self.executor.set_quantized_layers(decision.plan.quantized_layers)
            if self._efficient:
                self._memory_snapshot = self.executor.memory_report()
        cpu = set(decision.plan.cpu_request_ids)
        inputs = []
        for request in candidates:
            is_prefill = request.request_id in chunks
            if is_prefill:
                tokens = request.prompt_ids[request.cursor:request.cursor + chunks[request.request_id]]
            else:
                tokens = [request.output_ids[-1]]
            # Placement conversion is explicit so its cost belongs to this tick.
            destination = torch.device("cpu") if request.request_id in cpu else self.executor.device
            if request.cache is not None and (not self._efficient or request.cache[0][0].device != destination):
                request.cache = [(k.to(destination), v.to(destination)) for k, v in request.cache]
            inputs.append(BatchInput(request.request_id, tokens, request.cache, request.request_id in cpu, is_prefill))
        if self.executor.device.type == "cuda":
            torch.cuda.synchronize(self.executor.device)
        application_finished = time.perf_counter()
        batch = execute_batch(self.executor, inputs)
        # Optional teacher-forced quality audit on independent held-out targets.
        likelihoods = {}
        scored = []
        for request, output in zip(candidates, batch.outputs):
            will_emit = request.cursor + chunks.get(request.request_id, 0) >= len(request.prompt_ids)
            if will_emit and request.forced_tokens is not None:
                scored.append((request, output))
        if scored:
            from torch.nn import functional as F
            logits = torch.cat([out.logits for _, out in scored], dim=0).float()
            targets = torch.tensor([r.forced_tokens[len(r.output_ids)] for r, _ in scored], device=logits.device)
            nll = F.cross_entropy(logits, targets, reduction="none").cpu().tolist()
            likelihoods = {r.request_id: value for (r, _), value in zip(scored, nll)}
        if self.executor.device.type == "cuda":
            torch.cuda.synchronize(self.executor.device)
        finish = time.perf_counter()
        execution_ms = (finish - application_finished) * 1000
        self.controller.observe(decision, execution_ms)
        changed = (decision.plan.quantized_layers != self._state.quantized_layers
                   or decision.plan.cpu_request_ids != self._state.cpu_request_ids)
        self._state = ControlState(decision.plan.quantized_layers, decision.plan.cpu_request_ids,
                                   finish - self._origin if changed else self._state.last_change_s)
        done = []
        for request, output in zip(candidates, batch.outputs):
            request.cache = output.cache
            if request.request_id in chunks:
                request.cursor += chunks[request.request_id]
                if request.cursor < len(request.prompt_ids):
                    continue
            predicted = output.next_token_id
            token = request.forced_tokens[len(request.output_ids)] if request.forced_tokens is not None else predicted
            request.output_ids.append(token)
            request.predicted_ids.append(predicted)
            request.times.append(finish)
            request.precision.append(list(decision.plan.quantized_layers))
            request.placements.append("cpu" if request.request_id in cpu else str(self.executor.device))
            request.offloaded.append(request.request_id in cpu)
            if request.request_id in likelihoods:
                request.nll.append(likelihoods[request.request_id])
            eos = getattr(self.executor.tokenizer, "eos_token_id", None)
            if len(request.output_ids) == request.max_tokens or (request.stop_on_eos and token == eos):
                done.append(request)
        device_cache, host_cache = 0, 0
        for request in candidates:
            for k, v in request.cache:
                size = k.numel() * k.element_size() + v.numel() * v.element_size()
                if k.device.type == "cpu":
                    host_cache += size
                else:
                    device_cache += size
        event = {"iteration": len(self._events), "start_s": started - self._origin, "end_s": finish - self._origin,
                 "batch_size": len(candidates), "linear_tokens": batch.linear_tokens,
                 "projection_calls": batch.projection_calls, "prefill_tokens": sum(chunks.values()),
                 "decode_requests": len(candidates) - len(chunks), "active_ids": [r.request_id for r in candidates],
                 "quantized_layers": list(decision.plan.quantized_layers), "cpu_request_ids": list(decision.plan.cpu_request_ids),
                 "search_and_admission_ms": (decision_finished - started) * 1000,
                 "telemetry_ms": (telemetry_finished-started)*1000,
                 "planning_ms": (decision_finished-telemetry_finished)*1000,
                 **plan_stats,
                 "application_and_migration_ms": (application_finished - decision_finished) * 1000,
                 "execution_ms": execution_ms, "iteration_wall_ms": (finish - started) * 1000,
                 "predicted_execution_ms": decision.estimate.iteration_ms,
                 "device_reserved_bytes": decision.estimate.device_bytes,
                 "host_reserved_bytes": decision.estimate.host_bytes,
                 "live_device_kv_bytes": device_cache, "live_host_kv_bytes": host_cache,
                 "quality_proxy": decision.estimate.quality_risk, "telemetry": telemetry,
                 "estimate_components": decision.estimate.components}
        with self._condition:
            self._events.append(event)
            self._peak_cache_device = max(self._peak_cache_device, device_cache)
            self._peak_cache_host = max(self._peak_cache_host, host_cache)
            for request in done:
                row = self._finish(request)
                self._completed.append(row)
                if not request.future.done():
                    request.future.set_result(row)
                request.cache = None
            self._active = [r for r in candidates if r not in done]
            alive = {r.request_id for r in self._active}
            self._state = replace(self._state, cpu_request_ids=tuple(x for x in self._state.cpu_request_ids if x in alive))
            self._condition.notify_all()

    def _finish(self, request):
        gaps = [(b - a) * 1000 for a, b in zip(request.times, request.times[1:])]
        ttft = (request.times[0] - request.arrival) * 1000
        elapsed = (request.times[-1] - request.arrival) * 1000
        return {"request_id": request.request_id, "text": self.executor.decode(request.output_ids),
                "token_ids": request.output_ids, "predicted_token_ids": request.predicted_ids,
                "prompt_tokens": len(request.prompt_ids), "completion_tokens": len(request.output_ids),
                "finish_reason": "length" if len(request.output_ids) == request.max_tokens else "stop",
                "arrival_s": request.arrival - self._origin, "admitted_s": request.admitted - self._origin,
                "queue_wait_ms": (request.admitted - request.arrival) * 1000,
                "ttft_ms": ttft, "decode_token_ms": gaps, "tbt_p95_ms": percentile(gaps, .95),
                "elapsed_ms": elapsed, "output_token_times_s": [t - self._origin for t in request.times],
                "precision_history": request.precision, "cacheplacement_history": request.placements,
                "cpu_offload_history": request.offloaded,
                "slo_violated": ttft > self.config.ttft_slo_ms or max(gaps, default=0) > self.config.tbt_slo_ms,
                "evaluation_mode": "teacher_forced_quality_audit" if request.forced_tokens is not None else "greedy_generation",
                "target_nll": request.nll}
