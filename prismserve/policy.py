"""Portable finite candidate search over precision and CPU attention placement.

The controller enforces resource and quality-*proxy* constraints. SLO estimates
are preferences: workload queueing and model error prevent a hard latency bound.
"""
from __future__ import annotations

from dataclasses import replace
import math
import time

from .estimator import Estimator
from .types import ControlState, Decision, HardwareProfile, ModelSpec, Plan, PlannerConfig, WorkItem


POLICIES = ("full", "static_int4", "morph_only", "offload_only", "naive", "joint",
            "joint_no_morph", "joint_no_offload", "joint_no_hysteresis")


class Controller:
    def __init__(self, profile: HardwareProfile, model: ModelSpec,
                 config: PlannerConfig | None = None, policy: str = "joint"):
        if policy not in POLICIES:
            raise ValueError(f"unknown policy {policy!r}; expected one of {POLICIES}")
        self.profile, self.model = profile, model
        self.config = config or PlannerConfig()
        self.policy = policy
        self.estimator = Estimator(profile, model, self.config)
        self._order = tuple(sorted(range(model.layers), key=lambda i: (model.layer_sensitivities[i], i)))

    def choose(self, items: list[WorkItem], state: ControlState | None = None,
               now_s: float = 0.0) -> Decision:
        start = time.perf_counter_ns()
        state = state or ControlState()
        cfg, p, m, policy = self.config, self.profile, self.model, self.policy
        allow_morph = not cfg.disable_morph and policy not in ("full", "offload_only", "joint_no_morph")
        allow_offload = (not cfg.disable_offload and p.device_kind != "cpu"
                         and policy not in ("full", "static_int4", "morph_only", "joint_no_offload"))
        max_q = math.floor(m.layers * cfg.max_quantized_fraction + 1e-9) if allow_morph else 0
        q_counts = range(max_q + 1)
        if policy == "static_int4":
            q_counts = (m.layers,) if not cfg.disable_morph else (0,)
        # Short contexts cap host scan cost; candidate family is a heuristic,
        # not an exact search over all 2^N per-request placements.
        eligible = sorted((item for item in items if not item.prefill_tokens),
                          key=lambda item: (item.context_tokens, item.request_id))
        cpu_counts = sorted({0} | {math.ceil(len(eligible) * f) for f in cfg.offload_fractions}) if allow_offload else [0]
        candidates: list[tuple[Plan, object]] = []
        for count in q_counts:
            quantized = tuple(sorted(self._order[:count]))
            for cpu_count in cpu_counts:
                plan = Plan(quantized, tuple(sorted(item.request_id for item in eligible[:cpu_count])))
                estimate = self.estimator.estimate(items, plan, state)
                if policy == "static_int4" and estimate.components["memory_feasible"]:
                    # Baseline intentionally has no adaptive quality constraint;
                    # its proxy violation remains visible in the record.
                    estimate = replace(estimate, feasible=True,
                                       reason="static baseline: memory feasible; quality proxy reported, unconstrained")
                candidates.append((plan, estimate))
        feasible = [(plan, estimate) for plan, estimate in candidates if estimate.feasible]
        if not feasible:
            plan, estimate = min(candidates, key=lambda row: (
                max(0, row[1].device_bytes - p.device_memory_bytes * cfg.memory_fraction)
                + max(0, row[1].host_bytes - p.host_memory_bytes * cfg.memory_fraction),
                row[1].quality_risk, row[1].iteration_ms))
            return Decision(plan, estimate, policy, (time.perf_counter_ns() - start) / 1000,
                            "no feasible candidate; caller must reduce batch or reject")

        def score(row):
            plan, estimate = row
            # Fixed receding horizon amortizes a one-time transfer rather than
            # charging its full cost every future decision and never restoring.
            latency = (estimate.iteration_ms + cfg.transition_penalty * estimate.transition_ms
                       / cfg.transition_horizon)
            quality_cost = cfg.quality_penalty * estimate.quality_risk / max(cfg.quality_budget, 1e-12)
            return (latency / cfg.tbt_slo_ms + quality_cost,
                    estimate.quality_risk, len(plan.cpu_request_ids), plan.quantized_layers)

        # Deliberately simple stacking baseline: maximize CPU fraction first,
        # then use the minimum quantization needed to satisfy capacity.
        if policy == "naive":
            best = min(feasible, key=lambda row: (-len(row[0].cpu_request_ids),
                                                 len(row[0].quantized_layers), row[1].iteration_ms))
        else:
            best = min(feasible, key=score)
        current_cpu = tuple(sorted(item.request_id for item in eligible if item.request_id in state.cpu_request_ids))
        current = Plan(tuple(sorted(state.quantized_layers)), current_cpu)
        same_plan = next((row for row in feasible if row[0] == current), None)
        reason = "minimum estimated latency plus quality and transition cost"
        if (policy not in ("full", "static_int4", "naive", "joint_no_hysteresis")
                and now_s - state.last_change_s < cfg.cooldown_s and same_plan is not None
                and same_plan[1].iteration_ms <= cfg.tbt_slo_ms):
            best = same_plan
            reason = "cooldown retains feasible current plan"
        plan, estimate = best
        return Decision(plan, estimate, policy, (time.perf_counter_ns() - start) / 1000, reason)
