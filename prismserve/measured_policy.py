"""Measured batch latency lookup with bounded feedback and resource accounting.

Latency features use current contexts, while capacity reserves future contexts.
Calibration predicts synchronous reference execution: no analytical CPU/GPU
overlap or fused-INT4 speedup is substituted for measured batch timings.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
import time
from typing import Any

from .estimator import Estimator
from .types import ControlState, Decision, Estimate, HardwareProfile, ModelSpec, Plan, PlannerConfig, WorkItem


def _integer(name: str, value: Any, minimum: int = 0) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _finite(name: str, value: float, positive: bool = False) -> None:
    if not math.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")


@dataclass(frozen=True)
class TimingSample:
    quantized_layers: tuple[int, ...]
    batch_size: int
    prefill_tokens: int
    context_tokens: int
    cpu_requests: int
    elapsed_ms: float
    workspace_bytes: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "quantized_layers", tuple(sorted(self.quantized_layers)))
        if len(set(self.quantized_layers)) != len(self.quantized_layers):
            raise ValueError("quantized_layers must contain unique indices")
        for index in self.quantized_layers:
            _integer("quantized layer", index)
        _integer("batch_size", self.batch_size, 1)
        _integer("prefill_tokens", self.prefill_tokens)
        _integer("context_tokens", self.context_tokens)
        _integer("cpu_requests", self.cpu_requests)
        _integer("workspace_bytes", self.workspace_bytes)
        if self.cpu_requests > self.batch_size:
            raise ValueError("cpu_requests cannot exceed batch_size")
        _finite("elapsed_ms", self.elapsed_ms, positive=True)


@dataclass(frozen=True)
class TimingCalibration:
    samples: tuple[TimingSample, ...]
    to_int4_layer_ms: float = 0
    to_fp_layer_ms: float = 0
    source: str = "measured_actual_batches"

    def __post_init__(self) -> None:
        object.__setattr__(self, "samples", tuple(self.samples))
        if any(not isinstance(sample, TimingSample) for sample in self.samples):
            raise ValueError("samples must be TimingSample objects")
        _finite("to_int4_layer_ms", self.to_int4_layer_ms)
        _finite("to_fp_layer_ms", self.to_fp_layer_ms)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict) -> "TimingCalibration":
        return cls(samples=tuple(TimingSample(**sample) for sample in value["samples"]),
                   to_int4_layer_ms=value.get("to_int4_layer_ms", 0),
                   to_fp_layer_ms=value.get("to_fp_layer_ms", 0),
                   source=value.get("source", "measured_actual_batches"))


class MeasuredController:
    """Finite candidate search; measured timing plus EWMA, no exploration/RL.

    ``prefill_tokens`` is total batch prefill work and ``context_tokens`` is
    maximum current per-sequence context. Samples must match exact quantized
    layer sets and the presence/absence of CPU attention. Unknown families are
    infeasible rather than silently receiving roofline timings. Extrapolation
    only scales up: it is deliberately conservative, not an accuracy guarantee.
    """
    POLICIES = ("full", "morph_only", "offload_only", "naive", "joint", "joint_uncalibrated")

    def __init__(self, profile: HardwareProfile, model: ModelSpec, config: PlannerConfig,
                 calibration: TimingCalibration, policy: str = "joint",
                 quantization_candidates: tuple[tuple[int, ...], ...] = ((),), *,
                 ewma_alpha: float = .2, workspace_safety_factor: float = 1.25):
        if policy not in self.POLICIES:
            raise ValueError(f"unknown measured policy {policy!r}")
        if not 0 < ewma_alpha <= 1 or not math.isfinite(ewma_alpha):
            raise ValueError("ewma_alpha must be within (0,1]")
        if not math.isfinite(workspace_safety_factor) or workspace_safety_factor < 1.2:
            raise ValueError("workspace_safety_factor must be >= 1.2")
        candidates = {()}
        for candidate in quantization_candidates:
            if len(set(candidate)) != len(candidate):
                raise ValueError("quantization candidates cannot contain duplicate indices")
            if any(not isinstance(index, int) or isinstance(index, bool)
                   or not 0 <= index < model.layers for index in candidate):
                raise ValueError("quantization candidate contains an invalid layer")
            candidates.add(tuple(sorted(candidate)))
        if any(index >= model.layers for sample in calibration.samples for index in sample.quantized_layers):
            raise ValueError("calibration contains invalid model layer indices")
        self.profile, self.model, self.config = profile, model, config
        self.calibration, self.policy = calibration, policy
        self.quantization_candidates = tuple(sorted(candidates, key=lambda q: (len(q), q)))
        self.ewma_alpha, self.workspace_safety_factor = ewma_alpha, workspace_safety_factor
        # Measured workspace replaces the reference unpacker's gross full-layer
        # scratch bound. The original Estimator and analytical experiments stay
        # unchanged. The measured controller still charges all host backups.
        self.estimator = Estimator(profile, replace(model, reference_dequantization=False), config)
        self._feedback: dict[tuple[tuple[int, ...], bool], float] = {}

    @staticmethod
    def _features(items: list[WorkItem], cpu_count: int) -> dict:
        contexts = []
        for item in items:
            existing = item.existing_context_tokens
            if existing is None:
                existing = 0 if item.prefill_tokens else item.context_tokens
            contexts.append(min(item.context_tokens, existing + (item.prefill_tokens or 1)))
        return {"batch_size": len(items), "prefill_tokens": sum(item.prefill_tokens for item in items),
                "context_tokens": max(contexts, default=0), "cpu_requests": cpu_count}

    def _lookup(self, layers: tuple[int, ...], features: dict) -> tuple[TimingSample | None, float, float]:
        family = [sample for sample in self.calibration.samples
                  if sample.quantized_layers == layers
                  and bool(sample.cpu_requests) == bool(features["cpu_requests"])]
        if not family:
            return None, 0.0, 0.0
        def distance(sample):
            shape = (sample.batch_size, sample.prefill_tokens, sample.context_tokens, sample.cpu_requests)
            target = tuple(features[name] for name in ("batch_size", "prefill_tokens", "context_tokens", "cpu_requests"))
            return (sum(abs(math.log((a + 1) / (b + 1))) for a, b in zip(shape, target)),
                    sample.batch_size, sample.context_tokens, sample.prefill_tokens, sample.cpu_requests)
        nearest = min(family, key=distance)
        work = max(features["batch_size"], features["prefill_tokens"])
        reference_work = max(nearest.batch_size, nearest.prefill_tokens)
        cache_scale = (features["batch_size"] * max(1, features["context_tokens"])
                       / (nearest.batch_size * max(1, nearest.context_tokens)))
        scale = max(1.0, features["batch_size"] / nearest.batch_size,
                    work / reference_work, cache_scale,
                    features["cpu_requests"] / max(1, nearest.cpu_requests))
        return nearest, nearest.elapsed_ms * scale, scale

    def _estimate(self, items: list[WorkItem], plan: Plan, state: ControlState) -> Estimate:
        self.estimator.profile = self.profile
        base = self.estimator.estimate(items, plan, state)
        features = self._features(items, len(plan.cpu_request_ids))
        sample, lookup_ms, shape_scale = self._lookup(plan.quantized_layers, features)
        key = (plan.quantized_layers, bool(plan.cpu_request_ids))
        correction = self._feedback.get(key, 1.0)
        uncalibrated = self.policy == "joint_uncalibrated"
        known = sample is not None or not items
        predicted_ms = base.iteration_ms if uncalibrated else lookup_ms * correction
        # Capacity must remain safe as admitted contexts grow. A workspace
        # lookup at the *current* length can admit work that later fails despite
        # reserving its final KV. Fit a monotone upper envelope: per-precision
        # fixed scratch plus one extra copy of all reserved device KV. Subtract
        # each sample's new KV from its incremental peak to isolate scratch.
        # The bound is independent of the changing nearest latency sample.
        family = [row for row in self.calibration.samples if row.quantized_layers == plan.quantized_layers]
        kv_bytes = self.model.kv_bytes_per_token
        residuals = [(row.batch_size, max(0, row.workspace_bytes
                                           - (row.batch_size - row.cpu_requests) * row.context_tokens * kv_bytes))
                     for row in family]
        smallest_batch = min((b for b, _ in residuals), default=1)
        largest_batch = max((b for b, _ in residuals), default=1)
        fixed_scratch = max((value for b, value in residuals if b == smallest_batch), default=0)
        per_extra_request = max((max(0, value - fixed_scratch) / (b - smallest_batch)
                                 for b, value in residuals if b > smallest_batch), default=0)
        scratch_bound = (fixed_scratch + per_extra_request * max(0, len(items) - smallest_batch)
                         + fixed_scratch * max(0, len(items) - largest_batch) / smallest_batch)
        gpu_reserved = sum(item.context_tokens * kv_bytes for item in items if item.request_id not in plan.cpu_request_ids)
        cpu_reserved = sum(item.context_tokens * kv_bytes for item in items if item.request_id in plan.cpu_request_ids)
        measured_workspace = math.ceil((scratch_bound + gpu_reserved) * self.workspace_safety_factor)
        base_workspace = base.components["workspace_bytes"]
        workspace_delta = max(0, measured_workspace - base_workspace)
        device_peak = base.device_bytes + workspace_delta
        # The ragged reference keeps source and concatenated host KV alive
        # during a tick, just as it does for device histories.
        host_peak = base.host_bytes + cpu_reserved
        cfg, p = self.config, self.profile
        if p.device_kind == "cpu":
            memory_ok = device_peak + host_peak <= int(p.host_memory_bytes * cfg.memory_fraction)
        else:
            memory_ok = (device_peak <= int(p.device_memory_bytes * cfg.memory_fraction)
                         and host_peak <= int(p.host_memory_bytes * cfg.memory_fraction))
        q, old_q = set(plan.quantized_layers), set(state.quantized_layers)
        layer_transition_ms = (len(q - old_q) * self.calibration.to_int4_layer_ms
                               + len(old_q - q) * self.calibration.to_fp_layer_ms)
        transport = p.host_bandwidth_gbps if p.device_kind == "cpu" else p.link_bandwidth_gbps
        migration_ms = base.components["migration_bytes"] / (transport * 1e6)
        transition_ms = layer_transition_ms + migration_ms
        quality_ok = base.components["quality_proxy_feasible"]
        timing_ok = known or uncalibrated
        feasible = memory_ok and quality_ok and timing_ok
        reasons = []
        if not memory_ok:
            reasons.append("measured workspace, reservation, or transition peak exceeds capacity")
        if not quality_ok:
            reasons.append("quality proxy budget exceeded; not an accuracy guarantee")
        if not timing_ok:
            reasons.append("missing measured timing family for precision and CPU placement")
        return replace(base, device_bytes=device_peak, host_bytes=host_peak,
                       iteration_ms=predicted_ms, transition_ms=transition_ms,
                       feasible=feasible, reason="; ".join(reasons) or "measured timing with feasible resource reservations",
                       components={**base.components, "memory_feasible": memory_ok,
                           "analytical_iteration_ms_ignored": base.iteration_ms,
                           "timing_source": "analytical_uncalibrated_baseline" if uncalibrated else self.calibration.source,
                           "timing_features": features, "timing_family_available": known,
                           "nearest_sample": None if sample is None else asdict(sample),
                           "timing_shape_scale": shape_scale,
                           "lookup_base_ms": lookup_ms, "ewma_correction": correction,
                           "workspace_source": "calibrated_fixed_scratch_plus_reserved_kv_copy",
                           "workspace_bytes": max(base_workspace, measured_workspace),
                           "measured_workspace_bytes": measured_workspace,
                           "calibrated_fixed_scratch_bytes": fixed_scratch,
                           "calibrated_scratch_bound_bytes": scratch_bound,
                           "reserved_device_kv_copy_bytes": gpu_reserved,
                           "reserved_host_kv_copy_bytes": cpu_reserved,
                           "workspace_shape_scale": shape_scale,
                           "workspace_safety_factor": self.workspace_safety_factor,
                           "measured_layer_transition_ms": layer_transition_ms,
                           "migration_transition_ms": migration_ms,
                           "latency_proxy_feasible": predicted_ms + transition_ms <= cfg.tbt_slo_ms,
                           "model": "measured_batch_table_with_resource_reservations"})

    def choose(self, items: list[WorkItem], state: ControlState | None = None, now_s: float = 0.0) -> Decision:
        started = time.perf_counter_ns()
        state = state or ControlState()
        cfg, policy = self.config, self.policy
        allow_morph = not cfg.disable_morph and policy not in ("full", "offload_only")
        allow_offload = (not cfg.disable_offload and self.profile.device_kind != "cpu"
                         and policy not in ("full", "morph_only"))
        max_quantized = math.floor(cfg.max_quantized_fraction * self.model.layers + 1e-9)
        layer_sets = [q for q in self.quantization_candidates if (allow_morph or not q) and len(q) <= max_quantized]
        eligible = sorted((item for item in items if not item.prefill_tokens),
                          key=lambda item: ((item.existing_context_tokens
                                             if item.existing_context_tokens is not None else item.context_tokens),
                                            item.request_id))
        cpu_counts = ({0} | {math.ceil(len(eligible) * f) for f in cfg.offload_fractions}) if allow_offload else {0}
        plans = {Plan(q, tuple(sorted(item.request_id for item in eligible[:count])))
                 for q in layer_sets for count in cpu_counts}
        # Preserve an existing valid placement during cooldown even when the
        # active-set size changed and it no longer matches a fraction boundary.
        current = Plan(tuple(sorted(state.quantized_layers)),
                       tuple(sorted(item.request_id for item in eligible if item.request_id in state.cpu_request_ids)))
        if current.quantized_layers in layer_sets and (allow_offload or not current.cpu_request_ids):
            plans.add(current)
        evaluated = [(plan, self._estimate(items, plan, state))
                     for plan in sorted(plans, key=lambda p: (p.quantized_layers, p.cpu_request_ids))]
        feasible = [row for row in evaluated if row[1].feasible]
        if not feasible:
            def violation(row):
                estimate = row[1]
                p = self.profile
                if p.device_kind == "cpu":
                    excess = max(0, estimate.device_bytes + estimate.host_bytes - p.host_memory_bytes * cfg.memory_fraction)
                else:
                    excess = (max(0, estimate.device_bytes - p.device_memory_bytes * cfg.memory_fraction)
                              + max(0, estimate.host_bytes - p.host_memory_bytes * cfg.memory_fraction))
                return (not estimate.components["timing_family_available"], excess,
                        estimate.quality_risk, estimate.iteration_ms)
            plan, estimate = min(evaluated, key=violation)
            return Decision(plan, estimate, policy, (time.perf_counter_ns() - started) / 1000,
                            "no measured feasible candidate; reduce admission or reject")
        def score(row):
            plan, estimate = row
            latency = estimate.iteration_ms + cfg.transition_penalty * estimate.transition_ms / cfg.transition_horizon
            quality = cfg.quality_penalty * estimate.quality_risk / max(cfg.quality_budget, 1e-12)
            return (latency / cfg.tbt_slo_ms + quality, estimate.quality_risk,
                    len(plan.cpu_request_ids), plan.quantized_layers, plan.cpu_request_ids)
        if policy == "naive":
            best = min(feasible, key=lambda row: (-len(row[0].cpu_request_ids), len(row[0].quantized_layers), score(row)))
        else:
            best = min(feasible, key=score)
        reason = "minimum measured latency, amortized transition, and quality proxy cost"
        current_row = next((row for row in feasible if row[0] == current), None)
        if (policy not in ("full", "naive") and now_s - state.last_change_s < cfg.cooldown_s
                and current_row is not None and current_row[1].iteration_ms <= cfg.tbt_slo_ms):
            best = current_row
            reason = "cooldown retains a measured feasible current plan"
        plan, estimate = best
        return Decision(plan, estimate, policy, (time.perf_counter_ns() - started) / 1000, reason)

    def observe(self, decision: Decision, elapsed_ms: float) -> None:
        """Update per-precision/CPU-family EWMA using execution-only wall time.

        Transition time belongs to a separate measurement and must not be passed
        here. Clipping the ratio to [0.05,20] limits one outlier's influence.
        Unchosen candidates are never updated and no exploration is performed.
        """
        _finite("elapsed_ms", elapsed_ms, positive=True)
        baseline = decision.estimate.components.get("lookup_base_ms", 0)
        if not decision.estimate.feasible or baseline <= 0 or self.policy == "joint_uncalibrated":
            return
        key = (decision.plan.quantized_layers, bool(decision.plan.cpu_request_ids))
        ratio = min(20.0, max(.05, elapsed_ms / baseline))
        old = self._feedback.get(key, 1.0)
        self._feedback[key] = (1 - self.ewma_alpha) * old + self.ewma_alpha * ratio
