"""Validated, dependency-light contracts shared by policy and execution backends.

Bandwidth is decimal GB/s, compute is decimal TFLOP/s, memory is bytes, and
latencies are milliseconds unless a field explicitly ends in ``_s``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math


def _positive(name: str, value: float, *, zero: bool = False) -> None:
    if not math.isfinite(value) or (value < 0 if zero else value <= 0):
        raise ValueError(f"{name} must be finite and {'nonnegative' if zero else 'positive'}")


@dataclass(frozen=True)
class HardwareProfile:
    name: str
    device_memory_bytes: int
    host_memory_bytes: int
    device_bandwidth_gbps: float
    host_bandwidth_gbps: float
    link_bandwidth_gbps: float
    device_tflops: float
    host_tflops: float
    device_kind: str = "cuda"
    source: str = "synthetic"
    fixed_overhead_ms: float = 0.1

    def __post_init__(self) -> None:
        for name in ("device_memory_bytes", "host_memory_bytes", "device_bandwidth_gbps",
                     "host_bandwidth_gbps", "device_tflops", "host_tflops"):
            _positive(name, getattr(self, name))
        _positive("link_bandwidth_gbps", self.link_bandwidth_gbps, zero=True)
        _positive("fixed_overhead_ms", self.fixed_overhead_ms, zero=True)
        if self.device_kind not in ("cpu", "cuda", "mps", "synthetic"):
            raise ValueError("device_kind must be cpu, cuda, mps, or synthetic")
        if self.device_kind != "cpu" and self.link_bandwidth_gbps <= 0:
            raise ValueError("a discrete device requires positive link bandwidth")


@dataclass(frozen=True)
class ModelSpec:
    name: str
    layers: int
    hidden_size: int
    kv_heads: int
    head_dim: int
    parameters: int
    weight_bits: int = 16
    kv_bits: int = 16
    group_size: int = 64
    sensitivity: tuple[float, ...] = ()
    layer_parameters: tuple[int, ...] = ()
    reference_dequantization: bool = False

    def __post_init__(self) -> None:
        for name in ("layers", "hidden_size", "kv_heads", "head_dim", "parameters", "group_size"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.weight_bits not in (8, 16, 32) or self.kv_bits not in (8, 16, 32):
            raise ValueError("weight_bits and kv_bits must be 8, 16, or 32")
        if self.kv_heads * self.head_dim > self.hidden_size:
            raise ValueError("KV width cannot exceed hidden size")
        if self.sensitivity and len(self.sensitivity) != self.layers:
            raise ValueError("sensitivity must have one score per layer")
        for score in self.sensitivity:
            _positive("sensitivity", score, zero=True)
        if self.layer_parameters:
            if (len(self.layer_parameters) != self.layers
                    or any(not isinstance(n, int) or isinstance(n, bool) or n < 0
                           for n in self.layer_parameters)
                    or sum(self.layer_parameters) > self.parameters):
                raise ValueError("layer_parameters must be nonnegative layer counts totaling <= parameters")

    @property
    def fp_weight_bytes(self) -> int:
        return math.ceil(self.parameters * self.weight_bits / 8)

    @property
    def kv_bytes_per_token(self) -> int:
        return self.layers * 2 * self.kv_heads * self.head_dim * self.kv_bits // 8

    @property
    def fp_layer_bytes(self) -> int:
        return math.ceil(self.fp_weight_bytes / self.layers)

    @property
    def int4_layer_bytes(self) -> int:
        parameters = math.ceil(self.parameters / self.layers)
        groups = math.ceil(parameters / self.group_size)
        return math.ceil(groups * self.group_size / 2) + 4 * groups

    @property
    def layer_saving_bytes(self) -> int:
        return max(0, self.fp_layer_bytes - self.int4_layer_bytes)

    def fp_layer_bytes_for(self, index: int) -> int:
        if not 0 <= index < self.layers:
            raise IndexError(index)
        return (self.layer_parameters[index] * self.weight_bits // 8
                if self.layer_parameters else self.fp_layer_bytes)

    def int4_layer_bytes_for(self, index: int) -> int:
        if not 0 <= index < self.layers:
            raise IndexError(index)
        if not self.layer_parameters:
            return self.int4_layer_bytes
        groups = math.ceil(self.layer_parameters[index] / self.group_size)
        return math.ceil(groups * self.group_size / 2) + 4 * groups

    def layer_saving_bytes_for(self, index: int) -> int:
        return max(0, self.fp_layer_bytes_for(index) - self.int4_layer_bytes_for(index))

    @property
    def layer_sensitivities(self) -> tuple[float, ...]:
        # Default is a transparent prior, not a measured accuracy loss.
        return self.sensitivity or tuple(0.04 / self.layers for _ in range(self.layers))


@dataclass(frozen=True)
class RequestSpec:
    request_id: str
    arrival_s: float
    prompt_tokens: int
    output_tokens: int

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("request_id cannot be empty")
        _positive("arrival_s", self.arrival_s, zero=True)
        for name in ("prompt_tokens", "output_tokens"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class WorkItem:
    request_id: str
    context_tokens: int
    prefill_tokens: int = 0
    existing_context_tokens: int | None = None

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("request_id cannot be empty")
        if (not isinstance(self.context_tokens, int) or not isinstance(self.prefill_tokens, int)
                or self.context_tokens <= 0 or self.prefill_tokens < 0):
            raise ValueError("context_tokens must be positive and prefill_tokens nonnegative")
        if self.prefill_tokens > self.context_tokens:
            raise ValueError("prefill chunk cannot exceed reserved context")
        if self.existing_context_tokens is not None and (
                not isinstance(self.existing_context_tokens, int)
                or not 0 <= self.existing_context_tokens <= self.context_tokens):
            raise ValueError("existing_context_tokens must be within the reserved context")


@dataclass(frozen=True)
class ControlState:
    quantized_layers: tuple[int, ...] = ()
    cpu_request_ids: tuple[str, ...] = ()
    last_change_s: float = -1e9


@dataclass(frozen=True)
class Plan:
    quantized_layers: tuple[int, ...] = ()
    cpu_request_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class Estimate:
    device_bytes: int
    host_bytes: int
    iteration_ms: float
    transition_ms: float
    quality_risk: float
    feasible: bool
    reason: str
    components: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Decision:
    plan: Plan
    estimate: Estimate
    policy: str
    search_us: float
    reason: str


@dataclass(frozen=True)
class PlannerConfig:
    quality_budget: float = 0.03
    ttft_slo_ms: float = 2000
    tbt_slo_ms: float = 100
    memory_fraction: float = 0.9
    max_quantized_fraction: float = 0.75
    offload_fractions: tuple[float, ...] = (0, 0.25, 0.5, 0.75, 1)
    cooldown_s: float = 0.1
    transition_penalty: float = 1
    quality_penalty: float = 1
    disable_morph: bool = False
    disable_offload: bool = False
    transition_horizon: int = 16

    def __post_init__(self) -> None:
        for name in ("quality_budget", "cooldown_s", "transition_penalty", "quality_penalty"):
            _positive(name, getattr(self, name), zero=True)
        for name in ("ttft_slo_ms", "tbt_slo_ms", "memory_fraction"):
            _positive(name, getattr(self, name))
        if self.memory_fraction > 1 or not 0 <= self.max_quantized_fraction <= 1:
            raise ValueError("memory_fraction must be <=1 and max_quantized_fraction within [0,1]")
        if not self.offload_fractions or any(not math.isfinite(x) or not 0 <= x <= 1
                                             for x in self.offload_fractions):
            raise ValueError("offload_fractions must be a nonempty sequence within [0,1]")
        if (not isinstance(self.transition_horizon, int) or isinstance(self.transition_horizon, bool)
                or self.transition_horizon <= 0):
            raise ValueError("transition_horizon must be a positive integer")
