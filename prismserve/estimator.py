"""Conservative capacity accounting and an explicitly analytical roofline model.

This is not a GPU benchmark or a fitted NEO performance model. The overlap term
is an optimistic pipeline approximation; transitions are charged serially.
Reserved maximum contexts also upper-bound attention scans. When exact morphable
layer parameter counts are absent, uniform layer sizes are an approximation.
"""
from __future__ import annotations

import math
from .types import ControlState, Estimate, HardwareProfile, ModelSpec, Plan, PlannerConfig, WorkItem


class Estimator:
    def __init__(self, profile: HardwareProfile, model: ModelSpec, config: PlannerConfig):
        self.profile, self.model, self.config = profile, model, config

    def estimate(self, items: list[WorkItem], plan: Plan, state: ControlState) -> Estimate:
        p, m, cfg = self.profile, self.model, self.config
        ids = {item.request_id for item in items}
        if len(ids) != len(items):
            raise ValueError("duplicate WorkItem request IDs")
        q = set(plan.quantized_layers)
        old_q = set(state.quantized_layers)
        cpu = set(plan.cpu_request_ids)
        old_cpu = set(state.cpu_request_ids)
        if len(q) != len(plan.quantized_layers) or any(i < 0 or i >= m.layers for i in q | old_q):
            raise ValueError("invalid quantized layer indices")
        if len(cpu) != len(plan.cpu_request_ids) or not cpu <= ids:
            raise ValueError("CPU assignment must contain unique IDs present in items")
        if cpu and p.device_kind == "cpu":
            raise ValueError("CPU execution cannot offload attention to itself")
        if any(item.request_id in cpu and item.prefill_tokens for item in items):
            raise ValueError("prefill attention must execute on the primary device")

        kv_per_token = m.kv_bytes_per_token
        gpu_kv = sum(item.context_tokens * kv_per_token for item in items if item.request_id not in cpu)
        host_kv = sum(item.context_tokens * kv_per_token for item in items if item.request_id in cpu)
        def existing(item):
            if item.existing_context_tokens is not None:
                return item.existing_context_tokens
            return 0 if item.prefill_tokens else item.context_tokens

        old_gpu_kv = sum(existing(item) * kv_per_token for item in items
                         if item.request_id not in old_cpu)
        old_host_kv = sum(existing(item) * kv_per_token for item in items
                          if item.request_id in old_cpu)
        migrating = [item for item in items if existing(item)
                     and ((item.request_id in cpu) != (item.request_id in old_cpu))]
        migration_bytes = sum(existing(item) * kv_per_token for item in migrating)
        primary_weights = max(0, m.fp_weight_bytes - sum(m.layer_saving_bytes_for(i) for i in q))
        old_weights = max(0, m.fp_weight_bytes - sum(m.layer_saving_bytes_for(i) for i in old_q))
        # Pinned/restorable master weights and all prepacked variants are charged
        # to host memory even when the current candidate is full precision.
        backups = m.fp_weight_bytes + sum(m.int4_layer_bytes_for(i) for i in range(m.layers))
        compute_tokens = sum(item.prefill_tokens if item.prefill_tokens else 1 for item in items)
        workspace = max(1 << 20, 8 * m.hidden_size * max(1, compute_tokens))
        # The portable reference backend materializes FP weights and integer/
        # float unpack intermediates. Four full layer buffers upper-bound one
        # matrix's scratch. Fused-kernel scenarios must opt out explicitly.
        dequant_scratch = (4 * max((m.fp_layer_bytes_for(i) for i in q), default=0)
                          if m.reference_dequantization else 0)
        workspace += dequant_scratch
        device_steady = primary_weights + gpu_kv + workspace
        # Layer replacement is staged one layer at a time. New request cache is
        # allocated only after morphing. Copies retain existing KV until complete.
        changed = q.symmetric_difference(old_q)
        staging = (max(m.int4_layer_bytes_for(i) if i in q else m.fp_layer_bytes_for(i) for i in changed)
                   if changed else 0)
        device_peak = max(device_steady, old_weights + old_gpu_kv + workspace + staging)
        host_peak = backups + max(host_kv, old_host_kv) + migration_bytes
        if p.device_kind == "cpu":
            # Device and host are one allocation pool, not independent capacities.
            feasible_memory = device_peak + host_peak <= int(p.host_memory_bytes * cfg.memory_fraction)
        else:
            feasible_memory = (device_peak <= int(p.device_memory_bytes * cfg.memory_fraction)
                               and host_peak <= int(p.host_memory_bytes * cfg.memory_fraction))

        fp_transfer = sum(m.fp_layer_bytes_for(i) for i in changed if i not in q)
        int4_transfer = sum(m.int4_layer_bytes_for(i) for i in changed if i in q)
        transport_gbps = p.host_bandwidth_gbps if p.device_kind == "cpu" else p.link_bandwidth_gbps
        transition_ms = (fp_transfer + int4_transfer + migration_bytes) / (transport_gbps * 1e6)
        transition_ms += len(changed) * p.fixed_overhead_ms
        linear_flops = 2 * m.parameters * compute_tokens
        streamed_weights = primary_weights + (sum(m.fp_layer_bytes_for(i) for i in q)
                                               if m.reference_dequantization else 0)
        linear_ms = max(streamed_weights / (p.device_bandwidth_gbps * 1e6),
                        linear_flops / (p.device_tflops * 1e9)) if items else 0.0
        # Roofline attention: bytes scanned once; score and weighted-value FLOPs.
        gpu_decode = [item for item in items if not item.prefill_tokens and item.request_id not in cpu]
        cpu_decode = [item for item in items if item.request_id in cpu]
        gpu_scan = sum(item.context_tokens * kv_per_token for item in gpu_decode)
        cpu_scan = sum(item.context_tokens * kv_per_token for item in cpu_decode)
        gpu_attn_flops = sum(4 * m.layers * m.hidden_size * item.context_tokens for item in gpu_decode)
        cpu_attn_flops = sum(4 * m.layers * m.hidden_size * item.context_tokens for item in cpu_decode)
        prefill_flops = sum(4 * m.layers * m.hidden_size * item.prefill_tokens * item.context_tokens
                            for item in items if item.prefill_tokens)
        gpu_attention_ms = max(gpu_scan / (p.device_bandwidth_gbps * 1e6),
                               (gpu_attn_flops + prefill_flops) / (p.device_tflops * 1e9))
        cpu_attention_ms = max(cpu_scan / (p.host_bandwidth_gbps * 1e6),
                               cpu_attn_flops / (p.host_tflops * 1e9))
        # Decode transfers Q, new K/V, and attention output, not full histories.
        transfer_elements = len(cpu_decode) * m.layers * (2 * m.hidden_size + 2 * m.kv_heads * m.head_dim)
        steady_transfer_bytes = math.ceil(transfer_elements * m.kv_bits / 8)
        steady_transfer_ms = steady_transfer_bytes / (transport_gbps * 1e6)
        iteration_ms = (max(linear_ms + gpu_attention_ms, cpu_attention_ms + steady_transfer_ms)
                        + p.fixed_overhead_ms) if items else 0.0
        quality = sum(m.layer_sensitivities[i] for i in q)
        quality_ok = quality <= cfg.quality_budget + 1e-12
        reasons = []
        if not feasible_memory:
            reasons.append("memory capacity or transition peak exceeded")
        if not quality_ok:
            reasons.append("quality proxy budget exceeded")
        return Estimate(
            device_bytes=device_peak, host_bytes=host_peak,
            iteration_ms=iteration_ms, transition_ms=transition_ms,
            quality_risk=quality, feasible=feasible_memory and quality_ok,
            reason="; ".join(reasons) or "capacity and quality proxy constraints satisfied",
            components={
                "device_steady_bytes": device_steady, "weight_bytes": primary_weights,
                "gpu_kv_bytes": gpu_kv, "cpu_kv_bytes": host_kv,
                "host_backup_bytes": backups, "workspace_bytes": workspace,
                "dequantization_scratch_bytes": dequant_scratch,
                "transition_staging_bytes": staging, "migration_bytes": migration_bytes,
                "steady_transfer_bytes": steady_transfer_bytes,
                "linear_ms": linear_ms, "gpu_attention_ms": gpu_attention_ms,
                "cpu_attention_ms": cpu_attention_ms, "steady_transfer_ms": steady_transfer_ms,
                "memory_feasible": feasible_memory, "quality_proxy_feasible": quality_ok,
                "latency_proxy_feasible": iteration_ms + transition_ms <= cfg.tbt_slo_ms,
                "model": "analytical_roofline_optimistic_overlap",
            },
        )
