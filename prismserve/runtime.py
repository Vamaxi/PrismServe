"""Bridge the shared planner to real tensors under explicit diagnostic quotas.

This experiment changes a declared capacity quota, not physical GPU occupancy.
No dummy allocations or simulated token times are used. The cost model is
conservative and is not a calibrated performance predictor for the tiny decoder.
"""
from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path
import time

from .experiments import write_json
from .policy import Controller
from .types import ControlState, ModelSpec, Plan, PlannerConfig, WorkItem


def adaptive_demo(output: Path, checkpoint: Path | None = None, device: str = "auto") -> dict:
    import torch
    from .quantization import MorphLinear
    from .telemetry import profile_hardware, sample_resources
    from .torch_backend import generate, load_model, synchronize

    torch.set_num_threads(min(4, torch.get_num_threads()))
    model = load_model(checkpoint, device=device)
    profile = profile_hardware(output / "hardware_profile.json")
    if model.device.type == "cpu":
        profile = replace(profile, device_kind="cpu", device_memory_bytes=profile.host_memory_bytes,
                          device_bandwidth_gbps=profile.host_bandwidth_gbps,
                          device_tflops=profile.host_tflops, link_bandwidth_gbps=0)
    per_layer = tuple(sum(m.in_features * m.out_features for m in layer.modules() if isinstance(m, MorphLinear))
                      for layer in model.layers)
    spec = ModelSpec("real-tiny-byte-decoder", layers=model.config.layers,
                     hidden_size=model.config.hidden_size, kv_heads=model.config.heads,
                     head_dim=model.config.hidden_size // model.config.heads,
                     parameters=sum(p.numel() for p in model.parameters()), weight_bits=32, kv_bits=32,
                     group_size=model.config.group_size, layer_parameters=per_layer,
                     reference_dequantization=True)
    config = PlannerConfig(max_quantized_fraction=1, memory_fraction=1, cooldown_s=0,
                           quality_budget=.04, quality_penalty=1)
    normal = replace(profile, device_memory_bytes=max(8 * 2**20, spec.fp_weight_bytes * 8))
    state = ControlState()
    rows = []
    request_id = "real-request"
    prompt = "The serving controller observes memory and chooses a plan. "
    # Reserve more output than is exercised, so a tighter quota can affect future
    # admission without pretending that already-live bytes disappeared.
    reserved_tokens = model.config.max_seq_len - 1
    started = time.perf_counter()

    def control(current, cache, token_index):
        nonlocal state
        telemetry = sample_resources()
        existing = cache[0][0].shape[2]
        item = WorkItem(request_id, reserved_tokens, existing_context_tokens=existing)
        phase = "capacity_available" if token_index < 8 or token_index >= 20 else "declared_quota_tight"
        # Update available physical budgets on every token boundary. Add this
        # process's allocations back to free VRAM because estimates include them.
        physical_device_budget = (telemetry["device_free_bytes"] + telemetry["device_allocated_bytes"]
                                  if model.device.type == "cuda" else telemetry["host_available_bytes"])
        chosen_profile = replace(normal, device_memory_bytes=min(normal.device_memory_bytes, physical_device_budget),
                                 host_memory_bytes=telemetry["host_available_bytes"])
        if phase == "declared_quota_tight" and model.device.type == "cuda":
            base = Controller(normal, spec, config)
            full = base.estimator.estimate([item], Plan(), state)
            host = base.estimator.estimate([item], Plan((), (request_id,)), state)
            # Include migration peaks. Only impose a quota that can genuinely
            # admit CPU placement according to the same capacity model.
            quota = max(host.device_bytes, (host.device_bytes + full.device_bytes) // 2)
            chosen_profile = replace(chosen_profile, device_memory_bytes=min(chosen_profile.device_memory_bytes, quota))
        decision = Controller(chosen_profile, spec, config).choose([item], state, time.perf_counter() - started)
        if not decision.estimate.feasible:
            raise RuntimeError("diagnostic runtime plan infeasible: " + decision.estimate.reason)
        before = time.perf_counter()
        current.apply_plan(decision.plan, request_id)
        destination = torch.device("cpu") if current.cpu_attention else current.device
        cache = [(k.to(destination), v.to(destination)) for k, v in cache]
        synchronize(current.device)
        changed = decision.plan.quantized_layers != state.quantized_layers or decision.plan.cpu_request_ids != state.cpu_request_ids
        state = ControlState(decision.plan.quantized_layers, decision.plan.cpu_request_ids,
                             time.perf_counter() - started if changed else state.last_change_s)
        rows.append({"token_index": token_index, "phase": phase,
                     "quota_bytes": chosen_profile.device_memory_bytes,
                     "existing_cache_tokens": existing, "reserved_context_tokens": reserved_tokens,
                     "plan": asdict(decision.plan), "estimate": asdict(decision.estimate),
                     "planner_search_us": decision.search_us,
                     "apply_and_migrate_ms_measured": (time.perf_counter() - before) * 1000,
                     "cache_device": str(cache[0][0].device), "telemetry": telemetry,
                     **current.memory_report()})
        return cache

    generate(model, "Warm up", 3)
    result = generate(model, prompt, 32, callback=control)
    payload = {"measurement_kind": "real_adaptive_tiny_decoder_with_declared_quotas",
               "checkpoint": str(checkpoint) if checkpoint else None,
               "device": str(model.device), "hardware": asdict(profile), "model": asdict(spec),
               "generation": result, "decisions": rows,
               "notes": ["Capacity quotas are a deterministic diagnostic; physical GPU memory was not filled.",
                         "Every token is really executed, including CPU/device KV migration when selected.",
                         "Planner estimates use conservative reference dequantization scratch and optimistic overlap; measured timings are separate.",
                         "Precision may stay full because reference INT4 has scratch and unpack costs. Static mechanism demonstrations live in demo.",
                         "On CPU the controller correctly suppresses host offload."]}
    write_json(output / "adaptive_runtime.json", payload)
    return {"output": str((output / "adaptive_runtime.json").resolve()), "device": str(model.device),
            "cpu_attention_tokens": sum(bool(r["plan"]["cpu_request_ids"]) for r in rows),
            "quantized_tokens": sum(bool(r["plan"]["quantized_layers"]) for r in rows),
            "ttft_ms_measured": result["ttft_ms"], "elapsed_ms_measured": result["elapsed_ms"]}
