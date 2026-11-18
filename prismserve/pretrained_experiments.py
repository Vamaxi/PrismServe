"""Measured pretrained-model experiments; no analytical token timings here."""
from __future__ import annotations

from dataclasses import asdict, replace
from datetime import datetime, timezone
import gc
import importlib.metadata
import json
import math
from pathlib import Path
from statistics import mean
import time

from .experiments import write_json
from .pretrained_assets import sha256_file
from .simulator import percentile

PROMPTS = (
    "Artificial intelligence can help scientists understand",
    "The purpose of a computer operating system is to",
    "In a quiet village near the mountains, a young woman discovered",
)


def _sync(device):
    import torch
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _data(executor, data_path: Path, split: str, tokens: int):
    path = data_path / f"wiki.{split}.prefix.txt"
    if not path.exists():
        raise FileNotFoundError("WikiText prefixes are missing. Run pretrained-prepare --dataset, or supply --data-path.")
    ids = executor.tokenize(path.read_text(encoding="utf-8"))[:tokens]
    if len(ids) < min(tokens, 256):
        raise ValueError("evaluation input is too short")
    return ids, {"dataset": "WikiText-2 raw v1", "split": split,
                 "selection": "First 100 dataset rows, joined with two newlines; then fixed token prefix",
                 "file": str(path), "sha256": sha256_file(path), "input_tokens": len(ids)}


def calibrate(executor, ids: list[int]) -> dict:
    """Measure one-layer perturbations on validation data, never test labels."""
    import torch
    from torch.nn import functional as F
    executor.set_quantized_layers(())
    executor.set_cpu_attention(False)
    batch = torch.tensor([ids], device=executor.device)
    with torch.inference_mode():
        reference, _ = executor(batch)
        probability = reference.float().softmax(-1)
        rows = []
        for index in range(len(executor.layers)):
            executor.set_quantized_layers((index,))
            quantized, _ = executor(batch)
            kl = F.kl_div(quantized.float().log_softmax(-1), probability, reduction="sum") / batch.numel()
            rows.append({"layer": index, "mean_logit_kl": max(float(kl), 0)})
        executor.set_quantized_layers(())
        restored, _ = executor(batch)
        error = float((reference - restored).abs().max())
    return {"method": "one-layer validation logit KL relative to FP32; additive use is a surrogate, not a quality guarantee",
            "input_tokens": len(ids), "layers": rows,
            "layer_order": [r["layer"] for r in sorted(rows, key=lambda x: (x["mean_logit_kl"], x["layer"]))],
            "restore_max_abs_logit_error": error}


def evaluate_nll(executor, ids: list[int], context: int = 256, stride: int = 128) -> dict:
    """Sliding-window likelihood; each target after token zero is scored once."""
    import torch
    from torch.nn import functional as F
    if not 1 <= stride < context <= executor.context_limit:
        raise ValueError("require 1 <= stride < context <= model context")
    total_nll, count, previous_end = 0.0, 0, 0
    _sync(executor.device)
    started = time.perf_counter()
    with torch.inference_mode():
        for begin in range(0, len(ids), stride):
            end = min(begin + context, len(ids))
            target_length = end - previous_end
            batch = torch.tensor([ids[begin:end]], device=executor.device)
            logits, _ = executor(batch)
            first_target = max(1, batch.shape[1] - target_length)
            relevant = logits[:, first_target - 1:-1].float()
            targets = batch[:, first_target:]
            total_nll += float(F.cross_entropy(relevant.reshape(-1, relevant.shape[-1]),
                                              targets.reshape(-1), reduction="sum"))
            count += targets.numel()
            previous_end = end
            if end == len(ids):
                break
    _sync(executor.device)
    if count != len(ids) - 1:
        raise RuntimeError("sliding likelihood counted a target more than once or missed one")
    nll = total_nll / count
    return {"nll": nll, "perplexity": math.exp(nll), "scored_tokens": count,
            "context": context, "stride": stride, "target_accounting": "Each next-token target scored exactly once; first token unscored",
            "elapsed_s_measured": time.perf_counter() - started}


def verify_parity(executor, prompt: str = PROMPTS[0]) -> dict:
    import torch
    executor.set_quantized_layers(())
    executor.set_cpu_attention(False)
    ids = executor.tokenize(prompt)
    inputs = torch.tensor([ids], device=executor.device)
    with torch.inference_mode():
        # Original Transformers forward remains functional after wrapping the
        # Conv1D storage; it uses its own attention and cache implementations.
        original = executor.model(inputs, use_cache=False).logits
        custom, cache = executor(inputs)
        full_error = float((original - custom).abs().max())
        continuation = torch.tensor([[int(custom[0, -1].argmax())]], device=executor.device)
        cached, cache = executor(continuation, cache)
        full, _ = executor(torch.cat((inputs, continuation), dim=1))
        cached_error = float((cached[:, -1] - full[:, -1]).abs().max())
        executor.set_cpu_attention(True)
        cpu_cache = [(k.cpu(), v.cpu()) for k, v in cache]
        continuation2 = torch.tensor([[int(cached[0, -1].argmax())]], device=executor.device)
        offloaded, _ = executor(continuation2, cpu_cache)
        executor.set_cpu_attention(False)
        on_device, _ = executor(continuation2, [(k.to(executor.device), v.to(executor.device)) for k, v in cache])
        placement_error = float((offloaded - on_device).abs().max())
    # Allow small CPU/GPU accumulation differences in a real FP32 transformer.
    limits = {"original_forward": 0.002, "cached_decode": 0.002, "cpu_attention": 0.002}
    passed = full_error < limits["original_forward"] and cached_error < limits["cached_decode"] and placement_error < limits["cpu_attention"]
    result = {"measurement_kind": "real_pretrained_correctness", "device": str(executor.device),
              "hf_forward_max_abs_error": full_error, "cached_decode_max_abs_error": cached_error,
              "cpu_attention_max_abs_error": placement_error, "absolute_tolerances": limits, "passed": passed,
              "note": "FP32 custom path compared with Transformers eager attention on the same actual checkpoint"}
    if not passed:
        raise RuntimeError("pretrained correctness verification failed: " + json.dumps(result))
    return result


def _benchmark_variant(executor, variant: str, selected: tuple[int, ...], cpu: bool,
                       repeats: int, max_tokens: int) -> dict:
    import torch
    started = time.perf_counter()
    executor.set_quantized_layers(selected)
    executor.set_cpu_attention(cpu)
    _sync(executor.device)
    setup_ms = (time.perf_counter() - started) * 1000
    executor.generate("The computer", 3, stop_on_eos=False)  # excluded warmup
    if executor.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(executor.device)
    rows = []
    for prompt_id, prompt in enumerate(PROMPTS):
        for repeat in range(repeats):
            result = executor.generate(prompt, max_tokens, stop_on_eos=False)
            rows.append({"prompt_id": prompt_id, "prompt": prompt, "repeat": repeat, **result})
    gaps = [gap for row in rows for gap in row["decode_token_ms"]]
    return {"variant": variant, "device": str(executor.device), "quantized_layers": list(selected),
            "cpu_attention": cpu, "cpu_offload_effective": cpu and executor.device.type == "cuda",
            "ttft_mean_ms": mean(r["ttft_ms"] for r in rows),
            "ttft_p95_ms": percentile([r["ttft_ms"] for r in rows], .95),
            "tbt_mean_ms": mean(gaps), "tbt_p95_ms": percentile(gaps, .95),
            "output_tokens_per_s": sum(r["completion_tokens"] for r in rows) / (sum(r["elapsed_ms"] for r in rows) / 1000),
            "variant_setup_ms_excluded": setup_ms,
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(executor.device) if executor.device.type == "cuda" else None,
            "cuda_reserved_bytes": torch.cuda.memory_reserved(executor.device) if executor.device.type == "cuda" else None,
            "memory": executor.memory_report(), "runs": rows}


def benchmark_pretrained(model_path: Path, data_path: Path, output: Path, device: str = "auto",
                         quality_tokens: int = 4096, repeats: int = 3, max_tokens: int = 32) -> dict:
    import torch
    from torch.nn import functional as F
    from .pretrained import GPT2Executor
    if quality_tokens < 256 or repeats < 1 or max_tokens < 2:
        raise ValueError("require quality_tokens>=256, repeats>=1, max_tokens>=2")
    torch.set_num_threads(min(4, torch.get_num_threads()))
    torch.manual_seed(7)
    output.mkdir(parents=True, exist_ok=True)
    print("Loading local pretrained GPT-2...", flush=True)
    executor = GPT2Executor.from_local(model_path, device)
    print("Checking original-model and CPU-attention parity...", flush=True)
    parity = verify_parity(executor)
    write_json(output / "parity.json", parity)
    validation_ids, validation_info = _data(executor, data_path, "validation", 128)
    test_ids, test_info = _data(executor, data_path, "test", quality_tokens)
    print("Calibrating one-layer quality sensitivity on validation only...", flush=True)
    calibration = calibrate(executor, validation_ids)
    calibration["data"] = validation_info
    write_json(output / "calibration.json", calibration)
    selected = tuple(sorted(calibration["layer_order"][:3]))
    variants = [("fp32", (), False), ("int4_3_layers", selected, False),
                ("cpu_attention", (), True), ("hybrid_3_layers", selected, True),
                ("static_int4_all", tuple(range(len(executor.layers))), False)]
    measurements = []
    quality_rows = []
    with torch.inference_mode():
        batch = torch.tensor([test_ids[:128]], device=executor.device)
        baseline_logits, _ = executor(batch)
        baseline_probability = baseline_logits.float().softmax(-1)
        del baseline_logits
        for name, layers, offload in variants:
            print(f"Measuring {name}: real generation and WikiText-2 subset likelihood...", flush=True)
            measurements.append(_benchmark_variant(executor, name, layers, offload, repeats, max_tokens))
            # NLL uses fresh windows on primary-device prefill. Persistent CPU
            # decode quality is checked by parity, not mislabeled full-CPU NLL.
            nll = evaluate_nll(executor, test_ids)
            logits, _ = executor(batch)
            kl = float(F.kl_div(logits.float().log_softmax(-1), baseline_probability, reduction="sum") / batch.numel())
            quality_rows.append({"variant": name, "quantized_layers": list(layers), **nll,
                                 "diagnostic_kl_first_128_test_tokens": max(kl, 0),
                                 "memory": executor.memory_report()})
            del logits
        executor.set_quantized_layers(())
        executor.set_cpu_attention(False)
        restored, _ = executor(batch)
        # Regenerate the original unquantized baseline for direct exact checking.
        original = executor.model(batch, use_cache=False).logits
        restoration_error = float((original - restored).abs().max())
    base_nll = quality_rows[0]["nll"]
    for row in quality_rows:
        row["relative_perplexity_change_pct"] = 100 * (math.exp(row["nll"] - base_nll) - 1)
    quality = {"measurement_kind": "real_pretrained_wikitext2_subset", "model": "GPT-2", "tokenizer": "original GPT-2 BPE",
               "data": test_info, "calibration_data": validation_info, "variants": quality_rows,
               "restored_vs_hf_max_abs_logit_error": restoration_error,
               "notes": ["Subset diagnostic; not the full canonical WikiText-2 benchmark.",
                         "Single-layer calibration uses validation; held-out test is never used for layer selection.",
                         "Sliding windows preserve context up to256 tokens and count every target once.",
                         "CPU-placement variants evaluate fresh prefill; decode equivalence is tested separately.",
                         "Static all-layer INT4 has no guaranteed accuracy budget."]}
    write_json(output / "quality.json", quality)
    spec = asdict(executor.model_spec(tuple(r["mean_logit_kl"] for r in calibration["layers"])))
    initial_device = str(executor.device)
    if executor.device.type == "cuda":
        print("Measuring the same pretrained FP32 baseline on CPU...", flush=True)
        executor.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()
        measurements.append(_benchmark_variant(executor, "fp32", (), False, repeats, max_tokens))
    result = {"measurement_kind": "real_pretrained_microbenchmark", "model": "GPT-2", "model_path": str(model_path),
              "model_spec": spec, "primary_device": initial_device, "seed": 7, "repeats": repeats,
              "prompts": list(PROMPTS), "max_new_tokens": max_tokens, "measurements": measurements,
              "notes": ["Every reported latency executes actual pretrained weights; no simulator costs used.",
                        "Single serialized requests; no batching or production load-test claim.",
                        "Same prompts/token budget/repeat count; EOS stopping disabled so work lengths match.",
                        "Variant transition/setup cost is recorded separately and excluded from warm steady-state TTFT.",
                        "All CPU recovery masters touched during calibration remain retained and are counted.",
                        "Packed INT4 dequantizes to floating point; synchronous CPU offload does not implement NEO overlap."]}
    write_json(output / "benchmark.json", result)
    source = model_path / "prismserve_source.json"
    manifest = {"version": "0.2.0", "created_utc": datetime.now(timezone.utc).isoformat(),
                "model_source": json.loads(source.read_text()) if source.exists() else {"path": str(model_path)},
                "dataset_source": json.loads((data_path / "source.json").read_text()),
                "dependencies": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "numpy", "safetensors")},
                "source_sha256": {p.name: sha256_file(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
                "artifacts": {name: sha256_file(output / name) for name in ("parity.json", "quality.json", "calibration.json", "benchmark.json")}}
    write_json(output / "manifest.json", manifest)
    _report(output, result, quality, parity)
    return {"output": str(output.resolve()), "model_parameters": spec["parameters"],
            "parity_passed": parity["passed"], "quality_scored_tokens": quality_rows[0]["scored_tokens"],
            "variants_measured": len(measurements), "selected_layers": list(selected)}


def _report(output, benchmark, quality, parity):
    lines = ["# Real pretrained GPT-2 results", "", "All timings execute actual pretrained weights on this machine. WikiText-2 results are a fixed prefix diagnostic, not a full benchmark score.", "",
             "| Device | Variant | TTFT mean ms | TBT mean ms | TBT P95 ms | Output token/s | Resident MiB |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for row in benchmark["measurements"]:
        lines.append(f"| {row['device']} | {row['variant']} | {row['ttft_mean_ms']:.2f} | {row['tbt_mean_ms']:.2f} | {row['tbt_p95_ms']:.2f} | {row['output_tokens_per_s']:.2f} | {row['memory']['resident_model_bytes']/2**20:.2f} |")
    lines += ["", "| Variant | NLL | Perplexity | Relative PPL change % | Scored tokens |", "|---|---:|---:|---:|---:|"]
    for row in quality["variants"]:
        lines.append(f"| {row['variant']} | {row['nll']:.5f} | {row['perplexity']:.3f} | {row['relative_perplexity_change_pct']:.3f} | {row['scored_tokens']} |")
    lines += ["", f"Original-forward correctness passed: {parity['passed']}. Layer selection uses separate validation data. See JSON for exact inputs, hashes, numeric errors, generated text, setup costs, host backups and allocator peaks.", "",
              "Memory savings do not establish a speedup. This reference backend uses floating-point matmul after INT4 unpacking and synchronized CPU attention. Serialized timings do not establish continuous-batching SLOs or large-model performance.", ""]
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        rows = [r for r in benchmark["measurements"] if r["device"] == benchmark["primary_device"]]
        figure_dir = output / "figures"
        figure_dir.mkdir(exist_ok=True)
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        labels = [r["variant"] for r in rows]
        metrics = [("Resident model MiB", [r["memory"]["resident_model_bytes"] / 2**20 for r in rows]),
                   ("Measured mean TBT ms", [r["tbt_mean_ms"] for r in rows]),
                   ("WikiText-2 subset PPL", [r["perplexity"] for r in quality["variants"]])]
        for ax, (label, values) in zip(axes, metrics):
            ax.bar(labels, values, color="#176b87")
            ax.set_ylabel(label)
            ax.tick_params(axis="x", rotation=65)
            ax.grid(axis="y", alpha=.2)
        fig.suptitle("REAL pretrained GPT-2 — measured storage, latency and held-out subset quality")
        fig.tight_layout()
        fig.savefig(figure_dir / "pretrained_tradeoffs.png", dpi=160)
        plt.close(fig)
        lines += ["![Measured pretrained tradeoffs](figures/pretrained_tradeoffs.png)"]
    except ImportError:
        pass
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def adaptive_pretrained(model_path: Path, output: Path, device: str = "auto") -> dict:
    """Apply the common joint controller to real GPT-2 token boundaries.

    Live physical capacities bound every decision. A declared quota episode
    exercises migration without filling the user's GPU with dummy allocations.
    """
    import torch
    from .pretrained import GPT2Executor
    from .telemetry import profile_hardware, sample_resources
    from .types import ControlState, Plan, PlannerConfig, WorkItem
    from .policy import Controller
    torch.set_num_threads(min(4, torch.get_num_threads()))
    executor = GPT2Executor.from_local(model_path, device)
    calibration_path = output.parent / "calibration.json"
    sensitivity = ()
    if calibration_path.exists():
        calibration = json.loads(calibration_path.read_text())
        sensitivity = tuple(row["mean_logit_kl"] for row in calibration["layers"])
    spec = executor.model_spec(sensitivity)
    profile = profile_hardware()
    if executor.device.type == "cpu":
        profile = replace(profile, device_kind="cpu", link_bandwidth_gbps=0,
                          device_bandwidth_gbps=profile.host_bandwidth_gbps, device_tflops=profile.host_tflops)
    budget = max(.005, sum(sorted(spec.layer_sensitivities)[:3]) * 1.1)
    config = PlannerConfig(memory_fraction=1, quality_budget=budget, max_quantized_fraction=.75,
                           cooldown_s=.1, offload_fractions=(0, 1), tbt_slo_ms=100)
    state = ControlState()
    decisions = []
    request_id = "pretrained-request"
    started = time.perf_counter()

    def control(current, cache, index):
        nonlocal state
        telemetry = sample_resources()
        existing = cache[0][0].shape[2]
        item = WorkItem(request_id, executor.context_limit, existing_context_tokens=existing)
        actual = current.memory_report()
        buffers = actual["resident_buffer_bytes"]
        host_available = telemetry["host_available_bytes"] + actual["host_master_bytes"]
        if executor.device.type == "cuda":
            device_available = telemetry["device_free_bytes"] + telemetry["device_allocated_bytes"]
        else:
            host_available += actual["resident_model_bytes"]
            device_available = host_available
        # The estimator handles weights/workspaces. Reserve original attention
        # buffers and a separate64MiB backend margin outside its parameter model.
        buffer_margin = buffers + 64 * 2**20
        candidate_profile = replace(profile, device_memory_bytes=max(1, device_available - buffer_margin),
                                    host_memory_bytes=max(1, host_available))
        phase = "live_capacity" if index < 8 or index >= 20 else "declared_quota_tight"
        if phase == "declared_quota_tight" and executor.device.type == "cuda":
            estimates = Controller(candidate_profile, spec, config).estimator
            gpu = estimates.estimate([item], Plan(), state)
            cpu = estimates.estimate([item], Plan((), (request_id,)), state)
            quota = max(cpu.device_bytes, (cpu.device_bytes + gpu.device_bytes) // 2)
            candidate_profile = replace(candidate_profile, device_memory_bytes=min(candidate_profile.device_memory_bytes, quota))
        decision = Controller(candidate_profile, spec, config).choose([item], state, time.perf_counter() - started)
        if not decision.estimate.feasible:
            raise RuntimeError("insufficient capacity for the pretrained adaptive plan: " + decision.estimate.reason)
        application_start = time.perf_counter()
        current.apply_plan(decision.plan, request_id)
        destination = torch.device("cpu") if current.cpu_attention else current.device
        cache = [(k.to(destination), v.to(destination)) for k, v in cache]
        _sync(current.device)
        changed = decision.plan.quantized_layers != state.quantized_layers or decision.plan.cpu_request_ids != state.cpu_request_ids
        state = ControlState(decision.plan.quantized_layers, decision.plan.cpu_request_ids,
                             time.perf_counter() - started if changed else state.last_change_s)
        decisions.append({"token_index": index, "phase": phase, "cache_tokens": existing,
                          "plan": asdict(decision.plan), "estimate": asdict(decision.estimate),
                          "declared_parameter_budget_bytes": candidate_profile.device_memory_bytes,
                          "backend_buffer_margin_bytes": buffer_margin, "telemetry": telemetry,
                          "search_us_measured": decision.search_us,
                          "application_and_migration_ms_measured": (time.perf_counter() - application_start) * 1000,
                          "cache_device": str(destination)})
        return cache

    executor.generate("A computer", 3, stop_on_eos=False)
    generation = executor.generate(PROMPTS[0], 32, callback=control, stop_on_eos=False)
    result = {"measurement_kind": "real_pretrained_controller_execution", "model": "GPT-2", "device": str(executor.device),
              "model_path": str(model_path), "generation": generation, "decisions": decisions,
              "model_spec": asdict(spec), "planner": asdict(config),
              "sensitivity_source": str(calibration_path) if sensitivity else "default surrogate prior",
              "notes": ["Actual pretrained tensor execution and migration; per-token timing includes controller work.",
                        "Tight episode is an explicit reservation quota, not physical GPU exhaustion.",
                        "Physical free memory and own allocations are sampled every decision.",
                        "A conservative reference-kernel model may prefer offloading over quantization.",
                        "The additive validation KL budget is not a perplexity or task-accuracy guarantee."]}
    write_json(output, result)
    return {"output": str(output.resolve()), "device": str(executor.device),
            "cpu_attention_decisions": sum(bool(r["plan"]["cpu_request_ids"]) for r in decisions),
            "quantized_decisions": sum(bool(r["plan"]["quantized_layers"]) for r in decisions),
            "ttft_ms_measured": generation["ttft_ms"], "elapsed_ms_measured": generation["elapsed_ms"]}
