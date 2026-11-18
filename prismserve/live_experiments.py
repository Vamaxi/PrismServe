"""Real batch calibration and wall-clock request replay for GPT-2.

Synthetic arrival patterns drive real tensor execution. No virtual or roofline
time enters reported TTFT/TBT/throughput. Fixed references are used only in a
separately labeled teacher-forced quality audit.
"""
from __future__ import annotations

from concurrent.futures import wait
from dataclasses import asdict, replace
from datetime import datetime, timezone
import gc
import json
import math
from pathlib import Path
import random
from statistics import mean
import time
import zipfile

from .experiments import write_json
from .pretrained_assets import sha256_file
from .simulator import percentile


def _sync(executor):
    import torch
    if executor.device.type == "cuda":
        torch.cuda.synchronize(executor.device)


def calibrate_live(model_path: Path, output: Path, device: str = "auto", repeats: int = 3) -> dict:
    from .pretrained import GPT2Executor
    import torch
    torch.set_num_threads(min(4, torch.get_num_threads()))
    executor = GPT2Executor.from_local(model_path, device)
    payload = _calibrate(executor, repeats)
    payload["model_path"] = str(model_path)
    write_json(output, payload)
    return {"output": str(output.resolve()), "samples": len(payload["timing"]["samples"]), "device": str(executor.device)}


def _calibrate(executor, repeats):
    import torch
    from .batching import BatchInput, execute_batch
    from .measured_policy import TimingCalibration, TimingSample
    from .telemetry import profile_hardware
    if repeats < 1:
        raise ValueError("calibration repeats must be positive")
    profile = profile_hardware()
    if executor.device.type == "cpu":
        profile = replace(profile, device_kind="cpu", link_bandwidth_gbps=0,
                          device_bandwidth_gbps=profile.host_bandwidth_gbps, device_tflops=profile.host_tflops)
    calibration_file = Path("results/pretrained/calibration.json")
    sensitivity = ()
    if calibration_file.exists():
        old = json.loads(calibration_file.read_text())
        sensitivity = tuple(row["mean_logit_kl"] for row in old["layers"])
        order = old["layer_order"]
    else:
        # No claim of measured layer risk without that independent validation run.
        order = list(range(len(executor.layers)))
    qsets = ((), (order[0],), tuple(sorted(order[:3])))
    transition_to, transition_from, samples = [], [], []
    token = executor.tokenize("The computer processes information.")
    if not token:
        raise RuntimeError("tokenizer returned no calibration tokens")
    executor.set_quantized_layers(())
    executor.set_cpu_attention(False)
    executor.generate("The computer", 3, stop_on_eos=False)
    with torch.inference_mode():
        for qset in qsets:
            _sync(executor)
            begin = time.perf_counter()
            executor.set_quantized_layers(qset)
            _sync(executor)
            if qset:
                transition_to.append((time.perf_counter() - begin) * 1000 / len(qset))
            # batch1/4/8, decodecontext64/384 with
            # device, mixed, or host attention; plus shared128-token prefill.
            for count in (1, 4, 8):
                for context in (64, 384):
                    ids = (token * (context // len(token) + 1))[:context]
                    prepared = execute_batch(executor, [BatchInput(f"c{i}", ids, prefill=True) for i in range(count)])
                    base_cache = [row.cache for row in prepared.outputs]
                    for cpu_count in sorted({0, count // 2, count}):
                        if executor.device.type == "cpu" and cpu_count:
                            continue
                        caches = [[(k.cpu(), v.cpu()) for k, v in cache] if i < cpu_count else cache
                                  for i, cache in enumerate(base_cache)]
                        inputs = [BatchInput(f"c{i}", [token[0]], caches[i], i < cpu_count, False) for i in range(count)]
                        measured, peaks = [], []
                        for repeat in range(repeats + 1):
                            _sync(executor)
                            before_bytes = torch.cuda.memory_allocated(executor.device) if executor.device.type == "cuda" else 0
                            if executor.device.type == "cuda":
                                torch.cuda.reset_peak_memory_stats(executor.device)
                            begin = time.perf_counter()
                            result = execute_batch(executor, inputs)
                            _sync(executor)
                            elapsed = (time.perf_counter() - begin) * 1000
                            peak = max(0, torch.cuda.max_memory_allocated(executor.device) - before_bytes) if executor.device.type == "cuda" else 0
                            if repeat:
                                measured.append(elapsed)
                                peaks.append(peak)
                            del result
                        samples.append(TimingSample(qset, count, 0, context + 1, cpu_count,
                                                    mean(measured), max(peaks, default=0)))
                    del prepared, base_cache, caches, inputs
                # Prompt-only chunks: total128tokens across all requests.
                length = 128 // count
                ids = (token * (length // len(token) + 1))[:length]
                inputs = [BatchInput(f"p{i}", ids, prefill=True) for i in range(count)]
                measured, peaks = [], []
                for repeat in range(repeats + 1):
                    _sync(executor)
                    before_bytes = torch.cuda.memory_allocated(executor.device) if executor.device.type == "cuda" else 0
                    if executor.device.type == "cuda":
                        torch.cuda.reset_peak_memory_stats(executor.device)
                    begin = time.perf_counter()
                    result = execute_batch(executor, inputs)
                    _sync(executor)
                    if repeat:
                        measured.append((time.perf_counter() - begin) * 1000)
                        peaks.append(max(0, torch.cuda.max_memory_allocated(executor.device) - before_bytes) if executor.device.type == "cuda" else 0)
                    del result
                samples.append(TimingSample(qset, count, count * length, length, 0,
                                            mean(measured), max(peaks, default=0)))
                del inputs
            # Mixed iterations matter: context-heavy decode plus prompt chunks
            # cannot be inferred reliably from prompt-only/decode-only timing.
            for count in (4, 8):
                context = 384
                ids = (token * (context // len(token) + 1))[:context]
                old = execute_batch(executor, [BatchInput(f"m{i}", ids, prefill=True) for i in range(count - 2)])
                for cpu_count in sorted({0, (count - 2) // 2, count - 2}):
                    if executor.device.type == "cpu" and cpu_count:
                        continue
                    inputs = [BatchInput(f"m{i}", [token[0]],
                                         [(k.cpu(), v.cpu()) for k, v in out.cache] if i < cpu_count else out.cache,
                                         i < cpu_count, False) for i, out in enumerate(old.outputs)]
                    chunk = (token * (64 // len(token) + 1))[:64]
                    inputs += [BatchInput(f"new{i}", chunk, prefill=True) for i in range(2)]
                    measured, peaks = [], []
                    for repeat in range(repeats + 1):
                        _sync(executor)
                        before_bytes = torch.cuda.memory_allocated(executor.device) if executor.device.type == "cuda" else 0
                        if executor.device.type == "cuda":
                            torch.cuda.reset_peak_memory_stats(executor.device)
                        begin = time.perf_counter()
                        result = execute_batch(executor, inputs)
                        _sync(executor)
                        if repeat:
                            measured.append((time.perf_counter() - begin) * 1000)
                            peaks.append(max(0, torch.cuda.max_memory_allocated(executor.device) - before_bytes) if executor.device.type == "cuda" else 0)
                        del result
                    samples.append(TimingSample(qset, count, 128, context + 1, cpu_count,
                                                mean(measured), max(peaks, default=0)))
                    del inputs
                del old
            _sync(executor)
            begin = time.perf_counter()
            executor.set_quantized_layers(())
            _sync(executor)
            if qset:
                transition_from.append((time.perf_counter() - begin) * 1000 / len(qset))
            print(f"Measured precision {qset}: {len(samples)} batch shapes so far", flush=True)
    calibration = TimingCalibration(tuple(samples), mean(transition_to), mean(transition_from))
    return {"measurement_kind": "actual_selectively_batched_gpt2_calibration", "hardware": asdict(profile),
            "device": str(executor.device), "timing": calibration.to_dict(), "sensitivity": list(sensitivity),
            "repeats": repeats, "source": "Real tensor batches with warmup, synchronized timing and allocator deltas",
            "notes": ["Workspace records incremental CUDA peak beyond already-live allocations; safety headroom added by controller.",
                      "Model and KV bytes are independently reserved by the planner; workspace deltas may conservatively count new KV twice.",
                      "Ragged attention loops per request; dense projections are batched across all current tokens.",
                      "Calibration contains no held-out test continuations."]}


def _requests(executor, number, prompt_tokens, output_tokens, duration, kind, seed):
    data_path = Path("data/wikitext2/wiki.test.prefix.txt")
    text = data_path.read_text(encoding="utf-8")
    ids = executor.tokenizer.encode(text, add_special_tokens=False, verbose=False)
    required = prompt_tokens + output_tokens
    if required > executor.context_limit or len(ids) < required:
        raise ValueError("workload token lengths exceed local model/data limits")
    rng = random.Random(seed)
    if kind == "bursty":
        arrivals = [rng.uniform(duration * .2, duration * .35) if rng.random() < .8 else rng.uniform(0, duration)
                    for _ in range(number)]
    elif kind == "steady":
        arrivals = [i * duration / max(1, number - 1) for i in range(number)]
    else:
        raise ValueError("traffic must be steady or bursty")
    arrivals.sort()
    requests = []
    for index, arrival in enumerate(arrivals):
        start = (index * 317) % (len(ids) - required + 1)
        requests.append({"request_id": f"r{index:04d}", "arrival_s": arrival,
                         "prompt_ids": ids[start:start + prompt_tokens],
                         "reference_ids": ids[start + prompt_tokens:start + required],
                         "output_tokens": output_tokens})
    return requests


def replay(executor, calibration_payload, trace, serving_config, policy, quality=False, *, failure_output=None):
    import torch
    from .live_engine import ContinuousEngine
    from .measured_policy import TimingCalibration
    from .types import HardwareProfile
    executor.set_quantized_layers(())
    executor.set_cpu_attention(False)
    gc.collect()
    if executor.device.type == "cuda":
        torch.cuda.empty_cache()
    _warm_batch_shapes(executor, serving_config)
    if executor.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(executor.device)
    calibration = TimingCalibration.from_dict(calibration_payload["timing"])
    profile = HardwareProfile(**calibration_payload["hardware"])
    engine_policy = "full" if policy == "serial_full" else policy
    if policy == "serial_full":
        serving_config = replace(serving_config, max_batch_size=1)
    engine = ContinuousEngine(executor, calibration, profile, config=serving_config, policy=engine_policy,
                              sensitivity=tuple(calibration_payload.get("sensitivity", ())))
    engine.start()
    start = time.perf_counter()
    start_unix_s = time.time()
    futures = []
    submission_lag_ms = []
    try:
        for row in trace:
            deadline = start + row["arrival_s"]
            while time.perf_counter() < deadline:
                time.sleep(min(.01, max(0, deadline - time.perf_counter())))
            submission_lag_ms.append(max(0, time.perf_counter() - deadline) * 1000)
            futures.append(engine.submit_tokens(row["prompt_ids"], row["output_tokens"], row["request_id"],
                                                 deadline, stop_on_eos=False,
                                                 forced_tokens=row["reference_ids"] if quality else None))
        completed, unfinished = wait(futures, timeout=180)
        if unfinished:
            raise RuntimeError("real workload did not drain within180 seconds; no partial success")
        errors = []
        for future in futures:
            if future.exception() is not None:
                if isinstance(future.exception(), MemoryError):
                    errors.append(str(future.exception()))
                else:
                    raise future.exception()
        elapsed = time.perf_counter() - start
    except BaseException as error:
        if failure_output is not None:
            write_json(Path(failure_output), {'status': 'failed_not_a_successful_benchmark',
                       'error': f'{type(error).__name__}: {error}', 'policy': policy, 'trace': trace,
                       'partial_snapshot': engine.snapshot(), 'elapsed_wall_s': time.perf_counter()-start,
                       'notes': ['Partial completion is not accepted as successful throughput or a quality audit.']})
        raise
    finally:
        engine.close(drain=False)
    result = engine.snapshot()
    rows = result["requests"]
    gaps = [ms for row in rows for ms in row["decode_token_ms"]]
    good = sum(not row["slo_violated"] for row in rows)
    nll = [value for row in rows for value in row["target_nll"]]
    events = result["iterations"]
    result["summary"] = {"measurement_kind": "real_continuous_batched_pretrained_execution",
                         "mode": "teacher_forced_quality_audit" if quality else "greedy_generation",
                         "policy": policy, "requests": len(trace), "completed": len(rows),
                         "rejected": len(result["rejections"]), "elapsed_wall_s": elapsed,
                         "measurement_start_unix_s": start_unix_s,
                         "measurement_end_unix_s": start_unix_s + elapsed,
                         "ttft_p50_ms": percentile([r["ttft_ms"] for r in rows], .5),
                         "ttft_p95_ms": percentile([r["ttft_ms"] for r in rows], .95),
                         "tbt_p95_ms": percentile(gaps, .95), "max_tbt_ms": max(gaps, default=0),
                         "output_tokens_per_s": sum(r["completion_tokens"] for r in rows) / elapsed,
                         "goodput_requests_per_s": good / elapsed,
                         "slo_failure_rate": (len(trace) - good) / len(trace),
                         "max_active": result["metrics"]["max_active"],
                         "max_execution_batch_size": result["metrics"]["max_execution_batch_size"],
                         "mean_batch_size": mean(e["batch_size"] for e in events) if events else 0,
                         "quantized_output_fraction": sum(bool(p) for r in rows for p in r["precision_history"]) / max(1, sum(r["completion_tokens"] for r in rows)),
                         "cpu_attention_output_fraction": sum(p for r in rows for p in r["cpu_offload_history"]) / max(1, sum(r["completion_tokens"] for r in rows)),
                         "search_admission_wall_fraction": sum(e["search_and_admission_ms"] for e in events) / max(1, elapsed * 1000),
                         "telemetry_wall_fraction": sum(e.get("telemetry_ms", 0) for e in events) / max(1, elapsed * 1000),
                         "planning_wall_fraction": sum(e.get("planning_ms", 0) for e in events) / max(1, elapsed * 1000),
                         "plan_reuse_fraction": sum(e.get("plan_reused", False) for e in events) / max(1, len(events)),
                         "gate_fallback_count": sum(e.get("gate_fallback", False) for e in events),
                         "prediction_mae_ms": mean(abs(e["execution_ms"] - e["predicted_execution_ms"]) for e in events) if events else None,
                         "peak_live_device_cache_bytes": result["metrics"]["peak_live_device_cache_bytes"],
                         "peak_live_host_cache_bytes": result["metrics"]["peak_live_host_cache_bytes"],
                         "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(executor.device) if executor.device.type == "cuda" else None,
                         "submission_lag_p95_ms": percentile(submission_lag_ms, .95),
                         "teacher_forced_targets": len(nll), "teacher_forced_nll": mean(nll) if nll else None,
                         "teacher_forced_perplexity": math.exp(mean(nll)) if nll else None}
    result["trace"] = trace
    return result


def _warm_batch_shapes(executor, serving_config):
    """Warm the actual batch primitive before each timed policy replay."""
    import torch
    from .batching import BatchInput, execute_batch
    token = executor.tokenize("A computer processes information.")
    with torch.inference_mode():
        for count in sorted({1, min(4, serving_config.max_batch_size), serving_config.max_batch_size}):
            length = max(1, min(64, serving_config.prefill_token_budget // count))
            ids = (token * (length // len(token) + 1))[:length]
            result = execute_batch(executor, [BatchInput(f"warm{i}", ids, prefill=True) for i in range(count)])
            decode = execute_batch(executor, [BatchInput(row.request_id, [row.next_token_id], row.cache)
                                              for row in result.outputs])
            del result, decode
    _sync(executor)


def run_live_experiments(config_path: Path, output: Path, device: str = "auto",
                         calibration_path: Path | None = None) -> dict:
    import csv
    import torch
    from .live_engine import ServingConfig
    from .pretrained import GPT2Executor
    config = json.loads(config_path.read_text())
    if config.get("schema_version") != 1:
        raise ValueError("live experiment schema_version must be1")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "config.json", config)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    torch.manual_seed(7)
    executor = GPT2Executor.from_local(Path(config.get("model_path", "models/gpt2")), device)
    if calibration_path is not None:
        payload = json.loads(calibration_path.read_text())
        if payload["device"].split(":")[0] != executor.device.type:
            raise ValueError("timing calibration device does not match execution device")
    else:
        print("Measuring actual batch costs before replay...", flush=True)
        payload = _calibrate(executor, config.get("calibration_repeats", 3))
    write_json(output / "calibration.json", payload)
    summaries = []
    for scenario in config["scenarios"]:
        for seed in config.get("seeds", [7]):
            w = config["workload"]
            trace = _requests(executor, w["requests"], w["prompt_tokens"], w["output_tokens"],
                              w["duration_s"], scenario["traffic"], seed)
            write_json(output / "traces" / f"{scenario['name']}__{seed}.json", trace)
            serving = ServingConfig(**config.get("serving", {}),
                                    device_memory_budget_bytes=int(scenario["device_budget_mib"] * 2**20))
            # Rotate policy order by seed to reduce systematic warm/thermal bias.
            policies = list(config["policies"])
            random.Random(seed).shuffle(policies)
            for policy in policies:
                print(f"REAL replay {scenario['name']} / seed{seed} / {policy}", flush=True)
                result = replay(executor, payload, trace, serving, policy)
                result["summary"].update(scenario=scenario["name"], seed=seed,
                                          device_budget_mib=scenario["device_budget_mib"])
                summaries.append(result["summary"])
                write_json(output / "runs" / f"{scenario['name']}__{policy}__{seed}.json", result)
    quality_runs = []
    audit = config.get("quality_audit", {})
    if audit.get("enabled", False):
        trace = _requests(executor, audit["requests"], audit["prompt_tokens"], audit["output_tokens"],
                          audit["duration_s"], "bursty", 101)
        serving = ServingConfig(**config.get("serving", {}),
                                device_memory_budget_bytes=int(audit["device_budget_mib"] * 2**20))
        for policy in config["policies"]:
            print(f"REAL teacher-forced quality audit / {policy}", flush=True)
            result = replay(executor, payload, trace, serving, policy, quality=True)
            quality_runs.append(result["summary"])
            write_json(output / "quality_runs" / f"{policy}.json", result)
        base = next(row for row in quality_runs if row["policy"] == "full")
        for row in quality_runs:
            row["relative_ppl_change_pct"] = ((math.exp(row["teacher_forced_nll"] - base["teacher_forced_nll"]) - 1) * 100
                                               if row["teacher_forced_nll"] is not None and base["teacher_forced_nll"] is not None else None)
        write_json(output / "quality.json", {"mode": "separate_teacher_forced_live_quality_audit",
                   "dataset": "WikiText-2 test prefix", "calibration_split": "validation",
                   "policy_summaries": quality_runs,
                   "notes": ["Teacher-forced contexts evaluate actual dynamically selected weights and retained cache history.",
                             "Each policy executes its own wall-clock schedule; audit decisions can differ from greedy replay.",
                             "A short audit is not full WikiText accuracy or proof of a quality bound."]})
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    # Preserve the exact module snapshot so later extensions do not erase the
    # implementation that generated these physical timing results.
    sources = {p.name: sha256_file(p) for p in sorted(Path(__file__).parent.glob("*.py"))}
    with zipfile.ZipFile(output / "source_snapshot.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(Path(__file__).parent.glob("*.py")):
            archive.write(path, "prismserve/" + path.name)
    from . import __version__
    write_json(output / "manifest.json", {"version": __version__, "measurement_kind": "actual_pretrained_continuous_batch_serving",
               "created_utc": datetime.now(timezone.utc).isoformat(), "config_sha256": sha256_file(config_path),
               "source_sha256": sources, "source_snapshot_sha256": sha256_file(output / "source_snapshot.zip"),
               "calibration_sha256": sha256_file(output / "calibration.json"), "serving_runs": len(summaries),
               "quality_audits": len(quality_runs), "device": str(executor.device),
               "model_source": json.loads(Path(config.get("model_path", "models/gpt2"), "prismserve_source.json").read_text()),
               "notes": ["Actual tensor execution and host wall times, including scheduling/precision/migration overhead.",
                         "Memory budgets are enforced admission/reservation quotas; no dummy physical-VRAM exhaustion.",
                         "Serial baseline uses the same optimized last-token-head batch primitive with max_batch_size1.",
                         "Single physical machine and few seeds: exploratory performance evidence."]})
    _live_report(output, summaries, quality_runs)
    return {"output": str(output.resolve()), "serving_runs": len(summaries), "quality_audits": len(quality_runs)}


def _live_report(output, rows, quality):
    from collections import defaultdict
    groups = defaultdict(list)
    for row in rows:
        groups[(row["scenario"], row["policy"])].append(row)
    lines = ["# Real continuous-batching experiments", "",
             "**Actual pretrained GPT-2 tensor execution. All reported times are measured wall-clock observations.**", "",
             "Same prompt/arrival/output-budget traces across policies. Dense projections batch all current tokens. Ragged attention loops per request. Quotas are enforced admission budgets, not physical OOM events.", "",
             "| Scenario | Policy | P95 TTFT ms | P95 TBT ms | Output token/s | SLO failure % | Max active | INT4 output % | CPU offload output % |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for (scenario, policy), group in sorted(groups.items()):
        def avg(key):
            values = [r[key] for r in group if r[key] is not None]
            return mean(values) if values else None
        fmt = lambda value: "N/A" if value is None else f"{value:.2f}"
        lines.append(f"| {scenario} | {policy} | {fmt(avg('ttft_p95_ms'))} | {fmt(avg('tbt_p95_ms'))} | {fmt(avg('output_tokens_per_s'))} | {fmt(100*avg('slo_failure_rate'))} | {max(r['max_active'] for r in group)} | {100*avg('quantized_output_fraction'):.2f} | {100*avg('cpu_attention_output_fraction'):.2f} |")
    lines += ["", "## Quality audit", "",
              "A separate teacher-forced replay uses the same held-out continuations for each policy and records token NLL under the actual selected precision and historical KV. It has its own policy trajectory. See quality.json and quality_runs/; these results do not label greedy generation as teacher-forced work.", "",
              "| Policy | Audit targets | Audit perplexity | Relative PPL % | INT4 output % |", "|---|---:|---:|---:|---:|"]
    for row in quality:
        ppl = "N/A" if row["teacher_forced_perplexity"] is None else f"{row['teacher_forced_perplexity']:.3f}"
        change = "N/A" if row["relative_ppl_change_pct"] is None else f"{row['relative_ppl_change_pct']:.3f}"
        lines.append(f"| {row['policy']} | {row['teacher_forced_targets']} | {ppl} | {change} | {100*row['quantized_output_fraction']:.2f} |")
    lines += ["", "## Interpretation limits", "",
              "Do not assume a controller win: inspect the full baseline, admission, overhead, transition, and failure records. The measured nearest-shape model and EWMA feedback are estimates; real elapsed timings remain authoritative. Quality uses an additive validation-KL surrogate and a small separate audit, not a guaranteed task bound.", "",
              "TTFT starts at scheduled arrival, including delayed submission and queueing. All token gaps include scheduling, precision transitions and migrations. Throughput includes the complete trace and queue drain. Rejected requests count as SLO failures. Memory includes independently reported live KV, reserved budgets and CUDA peak allocations.", ""]
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        scenarios = sorted({r["scenario"] for r in rows})
        figure_dir = output / "figures"
        figure_dir.mkdir(exist_ok=True)
        for key, label in (("output_tokens_per_s", "Actual output token/s"), ("ttft_p95_ms", "Measured P95 TTFT ms")):
            fig, axes = plt.subplots(1, len(scenarios), figsize=(5 * len(scenarios), 4), squeeze=False)
            for ax, scenario in zip(axes[0], scenarios):
                selected = [(policy, group) for (s, policy), group in sorted(groups.items()) if s == scenario]
                labels = [p for p, _ in selected]
                values = [[r[key] for r in group if r[key] is not None] for _, group in selected]
                ax.bar(labels, [mean(v) if v else float("nan") for v in values], color=["#176b87" if p == "joint" else "#83b3c4" for p in labels])
                for index, (_, group) in enumerate(selected):
                    valid = [r[key] for r in group if r[key] is not None]
                    ax.scatter([index] * len(valid), valid, s=13, color="#203c4d")
                ax.set_title(scenario)
                ax.set_ylabel(label)
                ax.tick_params(axis="x", rotation=55)
                ax.grid(axis="y", alpha=.2)
            fig.suptitle("REAL continuous-batched pretrained GPT-2 — individual seeds shown")
            fig.tight_layout()
            fig.savefig(figure_dir / f"{key}.png", dpi=150)
            plt.close(fig)
            lines += [f"![{label}](figures/{key}.png)", ""]
    except ImportError:
        pass
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
