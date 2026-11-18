"""Insert final measured live results into the course report and presentation."""
from pathlib import Path
import csv
import json
import re
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]


def replace_block(text, begin, end, replacement, before):
    if begin in text:
        left, rest = text.split(begin, 1)
        _, right = rest.split(end, 1)
        return left + replacement + right
    return text.replace(before, replacement + "\n\n" + before, 1)


def main():
    output = ROOT / "results/live"
    manifest = json.loads((output / "manifest.json").read_text())
    if manifest.get("serving_runs") != 42:
        raise ValueError("final42-run serving suite must complete before documentation export")
    with (output / "summary.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    groups = {}
    for row in rows:
        groups.setdefault((row["scenario"], row["policy"]), []).append(row)
    scenarios = ["steady_roomy", "bursty_roomy", "bursty_constrained"]
    def avg(scenario, policy, key):
        return mean(float(row[key]) for row in groups[(scenario, policy)] if row[key] not in {"", "None"})
    def batch(scenario, policy):
        return max(int(row["max_active"]) for row in groups[(scenario, policy)])
    tests = json.loads((ROOT / "results/validation.json").read_text())["tests_run"]
    quality = json.loads((output / "quality.json").read_text())["policy_summaries"]
    http = json.loads((output / "http_smoke.json").read_text())
    assert http["passed"]
    completed = sum(int(row["completed"]) for row in rows)
    rejected = sum(int(row["rejected"]) for row in rows)
    constrained_peak = max(float(row['peak_cuda_allocated_bytes']) / 2**20
                           for row in rows if row['scenario'] == 'bursty_constrained')
    lines = ["<!-- LIVE_RESULTS_BEGIN -->", "## 6. Real continuous-batching traffic experiments", "",
             "Version 0.3 adds a shared background worker with independent per-request KV histories. Current prefill/decode tokens are flattened for each layer's four dense projections; ragged attention is evaluated per request. A real GPT-2 comparison of unequal contexts and mixed CPU/GPU placement has maximum logit error 3.05e-4 against independent inference, within the declared 0.002 tolerance. One iteration uses 48 dense projection calls across 12 blocks, regardless of request count; it does not invoke a complete model separately for each request.", "",
             "The controller starts from 75 synchronized measured batch shapes, with three repeats plus warmup, covering batches 1/4/8, contexts, mixed prefill/decode, precision sets and host placement. An EWMA corrects execution estimates. It evaluates admission alternatives using predicted time per serviced request and a TBT penalty. Layer changes and migrations execute at token boundaries. All request timings include actual controller/admission overhead, execution and transitions; analytical durations never advance the live clock.", "",
             "The final comparison has 42 greedy-serving runs: seven policies, two replay identifiers and three scenarios. Burst timestamps depend on seed; the steady trace is identical between repetitions, with policy order changed. Each receives identical 12-request traces with 384-token prompts, 24 outputs, two-second arrival windows and disabled EOS stopping. A maximum of eight requests is active, with 64-token prefill chunks and 128 total prefill tokens per tick. Roomy/constrained budgets are 1400/700 MiB enforced reservation quotas, not physical-VRAM exhaustion. The serial baseline uses the same last-token-head batch primitive at batch size one. Actual batch kernels are warmed before each timed run. TTFT measures engine token readiness; the HTTP endpoint returns a complete non-streaming response, whose elapsed time is a separate metric.", "",
             "| Scenario | Policy | Mean run P95 TTFT ms | Mean run P95 TBT ms | Output token/s | SLO failure % | Max active |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for scenario in scenarios:
        for policy in ("serial_full", "full", "joint"):
            lines.append(f"| {scenario} | {policy} | {avg(scenario,policy,'ttft_p95_ms'):.2f} | {avg(scenario,policy,'tbt_p95_ms'):.2f} | {avg(scenario,policy,'output_tokens_per_s'):.2f} | {100*avg(scenario,policy,'slo_failure_rate'):.2f} | {batch(scenario,policy)} |")
    ratios = [avg(s, "full", "output_tokens_per_s") / avg(s, "serial_full", "output_tokens_per_s") for s in scenarios]
    lines += ["", f"Across the three scenarios, batched full-precision throughput relative to the same-primitive serial baseline is {', '.join(f'{value:.2f}x' for value in ratios)}. These are measured observations on one laptop across two seeds, not universal speedups. The final runs complete {completed} requests and reject {rejected}; all rejections remain SLO failures. Percentile columns average per-run percentiles across seeds rather than pooling tokens.", "",
              f"Under the 700 MiB quota, joint control reaches {batch('bursty_constrained','joint')} active requests versus {batch('bursty_constrained','full')} for full precision. Its mean throughput is {avg('bursty_constrained','joint','output_tokens_per_s'):.2f} versus {avg('bursty_constrained','full','output_tokens_per_s'):.2f} output tokens/s. Increased concurrency is therefore reported separately from latency and throughput. Quantized-output exposure is {100*avg('bursty_constrained','joint','quantized_output_fraction'):.2f}% and CPU-offloaded exposure is {100*avg('bursty_constrained','joint','cpu_attention_output_fraction'):.2f}% for the joint arm. The largest observed CUDA tensor-allocation peak in that scenario is {constrained_peak:.2f} MiB, below the 700 MiB budget; this describes observed Torch allocations, not total machine VRAM use or a formal isolation guarantee.", "",
              "| Scenario | Joint search/admission wall fraction % | Joint prediction MAE ms | Uncalibrated prediction MAE ms |",
              "|---|---:|---:|---:|"]
    for scenario in scenarios:
        lines.append(f"| {scenario} | {100*avg(scenario,'joint','search_admission_wall_fraction'):.2f} | {avg(scenario,'joint','prediction_mae_ms'):.2f} | {avg(scenario,'joint_uncalibrated','prediction_mae_ms'):.2f} |")
    steady_full = [float(row['output_tokens_per_s']) for row in groups[('steady_roomy', 'full')]]
    lines += ["", f"The identical steady trace has substantial timing variation: full-precision throughput ranges from {min(steady_full):.2f} to {max(steady_full):.2f} tokens/s. Two repetitions do not establish a stable steady-load speedup; device clocks, thermal state and background activity were not independently controlled. Preserve the individual runs rather than interpreting the mean as a guarantee.", "",
              "No general superiority is inferred from calibration or online feedback. The uncalibrated arm executes the same real engine with analytical decision-time predictions, including the earlier model's optimistic overlap and weight-streaming assumptions. It is a comparison between prediction designs, not an isolation of calibration, context features and overlap assumptions individually. Search/admission overhead is a measured part of request latency, not excluded as in the analytical simulator. Full policy tables, individual seeds, per-token timestamps and plans are retained in `results/live/`.", "",
              "![Actual continuously batched output throughput](../results/live/figures/output_tokens_per_s.png)", "",
              "![Actual continuously batched tail latency](../results/live/figures/ttft_p95_ms.png)", "",
              "### Live policy quality and failure handling", "",
              "A separate teacher-forced audit evaluates the same four held-out WikiText-2 prompts and 16-token reference continuations for each policy. It scores actual token likelihoods under each audit's selected precision and historical KV, rather than assigning static-quality numbers to a dynamic run. The audit trajectory is separate from the greedy serving trajectory.", "",
              "| Policy | Scored targets | Audit PPL | Relative PPL change % | Quantized output % |", "|---|---:|---:|---:|---:|"]
    for row in quality:
        ppl = "N/A" if row["teacher_forced_perplexity"] is None else f"{row['teacher_forced_perplexity']:.3f}"
        delta = "N/A" if row["relative_ppl_change_pct"] is None else f"{row['relative_ppl_change_pct']:.3f}"
        lines.append(f"| {row['policy']} | {row['teacher_forced_targets']} | {ppl} | {delta} | {100*row['quantized_output_fraction']:.2f} |")
    lines += ["", "A 64-target audit per policy is a narrow diagnostic. Precision exposure must accompany perplexity: a policy that does not quantize in its audit supplies no quality evidence for a different trajectory that does quantize. The additive validation-KL budget still does not prove a task-accuracy bound.", "",
              f"The actual HTTP check completes four concurrent clients in a shared execution batch of {http['metrics']['max_execution_batch_size']}. The full suite has {tests} passing tests. A constrained-memory diagnostic exposed a bug in current-length workspace reservation; the fix reserves a monotone calibrated scratch bound plus temporary copies of fully reserved device/host KV. Regression tests cover future growth and request removal. The corrected targeted workload completes before the full suite is run. The initial incomplete attempt and its source snapshot remain in `results/live_initial/`; those measurements are excluded from the final table.", "",
              "The engine remains a synchronous reference: attention loops per request, CPU/GPU work is not pipelined, calibration extrapolation is approximate, and only one physical machine is measured. The source snapshot and calibration hashes in the live manifest describe this new experiment independently of the earlier simulation and serialized benchmarks.", "<!-- LIVE_RESULTS_END -->"]
    section = "\n".join(lines)
    report_path = ROOT / "docs/final_report.md"
    report = report_path.read_text(encoding="utf-8")
    if "<!-- LIVE_RESULTS_BEGIN -->" not in report:
        report = report.replace("## 8. Conclusion and continuation", "## 9. Conclusion and continuation")
        report = report.replace("## 7. Interpretation and threats to validity", "## 8. Interpretation and threats to validity")
        report = report.replace("## 6. Analytical simulation results", "## 7. Analytical simulation results")
    report = replace_block(report, "<!-- LIVE_RESULTS_BEGIN -->", "<!-- LIVE_RESULTS_END -->", section,
                           "## 7. Analytical simulation results")
    report = report.replace("All 58 recorded tests", f"All {tests} recorded tests")
    report = report.replace("Bounded completion APIs serialize mutable model state.",
                            "The earlier completion APIs serialize requests; version 0.3 adds a concurrent API with one shared model worker.")
    report = report.replace("version0.3", "version 0.3")
    if "The new continuous-batching experiment" not in report:
        report = report.replace("## 1. Motivation", "The new continuous-batching experiment adds 42 real traffic runs, 75 measured calibration shapes and separate policy-quality audits. It measures actual scheduling and transition overhead and preserves per-request state. See Section 6 for the new evidence.\n\n## 1. Motivation", 1)
    report_path.write_text(report, encoding="utf-8")
    slide_lines = ["<!-- LIVE_SLIDES_BEGIN -->", "# 12. Real continuous-batching traffic", "",
                   "42 actual traffic runs · 7 policies · 2 seeds · identical prompts/arrivals", "",
                   "| Scenario | Serial token/s | Batched full token/s | Joint token/s |", "|---|---:|---:|---:|"]
    for scenario in scenarios:
        slide_lines.append(f"| {scenario} | {avg(scenario,'serial_full','output_tokens_per_s'):.1f} | {avg(scenario,'full','output_tokens_per_s'):.1f} | {avg(scenario,'joint','output_tokens_per_s'):.1f} |")
    slide_lines += ["", "True shared dense execution:48 projections per iteration across 12 blocks; separate KV per request.", "",
                    "1400/700 MiB are enforced admission budgets. All latency includes controller and transition costs.", "",
                    "<!-- Speaker notes: Means across two workload seeds. Same optimized batch primitive for serial and batched controls. See results/live/report.md for all arms, per-run percentiles, failures, overhead and memory. Increased concurrency need not improve throughput. All tensor execution is real; no virtual token time. -->", "", "---", "",
                    "# 13. Live controller, quality and validation", "",
                    "- 75 measured shapes: batch sizes 1/4/8, mixed prefill/decode, precision and placement.",
                    "- Nearest timing lookup plus EWMA feedback; measured precision transitions.",
                    "- Reserve future KV and temporary copies before admission.",
                    f"- Four concurrent HTTP clients share an actual batch of {http['metrics']['max_execution_batch_size']}.",
                    f"- {tests} passing tests, including memory-growth and cancellation regressions.",
                    "- Separate teacher-forced audit: 64 held-out targets per policy; report precision exposure.", "",
                    "Reference attention remains synchronous and ragged. Quality and speedup guarantees remain unproven.", "",
                    "<!-- Speaker notes: The initial diagnostic suite exposed insufficient future workspace reservation and is retained separately in results/live_initial. Final results use corrected memory bounds and actual batching warmup. Calibration and resource estimates do not make a production allocator or fused paged-attention kernel. Keep audit trajectories separate from greedy output histories. -->", "<!-- LIVE_SLIDES_END -->", "", "---", ""]
    slides_path = ROOT / "docs/presentation.md"
    slides = slides_path.read_text(encoding="utf-8")
    if "<!-- LIVE_SLIDES_BEGIN -->" not in slides:
        slides = slides.replace("# 13. Answer and next step", "# 15. Answer and next step")
        slides = slides.replace("# 12. What remains uncertain", "# 14. What remains uncertain")
    slides = replace_block(slides, "<!-- LIVE_SLIDES_BEGIN -->", "<!-- LIVE_SLIDES_END -->",
                           "\n".join(slide_lines), "# 14. What remains uncertain")
    slides = slides.replace("**58 tests pass**", f"**{tests} tests pass**")
    while re.search(r"\n---\s*\n---\s*\n", slides):
        slides = re.sub(r"\n---\s*\n---\s*\n", "\n---\n\n", slides)
    slides_path.write_text(slides, encoding="utf-8")
    print(f"Updated report and presentation from {len(rows)} live serving runs")


if __name__ == "__main__":
    main()
