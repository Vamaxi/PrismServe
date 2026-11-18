"""Build v0.4 tables/figures and insert measured results in submission sources."""
from collections import defaultdict
from pathlib import Path
from statistics import mean
import csv
import json
import re

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/controller"
POLICIES = ["full_v03", "joint_v03", "full", "joint", "joint_no_gate", "joint_no_reuse"]


def insert_block(path, name, content, before):
    text = path.read_text(encoding="utf-8")
    block = f"<!-- {name}_BEGIN -->\n{content}\n<!-- {name}_END -->\n\n"
    pattern = rf"<!-- {name}_BEGIN -->.*?<!-- {name}_END -->\s*"
    if re.search(pattern, text, flags=re.S):
        text = re.sub(pattern, lambda match: block, text, flags=re.S)
    else:
        assert before in text
        text = text.replace(before, block + before, 1)
    path.write_text(text, encoding="utf-8")


def main():
    rows = list(csv.DictReader((OUT / "summary.csv").open(encoding="utf-8")))
    groups = defaultdict(list)
    for row in rows:
        groups[(row["scenario"], row["policy"])].append(row)
    def avg(scenario, policy, key):
        return mean(float(row[key]) for row in groups[(scenario, policy)])
    scenarios = [s["name"] for s in json.loads((OUT / "config.json").read_text())["scenarios"]]
    quality = json.loads((OUT / "quality.json").read_text())["policy_summaries"]
    validation = json.loads((OUT / "validation.json").read_text())
    assert validation["passed"]
    table = ["| Scenario | Policy | Token/s | Control % | Reuse % | P95 TTFT ms | P95 TBT ms | Prediction MAE ms |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for scenario in scenarios:
        for policy in POLICIES:
            values = [avg(scenario, policy, key) * scale for key, scale in
                      (("output_tokens_per_s", 1), ("search_admission_wall_fraction", 100),
                       ("plan_reuse_fraction", 100), ("ttft_p95_ms", 1), ("tbt_p95_ms", 1),
                       ("prediction_mae_ms", 1))]
            table.append(f"| {scenario} | {policy} | " + " | ".join(f"{v:.2f}" for v in values) + " |")
    comparison = ["| Scenario | Old joint control % | New joint control % | Old joint token/s | New joint token/s | Change % | New full token/s |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
    for scenario in scenarios:
        old = avg(scenario, "joint_v03", "output_tokens_per_s")
        new = avg(scenario, "joint", "output_tokens_per_s")
        comparison.append(f"| {scenario} | {100*avg(scenario,'joint_v03','search_admission_wall_fraction'):.2f} | {100*avg(scenario,'joint','search_admission_wall_fraction'):.2f} | {old:.2f} | {new:.2f} | {100*(new/old-1):+.2f} | {avg(scenario,'full','output_tokens_per_s'):.2f} |")
    qtable = ["| Policy | Audit targets | Perplexity | Relative PPL % | INT4 output % | INT4 iterations |",
              "|---|---:|---:|---:|---:|---:|"]
    for row in quality:
        audit_run = json.loads((OUT / "quality_runs" / f"{row['policy']}.json").read_text())
        quantized_iterations = sum(bool(e['quantized_layers']) for e in audit_run['iterations'])
        qtable.append(f"| {row['policy']} | {row['teacher_forced_targets']} | {row['teacher_forced_perplexity']:.3f} | {row['relative_ppl_change_pct']:+.3f} | {100*row['quantized_output_fraction']:.2f} | {quantized_iterations} |")
    throughput_wins = sum(avg(s,'joint','output_tokens_per_s') > avg(s,'joint_v03','output_tokens_per_s') for s in scenarios)
    mae_worse = sum(avg(s,'joint','prediction_mae_ms') > avg(s,'joint_v03','prediction_mae_ms') for s in scenarios)
    tight_faster = 'faster' if avg('bursty_constrained','full','output_tokens_per_s') > avg('bursty_constrained','joint','output_tokens_per_s') else 'slower'
    interpretation = (
        "Lower overhead does not establish a general throughput or prediction-accuracy improvement. "
        f"The optimized joint controller improves mean throughput over its old counterpart in {throughput_wins} of {len(scenarios)} scenarios. "
        "The steady scenario's old joint repeats range from "
        f"{min(float(r['output_tokens_per_s']) for r in groups[('steady_roomy','joint_v03')]):.1f} to "
        f"{max(float(r['output_tokens_per_s']) for r in groups[('steady_roomy','joint_v03')]):.1f} token/s. "
        f"Optimized full precision is {tight_faster} than optimized joint control under the tight quota. "
        f"Joint prediction MAE increases in {mae_worse} of {len(scenarios)} scenarios; phase isolation passes its correctness test "
        "but does not demonstrate better overall prediction in these different executed trajectories. "
        "Disabling reuse increases control cost, yet its throughput result varies by scenario. "
        "The evidence supports reducing scheduling cost, not a universal performance win.")
    joint_audit = json.loads((OUT / 'quality_runs/joint.json').read_text())
    q_prefill = sum(bool(e['quantized_layers']) and bool(e['prefill_tokens']) for e in joint_audit['iterations'])
    joint_quality = next(r for r in quality if r['policy'] == 'joint')
    quality_note = (
        f"The joint audit uses INT4 during {q_prefill} iterations containing prefill and emits "
        f"{100*joint_quality['quantized_output_fraction']:.2f}% of outputs under INT4. "
        "Historical KV can remain affected after FP32 restoration. "
        "A small perplexity decrease is not evidence of improved model quality. Greedy serving can "
        "use different precision histories; this short audit does not certify all quantized serving outputs.")
    new_rows = [r for r in rows if r["policy"] == "joint"]
    max_overhead = 100*max(float(r["search_admission_wall_fraction"]) for r in new_rows)
    met = max_overhead < 5
    verdict = (f"All six optimized joint runs meet the exploratory <5% control-overhead target (worst {max_overhead:.2f}%)."
               if met else f"The <5% target is not met in every run; worst optimized joint overhead is {max_overhead:.2f}%.")
    evidence = (f"{validation['serving_runs']} actual serving runs complete {validation['completed_requests']} requests "
                f"with {validation['rejected_requests']} rejections, plus {validation['quality_audits']} independent "
                f"teacher-forced audits ({validation['teacher_forced_targets']} targets total). "
                f"{validation['reused_iterations_with_fresh_resource_checks']} reused iterations retain fresh resource checks. "
                f"{validation['tests_passed']} tests and four-client real HTTP verification pass. "
                f"Actual CUDA peaks exceed the declared device quota in {validation['actual_cuda_peak_over_quota_runs']} runs.")
    limits = ("Means summarize two seeds per scenario; individual points are retained. Policy order is shuffled, but laptop "
              "thermal state, OS activity and limited repeats remain confounders. Throughput includes the entire scheduled "
              "trace and drain. Control overhead includes live telemetry, planning and admission. TTFT is engine token "
              "readiness from scheduled arrival, not streaming HTTP delivery. Quotas are enforced reservation budgets, "
              "not physical VRAM exhaustion. The small quality audit and heuristic uncertainty margins do not prove a "
              "task-quality bound or statistical confidence. Fresh v0.3 comparators run in the new study; historical "
              "results/live measurements remain separately archived.")
    # Retain phase-specific prediction errors alongside the full raw events.
    phase_errors = defaultdict(list)
    for path in (OUT / "runs").glob("*.json"):
        run = json.loads(path.read_text())
        for event in run["iterations"]:
            phase = ("prefill" if event["decode_requests"] == 0 else
                     "mixed" if event["prefill_tokens"] else "decode")
            phase_errors[(run["summary"]["policy"], phase)].append(abs(event["execution_ms"]-event["predicted_execution_ms"]))
    (OUT / "prediction_by_phase.json").write_text(json.dumps([
        {"policy": p, "phase": phase, "iterations": len(values), "mae_ms": mean(values)}
        for (p, phase), values in sorted(phase_errors.items())], indent=2), encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figures = OUT / "figures"
    figures.mkdir(exist_ok=True)
    for key, label, scale, filename in (("search_admission_wall_fraction", "Control / workload wall time (%)", 100, "control_overhead"),
                                         ("output_tokens_per_s", "Generated output tokens/s", 1, "controller_throughput")):
        fig, axes = plt.subplots(1, len(scenarios), figsize=(14, 4), squeeze=False)
        for ax, scenario in zip(axes[0], scenarios):
            ax.bar(POLICIES, [scale*avg(scenario,p,key) for p in POLICIES],
                   color=["#176b87" if p == "joint" else "#91b9c7" for p in POLICIES])
            for i, policy in enumerate(POLICIES):
                ax.scatter([i]*len(groups[(scenario,policy)]),
                           [scale*float(r[key]) for r in groups[(scenario,policy)]], color="#203c4d", s=16)
            if filename == "control_overhead":
                ax.axhline(5, color="#bb683c", linestyle="--", linewidth=1)
            ax.set_title(scenario)
            ax.set_ylabel(label)
            ax.tick_params(axis="x", rotation=55)
            ax.grid(axis="y", alpha=.2)
        fig.suptitle("Real GPT-2 controller comparison; individual seeds shown")
        fig.tight_layout()
        fig.savefig(figures / f"{filename}.png", dpi=160)
        plt.close(fig)
    report = ["# Controller efficiency: real v0.4 comparison", "", verdict, "", evidence, "",
              *comparison, "", interpretation, "", "## Complete policy results", "", *table, "",
              "![Actual control overhead](figures/control_overhead.png)", "",
              "![Actual throughput](figures/controller_throughput.png)", "",
              "## Separate policy quality audits", "", *qtable, "", quality_note, "", limits, ""]
    (OUT / "comparison.md").write_text("\n".join(report), encoding="utf-8")
    content = "\n".join(["### Controller efficiency follow-up (v0.4)", "",
        "The new controller compiles resource accounting and calibration lookup, reuses stable plans for up to three intervening iterations with fresh resource checks, separates feedback by phase, and gates adaptive choices against a feasible full-precision reference with heuristic uncertainty margins. Adaptive admission searches a bounded prefix set. The comparison retains the same model, calibration and workload as M8 and adds fresh old-controller comparators and gate/reuse ablations.", "",
        evidence, "", *comparison, "", verdict, "", interpretation, "",
        "![Measured controller overhead: old, optimized and ablated policies](../results/controller/figures/control_overhead.png)", "",
        "![Measured throughput: old, optimized and ablated policies](../results/controller/figures/controller_throughput.png)", "",
        *qtable, "", quality_note, "", limits, "",
        "All arms, latency tails, reuse fractions and prediction errors appear in `results/controller/comparison.md`; raw events and exact source hashes accompany each run. `docs/controller_efficiency.md` documents the algorithm and reproduction commands.", ""])
    insert_block(ROOT / "docs/final_report.md", "CONTROLLER_RESULTS", content, "## 7. Analytical simulation results")
    slide_table = ["| Scenario | Old control % | New control % | Old token/s | New token/s |", "|---|---:|---:|---:|---:|"]
    for s in scenarios:
        slide_table.append(f"| {s} | {100*avg(s,'joint_v03','search_admission_wall_fraction'):.1f} | {100*avg(s,'joint','search_admission_wall_fraction'):.1f} | {avg(s,'joint_v03','output_tokens_per_s'):.1f} | {avg(s,'joint','output_tokens_per_s'):.1f} |")
    slides = "\n".join(["# 14. Controller efficiency: measured follow-up", "",
        "36 real traffic runs, fresh old-controller controls, gate/reuse ablations.", "", *slide_table, "", verdict, "",
        "<!-- Speaker notes: Compare actual full-precision arms as well as joint. Means of two seeds do not establish a reliable speedup. Complete latency, per-seed variability and prediction errors are in results/controller/comparison.md. -->", "", "---", "",
        "# 15. Faster decisions with fresh resource checks", "",
        "- Compile safe reservation formulas and cache shape lookups.",
        "- Reuse stable plans; sample capacity and check reservations each tick.",
        "- Separate prefill, mixed and decode feedback.",
        "- Conservative full-precision reference; capacity necessity bypasses the gate.",
        f"- {validation['tests_passed']} tests; four-client real HTTP check passes.", "",
        "Reference kernels remain synchronous. Short audits do not guarantee task quality.", "",
        "<!-- Speaker notes: Reuse can delay reconsidering existing pending arrivals by three iterations. Heuristic uncertainty is not a confidence interval. Compilation, pruning and phase feedback are combined; only reuse and gate have separate ablations. Every reused event records at least one fresh resource check. -->", "", "---", ""])
    deck = ROOT / "docs/presentation.md"
    deck_text = deck.read_text(encoding="utf-8").replace("# 14. What remains uncertain", "# 16. What remains uncertain").replace("# 15. Answer and next step", "# 17. Answer and next step")
    deck.write_text(deck_text, encoding="utf-8")
    insert_block(deck, "CONTROLLER_SLIDES", slides, "# 16. What remains uncertain")
    print(verdict)
    print("Updated comparison, two figures, report section and two slides.")


if __name__ == "__main__":
    main()
