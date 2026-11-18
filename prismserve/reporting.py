"""Generate course report tables and static figures from saved raw experiment results."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean


def build_report(output: Path) -> Path:
    with (output / "summary.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("no experiment rows")
    groups = {}
    for row in rows:
        groups.setdefault((row["profile"], row["policy"]), []).append(row)
    lines = ["# PrismServe experiment results", "", "**Analytical simulation. These are not measured LLM serving results.**", "",
             "The rates in config.json describe hypothetical effective hardware. The same traces and seeds are replayed for every policy.", "",
             "| Profile | Policy | P95 TTFT ms (mean across seeds) | P95 TBT ms | SLO failure % | Output token/s | Good request/s |",
             "|---|---|---:|---:|---:|---:|---:|"]
    def avg(group, key):
        vals = [float(r[key]) for r in group if r[key] not in {"", "None"}]
        return mean(vals) if vals else None
    def fmt(v):
        return f"{v:.2f}" if v is not None else "N/A"
    for (profile, policy), group in groups.items():
        numbers = [avg(group, x) for x in ["ttft_p95_ms", "tbt_p95_ms", "slo_violation_rate", "output_tokens_per_s", "goodput_requests_per_s"]]
        numbers[2] = 100 * numbers[2] if numbers[2] is not None else None
        lines.append(f"| {profile} | {policy} | " + " | ".join(fmt(x) for x in numbers) + " |")
    lines += ["", "## Interpretation and evidence limits", "",
              "SLO failure counts each rejected request as a failure, plus completed requests whose TTFT or worst intertoken gap exceeds the configured SLO. TTFT percentiles cover completed requests; always inspect rejection counts in summary.csv.", "",
              "Full reserves full-precision weights. static_int4 is an unconstrained quality reference. morph_only and offload_only are mechanism-inspired analytical baselines, not the original MorphServe or NEO systems. naive uses a fixed combination; joint searches combinations with one common hardware interface.", "",
              "The model accounts for host weight backups, packed weights with scales, full context reservations, workspace, migration, and transition peaks. Precision selection uses a synthetic additive risk proxy; this is not perplexity, accuracy, or a guaranteed quality bound.", "",
              "The CPU/device pipeline overlap assumption can overestimate achieved throughput. The real backend synchronizes transfers and does not implement NEO's ISPC kernels or MorphServe's asynchronous in-place allocator.", "",
              "aggregate.json includes bootstrap intervals and paired differences across seeds. Three seeds provide exploratory variability estimates only. Planner search times are measured host wall times and excluded from virtual token latency; they cannot establish a production scheduler overhead percentage.", "",
              "Inspect runs/*.json for every request's token timestamps and every plan transition. Reproduce from config.json and traces/, with hashes recorded in manifest.json.", ""]
    figures = _plots(output, groups)
    if figures:
        lines += ["## Figures", ""] + [f"![{name}]({path})" for name, path in figures]
    dest = output / "report.md"
    dest.write_text("\n".join(lines), encoding="utf-8")
    return dest


def _plots(output, groups):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    figure_dir = output / "figures"
    figure_dir.mkdir(exist_ok=True)
    profiles = list(dict.fromkeys(p for p, _ in groups))
    figure_files = []
    for metric, label in [("ttft_p95_ms", "P95 TTFT (ms)"), ("slo_violation_rate", "SLO failure fraction")]:
        fig, axes = plt.subplots(1, len(profiles), figsize=(5 * len(profiles), 4), squeeze=False)
        for ax, profile in zip(axes[0], profiles):
            selected = [(policy, rows) for (p, policy), rows in groups.items() if p == profile]
            policies = [p for p, _ in selected]
            values = [[float(r[metric]) for r in rows if r[metric] not in {"", "None"}] for _, rows in selected]
            means = [mean(v) if v else 0 for v in values]
            ax.bar(policies, means, color=["#176b87" if p == "joint" else "#88b5c6" for p in policies])
            for i, vals in enumerate(values):
                ax.scatter([i] * len(vals), vals, s=10, color="#192e3d", zorder=3)
            ax.set_title(profile)
            ax.set_ylabel(label)
            ax.tick_params(axis="x", rotation=65)
            ax.grid(axis="y", alpha=.2)
        fig.suptitle("ANALYTICAL SIMULATION — individual seeds shown")
        fig.tight_layout()
        path = figure_dir / f"{metric}.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        figure_files.append((label, f"figures/{path.name}"))
    run_paths = sorted((output / "runs").glob("*__joint__*.json"))
    if run_paths:
        run = json.loads(run_paths[0].read_text(encoding="utf-8"))
        iters = run["iterations"]
        if iters:
            fig, axes = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
            t = [x["start_s"] for x in iters]
            axes[0].step(t, [len(x["quantized_layers"]) for x in iters], where="post", label="INT4 layers")
            axes[0].step(t, [len(x["cpu_request_ids"]) for x in iters], where="post", label="CPU attention requests")
            axes[0].legend()
            axes[1].plot(t, [x["device_bytes"] / 2**30 for x in iters], label="Reserved device GiB")
            axes[1].plot(t, [x["pending"] for x in iters], label="Queued requests")
            axes[1].set_xlabel("Virtual time (s)")
            axes[1].legend()
            fig.suptitle(f"SIMULATED joint controller: {run['summary']['profile']}, seed {run['summary']['seed']}")
            fig.tight_layout()
            fig.savefig(figure_dir / "controller_timeline.png", dpi=160)
            plt.close(fig)
            figure_files.append(("Simulated controller timeline", "figures/controller_timeline.png"))
    return figure_files
