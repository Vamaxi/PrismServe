"""Exploratory offered-load sweep, using the same requests at every rate.

Run from the repository root without installing the package::

    python scripts/load_sweep.py --output results/load_sweep

No model is executed. All service times and hardware profiles are analytical.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import time


REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY))

from prismserve.experiments import write_json
from prismserve.simulator import simulate
from prismserve.types import HardwareProfile, ModelSpec, PlannerConfig
from prismserve.workload import synthetic_trace, write_trace


POLICIES = ("full", "morph_only", "offload_only", "naive", "joint")
LOAD_FACTORS = (0.125, 0.25, 0.5, 1.0, 2.0)


def plot_results(rows: list[dict], output: Path, profile_name: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"full": "#52606d", "morph_only": "#b26a00", "offload_only": "#00857c",
              "naive": "#a44b83", "joint": "#176bcb"}
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for policy in POLICIES:
        selected = sorted((r for r in rows if r["policy"] == policy), key=lambda r: r["load_factor"])
        visible = [r for r in selected if r["ttft_p95_ms"] is not None]
        ax.plot([r["output_tokens_per_s"] for r in visible], [r["ttft_p95_ms"] for r in visible],
                marker="o", color=colors[policy], label=policy, linewidth=1.8)
        for row in visible:
            ax.annotate(f'{row["load_factor"]:g}x',
                        (row["output_tokens_per_s"], row["ttft_p95_ms"]),
                        xytext=(4, 4), textcoords="offset points", fontsize=8, color=colors[policy])
    ax.set_xlabel("Achieved output throughput (tokens/s, including drain time)")
    ax.set_ylabel("P95 time to first token (ms, completed requests)")
    ax.set_title(f"ANALYTICAL SIMULATION | {profile_name}\nSame requests, timestamp scaling; seed 7 is the default")
    ax.grid(alpha=.22)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "throughput_latency.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for policy in POLICIES:
        selected = sorted((r for r in rows if r["policy"] == policy), key=lambda r: r["load_factor"])
        ax.plot([r["offered_requests_per_s"] for r in selected],
                [100 * r["slo_violation_rate"] for r in selected],
                marker="o", color=colors[policy], label=policy, linewidth=1.8)
    ax.set_xlabel("Offered load (requests/s over configured arrival window)")
    ax.set_ylabel("SLO failures (% of all offered requests)")
    ax.set_ylim(-2, 102)
    ax.set_title(f"ANALYTICAL SIMULATION | {profile_name}\nRejected requests count as SLO failures")
    ax.grid(alpha=.22)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "failures_vs_load.png", dpi=170)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPOSITORY / "configs" / "experiment.json")
    parser.add_argument("--output", type=Path, default=REPOSITORY / "results" / "load_sweep")
    parser.add_argument("--requests", type=int, default=48)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--duration", type=float, default=10.0,
                        help="Base arrival-window duration in seconds; divided by each load factor")
    args = parser.parse_args()
    if args.requests <= 0 or args.duration <= 0:
        parser.error("--requests and --duration must be positive")
    start = time.perf_counter()
    raw = args.config.read_bytes()
    config = json.loads(raw)
    if config.get("schema_version") != 1:
        raise ValueError("config schema_version must be 1")
    model = ModelSpec(**config["model"])
    profile = HardwareProfile(**config["profiles"][0])
    planner = PlannerConfig(**config.get("planner", {}))
    workload = {**config.get("workload", {}), "requests": args.requests, "duration_s": args.duration}
    base_trace = synthetic_trace(seed=args.seed, **workload)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "config.json", config)
    write_trace(base_trace, output / "traces" / "base.csv")
    summaries = []
    run_files = []
    for factor in LOAD_FACTORS:
        # All timestamps share t=0 as origin; token lengths and request IDs are
        # untouched. Bursts and quiet intervals compress by precisely 1/factor.
        trace = [replace(request, arrival_s=request.arrival_s / factor) for request in base_trace]
        trace_name = f"load_{factor:g}x.csv"
        write_trace(trace, output / "traces" / trace_name)
        window_s = args.duration / factor
        for policy in POLICIES:
            print(f"ANALYTICAL SIMULATION [{len(summaries)+1}/{len(LOAD_FACTORS)*len(POLICIES)}] "
                  f"{profile.name} | {policy} | {factor:g}x",
                  flush=True)
            result = simulate(trace, profile, model, planner, policy=policy, **config.get("simulation", {}))
            result["summary"].update(seed=args.seed, load_factor=factor,
                                     arrival_window_s=window_s,
                                     offered_requests_per_s=args.requests / window_s)
            result["load_sweep"] = {"base_trace": "traces/base.csv", "arrival_timestamp_scale": 1 / factor,
                                    "arrival_window_s": window_s, "seed": args.seed}
            run_name = f"{policy}__load_{factor:g}x.json"
            write_json(output / "runs" / run_name, result)
            run_files.append(run_name)
            summaries.append(result["summary"])
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    plot_results(summaries, output, profile.name)
    elapsed = time.perf_counter() - start
    trace_paths = sorted((output / "traces").glob("*.csv"))
    write_json(output / "manifest.json", {
        "mode": "analytical_simulation", "experiment": "offered_load_sweep",
        "created_utc": datetime.now(timezone.utc).isoformat(), "python": sys.version,
        "platform": platform.platform(), "profile": profile.name,
        "seed": args.seed, "requests_per_run": args.requests, "base_duration_s": args.duration,
        "load_factors": LOAD_FACTORS, "policies": POLICIES, "runs": len(summaries),
        "wall_seconds_measured": elapsed, "config_sha256": hashlib.sha256(raw).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in sorted((REPOSITORY / "prismserve").glob("*.py"))},
        "trace_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in trace_paths},
        "run_files": run_files,
        "limits": ["Single-seed exploratory sweep; no confidence intervals or statistical generalization.",
                   "Synthetic effective hardware and analytical service times; no LLM or GPU benchmark.",
                   "Same prompts, outputs, IDs and burst structure in every run; only arrival timestamps scale.",
                   "Throughput denominator includes idle gaps after first arrival and completion drain time.",
                   "Completed-request TTFT can hide rejection; inspect rejected and SLO failure rate together.",
                   "Policy search wall time is measured separately and excluded from virtual service latency.",
                   "Quality is an additive risk proxy, not measured task accuracy."]})
    lines = ["# Exploratory offered-load sweep", "",
             "**ANALYTICAL SIMULATION. These are not measured LLM/GPU performance results.**", "",
             f"Profile: `{profile.name}`. One seed ({args.seed}), {args.requests} requests per run, "
             f"base arrival window {args.duration:g} seconds. Load factors: "
             + ", ".join(f"{factor:g}x" for factor in LOAD_FACTORS) + ".", "",
             f"The exact same request IDs and token lengths are replayed in all {len(summaries)} runs. "
             "Multiplying offered load by f divides every arrival timestamp by f, preserving the burst structure.", "",
             "| Policy | Load factor | Offered request/s | Achieved output token/s | P95 TTFT ms | SLO failure % | Rejected |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for row in summaries:
        ttft = "N/A" if row["ttft_p95_ms"] is None else f'{row["ttft_p95_ms"]:.2f}'
        lines.append(f'| {row["policy"]} | {row["load_factor"]:g}x | {row["offered_requests_per_s"]:.2f} '
                     f'| {row["output_tokens_per_s"]:.2f} | {ttft} | {100*row["slo_violation_rate"]:.2f} '
                     f'| {row["rejected"]} |')
    lines += ["", "![Throughput versus P95 TTFT](throughput_latency.png)", "",
              "![SLO failures versus offered load](failures_vs_load.png)", "",
              "Throughput includes the completion drain after the arrival window. TTFT covers completed requests; "
              "the failure rate includes rejected requests and completed requests exceeding either configured SLO.", "",
              "This one-seed sweep is exploratory. It does not establish confidence intervals, hardware speedups, "
              "paper reproduction, an accuracy bound, or a production scheduling overhead percentage. "
              "Inspect manifest.json for all assumptions and raw runs/*.json for per-token timestamps.", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"output": str(output.resolve()), "runs": len(summaries),
                      "wall_seconds": round(elapsed, 2)}), flush=True)


if __name__ == "__main__":
    main()
