"""Reproducible experiment runner, paired comparisons, and bootstrap intervals."""
from __future__ import annotations

import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import random
from statistics import mean
import sys

from .simulator import percentile, simulate
from .types import HardwareProfile, ModelSpec, PlannerConfig
from .workload import read_trace, synthetic_trace, write_trace


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def bootstrap_mean(values: list[float], seed: int = 2026, samples: int = 2000) -> dict:
    if not values:
        return {"mean": None, "ci95_low": None, "ci95_high": None, "n": 0}
    rng = random.Random(seed)
    draws = [mean(rng.choices(values, k=len(values))) for _ in range(samples)]
    return {"mean": mean(values), "ci95_low": percentile(draws, .025),
            "ci95_high": percentile(draws, .975), "n": len(values)}


def run_experiments(config_path: Path, output: Path, trace_path: Path | None = None,
                    progress=print) -> dict:
    raw = config_path.read_bytes()
    config = json.loads(raw)
    if config.get("schema_version") != 1:
        raise ValueError("config schema_version must be 1")
    model = ModelSpec(**config["model"])
    planner = PlannerConfig(**config.get("planner", {}))
    simulation = config.get("simulation", {})
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "config.json", config)
    summaries = []
    traces = {}
    seeds = config.get("seeds", [7])
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be nonempty and unique")
    if trace_path and len(seeds) != 1:
        raise ValueError("external CSV is one deterministic trace: configure exactly one seed")
    for seed in seeds:
        trace = read_trace(trace_path) if trace_path else synthetic_trace(seed=seed, **config.get("workload", {}))
        traces[seed] = trace
        write_trace(trace, output / "traces" / f"seed_{seed}.csv")
    run_index = 0
    for profile_data in config["profiles"]:
        profile = HardwareProfile(**profile_data)
        # Reject path separators from externally editable profile/policy names.
        if any(c in profile.name for c in '/\\:') or profile.name in {".", ".."}:
            raise ValueError("profile name must be a simple filename component")
        for policy in config.get("policies", ["joint"]):
            if not isinstance(policy, str) or any(c in policy for c in '/\\:') or policy in {".", ".."}:
                raise ValueError("policy name must be a simple filename component")
            for seed, trace in traces.items():
                run_index += 1
                progress(f"[{run_index}] {profile.name} / {policy} / seed {seed}")
                result = simulate(trace, profile, model, planner, policy=policy, **simulation)
                result["summary"]["seed"] = seed
                summaries.append(result["summary"])
                write_json(output / "runs" / f"{profile.name}__{policy}__{seed}.json", result)
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    groups = {}
    metrics = ["ttft_p95_ms", "tbt_p95_ms", "slo_violation_rate", "output_tokens_per_s", "goodput_requests_per_s"]
    for row in summaries:
        groups.setdefault((row["profile"], row["policy"]), []).append(row)
    aggregates = []
    for (profile_name, policy), rows in groups.items():
        aggregates.append({"profile": profile_name, "policy": policy,
                           **{key: bootstrap_mean([r[key] for r in rows if r[key] is not None]) for key in metrics}})
    pairs = []
    lookup = {(r["profile"], r["policy"], r["seed"]): r for r in summaries}
    for profile_name in {r["profile"] for r in summaries}:
        for baseline in config.get("policies", []):
            if baseline == "joint":
                continue
            differences = []
            for seed in seeds:
                a = lookup.get((profile_name, baseline, seed))
                b = lookup.get((profile_name, "joint", seed))
                if a and b and a["ttft_p95_ms"] is not None and b["ttft_p95_ms"] is not None:
                    differences.append(a["ttft_p95_ms"] - b["ttft_p95_ms"])
            pairs.append({"profile": profile_name, "baseline": baseline,
                          "ttft_ms_baseline_minus_joint": bootstrap_mean(differences)})
    write_json(output / "aggregate.json", {"groups": aggregates, "paired_comparisons": pairs,
               "ci_note": "Seed-level nonparametric bootstrap; n=3 is exploratory, not strong evidence. Positive paired TTFT means joint improves."})
    manifest = {"mode": "analytical_simulation", "created_utc": datetime.now(timezone.utc).isoformat(),
                "python": sys.version, "platform": platform.platform(),
                "config_sha256": hashlib.sha256(raw).hexdigest(), "runs": len(summaries),
                "trace_sha256": {str(s): hashlib.sha256((output / 'traces' / f'seed_{s}.csv').read_bytes()).hexdigest() for s in seeds},
                "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(__file__).parent.glob('*.py'))}}
    write_json(output / "manifest.json", manifest)
    from .reporting import build_report
    build_report(output)
    return {"output": str(output.resolve()), "runs": len(summaries)}
