"""Check the v0.4 raw traffic evidence, quotas, reuse and archived sources."""
from pathlib import Path
import hashlib
import json
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    output = ROOT / "results/controller"
    manifest = json.loads((output / "manifest.json").read_text())
    config = json.loads((output / "config.json").read_text())
    runs = [json.loads(p.read_text()) for p in sorted((output / "runs").glob("*.json"))]
    expected = len(config["policies"]) * len(config["seeds"]) * len(config["scenarios"])
    assert len(runs) == manifest["serving_runs"] == expected
    traces, completed, rejected, reused, peaks_over = {}, 0, 0, 0, 0
    for run in runs:
        row = run["summary"]
        key = (row["scenario"], row["seed"])
        assert traces.setdefault(key, run["trace"]) == run["trace"]
        assert row["requests"] == row["completed"] + row["rejected"] == config["workload"]["requests"]
        assert not run["metrics"]["fatal_error"]
        assert row["mode"] == "greedy_generation"
        assert run["config"]["scheduler_strategy"] == ("legacy" if row["policy"].endswith("_v03") else "efficient")
        for request in run["requests"]:
            count = config["workload"]["output_tokens"]
            assert request["completion_tokens"] == count
            assert len(request["output_token_times_s"]) == count
            assert len(request["decode_token_ms"]) == count-1
            assert len(request["precision_history"]) == count
            assert request["admitted_s"] >= request["arrival_s"]
            assert not request["target_nll"]
        for event in run["iterations"]:
            assert event["projection_calls"] == 48
            assert event["device_reserved_bytes"] <= run["config"]["device_memory_budget_bytes"]
            assert event["quality_proxy"] <= run["config"]["quality_budget"] + 1e-12
            assert event["telemetry"]["host_available_bytes"] > 0
            assert abs(event["search_and_admission_ms"]-event["telemetry_ms"]-event["planning_ms"]) < 1e-6
            if event["plan_reused"]:
                assert event["resource_checks"] >= 1
                assert event["estimate_components"]["memory_feasible"]
                assert event["estimate_components"]["quality_proxy_feasible"]
                reused += 1
        if row["policy"] == "joint_no_reuse":
            assert row["plan_reuse_fraction"] == 0
        completed += row["completed"]
        rejected += row["rejected"]
        peaks_over += row["peak_cuda_allocated_bytes"] > row["device_budget_mib"] * 2**20
    assert reused > 0
    quality = json.loads((output / "quality.json").read_text())["policy_summaries"]
    assert len(quality) == manifest["quality_audits"] == len(config["policies"])
    targets = config["quality_audit"]["requests"] * config["quality_audit"]["output_tokens"]
    assert all(r["teacher_forced_targets"] == targets for r in quality)
    audit_trace = None
    for policy in config["policies"]:
        audit = json.loads((output / "quality_runs" / f"{policy}.json").read_text())
        if audit_trace is None:
            audit_trace = audit["trace"]
        assert audit["trace"] == audit_trace
        assert not audit["metrics"]["fatal_error"]
        assert audit["summary"]["mode"] == "teacher_forced_quality_audit"
        assert sum(len(r["target_nll"]) for r in audit["requests"]) == targets
        for event in audit["iterations"]:
            assert event["device_reserved_bytes"] <= audit["config"]["device_memory_budget_bytes"]
        assert audit["summary"]["peak_cuda_allocated_bytes"] <= config["quality_audit"]["device_budget_mib"] * 2**20
    with zipfile.ZipFile(output / "source_snapshot.zip") as archive:
        for name, expected_hash in manifest["source_sha256"].items():
            assert hashlib.sha256(archive.read("prismserve/"+name)).hexdigest() == expected_hash
            # Historical M9 measurements are checked against their snapshot;
            # later milestones can extend the working runtime independently.
    assert digest(output / "source_snapshot.zip") == manifest["source_snapshot_sha256"]
    assert digest(output / "calibration.json") == manifest["calibration_sha256"]
    assert json.loads((output / "http_smoke.json").read_text())["passed"]
    tests = json.loads((ROOT / "results/validation.json").read_text())
    assert tests["successful"]
    result = {"passed": True, "serving_runs": len(runs), "quality_audits": len(quality),
              "completed_requests": completed, "rejected_requests": rejected,
              "greedy_output_tokens": completed * config["workload"]["output_tokens"],
              "teacher_forced_targets": sum(r["teacher_forced_targets"] for r in quality),
              "reused_iterations_with_fresh_resource_checks": reused,
              "actual_cuda_peak_over_quota_runs": peaks_over,
              "tests_passed": tests["tests_run"], "paired_traces_verified": True,
              "archived_source_hashes_verified": True, "concurrent_http_passed": True}
    (output / "validation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
