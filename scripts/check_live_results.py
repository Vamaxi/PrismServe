"""Validate saved actual serving evidence without rerunning the GPU workload."""
from pathlib import Path
import hashlib
import json
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main():
    output = ROOT / "results/live"
    manifest = json.loads((output / "manifest.json").read_text())
    paths = sorted((output / "runs").glob("*.json"))
    assert len(paths) == manifest["serving_runs"] == 42
    traces = {}
    total_completed, total_rejected = 0, 0
    batches = []
    peak_budget_exceedances = 0
    for path in paths:
        result = json.loads(path.read_text())
        summary = result["summary"]
        assert summary["mode"] == "greedy_generation"
        assert summary["requests"] == summary["completed"] + summary["rejected"] == 12
        assert not result["metrics"]["fatal_error"]
        key = (summary["scenario"], summary["seed"])
        previous = traces.setdefault(key, result["trace"])
        assert previous == result["trace"], "policy inputs differ"
        for row in result["requests"]:
            assert row["completion_tokens"] == 24
            assert len(row["output_token_times_s"]) == 24
            assert len(row["decode_token_ms"]) == 23
            assert len(row["precision_history"]) == 24
            assert row["admitted_s"] >= row["arrival_s"]
            assert all(t >= row["arrival_s"] for t in row["output_token_times_s"])
            assert all(b >= a for a, b in zip(row["output_token_times_s"], row["output_token_times_s"][1:]))
            assert not row["target_nll"], "greedy benchmark mislabeled as teacher forcing"
        for event in result["iterations"]:
            assert event["projection_calls"] == 48
            assert event["linear_tokens"] >= event["batch_size"]
            assert event["device_reserved_bytes"] <= result["config"]["device_memory_budget_bytes"]
        if summary["peak_cuda_allocated_bytes"] > summary["device_budget_mib"] * 2**20:
            peak_budget_exceedances += 1
        batches.append(summary["max_execution_batch_size"])
        total_completed += summary["completed"]
        total_rejected += summary["rejected"]
    quality = json.loads((output / "quality.json").read_text())
    assert len(quality["policy_summaries"]) == 7
    assert all(row["teacher_forced_targets"] == 64 for row in quality["policy_summaries"])
    source = json.loads((ROOT / "data/wikitext2/source.json").read_text())
    for split in ("validation", "test"):
        record = source["splits"][split]
        assert digest(ROOT / "data/wikitext2" / record["file"]) == record["text_sha256"]
    with zipfile.ZipFile(output / "source_snapshot.zip") as archive:
        for name, expected in manifest["source_sha256"].items():
            assert hashlib.sha256(archive.read("prismserve/" + name)).hexdigest() == expected
            # Historical physical timings belong to their archived source.
            # The current working tree can contain a newer controller.
    assert digest(output / "calibration.json") == manifest["calibration_sha256"]
    assert digest(output / "source_snapshot.zip") == manifest["source_snapshot_sha256"]
    assert json.loads((output / "http_smoke.json").read_text())["passed"]
    parity = json.loads((output / "real_batch_parity.json").read_text())
    assert parity["passed"]
    tests = json.loads((ROOT / "results/validation.json").read_text())
    assert tests["successful"]
    result = {"passed": True, "serving_runs": 42, "quality_audits": 7,
              "completed_requests": total_completed, "rejected_requests": total_rejected,
              "greedy_output_tokens": total_completed * 24, "teacher_forced_audit_targets": 7 * 64,
              "max_actual_execution_batch": max(batches), "projection_calls_per_iteration_verified": 48,
              "calibration_shapes": len(json.loads((output / "calibration.json").read_text())["timing"]["samples"]),
              "reservation_quota_checks_passed": True, "runs_with_actual_cuda_peak_above_declared_quota": peak_budget_exceedances,
              "tests_passed": tests["tests_run"], "exact_source_snapshot_verified": True,
              "paired_traces_verified": True, "dataset_hashes_verified": True,
              "real_concurrent_http_passed": True,
              "note": "Quotas govern admission estimates. Actual CUDA peaks are reported separately; no physical OOM is forced."}
    (output / "validation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
