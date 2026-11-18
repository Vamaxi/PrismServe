import json
from pathlib import Path
import tempfile
import unittest

from prismserve.simulator import simulate, percentile
from prismserve.types import HardwareProfile, ModelSpec, PlannerConfig, RequestSpec
from prismserve.workload import synthetic_trace, read_trace, write_trace


class SimulatorTests(unittest.TestCase):
    def setUp(self):
        self.profile = HardwareProfile("test", 128 * 2**20, 2**30, 20, 10, 4, 1, .1)
        self.model = ModelSpec("test", 2, 32, 2, 16, 10000)
        self.config = PlannerConfig()

    def test_one_output_finishes_at_prefill(self):
        result = simulate([RequestSpec("a", 0, 3, 1)], self.profile, self.model, self.config,
                          prefill_chunk_size=2)
        self.assertEqual(result["summary"]["iterations"], 2)
        self.assertEqual(result["requests"][0]["completed_s"], result["requests"][0]["output_token_times_s"][0])
        self.assertIsNone(result["requests"][0]["tbt_p95_ms"])

    def test_rejection_counts_in_slo_failure(self):
        model = ModelSpec("oversize", 2, 32, 2, 16, 10**10)
        result = simulate([RequestSpec("a", 0, 3, 2)], self.profile, model, self.config)
        self.assertEqual(result["summary"]["completed"], 0)
        self.assertEqual(result["summary"]["rejected"], 1)
        self.assertEqual(result["summary"]["slo_violation_rate"], 1)
        self.assertIsNone(result["summary"]["ttft_p95_ms"])

    def test_output_count_no_early_arrival_and_repeatable_virtual_time(self):
        trace = synthetic_trace(7, 5, .1, prompt_min=2, prompt_max=4, output_min=2, output_max=3)
        a = simulate(trace, self.profile, self.model, self.config)
        b = simulate(trace, self.profile, self.model, self.config)
        self.assertEqual(a["requests"], b["requests"])
        self.assertEqual(a["summary"]["completed"], len(trace))
        for row in a["requests"]:
            self.assertEqual(len(row["output_token_times_s"]), row["output_tokens"])
            self.assertGreaterEqual(row["admitted_s"], row["arrival_s"])
            self.assertTrue(all(t >= row["arrival_s"] for t in row["output_token_times_s"]))

    def test_csv_roundtrip_and_duplicate_validation(self):
        trace = synthetic_trace(requests=4)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.csv"
            write_trace(trace, path)
            self.assertEqual(read_trace(path), trace)
            write_trace(trace + trace, path)
            with self.assertRaises(ValueError):
                read_trace(path)

    def test_percentile(self):
        self.assertEqual(percentile([1, 3], .5), 2)
        self.assertIsNone(percentile([], .95))

    def test_small_trace_all_policies_respect_capacity(self):
        trace = synthetic_trace(requests=5, duration_s=.1, prompt_min=4, prompt_max=16,
                                output_min=2, output_max=5)
        for policy in ["full", "static_int4", "morph_only", "offload_only", "naive", "joint"]:
            result = simulate(trace, self.profile, self.model, self.config, policy=policy)
            self.assertEqual(result["summary"]["completed"] + result["summary"]["rejected"], 5)
            self.assertLessEqual(result["summary"]["peak_device_bytes"], self.profile.device_memory_bytes * .9)
            json.dumps(result, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
