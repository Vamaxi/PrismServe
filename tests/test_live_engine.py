"""Actual tiny-model worker tests: request isolation, accounting and cleanup."""
import os
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

import unittest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from prismserve.pretrained import GPT2Executor
from prismserve.live_engine import ContinuousEngine, ServingConfig
from prismserve.measured_policy import TimingCalibration, TimingSample
from prismserve.types import HardwareProfile


class Tokenizer:
    eos_token_id = 63

    def encode(self, text, **kwargs):
        return [ord(char) % 60 for char in text]

    def decode(self, tokens, **kwargs):
        return " ".join(map(str, tokens))


class LiveEngineTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(min(4, torch.get_num_threads()))
        torch.manual_seed(3)
        config = GPT2Config(vocab_size=64, n_positions=32, n_embd=16, n_layer=2, n_head=2,
                            resid_pdrop=0, embd_pdrop=0, attn_pdrop=0)
        config._attn_implementation = "eager"
        self.executor = GPT2Executor(GPT2LMHeadModel(config).eval(), Tokenizer())
        self.profile = HardwareProfile("test_cpu", 2**30, 2**30, 10, 10, 0, .1, .1, device_kind="cpu")
        self.timing = TimingCalibration(tuple(TimingSample((), b, p, 8, 0, 1, 0)
                                              for b in [1, 2, 4] for p in [0, 4, 16]))
        self.config = ServingConfig(max_batch_size=4, prefill_chunk_size=2, prefill_token_budget=8,
                                    backend_margin_bytes=0)

    def test_multirequest_matches_independent_generation(self):
        prompts = ["abcde", "ghi", "jklm", "no"]
        expected = [self.executor.generate(prompt, 4, stop_on_eos=False)["token_ids"] for prompt in prompts]
        engine = ContinuousEngine(self.executor, self.timing, self.profile, self.config, policy="full")
        futures = [engine.submit(prompt, 4, str(i), stop_on_eos=False) for i, prompt in enumerate(prompts)]
        engine.start()
        try:
            rows = [future.result(timeout=15) for future in futures]
        finally:
            engine.close()
        self.assertEqual([r["token_ids"] for r in rows], expected)
        self.assertEqual(engine.metrics()["completed_requests"], 4)
        self.assertGreater(engine.metrics()["max_execution_batch_size"], 1)
        for row in rows:
            self.assertEqual(len(row["output_token_times_s"]), 4)
            self.assertEqual(len(row["decode_token_ms"]), 3)
            self.assertTrue(all(t >= row["arrival_s"] for t in row["output_token_times_s"]))
            self.assertFalse(any(row["cpu_offload_history"]))
        self.assertTrue(all(e["projection_calls"] == 8 for e in engine.snapshot()["iterations"]))

    def test_teacher_forced_targets_are_scored_under_live_state(self):
        engine = ContinuousEngine(self.executor, self.timing, self.profile, self.config, policy="full")
        future = engine.submit_tokens([1, 2, 3], 3, forced_tokens=[4, 5, 6], stop_on_eos=False)
        engine.start()
        try:
            row = future.result(timeout=15)
        finally:
            engine.close()
        self.assertEqual(row["token_ids"], [4, 5, 6])
        self.assertEqual(len(row["target_nll"]), 3)
        self.assertTrue(all(value > 0 for value in row["target_nll"]))
        self.assertEqual(row["evaluation_mode"], "teacher_forced_quality_audit")

    def test_close_cancels_queued_work_and_forbids_new_work(self):
        engine = ContinuousEngine(self.executor, self.timing, self.profile, self.config, policy="full")
        futures = [engine.submit("abcd", 3) for _ in range(2)]
        engine.close(drain=False)
        self.assertTrue(all(future.cancelled() for future in futures))
        self.assertFalse(engine.metrics()["worker_running"])
        with self.assertRaises(RuntimeError):
            engine.submit("a", 1)

    def test_impossible_reservation_is_rejected_and_counted(self):
        from dataclasses import replace
        engine = ContinuousEngine(self.executor, self.timing, self.profile,
                                  replace(self.config, host_memory_budget_bytes=1), policy="full")
        future = engine.submit("abcd", 3)
        engine.start()
        try:
            with self.assertRaises(MemoryError):
                future.result(timeout=15)
        finally:
            engine.close()
        self.assertEqual(engine.metrics()["rejected_requests"], 1)

    def test_invalid_request_and_queue_bound(self):
        from dataclasses import replace
        engine = ContinuousEngine(self.executor, self.timing, self.profile,
                                  replace(self.config, max_pending_requests=1), policy="full")
        with self.assertRaises(ValueError):
            engine.submit("a" * 32, 2)
        engine.submit("a", 2)
        with self.assertRaises(OverflowError):
            engine.submit("b", 2)

    def test_mismatched_calibration_device_is_rejected(self):
        from dataclasses import replace
        with self.assertRaises(ValueError):
            ContinuousEngine(self.executor, self.timing,
                             replace(self.profile, device_kind="cuda", link_bandwidth_gbps=4), self.config)

    def test_stable_plan_reuse_still_checks_resources_every_tick(self):
        from unittest.mock import patch
        engine = ContinuousEngine(self.executor, self.timing, self.profile, self.config, policy='full')
        futures = [engine.submit('ab',12,stop_on_eos=False) for _ in range(2)]
        with patch.object(engine,'_resource_profile',wraps=engine._resource_profile) as checks:
            engine.start()
            try:
                for future in futures:
                    future.result(timeout=15)
            finally:
                engine.close()
            events = engine.snapshot()['iterations']
            self.assertEqual(checks.call_count,len(events))
            self.assertGreater(sum(e['plan_reused'] for e in events),0)

    def test_capacity_loss_invalidates_a_reused_plan_before_execution(self):
        from dataclasses import replace
        from unittest.mock import patch
        engine = ContinuousEngine(self.executor, self.timing, self.profile, self.config, policy='full')
        futures = [engine.submit('ab',12,stop_on_eos=False) for _ in range(2)]
        original = engine._resource_profile
        def changing_profile():
            resource, telemetry = original()
            if len(engine._events)>=3:
                resource = replace(resource,host_memory_bytes=1)
            return resource,telemetry
        with patch.object(engine,'_resource_profile',side_effect=changing_profile):
            engine.start()
            try:
                for future in futures:
                    with self.assertRaises(RuntimeError):
                        future.result(timeout=15)
            finally:
                engine.close()
        self.assertEqual(len(engine.snapshot()['iterations']),3)
        self.assertEqual(engine.metrics()['worker_status'],'failed')
        detail=engine.snapshot()['failure_diagnostic']
        self.assertEqual(detail['profile_at_failure']['host_memory_bytes'],1)
        self.assertFalse(detail['current_plan_estimate']['components']['memory_feasible'])
        self.assertEqual(detail['iteration'],3)
        self.assertTrue(detail['active_work'])


if __name__ == "__main__":
    unittest.main()
