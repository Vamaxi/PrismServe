"""Measured policies must follow timings, resource limits, and live feedback."""
from dataclasses import replace
import json
import unittest

from prismserve.measured_policy import MeasuredController, TimingCalibration, TimingSample
from prismserve.types import ControlState, HardwareProfile, ModelSpec, PlannerConfig, WorkItem


def profile(**changes):
    values = dict(name="test", device_memory_bytes=2**30, host_memory_bytes=2**30,
                  device_bandwidth_gbps=200, host_bandwidth_gbps=30,
                  link_bandwidth_gbps=8, device_tflops=10, host_tflops=.2)
    values.update(changes)
    return HardwareProfile(**values)


def model():
    return ModelSpec("small", layers=4, hidden_size=64, kv_heads=1, head_dim=16,
                     parameters=1_000_000, sensitivity=(.001, .002, .003, .004),
                     reference_dequantization=True)


def sample(q=(), batch=1, prefill=0, context=64, cpu=0, ms=5, workspace=100_000):
    return TimingSample(q, batch, prefill, context, cpu, ms, workspace)


def controller(samples, *, hardware=None, config=None, policy="joint", candidates=((), (0,)), **kwargs):
    return MeasuredController(hardware or profile(), model(),
                              config or PlannerConfig(quality_penalty=0),
                              TimingCalibration(tuple(samples)), policy,
                              quantization_candidates=candidates, **kwargs)


class MeasuredPolicyTests(unittest.TestCase):
    def test_slower_reference_quantization_is_not_selected(self):
        policy = controller([sample(ms=5), sample(q=(0,), ms=20)])
        decision = policy.choose([WorkItem("a", 1024, existing_context_tokens=63)])
        self.assertTrue(decision.estimate.feasible)
        self.assertEqual(decision.plan.quantized_layers, ())
        self.assertEqual(decision.estimate.iteration_ms, 5)
        self.assertEqual(decision.estimate.components["timing_source"], "measured_actual_batches")

    def test_memory_pressure_selects_compressed_capacity_with_measured_scratch(self):
        samples = [sample(prefill=64, context=64, workspace=1_200_000),
                   sample(q=(0,), prefill=64, context=64, ms=30, workspace=1_200_000)]
        policy = controller(samples, hardware=profile(device_memory_bytes=5_600_000),
                            config=PlannerConfig(memory_fraction=1, quality_penalty=0))
        # Memory feasibility overrides the recently changed full-precision plan.
        decision = policy.choose([WorkItem("a", 4000, 64, 0)], ControlState(last_change_s=0), now_s=.01)
        self.assertTrue(decision.estimate.feasible)
        self.assertEqual(decision.plan.quantized_layers, (0,))
        self.assertGreater(decision.estimate.components["measured_workspace_bytes"], 1_500_000)
        self.assertEqual(decision.estimate.components["reserved_device_kv_copy_bytes"], 4000 * model().kv_bytes_per_token)
        self.assertLessEqual(decision.estimate.device_bytes, policy.profile.device_memory_bytes)
        self.assertEqual(decision.estimate.components["dequantization_scratch_bytes"], 0)

    def test_observed_workspace_can_reject_previously_fitting_candidates(self):
        policy = controller([sample(prefill=64, context=64, workspace=5_000_000),
                             sample(q=(0,), prefill=64, context=64, workspace=5_000_000)],
                            hardware=profile(device_memory_bytes=8_400_000),
                            config=PlannerConfig(memory_fraction=1))
        decision = policy.choose([WorkItem("a", 20000, 64, 0)])
        self.assertFalse(decision.estimate.feasible)
        self.assertIn("workspace", decision.estimate.reason)

    def test_latency_uses_current_context_while_capacity_reserves_future(self):
        policy = controller([sample(context=64, ms=5), sample(context=384, ms=50)])
        short = policy.choose([WorkItem("a", 1024, 0, 63)])
        future_long = policy.choose([WorkItem("a", 100000, 0, 63)])
        self.assertEqual(short.estimate.iteration_ms, future_long.estimate.iteration_ms)
        self.assertEqual(future_long.estimate.components["timing_features"]["context_tokens"], 64)
        self.assertGreater(future_long.estimate.device_bytes, short.estimate.device_bytes)

    def test_live_feedback_changes_next_choice_without_exploration(self):
        policy = controller([sample(ms=5), sample(q=(0,), ms=6)], ewma_alpha=1)
        items = [WorkItem("a", 512, 0, 63)]
        before = policy.choose(items)
        self.assertEqual(before.plan.quantized_layers, ())
        policy.observe(before, elapsed_ms=20)
        after = policy.choose(items)
        self.assertEqual(after.plan.quantized_layers, (0,))
        full_only = controller([sample(ms=5)], candidates=((),), ewma_alpha=1)
        initial = full_only.choose(items)
        full_only.observe(initial, 20)
        corrected = full_only.choose(items)
        self.assertEqual(corrected.estimate.iteration_ms, 20)
        self.assertEqual(corrected.estimate.components["ewma_correction"], 4)

    def test_one_time_transitions_and_cooldown_affect_selection(self):
        samples = (sample(ms=5), sample(q=(0,), ms=4))
        calibration = TimingCalibration(samples, to_int4_layer_ms=100, to_fp_layer_ms=40)
        policy = MeasuredController(profile(), model(), PlannerConfig(quality_penalty=0),
                                    calibration, quantization_candidates=((), (0,)))
        items = [WorkItem("a", 512, 0, 63)]
        initial = policy.choose(items)
        self.assertEqual(initial.plan.quantized_layers, ())
        retained = policy.choose(items, ControlState((0,), (), 0), now_s=.01)
        self.assertEqual(retained.plan.quantized_layers, (0,))
        # A current plan exceeding the quality proxy constraint cannot be kept
        # merely because its cooldown has not elapsed.
        strict = MeasuredController(profile(), model(), PlannerConfig(quality_budget=0), calibration,
                                    quantization_candidates=((), (0,)))
        restored = strict.choose(items, ControlState((0,), (), 0), now_s=.01)
        self.assertEqual(restored.plan.quantized_layers, ())
        self.assertEqual(restored.estimate.transition_ms, 40)

    def test_cpu_never_offloads_to_itself(self):
        p = profile(device_kind="cpu", link_bandwidth_gbps=0)
        policy = controller([sample(ms=20), sample(cpu=1, ms=1)], hardware=p, candidates=((),))
        decision = policy.choose([WorkItem("a", 512, 0, 63)])
        self.assertEqual(decision.plan.cpu_request_ids, ())

    def test_missing_offload_timing_family_is_rejected(self):
        policy = controller([sample(ms=10)], policy="offload_only", candidates=((),))
        decision = policy.choose([WorkItem("a", 512, 0, 63)])
        self.assertEqual(decision.plan.cpu_request_ids, ())
        self.assertTrue(decision.estimate.feasible)
        unknown = controller([sample(q=(0,), ms=1)], policy="full")
        decision = unknown.choose([WorkItem("a", 512, 0, 63)])
        self.assertFalse(decision.estimate.feasible)
        self.assertFalse(decision.estimate.components["timing_family_available"])

    def test_workspace_is_shape_specific_not_largest_batch_blanket(self):
        policy = controller([sample(batch=1, context=64, workspace=2_000_000),
                             sample(batch=8, context=384, workspace=80_000_000)], candidates=((),))
        decision = policy.choose([WorkItem("a", 512, 0, 63)])
        self.assertLess(decision.estimate.components["measured_workspace_bytes"], 5_000_000)
        self.assertEqual(decision.estimate.components["nearest_sample"]["batch_size"], 1)

    def test_larger_shapes_conservatively_scale_measured_table(self):
        policy = controller([sample(batch=1, context=64, workspace=2_000_000)], candidates=((),))
        decision = policy.choose([WorkItem("a", 512, 0, 127), WorkItem("b", 512, 0, 127)])
        self.assertEqual(decision.estimate.components["timing_shape_scale"], 4)
        self.assertEqual(decision.estimate.iteration_ms, 20)
        self.assertGreater(decision.estimate.components["measured_workspace_bytes"], 2_500_000)

    def test_future_workspace_is_reserved_before_context_growth(self):
        samples = [sample(prefill=32, context=32, workspace=2_000_000),
                   sample(context=200, workspace=20_000_000)]
        policy = controller(samples, candidates=((),), policy="full")
        initial = policy.choose([WorkItem("a", 200, 32, 0)])
        later = policy.choose([WorkItem("a", 200, 0, 180)])
        self.assertTrue(initial.estimate.feasible)
        self.assertTrue(later.estimate.feasible)
        self.assertEqual(initial.estimate.device_bytes, later.estimate.device_bytes)
        # A budget between the initial small working set and final workspace
        # must reject immediately, not fail an already-running request later.
        policy.profile = replace(policy.profile, device_memory_bytes=10_000_000)
        self.assertFalse(policy.choose([WorkItem("a", 200, 32, 0)]).estimate.feasible)

    def test_removing_requests_does_not_increase_workspace_reservation(self):
        policy = controller([sample(batch=1, context=64, workspace=2_000_000),
                             sample(batch=8, context=384, workspace=80_000_000)], candidates=((),))
        larger = policy.choose([WorkItem(str(i), 512, 0, 100) for i in range(4)])
        smaller = policy.choose([WorkItem(str(i), 512, 0, 100) for i in range(3)])
        self.assertLessEqual(smaller.estimate.device_bytes, larger.estimate.device_bytes)

    def test_live_profile_replacement_changes_capacity(self):
        policy = controller([sample()], candidates=((),))
        items = [WorkItem("a", 1000, 0, 63)]
        self.assertTrue(policy.choose(items).estimate.feasible)
        policy.profile = replace(policy.profile, device_memory_bytes=1024)
        self.assertFalse(policy.choose(items).estimate.feasible)

    def test_prefill_is_not_offloaded_even_with_faster_cpu_sample(self):
        policy = controller([sample(prefill=64, ms=5), sample(prefill=64, cpu=1, ms=1)],
                            candidates=((),), policy="offload_only")
        decision = policy.choose([WorkItem("a", 512, 64, 0)])
        self.assertEqual(decision.plan.cpu_request_ids, ())


class CalibrationValidationTests(unittest.TestCase):
    def test_serialization_roundtrip(self):
        calibration = TimingCalibration((sample(q=(2, 0)),), to_int4_layer_ms=3, to_fp_layer_ms=4)
        restored = TimingCalibration.from_dict(json.loads(json.dumps(calibration.to_dict())))
        self.assertEqual(restored, calibration)

    def test_invalid_inputs_and_feedback_rejected(self):
        for kwargs in ({"elapsed_ms": float("nan")}, {"cpu_requests": 2}, {"workspace_bytes": -1}):
            base = dict(quantized_layers=(), batch_size=1, prefill_tokens=0, context_tokens=64,
                        cpu_requests=0, elapsed_ms=5)
            with self.assertRaises(ValueError):
                TimingSample(**{**base, **kwargs})
        with self.assertRaises(ValueError):
            controller([sample()], candidates=((4,),))
        with self.assertRaises(ValueError):
            controller([sample()], workspace_safety_factor=1)
        policy = controller([sample()], candidates=((),))
        with self.assertRaises(ValueError):
            policy.observe(policy.choose([WorkItem("a", 512, 0, 63)]), -1)


if __name__ == "__main__":
    unittest.main()
