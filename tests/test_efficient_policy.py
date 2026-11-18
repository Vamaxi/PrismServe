"""Resource parity, safe cache invalidation, phase feedback and fallback checks."""
from dataclasses import replace
import random
import unittest

from prismserve.efficient_policy import EfficientController
from prismserve.measured_policy import MeasuredController, TimingSample, TimingCalibration
from prismserve.types import HardwareProfile, ModelSpec, PlannerConfig, Plan, WorkItem, ControlState


def setup_controllers(**kwargs):
    hardware = HardwareProfile('test', 2**30, 4*2**30, 100, 30, 8, 10, .2)
    model = ModelSpec('test', 4, 64, 1, 16, 1000000, sensitivity=(.001,.002,.003,.004))
    samples = tuple(TimingSample(q, b, p, c, cpu, 5 + len(q)*3 + cpu, 500000 + b*20000)
                    for q in ((),(0,),(0,1)) for b in (1,4,8) for p in (0,64)
                    for c in (64,384) for cpu in (0,b) if not p or not cpu)
    calibration = TimingCalibration(samples, 20, 5)
    config = PlannerConfig(quality_budget=.03, memory_fraction=1, quality_penalty=0)
    old = MeasuredController(hardware, model, config, calibration, quantization_candidates=((),(0,),(0,1)))
    new = EfficientController(hardware, model, config, calibration, quantization_candidates=((),(0,),(0,1)), **kwargs)
    return old, new


class EfficientPolicyTests(unittest.TestCase):
    def test_compiled_resources_match_previous_safe_formulas(self):
        old, new = setup_controllers()
        rng = random.Random(13)
        for _ in range(100):
            count = rng.randint(1,8)
            items = [WorkItem(str(i), rng.randint(400,800), 0, rng.randint(0,390)) for i in range(count)]
            plan = Plan(rng.choice(((),(0,),(0,1))), tuple(str(i) for i in range(count//2)))
            state = ControlState(rng.choice(((),(0,),(0,1))), tuple(str(i) for i in range(count//3)))
            before = old._estimate(items, plan, state)
            after = new.evaluate_plan(items, plan, state).estimate
            self.assertEqual(before.device_bytes, after.device_bytes)
            self.assertEqual(before.host_bytes, after.host_bytes)
            self.assertEqual(before.quality_risk, after.quality_risk)
            self.assertAlmostEqual(before.transition_ms, after.transition_ms)
            self.assertEqual(before.feasible, after.feasible)

    def test_cached_reservations_check_new_capacity_each_time(self):
        _, policy = setup_controllers()
        items = [WorkItem('a',512,0,100)]
        before = policy.evaluate_plan(items, Plan(), ControlState())
        self.assertTrue(before.estimate.feasible)
        policy.evaluate_plan(items, Plan(), ControlState())
        self.assertGreater(policy.reservation_hits,0)
        policy.profile = replace(policy.profile, device_memory_bytes=1)
        after = policy.evaluate_plan(items, Plan(), ControlState())
        self.assertFalse(after.estimate.feasible)

    def test_cpu_primary_uses_combined_host_capacity_like_previous_controller(self):
        old, new = setup_controllers()
        old.profile = replace(old.profile, device_kind='cpu', host_memory_bytes=20_000_000)
        new.profile = old.profile
        items = [WorkItem(str(i),512,0,100) for i in range(4)]
        for q in ((),(0,),(0,1)):
            before = old._estimate(items, Plan(q,()), ControlState())
            after = new.evaluate_plan(items, Plan(q,()), ControlState()).estimate
            self.assertEqual((before.device_bytes,before.host_bytes,before.feasible),
                             (after.device_bytes,after.host_bytes,after.feasible))
        new.profile = replace(new.profile, host_memory_bytes=1)
        self.assertFalse(new.evaluate_plan(items, Plan(), ControlState()).estimate.feasible)

    def test_prefill_feedback_does_not_distort_decode(self):
        _, policy = setup_controllers()
        decode = [WorkItem('a',512,0,63)]
        prefill = [WorkItem('a',512,64,0)]
        initial = policy.evaluate_plan(decode, Plan(), ControlState()).estimate.iteration_ms
        observed = policy.evaluate_plan(prefill, Plan(), ControlState())
        policy.observe(observed, observed.estimate.iteration_ms*4)
        self.assertEqual(initial, policy.evaluate_plan(decode, Plan(), ControlState()).estimate.iteration_ms)
        self.assertGreater(policy.evaluate_plan(prefill, Plan(), ControlState()).estimate.iteration_ms,
                           observed.estimate.iteration_ms)

    def test_uncertain_small_predicted_gain_keeps_full_precision(self):
        _, policy = setup_controllers()
        items = [WorkItem('a',512,0,63)]
        full = policy.evaluate_plan(items, Plan(), ControlState())
        alternative = policy.evaluate_plan(items, Plan((0,),()), ControlState())
        alternative = replace(alternative, estimate=replace(alternative.estimate,
                              iteration_ms=full.estimate.iteration_ms*.95, transition_ms=0))
        self.assertFalse(policy.supports_change(alternative,1,full,1))

    def test_fallback_does_not_override_a_capacity_necessity(self):
        _, policy = setup_controllers()
        items = [WorkItem('a',512,0,63)]
        full = policy.evaluate_plan(items, Plan(), ControlState())
        impossible_full = replace(full, estimate=replace(full.estimate, feasible=False))
        alternative = policy.evaluate_plan(items, Plan((0,),()), ControlState())
        self.assertTrue(policy.supports_change(alternative,1,impossible_full,1))

    def test_current_context_predicts_time_but_future_context_reserves_memory(self):
        _, policy = setup_controllers()
        a = policy.evaluate_plan([WorkItem('a',512,0,63)], Plan(), ControlState())
        b = policy.evaluate_plan([WorkItem('a',100000,0,63)], Plan(), ControlState())
        self.assertEqual(a.estimate.iteration_ms,b.estimate.iteration_ms)
        self.assertGreater(b.estimate.device_bytes,a.estimate.device_bytes)


if __name__ == '__main__':
    unittest.main()
