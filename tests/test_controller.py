"""Policy and accounting invariants; runnable with Python's unittest."""
from dataclasses import replace
import unittest

from prismserve.estimator import Estimator
from prismserve.policy import Controller
from prismserve.types import ControlState, HardwareProfile, ModelSpec, Plan, PlannerConfig, RequestSpec, WorkItem


def hardware(**changes):
    values = dict(name="test", device_memory_bytes=8 * 2**30, host_memory_bytes=64 * 2**30,
                  device_bandwidth_gbps=400, host_bandwidth_gbps=50,
                  link_bandwidth_gbps=16, device_tflops=20, host_tflops=1)
    values.update(changes)
    return HardwareProfile(**values)


def model(**changes):
    values = dict(name="test", layers=4, hidden_size=512, kv_heads=4, head_dim=64,
                  parameters=1_000_000_000, sensitivity=(.002, .004, .008, .016))
    values.update(changes)
    return ModelSpec(**values)


class AccountingTests(unittest.TestCase):
    def test_kv_counts_keys_values_and_layers(self):
        self.assertEqual(model().kv_bytes_per_token, 4096)

    def test_quantization_includes_group_scales(self):
        m = model()
        self.assertGreater(m.int4_layer_bytes, m.parameters / m.layers / 2)
        self.assertLess(m.layer_saving_bytes, m.fp_layer_bytes * .75)

    def test_only_morphable_weights_are_compressed(self):
        m = model(layer_parameters=(10_000, 20_000, 30_000, 40_000))
        estimate = Estimator(hardware(), m, PlannerConfig()).estimate([], Plan((0, 1)), ControlState())
        expected = m.fp_weight_bytes - m.layer_saving_bytes_for(0) - m.layer_saving_bytes_for(1)
        self.assertEqual(estimate.components["weight_bytes"], expected)
        self.assertEqual(m.fp_layer_bytes_for(0), 20_000)

    def test_reference_dequantization_charges_scratch(self):
        m = model()
        ideal = Estimator(hardware(), m, PlannerConfig()).estimate([], Plan((0,)), ControlState())
        reference = Estimator(hardware(), replace(m, reference_dequantization=True), PlannerConfig()).estimate(
            [], Plan((0,)), ControlState())
        self.assertGreater(reference.device_bytes, ideal.device_bytes)
        self.assertGreater(reference.components["dequantization_scratch_bytes"], 0)

    def test_steady_transfer_independent_of_context_length(self):
        estimator = Estimator(hardware(), model(), PlannerConfig())
        short = estimator.estimate([WorkItem("a", 100)], Plan((), ("a",)), ControlState(cpu_request_ids=("a",)))
        long = estimator.estimate([WorkItem("a", 10000)], Plan((), ("a",)), ControlState(cpu_request_ids=("a",)))
        self.assertEqual(short.components["steady_transfer_bytes"], long.components["steady_transfer_bytes"])
        self.assertGreater(long.components["cpu_attention_ms"], short.components["cpu_attention_ms"])

    def test_migration_charged_once(self):
        estimator = Estimator(hardware(), model(), PlannerConfig())
        items, plan = [WorkItem("a", 1000)], Plan((), ("a",))
        first = estimator.estimate(items, plan, ControlState())
        subsequent = estimator.estimate(items, plan, ControlState(cpu_request_ids=("a",)))
        self.assertGreater(first.transition_ms, subsequent.transition_ms)
        self.assertEqual(subsequent.components["migration_bytes"], 0)

    def test_transition_peak_can_reject_feasible_steady_state(self):
        p, m = hardware(device_memory_bytes=2_100_000_000), model()
        estimator = Estimator(p, m, PlannerConfig(memory_fraction=1))
        result = estimator.estimate([WorkItem("a", 10, 10)], Plan((0,), ()), ControlState())
        self.assertLess(result.components["device_steady_bytes"], p.device_memory_bytes)
        self.assertFalse(result.feasible)

    def test_chunked_prefill_accounts_existing_cache(self):
        estimator = Estimator(hardware(), model(), PlannerConfig())
        new = estimator.estimate([WorkItem("a", 1000, 128, 0)], Plan((0,), ()), ControlState())
        existing = estimator.estimate([WorkItem("a", 1000, 128, 1000)], Plan((0,), ()), ControlState())
        self.assertGreater(existing.device_bytes, new.device_bytes)

    def test_cpu_shared_ram_not_double_capacity(self):
        p = hardware(device_kind="cpu", link_bandwidth_gbps=0,
                     host_memory_bytes=3_000_000_000, device_memory_bytes=3_000_000_000)
        result = Estimator(p, model(), PlannerConfig()).estimate([], Plan(), ControlState())
        self.assertLess(result.device_bytes, p.device_memory_bytes)
        self.assertLess(result.host_bytes, p.host_memory_bytes)
        self.assertFalse(result.feasible)


class PolicyTests(unittest.TestCase):
    def test_no_cpu_self_offload(self):
        p = hardware(device_kind="cpu", link_bandwidth_gbps=0)
        result = Controller(p, model()).choose([WorkItem("a", 100)])
        self.assertEqual(result.plan.cpu_request_ids, ())

    def test_prefill_cannot_offload(self):
        result = Controller(hardware(), model(), policy="offload_only").choose([WorkItem("a", 100, 100)])
        self.assertEqual(result.plan.cpu_request_ids, ())

    def test_risk_budget_is_enforced(self):
        result = Controller(hardware(), model(), PlannerConfig(quality_budget=0)).choose([WorkItem("a", 100)])
        self.assertEqual(result.plan.quantized_layers, ())
        self.assertTrue(result.estimate.feasible)

    def test_static_baseline_reports_unconstrained_proxy(self):
        result = Controller(hardware(), model(), PlannerConfig(quality_budget=0), "static_int4").choose([])
        self.assertEqual(len(result.plan.quantized_layers), model().layers)
        self.assertTrue(result.estimate.feasible)
        self.assertGreater(result.estimate.quality_risk, 0)

    def test_precision_restores_after_cooldown(self):
        controller = Controller(hardware(), model())
        result = controller.choose([WorkItem("a", 100)], ControlState((0, 1), (), 0), now_s=1)
        self.assertEqual(result.plan.quantized_layers, ())

    def test_cooldown_retains_feasible_plan(self):
        controller = Controller(hardware(), model())
        result = controller.choose([WorkItem("a", 100)], ControlState((0, 1), (), 0), now_s=.01)
        self.assertEqual(result.plan.quantized_layers, (0, 1))

    def test_cooldown_does_not_keep_invalid_plan(self):
        controller = Controller(hardware(), model(), PlannerConfig(quality_budget=0))
        result = controller.choose([WorkItem("a", 100)], ControlState((0,), (), 0), now_s=.01)
        self.assertEqual(result.plan.quantized_layers, ())

    def test_infeasibility_is_explicit(self):
        result = Controller(hardware(device_memory_bytes=1024), model()).choose([WorkItem("a", 100)])
        self.assertFalse(result.estimate.feasible)

    def test_deterministic_tie_breaking(self):
        controller = Controller(hardware(), model())
        items = [WorkItem("a", 100), WorkItem("b", 200)]
        first = controller.choose(items)
        second = controller.choose(items)
        self.assertEqual(first.plan, second.plan)
        self.assertEqual(first.estimate, second.estimate)


class ValidationTests(unittest.TestCase):
    def test_invalid_requests(self):
        with self.assertRaises(ValueError):
            RequestSpec("x", float("nan"), 1, 1)
        with self.assertRaises(ValueError):
            RequestSpec("x", 0, 0, 1)

    def test_invalid_profiles_and_configs(self):
        with self.assertRaises(ValueError):
            hardware(link_bandwidth_gbps=0)
        with self.assertRaises(ValueError):
            PlannerConfig(memory_fraction=2)
        with self.assertRaises(ValueError):
            PlannerConfig(offload_fractions=(float("nan"),))
        with self.assertRaises(ValueError):
            PlannerConfig(transition_horizon=0)


if __name__ == "__main__":
    unittest.main()
