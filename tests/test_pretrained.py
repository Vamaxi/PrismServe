"""Offline HF parity tests using a small, random GPT-2 configuration only."""
import importlib.util
import os
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
from types import SimpleNamespace
import unittest

AVAILABLE = all(importlib.util.find_spec(name) is not None for name in ("torch", "transformers"))


class MockTokenizer:
    eos_token_id = 63

    def encode(self, text, add_special_tokens=False):
        return [ord(character) % 63 for character in text]

    def decode(self, ids, **kwargs):
        return " ".join(map(str, ids))


@unittest.skipUnless(AVAILABLE, "install the pretrained extra")
class PretrainedTests(unittest.TestCase):
    def setUp(self):
        import torch
        torch.set_num_threads(min(2, torch.get_num_threads()))
        torch.manual_seed(71)

    def make_model(self, **kwargs):
        from transformers import GPT2Config, GPT2LMHeadModel
        config = GPT2Config(vocab_size=64, n_positions=32, n_embd=32, n_layer=2,
                            n_head=4, resid_pdrop=0, embd_pdrop=0, attn_pdrop=0, **kwargs)
        config._attn_implementation = "eager"
        return GPT2LMHeadModel(config).eval()

    def test_hf_eager_prefill_and_cached_parity(self):
        import torch
        from prismserve.pretrained import GPT2Executor
        original = self.make_model()
        ids = torch.tensor([[4, 5, 9, 7, 12]])
        following = torch.tensor([[11, 16]])
        with torch.inference_mode():
            reference = original(ids, use_cache=True)
            expected = reference.logits.clone()
            expected_cached = original(following, past_key_values=reference.past_key_values, use_cache=True).logits
            executor = GPT2Executor(original, MockTokenizer())
            actual, cache = executor(ids)
            cached, extended = executor(following, cache)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6, rtol=1e-5),
                        float((actual - expected).abs().max()))
        self.assertTrue(torch.allclose(cached, expected_cached, atol=1e-6, rtol=1e-5),
                        float((cached - expected_cached).abs().max()))
        self.assertEqual(extended[0][0].shape[2], 7)

    def test_custom_cache_matches_full_forward_and_causality(self):
        import torch
        from prismserve.pretrained import GPT2Executor
        executor = GPT2Executor(self.make_model(), MockTokenizer())
        ids = torch.tensor([[7, 10, 3, 13, 19]])
        with torch.inference_mode():
            full, _ = executor(ids)
            _, cache = executor(ids[:, :3])
            tail, _ = executor(ids[:, 3:], cache)
            ids[0, -1] = 1
            alternate, _ = executor(ids)
        self.assertTrue(torch.allclose(tail, full[:, 3:], atol=1e-6, rtol=1e-5))
        self.assertTrue(torch.equal(full[:, :-1], alternate[:, :-1]))

    def test_quantization_restores_exactly_and_preserves_cache(self):
        import torch
        from prismserve.pretrained import GPT2Executor
        executor = GPT2Executor(self.make_model(), MockTokenizer())
        ids = torch.tensor([[7, 9, 3]])
        with torch.inference_mode():
            expected, cache = executor(ids)
            previous = cache[0][0].clone()
            original_bytes = executor.memory_report()["resident_parameter_bytes"]
            executor.set_quantized_layers((0,))
            report = executor.memory_report()
            self.assertLess(report["resident_parameter_bytes"], original_bytes)
            self.assertGreater(report["host_master_bytes"], 0)
            self.assertGreater(report["max_dequantization_scratch_bytes"], 0)
            self.assertIsNone(executor.layers[0].attn.c_attn.weight)
            changed, extended = executor(torch.tensor([[15]]), cache)
            self.assertTrue(torch.equal(previous, extended[0][0][:, :, :3]))
            self.assertTrue(torch.isfinite(changed).all())
            executor.set_quantized_layers(())
            restored, _ = executor(ids)
            self.assertTrue(torch.equal(restored, expected))

    def test_tied_embedding_head_counted_once_in_model_spec(self):
        from prismserve.pretrained import GPT2Executor
        model = self.make_model()
        count = sum(p.numel() for p in model.parameters())
        bytes_count = sum(p.numel() * p.element_size() for p in model.parameters())
        executor = GPT2Executor(model, MockTokenizer())
        spec = executor.model_spec()
        report = executor.memory_report()
        self.assertIs(model.lm_head.weight, model.transformer.wte.weight)
        self.assertEqual(spec.parameters, count)
        self.assertEqual(report["resident_parameter_bytes"], bytes_count)
        self.assertEqual(spec.fp_weight_bytes, bytes_count)
        self.assertEqual(spec.weight_bits, 32)
        self.assertEqual(spec.kv_bits, 32)
        self.assertTrue(spec.reference_dequantization)
        self.assertEqual(len(spec.layer_parameters), 2)
        self.assertLess(sum(spec.layer_parameters), spec.parameters)

    def test_conv1d_layout_packing_and_dtype_restoration(self):
        import torch
        from torch import nn
        from prismserve.pretrained import MorphConv1D
        original = nn.Module()
        original.weight = nn.Parameter(torch.randn(5, 7))
        original.bias = nn.Parameter(torch.randn(7))
        projection = MorphConv1D(original, group_size=6)
        inputs = torch.randn(2, 3, 5)
        expected = torch.addmm(original.bias, inputs.reshape(-1, 5), original.weight).view(2, 3, 7)
        self.assertTrue(torch.equal(projection(inputs), expected))
        projection.set_quantized(True)
        self.assertEqual(projection(inputs).shape, expected.shape)
        projection.set_quantized(False)
        self.assertTrue(torch.equal(projection(inputs), expected))
        half_original = nn.Module()
        half_original.weight = nn.Parameter(torch.randn(5, 7).half())
        half_original.bias = nn.Parameter(torch.randn(7).half())
        half = MorphConv1D(half_original)
        values = half.weight.detach().clone()
        half.set_quantized(True)
        half.set_quantized(False)
        self.assertEqual(half.weight.dtype, torch.float16)
        self.assertTrue(torch.equal(half.weight, values))

    def test_generate_applies_plan_between_tokens(self):
        from prismserve.pretrained import GPT2Executor
        executor = GPT2Executor(self.make_model(), MockTokenizer())
        changes = []
        def callback(current, cache, index):
            current.apply_plan(SimpleNamespace(quantized_layers=(0,) if index == 1 else (),
                                               cpu_request_ids=("r",)), "r")
            changes.append(current.quantized_layers)
            return cache
        result = executor.generate("test", max_tokens=4, callback=callback, stop_on_eos=False)
        self.assertEqual(result["completion_tokens"], 4)
        self.assertEqual(result["cache_tokens"], 7)
        self.assertEqual(result["cache_device"], "cpu")
        self.assertEqual(changes, [(0,), (), ()])
        self.assertEqual(len(result["decode_token_ms"]), 3)
        self.assertEqual(result["finish_reason"], "length")

    def test_invalid_context_cache_and_config(self):
        import torch
        from prismserve.pretrained import GPT2Executor
        executor = GPT2Executor(self.make_model(), MockTokenizer())
        with self.assertRaises(ValueError):
            executor.generate("x" * 30, max_tokens=4)
        with self.assertRaises(ValueError):
            executor(torch.tensor([[1]]), cache=[])
        with self.assertRaises(ValueError):
            executor.set_quantized_layers((2,))
        with self.assertRaises(ValueError):
            GPT2Executor(self.make_model(add_cross_attention=True), MockTokenizer())
        with self.assertRaises(ValueError):
            GPT2Executor(self.make_model(reorder_and_upcast_attn=True), MockTokenizer())

    def test_inverse_layer_scaling_matches_hf(self):
        import torch
        from prismserve.pretrained import GPT2Executor
        model = self.make_model(scale_attn_by_inverse_layer_idx=True, scale_attn_weights=False)
        ids = torch.tensor([[1, 8, 3]])
        with torch.inference_mode():
            expected = model(ids).logits
            actual, _ = GPT2Executor(model, MockTokenizer())(ids)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6, rtol=1e-5))

    def test_cuda_persistent_cpu_kv_matches_device_attention(self):
        import torch
        from prismserve.pretrained import GPT2Executor
        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")
        executor = GPT2Executor(self.make_model().cuda(), MockTokenizer())
        ids = torch.tensor([[5, 7, 9]], device="cuda")
        with torch.inference_mode():
            _, gpu_cache = executor(ids)
            expected, _ = executor(torch.tensor([[11]], device="cuda"), gpu_cache)
            executor.set_cpu_attention(True)
            actual, host_cache = executor(torch.tensor([[11]], device="cuda"), gpu_cache)
            self.assertEqual(host_cache[0][0].device.type, "cpu")
            _, extended = executor(torch.tensor([[12]], device="cuda"), host_cache)
            self.assertEqual(extended[0][0].device.type, "cpu")
            executor.set_cpu_attention(False)
            _, restored = executor(torch.tensor([[13]], device="cuda"), extended)
        self.assertTrue(torch.allclose(actual, expected, atol=2e-5, rtol=2e-5))
        self.assertEqual(restored[0][0].device.type, "cuda")


if __name__ == "__main__":
    unittest.main()
