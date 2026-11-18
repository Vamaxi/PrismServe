"""Numerical mechanisms tests; no download or gated model is required."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH_AVAILABLE, "install the torch extra")
class QuantizationTests(unittest.TestCase):
    def setUp(self):
        import torch
        torch.set_num_threads(min(2, torch.get_num_threads()))
        torch.manual_seed(7)

    def test_odd_shape_and_group_roundtrip_error(self):
        import torch
        from prismserve.quantization import pack_int4, unpack_int4
        weight = torch.randn(3, 9)
        packed = pack_int4(weight, group_size=7)
        result = unpack_int4(packed)
        self.assertEqual(result.shape, weight.shape)
        max_error = float(packed.scales.max()) / 2 + 1e-6
        self.assertLessEqual(float((result - weight).abs().max()), max_error)
        self.assertEqual(packed.data.dtype, torch.uint8)

    def test_zero_matrix(self):
        import torch
        from prismserve.quantization import pack_int4, unpack_int4
        weight = torch.zeros(3, 5)
        self.assertTrue(torch.equal(unpack_int4(pack_int4(weight)), weight))

    def test_invalid_weight_and_group(self):
        import torch
        from prismserve.quantization import pack_int4
        for value, group in ((torch.tensor([float("nan")]), 64), (torch.ones(3), 0)):
            with self.assertRaises(ValueError):
                pack_int4(value, group)

    def test_reversible_weight_storage(self):
        import torch
        from prismserve.quantization import MorphLinear
        module = MorphLinear(64, 64)
        values = torch.randn(2, 64)
        reference = module(values).detach()
        fp_bytes = module.resident_weight_bytes
        module.set_quantized(True)
        self.assertIsNone(module.weight)
        self.assertLess(module.resident_weight_bytes, fp_bytes / 4)
        self.assertEqual(module.master_bytes, fp_bytes)
        module.set_quantized(False)
        self.assertTrue(torch.equal(module(values), reference))


@unittest.skipUnless(TORCH_AVAILABLE, "install the torch extra")
class DecoderTests(unittest.TestCase):
    def setUp(self):
        import torch
        torch.set_num_threads(min(2, torch.get_num_threads()))
        torch.manual_seed(7)

    def test_cached_forward_matches_causal_full_forward(self):
        import torch
        from prismserve.torch_backend import TinyDecoder
        model = TinyDecoder().eval()
        tokens = torch.tensor([[5, 9, 11, 15, 20, 22]])
        with torch.inference_mode():
            expected, _ = model(tokens)
            _, cache = model(tokens[:, :3])
            actual, cache = model(tokens[:, 3:], cache)
        self.assertTrue(torch.allclose(actual, expected[:, 3:], atol=1e-5, rtol=1e-5))
        self.assertEqual(cache[0][0].shape[2], 6)

    def test_future_tokens_cannot_change_prefix(self):
        import torch
        from prismserve.torch_backend import TinyDecoder
        model = TinyDecoder().eval()
        tokens = torch.tensor([[3, 4, 5, 6]])
        with torch.inference_mode():
            before, _ = model(tokens)
            tokens[0, -1] = 200
            after, _ = model(tokens)
        self.assertTrue(torch.equal(before[:, :-1], after[:, :-1]))

    def test_precision_and_placement_changes_preserve_cache(self):
        import torch
        from prismserve.torch_backend import TinyDecoder
        model = TinyDecoder().eval()
        with torch.inference_mode():
            baseline, cache = model(torch.tensor([[1, 2, 3]]))
            original = cache[0][0].clone()
            model.set_quantized_layers((0,))
            model.set_cpu_attention(True)
            _, extended = model(torch.tensor([[4]]), cache)
            self.assertTrue(torch.equal(extended[0][0][:, :, :3], original))
            self.assertEqual(extended[0][0].device.type, "cpu")
            self.assertEqual(extended[0][0].shape[2], 4)
            model.set_quantized_layers(())
            restored, _ = model(torch.tensor([[1, 2, 3]]))
            self.assertTrue(torch.equal(baseline, restored))

    def test_checkpoint_roundtrip_and_generation(self):
        import torch
        from prismserve.torch_backend import TinyDecoder, save_checkpoint, load_model, generate
        model = TinyDecoder().eval()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            save_checkpoint(model, path)
            loaded = load_model(path, device="cpu")
            first = generate(model, "Hello", max_tokens=4)
            second = generate(loaded, "Hello", max_tokens=4)
            self.assertEqual(first["token_ids"], second["token_ids"])
            self.assertEqual(first["cache_tokens"], len("Hello") + 3)

    def test_context_limits(self):
        from prismserve.torch_backend import TinyDecoder, generate
        model = TinyDecoder().eval()
        with self.assertRaises(ValueError):
            generate(model, "x" * 250, max_tokens=10)

    def test_shared_plan_is_applied_between_tokens(self):
        from types import SimpleNamespace
        import torch
        from prismserve.torch_backend import TinyDecoder, generate
        model = TinyDecoder().eval()
        changes = []
        def adapt(current, cache, index):
            before = cache[0][0].clone()
            current.apply_plan(SimpleNamespace(quantized_layers=(0,) if index == 1 else (),
                                              cpu_request_ids=("request",)), "request")
            self.assertTrue(torch.equal(before, cache[0][0]))
            changes.append(current.quantized_layers)
            return cache
        result = generate(model, "test", max_tokens=4, callback=adapt)
        self.assertEqual(changes, [(0,), (), ()])
        self.assertEqual(result["completion_tokens"], 4)

    def test_cuda_cpu_attention_agrees_with_device_attention(self):
        import torch
        from prismserve.torch_backend import TinyDecoder
        if not torch.cuda.is_available():
            self.skipTest("CUDA hardware not available")
        model = TinyDecoder().cuda().eval()
        with torch.inference_mode():
            _, cache = model(torch.tensor([[10, 11, 12]], device="cuda"))
            expected, _ = model(torch.tensor([[13]], device="cuda"), cache)
            model.set_cpu_attention(True)
            actual, cpu_cache = model(torch.tensor([[13]], device="cuda"), cache)
        self.assertTrue(torch.allclose(actual, expected, atol=2e-5, rtol=2e-5))
        self.assertEqual(cpu_cache[0][0].device.type, "cpu")


@unittest.skipUnless(TORCH_AVAILABLE and importlib.util.find_spec("fastapi"), "install the serve extra")
class ServiceTests(unittest.TestCase):
    def test_asgi_completion_validation_and_metrics(self):
        import asyncio
        import json
        from prismserve.service import create_app
        app = create_app(device="cpu")

        async def request(path, body=None):
            data = b"" if body is None else json.dumps(body).encode()
            sent = []
            scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
                     "method": "GET" if body is None else "POST", "scheme": "http",
                     "path": path, "raw_path": path.encode(), "query_string": b"",
                     "headers": [(b"content-type", b"application/json")],
                     "client": ("127.0.0.1", 1), "server": ("test", 80), "root_path": ""}
            async def receive():
                return {"type": "http.request", "body": data, "more_body": False}
            async def send(message):
                sent.append(message)
            await app(scope, receive, send)
            status = next(m["status"] for m in sent if m["type"] == "http.response.start")
            payload = json.loads(b"".join(m.get("body", b"") for m in sent))
            return status, payload

        self.assertEqual(asyncio.run(request("/health"))[0], 200)
        code, payload = asyncio.run(request("/v1/completions", {"prompt": "Hello", "max_tokens": 3,
                                                                "quantized_layers": [0]}))
        self.assertEqual(code, 200)
        self.assertEqual(payload["usage"]["completion_tokens"], 3)
        self.assertEqual(asyncio.run(request("/metrics"))[1]["completed_requests"], 1)
        self.assertEqual(asyncio.run(request("/v1/completions", {"prompt": "a", "max_tokens": -1}))[0], 422)
        self.assertEqual(asyncio.run(request("/v1/completions", {"prompt": "a", "stream": True}))[0], 400)


if __name__ == "__main__":
    unittest.main()
