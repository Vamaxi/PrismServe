"""Exercise API validation, serialization, timing, and EOS without a checkpoint."""
from __future__ import annotations

import asyncio
import importlib.util
import json
from pathlib import Path
import threading
import time
from types import SimpleNamespace
import unittest


class FakeExecutor:
    device = SimpleNamespace(type="cpu")
    dtype = "float32"
    context_limit = 1024

    def __init__(self):
        self.quantized_layers = ()
        self.cpu_attention = False
        self.generate_calls = 0
        self.active_calls = 0
        self.max_active_calls = 0
        self.calls_lock = threading.Lock()
        self.early_stop = False
        self.delay_s = .03

    def model_spec(self):
        return SimpleNamespace(layers=12)

    def tokenize(self, prompt):
        # A tokenizer can yield more tokens than Unicode character count.
        return [1] * (len(prompt) * 2)

    def set_quantized_layers(self, layers):
        time.sleep(.003)
        self.quantized_layers = layers

    def set_cpu_attention(self, enabled):
        # CPU-only execution is correctly reported as no offload.
        self.cpu_attention = False

    def memory_report(self):
        return {"resident_weight_bytes": 1000, "host_master_bytes": 2000,
                "dequant_scratch_bound_bytes": 3000}

    def generate(self, prompt, max_tokens=32, callback=None, stop_on_eos=True):
        assert stop_on_eos
        with self.calls_lock:
            self.active_calls += 1
            self.max_active_calls = max(self.max_active_calls, self.active_calls)
        time.sleep(self.delay_s)
        with self.calls_lock:
            self.active_calls -= 1
            self.generate_calls += 1
        tokens = 1 if self.early_stop else max_tokens
        return {"text": "generated text", "prompt_tokens": len(self.tokenize(prompt)),
                "completion_tokens": tokens, "elapsed_ms": self.delay_s * 1000,
                "ttft_ms": 2.0, "decode_token_ms": [1.0] * (tokens - 1),
                "cache_device": "cpu", "finish_reason": "stop" if self.early_stop else "length"}


async def asgi_request(app, path, body=None):
    data = b"" if body is None else json.dumps(body).encode()
    sent = []
    scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
             "method": "GET" if body is None else "POST", "scheme": "http", "path": path,
             "raw_path": path.encode(), "query_string": b"", "root_path": "",
             "headers": [(b"content-type", b"application/json")],
             "client": ("127.0.0.1", 1), "server": ("test", 80)}
    async def receive():
        return {"type": "http.request", "body": data, "more_body": False}
    async def send(message):
        sent.append(message)
    await app(scope, receive, send)
    status = next(message["status"] for message in sent if message["type"] == "http.response.start")
    payload = json.loads(b"".join(message.get("body", b"") for message in sent))
    return status, payload


@unittest.skipUnless(importlib.util.find_spec("fastapi"), "install the serve extra")
class PretrainedServiceTests(unittest.TestCase):
    def setUp(self):
        from prismserve.pretrained_service import create_pretrained_app
        self.executor = FakeExecutor()
        self.app = create_pretrained_app(Path("models/test-local"), executor=self.executor)

    def call(self, path, body=None):
        return asyncio.run(asgi_request(self.app, path, body))

    def test_health_models_and_memory_describe_actual_backend(self):
        code, health = self.call("/health")
        self.assertEqual(code, 200)
        self.assertTrue(health["pretrained"])
        self.assertEqual(health["checkpoint_source"]["kind"], "local")
        self.assertEqual(health["model_kind"], "pretrained_base_completion")
        self.assertEqual(health["model_dtype"], "float32")
        self.assertFalse(health["cpu_offload_effective"])
        self.assertEqual(health["memory"]["host_master_bytes"], 2000)
        code, models = self.call("/v1/models")
        self.assertEqual(models["data"][0]["id"], "prismserve-gpt2")

    def test_successful_completion_includes_transition_and_queue_times(self):
        code, result = self.call("/v1/completions", {"prompt": "Hello", "max_tokens": 3,
            "quantized_layers": [2, 0], "cpu_attention": True})
        self.assertEqual(code, 200)
        self.assertEqual(result["usage"], {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13})
        self.assertEqual(result["choices"][0]["finish_reason"], "length")
        info = result["prismserve"]
        self.assertEqual(info["quantized_layers"], [0, 2])
        self.assertFalse(info["cpu_offload_effective"])
        self.assertGreater(info["setup_ms"], 0)
        self.assertAlmostEqual(info["ttft_ms"], info["model_ttft_ms"] + info["setup_ms"] + info["queue_wait_ms"])
        self.assertGreaterEqual(info["service_elapsed_ms"], info["inference_elapsed_ms"])
        metrics = self.call("/metrics")[1]
        self.assertEqual(metrics["completed_requests"], 1)
        self.assertEqual(metrics["generated_tokens"], 3)
        self.assertGreaterEqual(metrics["total_inference_ms"], metrics["total_setup_ms"])
        self.assertEqual(metrics["dequant_scratch_bound_bytes"], 3000)

    def test_unsupported_model_and_stream_are_rejected(self):
        self.assertEqual(self.call("/v1/completions", {"prompt": "a", "model": "another"})[0], 404)
        self.assertEqual(self.call("/v1/completions", {"prompt": "a", "stream": True})[0], 400)
        self.assertEqual(self.executor.generate_calls, 0)

    def test_actual_token_limit_not_just_character_limit(self):
        # 500 characters tokenize to 1000 tokens; another 25 would exceed 1024.
        self.assertEqual(self.call("/v1/completions", {"prompt": "a" * 500, "max_tokens": 25})[0], 400)
        self.assertEqual(self.call("/v1/completions", {"prompt": "a" * 8193})[0], 422)
        self.assertEqual(self.executor.generate_calls, 0)

    def test_invalid_layers_tokens_and_unknown_options_are_rejected(self):
        for layers in ([-1], [12], [1, 1]):
            with self.subTest(layers=layers):
                self.assertEqual(self.call("/v1/completions", {"prompt": "a", "quantized_layers": layers})[0], 400)
        for value in (0, -1, 257, True, 1.2):
            with self.subTest(max_tokens=value):
                self.assertEqual(self.call("/v1/completions", {"prompt": "a", "max_tokens": value})[0], 422)
        self.assertEqual(self.call("/v1/completions", {"prompt": "a", "quantized_layers": [True]})[0], 422)
        self.assertEqual(self.call("/v1/completions", {"prompt": "a", "temperature": .7})[0], 422)
        self.assertEqual(self.executor.generate_calls, 0)

    def test_eos_stop_reports_actual_completion_usage(self):
        self.executor.early_stop = True
        code, result = self.call("/v1/completions", {"prompt": "Hello", "max_tokens": 10})
        self.assertEqual(code, 200)
        self.assertEqual(result["choices"][0]["finish_reason"], "stop")
        self.assertEqual(result["usage"]["completion_tokens"], 1)
        self.assertEqual(self.call("/metrics")[1]["generated_tokens"], 1)

    def test_overlapping_clients_are_serialized_and_queue_is_measured(self):
        self.executor.delay_s = .05
        async def clients():
            return await asyncio.gather(
                asgi_request(self.app, "/v1/completions", {"prompt": "first", "max_tokens": 2}),
                asgi_request(self.app, "/v1/completions", {"prompt": "second", "max_tokens": 3}))
        results = asyncio.run(clients())
        self.assertTrue(all(code == 200 for code, body in results))
        self.assertEqual(self.executor.max_active_calls, 1)
        self.assertGreater(max(body["prismserve"]["queue_wait_ms"] for code, body in results), 20)
        metrics = self.call("/metrics")[1]
        self.assertEqual(metrics["completed_requests"], 2)
        self.assertEqual(metrics["generated_tokens"], 5)
        self.assertGreater(metrics["total_queue_wait_ms"], 20)
        self.assertGreater(metrics["total_service_ms"], metrics["total_inference_ms"])


if __name__ == "__main__":
    unittest.main()
