"""Async batching API tests with no torch import or checkpoint allocation."""
from __future__ import annotations

import asyncio
from concurrent.futures import Future
import importlib.util
import json
from pathlib import Path
import queue
import threading
from types import SimpleNamespace
import unittest


async def asgi_request(app, path, body=None):
    payload = b"" if body is None else json.dumps(body).encode()
    messages = []
    scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
             "method": "GET" if body is None else "POST", "scheme": "http", "path": path,
             "raw_path": path.encode(), "query_string": b"", "root_path": "",
             "headers": [(b"content-type", b"application/json")],
             "client": ("127.0.0.1", 1), "server": ("test", 80)}
    async def receive():
        return {"type": "http.request", "body": payload, "more_body": False}
    async def send(message):
        messages.append(message)
    await app(scope, receive, send)
    status = next(m["status"] for m in messages if m["type"] == "http.response.start")
    return status, json.loads(b"".join(m.get("body", b"") for m in messages))


class FakeEngine:
    def __init__(self, batch_target=1):
        self.executor = SimpleNamespace(device="cpu", dtype="float32", context_limit=1024)
        self.batch_target = batch_target
        self.pending = []
        self.submitted = 0
        self.completed = 0
        self.max_pending = 0
        self.starts = 0
        self.closes = []
        self.worker_status = "not_started"
        self.submit_error = None
        self.future_error = None
        self.lock = threading.Lock()

    def start(self):
        self.starts += 1
        self.worker_status = "running"

    def close(self, drain=True):
        self.closes.append(drain)
        self.worker_status = "closed"
        with self.lock:
            for future, kwargs in self.pending:
                if not future.done():
                    future.set_exception(RuntimeError("worker closed"))
            self.pending.clear()

    def submit(self, prompt, max_tokens=32, request_id=None, arrival_time=None, stop_on_eos=True):
        if self.submit_error:
            raise self.submit_error
        if self.worker_status == "closed":
            raise RuntimeError("worker closed")
        if len(prompt) * 2 + max_tokens > self.executor.context_limit:
            raise ValueError("tokenized prompt exceeds context capacity")
        if not stop_on_eos:
            raise AssertionError("API should respect EOS")
        future = Future()
        kwargs = {"prompt": prompt, "max_tokens": max_tokens, "request_id": request_id,
                  "arrival_time": arrival_time}
        with self.lock:
            self.submitted += 1
            self.pending.append((future, kwargs))
            self.max_pending = max(self.max_pending, len(self.pending))
            if len(self.pending) >= self.batch_target:
                # Completing together only after both submissions proves that
                # the first HTTP handler did not block the event loop.
                current, self.pending = self.pending, []
                for item, data in current:
                    if self.future_error:
                        item.set_exception(self.future_error)
                        continue
                    count = min(data["max_tokens"], 2)
                    item.set_result({"request_id": data["request_id"], "text": "actual output",
                        "token_ids": [1] * count, "prompt_tokens": len(data["prompt"]) * 2,
                        "completion_tokens": count, "finish_reason": "stop", "ttft_ms": 12.0,
                        "decode_token_ms": [3.0] * (count - 1), "elapsed_ms": 15.0,
                        "queue_wait_ms": 9.0, "arrival_s": data["arrival_time"],
                        "admitted_s": data["arrival_time"] + .009,
                        "output_token_times_s": [data["arrival_time"] + .012 + i * .003 for i in range(count)],
                        "precision_history": [{"quantized_layers": [1]}],
                        "cacheplacement_history": [{"device": "cpu"}]})
                    self.completed += 1
        return future

    def metrics(self):
        return {"worker_status": self.worker_status, "completed": self.completed,
                "pending": len(self.pending), "max_active": self.max_pending,
                "max_execution_batch_size": self.max_pending}


@unittest.skipUnless(importlib.util.find_spec("fastapi"), "install the runtime dependencies")
class BatchServiceTests(unittest.TestCase):
    def app(self, engine=None):
        from prismserve.batch_service import create_batch_app
        return create_batch_app(Path("models/fake"), Path("calibration/fake.json"),
                                engine=engine or FakeEngine())

    def call(self, app, path, body=None):
        return asyncio.run(asgi_request(app, path, body))

    def test_async_clients_submit_before_shared_batch_completes(self):
        engine = FakeEngine(batch_target=2)
        app = self.app(engine)
        async def run():
            async with app.router.lifespan_context(app):
                return await asyncio.wait_for(asyncio.gather(
                    asgi_request(app, "/v1/completions", {"prompt": "first", "max_tokens": 3}),
                    asgi_request(app, "/v1/completions", {"prompt": "second", "max_tokens": 4})), timeout=2)
        responses = asyncio.run(run())
        self.assertEqual(engine.submitted, 2)
        self.assertEqual(engine.max_pending, 2)
        self.assertEqual(engine.completed, 2)
        self.assertTrue(all(code == 200 for code, payload in responses))
        self.assertEqual(engine.starts, 1)
        self.assertEqual(engine.closes, [False])

    def test_response_preserves_worker_timings_history_and_actual_usage(self):
        app = self.app()
        code, payload = self.call(app, "/v1/completions", {"prompt": "Hello", "max_tokens": 32})
        self.assertEqual(code, 200)
        self.assertEqual(payload["model"], "prismserve-gpt2-batched")
        self.assertEqual(payload["usage"], {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12})
        self.assertEqual(payload["choices"][0]["finish_reason"], "stop")
        self.assertEqual(payload["prismserve"]["ttft_ms"], 12.0)
        self.assertEqual(payload["prismserve"]["queue_wait_ms"], 9.0)
        self.assertEqual(payload["prismserve"]["precision_history"], [{"quantized_layers": [1]}])
        self.assertTrue(payload["prismserve"]["continuous_batching"])

    def test_health_models_and_metrics(self):
        engine = FakeEngine()
        engine.start()
        app = self.app(engine)
        code, payload = self.call(app, "/health")
        self.assertEqual(code, 200)
        self.assertEqual(payload["execution"], "single_worker_continuous_batching")
        self.assertEqual(payload["worker"]["worker_status"], "running")
        self.assertEqual(self.call(app, "/v1/models")[1]["data"][0]["id"], "prismserve-gpt2-batched")
        self.assertEqual(self.call(app, "/metrics")[1]["completed"], 0)
        engine.close(drain=False)
        self.assertEqual(self.call(app, "/health")[0], 503)

    def test_schema_forbids_per_request_global_controls_and_invalid_values(self):
        engine = FakeEngine()
        app = self.app(engine)
        bodies = [{"prompt": "a", "quantized_layers": [0]},
                  {"prompt": "a", "cpu_attention": True},
                  {"prompt": "a", "temperature": .7},
                  {"prompt": ""}, {"prompt": 2}, {"prompt": "a", "stream": 1}]
        bodies += [{"prompt": "a", "max_tokens": v} for v in (0, -1, 257, True, 1.1, "3")]
        for body in bodies:
            with self.subTest(body=body):
                self.assertEqual(self.call(app, "/v1/completions", body)[0], 422)
        self.assertEqual(engine.submitted, 0)

    def test_wrong_model_stream_and_real_token_limit(self):
        engine = FakeEngine()
        app = self.app(engine)
        self.assertEqual(self.call(app, "/v1/completions", {"prompt": "a", "model": "gpt2"})[0], 404)
        self.assertEqual(self.call(app, "/v1/completions", {"prompt": "a", "stream": True})[0], 400)
        self.assertEqual(self.call(app, "/v1/completions", {"prompt": "a" * 500, "max_tokens": 25})[0], 400)
        self.assertEqual(engine.submitted, 0)

    def test_queue_capacity_closed_worker_and_backend_errors(self):
        for error, expected in [(queue.Full("queue full"), 429), (OverflowError("full"), 429),
                                (MemoryError("budget exceeded"), 429),
                                (RuntimeError("closed"), 503), (ValueError("bad request"), 400)]:
            with self.subTest(error=type(error).__name__):
                engine = FakeEngine()
                engine.submit_error = error
                self.assertEqual(self.call(self.app(engine), "/v1/completions", {"prompt": "a"})[0], expected)
        for error, expected in [(RuntimeError("worker failed"), 503),
                                (ValueError("bad future request"), 400), (KeyError("backend defect"), 500)]:
            with self.subTest(future_error=type(error).__name__):
                engine = FakeEngine()
                engine.future_error = error
                self.assertEqual(self.call(self.app(engine), "/v1/completions", {"prompt": "a"})[0], expected)

    def test_lifespan_resolves_pending_requests_on_close(self):
        engine = FakeEngine(batch_target=10)
        app = self.app(engine)
        async def run():
            async with app.router.lifespan_context(app):
                request = asyncio.create_task(asgi_request(app, "/v1/completions", {"prompt": "a"}))
                await asyncio.sleep(.01)
                self.assertEqual(engine.submitted, 1)
            return await asyncio.wait_for(request, timeout=2)
        code, payload = asyncio.run(run())
        self.assertEqual(code, 503)
        self.assertEqual(engine.closes, [False])

    def test_invalid_server_limits_fail_without_loading_model(self):
        from prismserve.batch_service import create_batch_app
        for budget in (-1, float("nan"), float("inf"), True):
            with self.assertRaises(ValueError):
                create_batch_app(Path("missing"), Path("missing"), device_budget_mib=budget, engine=FakeEngine())
        for batch in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                create_batch_app(Path("missing"), Path("missing"), max_batch_size=batch, engine=FakeEngine())


if __name__ == "__main__":
    unittest.main()
