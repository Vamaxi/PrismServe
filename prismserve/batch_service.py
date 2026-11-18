"""Concurrent HTTP requests backed by one continuous-batching GPU worker.

The async API only validates and awaits thread-safe concurrent Futures. Model
execution and global precision changes belong to ContinuousEngine's worker.
Importing this module never loads a model or imports the torch runtime.
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import json
import math
from pathlib import Path
import queue
import time
from typing import Any
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field


MODEL_ID = "prismserve-gpt2-batched"


class BatchCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str = Field(default=MODEL_ID, strict=True)
    prompt: str = Field(min_length=1, max_length=8192, strict=True)
    max_tokens: int = Field(default=32, ge=1, le=256, strict=True)
    stream: bool = Field(default=False, strict=True)


def _http_error(error: Exception) -> HTTPException:
    if isinstance(error, (queue.Full, OverflowError, MemoryError)):
        return HTTPException(429, str(error) or "Serving queue or memory budget is full")
    if isinstance(error, ValueError):
        return HTTPException(400, str(error))
    if isinstance(error, (RuntimeError, asyncio.CancelledError)):
        return HTTPException(503, str(error) or "Serving worker is unavailable")
    return HTTPException(500, "Serving backend failed")


def create_batch_app(model_path: Path, calibration_path: Path, device: str = "auto",
                     policy: str = "joint", device_budget_mib: float = 0,
                     max_batch_size: int = 8, *, engine: Any = None) -> FastAPI:
    """Construct the app; lifespan owns worker startup and bounded cleanup.

    An injected engine supports lightweight API tests. Without one, all heavy
    imports and checkpoint loading occur explicitly here, never at import time.
    No checkpoint or calibration download is performed by the server.
    """
    if (not isinstance(max_batch_size, int) or isinstance(max_batch_size, bool)
            or max_batch_size < 1):
        raise ValueError("max_batch_size must be a positive integer")
    if (not isinstance(device_budget_mib, (int, float)) or isinstance(device_budget_mib, bool)
            or not math.isfinite(device_budget_mib) or device_budget_mib < 0):
        raise ValueError("device_budget_mib must be finite and nonnegative")
    if engine is None:
        print("Loading local GPT-2 and batch timing calibration...", flush=True)
        # Calibration/profile errors should fail before allocating model weights.
        payload = json.loads(Path(calibration_path).read_text(encoding="utf-8"))
        from .measured_policy import TimingCalibration
        from .types import HardwareProfile
        from .telemetry import detect_hardware
        timing = TimingCalibration.from_dict(payload.get("timing", payload))
        profile = HardwareProfile(**payload["hardware"]) if "hardware" in payload else detect_hardware()
        sensitivity = tuple(payload.get("sensitivity", ()))
        import torch
        from .pretrained import GPT2Executor
        from .live_engine import ContinuousEngine, ServingConfig
        torch.set_num_threads(min(4, torch.get_num_threads()))
        executor = GPT2Executor.from_local(model_path, device=device, dtype="float32")
        config = ServingConfig(max_batch_size=max_batch_size,
                               device_memory_budget_bytes=int(device_budget_mib * 2**20))
        engine = ContinuousEngine(executor, timing, profile, config=config,
                                  policy=policy, sensitivity=sensitivity)
        print("Continuous-batching model ready.", flush=True)

    @asynccontextmanager
    async def lifespan(app):
        engine.start()
        try:
            yield
        finally:
            # Worker joins and pending-Future resolution must not block the loop.
            await asyncio.to_thread(engine.close, drain=False)

    app = FastAPI(title="PrismServe continuous-batching GPT-2 API", version="1.0.0",
                  description="Concurrent completion requests sharing one model worker and controller.",
                  lifespan=lifespan)
    app.state.engine = engine
    app.state.checkpoint = Path(model_path).resolve()
    app.state.calibration = Path(calibration_path).resolve()
    actual_policy = getattr(engine, "policy", policy)
    if not isinstance(actual_policy, str):
        actual_policy = policy

    def metadata() -> dict:
        executor = getattr(engine, "executor", None)
        dtype = getattr(executor, "dtype", getattr(executor, "original_dtype", "unknown"))
        return {"pretrained": True, "backend": "continuous_batching_gpt2", "model": MODEL_ID,
                "model_kind": "pretrained_base_completion", "policy": actual_policy,
                "device": str(getattr(executor, "device", device)),
                "model_dtype": str(dtype).removeprefix("torch."),
                "context_capacity_tokens": getattr(executor, "context_limit", 1024),
                "checkpoint_source": {"kind": "local", "path": str(app.state.checkpoint)},
                "calibration_path": str(app.state.calibration),
                "execution": "single_worker_continuous_batching", "async_http": True,
                "precision_control": "global_controller", "streaming": False}

    @app.get("/health")
    async def health():
        snapshot = engine.metrics()
        state = snapshot.get("worker_status", snapshot.get("status", "unknown"))
        unavailable = (state in {"failed", "closed", "stopped"}
                       or bool(snapshot.get("worker_error")))
        payload = {"status": "unavailable" if unavailable else "ok", **metadata(),
                   "worker": snapshot}
        return JSONResponse(payload, status_code=503 if unavailable else 200)

    @app.get("/v1/models")
    async def models() -> dict:
        info = metadata()
        return {"object": "list", "data": [{"id": MODEL_ID, "object": "model", "owned_by": "local",
                "pretrained": True, "model_kind": info["model_kind"],
                "context_capacity_tokens": info["context_capacity_tokens"],
                "model_dtype": info["model_dtype"], "continuous_batching": True}]}

    @app.get("/metrics")
    async def metrics() -> dict:
        return {**engine.metrics(), **metadata()}

    @app.post("/v1/completions")
    async def complete(request: BatchCompletionRequest) -> dict:
        entered = time.perf_counter()
        if request.model != MODEL_ID:
            raise HTTPException(404, f"Only {MODEL_ID} is available")
        if request.stream:
            raise HTTPException(400, "Streaming is not implemented")
        request_id = "cmpl-" + uuid.uuid4().hex
        try:
            future = engine.submit(request.prompt, max_tokens=request.max_tokens,
                                   request_id=request_id, arrival_time=entered, stop_on_eos=True)
        except Exception as error:
            raise _http_error(error) from error
        try:
            # Do not use Future.result(): another HTTP handler must be able to
            # submit while this request is waiting for a shared execution tick.
            result = await asyncio.wrap_future(future)
        except asyncio.CancelledError:
            future.cancel()
            raise
        except Exception as error:
            raise _http_error(error) from error
        timing_fields = ("request_id", "token_ids", "ttft_ms", "decode_token_ms", "elapsed_ms", "queue_wait_ms",
                         "admitted_s", "arrival_s", "output_token_times_s", "precision_history",
                         "cacheplacement_history")
        info = {key: result[key] for key in timing_fields if key in result}
        info.update({"pretrained": True, "continuous_batching": True, "policy": actual_policy,
                     "precision_control": "global_controller",
                     "http_elapsed_ms_measured": (time.perf_counter() - entered) * 1000})
        finish_reason = result.get("finish_reason", "length")
        return {"id": result.get("request_id", request_id), "object": "text_completion",
                "created": int(time.time()), "model": MODEL_ID,
                "choices": [{"text": result["text"], "index": 0, "finish_reason": finish_reason,
                             "logprobs": None}],
                "usage": {"prompt_tokens": result["prompt_tokens"],
                          "completion_tokens": result["completion_tokens"],
                          "total_tokens": result["prompt_tokens"] + result["completion_tokens"]},
                "prismserve": info}

    return app
