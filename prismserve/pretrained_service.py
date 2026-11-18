"""Serialized local GPT-2 completions API backed by actual pretrained weights.

The executor is imported lazily when constructing the application. Importing
this module does not load Torch, Transformers, TensorFlow, or a checkpoint.
This is a base-model completion endpoint, not a chat or batching service.
"""
from __future__ import annotations

from pathlib import Path
import threading
import time
from typing import Any
import uuid

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, StrictInt


MODEL_ID = "prismserve-gpt2"


class PretrainedCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str = MODEL_ID
    prompt: str = Field(min_length=1, max_length=8192, strict=True)
    max_tokens: int = Field(default=32, ge=1, le=256, strict=True)
    stream: bool = Field(default=False, strict=True)
    quantized_layers: list[StrictInt] = Field(default_factory=list)
    cpu_attention: bool = Field(default=False, strict=True)


def _device_type(executor: Any) -> str:
    return getattr(executor.device, "type", str(executor.device).split(":", 1)[0])


def _dtype(executor: Any) -> str:
    value = getattr(executor, "dtype", getattr(executor, "original_dtype", "unknown"))
    return str(value).removeprefix("torch.")


def _synchronize(executor: Any) -> None:
    if _device_type(executor) == "cuda":
        import torch
        torch.cuda.synchronize(executor.device)


def create_pretrained_app(model_path: Path, device: str = "auto", *, executor: Any = None) -> FastAPI:
    """Load one local executor, or inject a test double without loading weights.

    No network or model downloads occur here. Every precision transition and
    generation shares one lock because the executor owns mutable layer state.
    Queue, setup, backend generation and service wall times are kept distinct.
    """
    checkpoint = Path(model_path).resolve()
    if executor is None:
        from .pretrained import GPT2Executor
        import torch
        torch.set_num_threads(min(4, torch.get_num_threads()))
        executor = GPT2Executor.from_local(checkpoint, device=device, dtype="float32")
    layer_count = executor.model_spec().layers
    lock = threading.Lock()
    metrics = {"completed_requests": 0, "generated_tokens": 0,
               "total_queue_wait_ms": 0.0, "total_setup_ms": 0.0,
               "total_inference_ms": 0.0, "total_model_inference_ms": 0.0,
               "total_service_ms": 0.0}
    app = FastAPI(title="PrismServe pretrained GPT-2 API", version="0.2.0",
                  description="Local pretrained base GPT-2 text completion; serialized synchronous execution.")
    app.state.executor = executor
    app.state.model = executor
    app.state.checkpoint = checkpoint
    app.state.metrics = metrics

    def metadata() -> dict:
        return {"pretrained": True, "backend": "local_pretrained_gpt2",
                "model": MODEL_ID, "model_kind": "pretrained_base_completion",
                "checkpoint_source": {"kind": "local", "path": str(checkpoint)},
                "device": str(executor.device), "model_dtype": _dtype(executor),
                "context_capacity_tokens": executor.context_limit,
                "cpu_offload_supported": _device_type(executor) == "cuda",
                "cpu_attention_enabled": bool(getattr(executor, "cpu_attention", False)),
                "cpu_offload_effective": (_device_type(executor) == "cuda"
                                          and bool(getattr(executor, "cpu_attention", False))),
                "quantized_layers": list(executor.quantized_layers),
                "execution": "serialized synchronous reference"}

    @app.get("/health")
    def health() -> dict:
        with lock:
            return {"status": "ok", **metadata(), "memory": executor.memory_report()}

    @app.get("/v1/models")
    def models() -> dict:
        return {"object": "list", "data": [{"id": MODEL_ID, "object": "model", "owned_by": "local",
                "pretrained": True, "model_kind": "pretrained_base_completion",
                "context_capacity_tokens": executor.context_limit, "model_dtype": _dtype(executor)}]}

    @app.get("/metrics")
    def get_metrics() -> dict:
        with lock:
            return {**metrics, **metadata(), **executor.memory_report(),
                    "timing_note": "total_inference_ms includes setup and measured generation wall time; "
                                   "total_model_inference_ms is backend-reported generation time; "
                                   "total_service_ms also includes serialization queue wait"}

    @app.post("/v1/completions")
    def complete(request: PretrainedCompletionRequest) -> dict:
        if request.model != MODEL_ID:
            raise HTTPException(404, f"Only the local {MODEL_ID} model is available")
        if request.stream:
            raise HTTPException(400, "Streaming is not implemented by this synchronous service")
        if (len(set(request.quantized_layers)) != len(request.quantized_layers)
                or any(i < 0 or i >= layer_count for i in request.quantized_layers)):
            raise HTTPException(400, "quantized_layers must contain unique valid decoder layer indices")
        ids = executor.tokenize(request.prompt)
        if not ids:
            raise HTTPException(400, "The tokenized prompt is empty")
        if len(ids) + request.max_tokens > executor.context_limit:
            raise HTTPException(400, "Tokenized prompt plus maximum output exceeds context capacity")
        queued_at = time.perf_counter()
        with lock:
            acquired_at = time.perf_counter()
            queue_wait_ms = (acquired_at - queued_at) * 1000
            executor.set_quantized_layers(tuple(sorted(request.quantized_layers)))
            executor.set_cpu_attention(request.cpu_attention)
            _synchronize(executor)
            setup_finished = time.perf_counter()
            setup_ms = (setup_finished - acquired_at) * 1000
            result = executor.generate(request.prompt, max_tokens=request.max_tokens, stop_on_eos=True)
            _synchronize(executor)
            finished_at = time.perf_counter()
            inference_ms = (finished_at - acquired_at) * 1000
            service_ms = (finished_at - queued_at) * 1000
            metrics["completed_requests"] += 1
            metrics["generated_tokens"] += result["completion_tokens"]
            metrics["total_queue_wait_ms"] += queue_wait_ms
            metrics["total_setup_ms"] += setup_ms
            metrics["total_inference_ms"] += inference_ms
            metrics["total_model_inference_ms"] += result["elapsed_ms"]
            metrics["total_service_ms"] += service_ms
            effective_cpu_offload = (_device_type(executor) == "cuda"
                                     and bool(getattr(executor, "cpu_attention", False)))
            applied_layers = list(executor.quantized_layers)
            memory = executor.memory_report()
        finish_reason = result.get("finish_reason")
        if finish_reason not in {"stop", "length"}:
            finish_reason = "stop" if result["completion_tokens"] < request.max_tokens else "length"
        return {"id": "cmpl-" + uuid.uuid4().hex, "object": "text_completion", "created": int(time.time()),
                "model": MODEL_ID, "choices": [{"text": result["text"], "index": 0,
                    "finish_reason": finish_reason, "logprobs": None}],
                "usage": {"prompt_tokens": result["prompt_tokens"],
                          "completion_tokens": result["completion_tokens"],
                          "total_tokens": result["prompt_tokens"] + result["completion_tokens"]},
                "prismserve": {"pretrained": True, "checkpoint_source": "local",
                    "model_dtype": _dtype(executor), "quantized_layers": applied_layers,
                    "precision": "mixed_packed_int4_reference" if applied_layers else _dtype(executor),
                    "cpu_attention_requested": request.cpu_attention,
                    "cpu_offload_effective": effective_cpu_offload,
                    "queue_wait_ms": queue_wait_ms, "setup_ms": setup_ms,
                    "ttft_ms": result["ttft_ms"] + queue_wait_ms + setup_ms,
                    "model_ttft_ms": result["ttft_ms"],
                    "decode_token_ms": result["decode_token_ms"],
                    "model_inference_ms": result["elapsed_ms"], "inference_elapsed_ms": inference_ms,
                    "service_elapsed_ms": service_ms, "cache_device": result["cache_device"],
                    "memory": memory}}

    return app
