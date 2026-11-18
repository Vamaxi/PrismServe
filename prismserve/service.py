"""Bounded, serialized FastAPI API for the real tiny reference backend."""
from __future__ import annotations

from pathlib import Path
import threading
import time
import uuid

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field
import torch

from .torch_backend import encode, generate, load_model


class CompletionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str = "prismserve-tiny"
    prompt: str = Field(min_length=1, max_length=1024)
    max_tokens: int = Field(default=32, ge=1, le=128)
    stream: bool = False
    quantized_layers: list[int] = Field(default_factory=list)
    cpu_attention: bool = False

def create_app(checkpoint: Path | None = None, device: str = "auto") -> FastAPI:
    torch.set_num_threads(min(4, torch.get_num_threads()))
    model = load_model(checkpoint, device)
    lock = threading.Lock()
    metrics = {"completed_requests": 0, "generated_tokens": 0, "total_inference_ms": 0.0}
    app = FastAPI(title="PrismServe reference API", version="0.1.0",
                  description="Serialized tiny byte decoder; no pretrained LLM or production batching.")
    app.state.model = model
    app.state.checkpoint = checkpoint
    app.state.metrics = metrics

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "backend": "tiny_reference", "device": str(model.device),
                "trained_checkpoint": checkpoint is not None,
                "context_capacity_tokens": model.config.max_seq_len,
                "cpu_offload_effective": model.device.type == "cuda",
                "execution": "serialized synchronous reference"}

    @app.get("/v1/models")
    def models() -> dict:
        return {"object": "list", "data": [{"id": "prismserve-tiny", "object": "model", "owned_by": "local"}]}

    @app.get("/metrics")
    def get_metrics() -> dict:
        with lock:
            return {**metrics, "device": str(model.device), **model.memory_report()}

    @app.post("/v1/completions")
    def complete(request: CompletionRequest) -> dict:
        if request.model != "prismserve-tiny":
            raise HTTPException(404, "Only the local prismserve-tiny model is available")
        if request.stream:
            raise HTTPException(400, "Streaming is not implemented by this reference service")
        if len(encode(request.prompt)) + request.max_tokens > model.config.max_seq_len:
            raise HTTPException(400, "UTF-8 byte prompt plus output exceeds context capacity")
        if any(i < 0 or i >= model.config.layers for i in request.quantized_layers):
            raise HTTPException(400, "quantized_layers contains an invalid decoder layer")
        queued_at = time.perf_counter()
        with lock:
            queue_wait_ms = (time.perf_counter() - queued_at) * 1000
            setup_started = time.perf_counter()
            model.set_quantized_layers(tuple(request.quantized_layers))
            model.set_cpu_attention(request.cpu_attention)
            if model.device.type == "cuda":
                torch.cuda.synchronize(model.device)
            setup_ms = (time.perf_counter() - setup_started) * 1000
            result = generate(model, request.prompt, request.max_tokens)
            metrics["completed_requests"] += 1
            metrics["generated_tokens"] += result["completion_tokens"]
            metrics["total_inference_ms"] += result["elapsed_ms"]
        return {"id": "cmpl-" + uuid.uuid4().hex, "object": "text_completion", "created": int(time.time()),
                "model": request.model, "choices": [{"text": result["text"], "index": 0,
                "finish_reason": "length", "logprobs": None}],
                "usage": {"prompt_tokens": result["prompt_tokens"],
                          "completion_tokens": result["completion_tokens"],
                          "total_tokens": result["prompt_tokens"] + result["completion_tokens"]},
                "prismserve": {"queue_wait_ms": queue_wait_ms, "setup_ms": setup_ms,
                               "ttft_ms": result["ttft_ms"] + queue_wait_ms + setup_ms,
                               "decode_token_ms": result["decode_token_ms"],
                               "cache_device": result["cache_device"],
                               "precision": "packed_int4_reference" if request.quantized_layers else "fp32",
                               "trained_checkpoint": checkpoint is not None}}

    return app
