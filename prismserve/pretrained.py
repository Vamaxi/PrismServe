"""Real GPT-2 weights with portable precision and attention placement hooks.

This reference executor implements the standard GPT-2 block using the loaded
model's projections, norms, activations and embeddings. Its KV representation is
independent of Transformers' version-specific Cache API. CPU attention is
synchronous and packed INT4 uses temporary floating-point dequantization.
"""
from __future__ import annotations

import os
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

from pathlib import Path
import math
import time
from typing import Any

import torch
from torch import nn

from .quantization import PackedInt4, pack_int4, unpack_int4
from .types import ModelSpec


Cache = list[tuple[torch.Tensor, torch.Tensor]]


class MorphConv1D(nn.Module):
    """GPT-2 Conv1D projection: ``x @ weight + bias``, weight=[in,out]."""

    def __init__(self, original: nn.Module, group_size: int = 64):
        super().__init__()
        if original.weight.ndim != 2 or original.bias.ndim != 1:
            raise ValueError("expected a GPT-2 Conv1D projection")
        self.shape = tuple(original.weight.shape)
        self.nf = self.shape[1]
        self.group_size = group_size
        self.original_dtype = original.weight.dtype
        self.weight = original.weight
        self.bias = original.bias
        self.register_buffer("packed_data", torch.empty(0, dtype=torch.uint8, device=self.weight.device), persistent=False)
        self.register_buffer("packed_scales", torch.empty(0, device=self.weight.device), persistent=False)
        self._host_master: torch.Tensor | None = None

    @property
    def quantized(self) -> bool:
        return self.weight is None

    @property
    def master_bytes(self) -> int:
        return 0 if self._host_master is None else self._host_master.numel() * self._host_master.element_size()

    @property
    def resident_weight_bytes(self) -> int:
        if self.quantized:
            return self.packed_data.numel() + self.packed_scales.numel() * self.packed_scales.element_size()
        return self.weight.numel() * self.weight.element_size()

    def set_quantized(self, enabled: bool) -> None:
        if enabled == self.quantized:
            return
        if enabled:
            device = self.weight.device
            # Preserve the original FP dtype for exact reversible restoration.
            # Only one matrix is packed at a time to bound host scratch space.
            self._host_master = self.weight.detach().cpu().clone()
            packed = pack_int4(self._host_master, self.group_size)
            self.packed_data = packed.data.to(device)
            self.packed_scales = packed.scales.to(device)
            self.register_parameter("weight", None)
        else:
            if self._host_master is None:
                raise RuntimeError("full-precision master weight is unavailable")
            device = self.packed_data.device
            self.weight = nn.Parameter(self._host_master.to(device).clone(), requires_grad=False)
            self.packed_data = torch.empty(0, dtype=torch.uint8, device=device)
            self.packed_scales = torch.empty(0, dtype=torch.float32, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.quantized:
            weight = unpack_int4(PackedInt4(self.packed_data, self.packed_scales, self.shape,
                                           self.group_size), x.dtype)
        else:
            weight = self.weight
        size_out = x.size()[:-1] + (self.nf,)
        # Match transformers.pytorch_utils.Conv1D rather than transposing weights.
        output = torch.addmm(self.bias, x.reshape(-1, x.size(-1)), weight)
        return output.view(size_out)


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class GPT2Executor(nn.Module):
    """Single-request reference execution for standard, unpruned causal GPT-2."""

    def __init__(self, model: nn.Module, tokenizer: Any, group_size: int = 64):
        super().__init__()
        config = model.config
        if getattr(config, "model_type", None) != "gpt2":
            raise ValueError("GPT2Executor supports only GPT-2 architectures")
        if getattr(config, "add_cross_attention", False):
            raise ValueError("cross-attention GPT-2 configurations are unsupported")
        if getattr(config, "reorder_and_upcast_attn", False):
            raise ValueError("reorder_and_upcast_attn must be false for the reference executor")
        if getattr(config, "pruned_heads", {}):
            raise ValueError("pruned attention heads are unsupported")
        if group_size <= 0:
            raise ValueError("group_size must be positive")
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.config = config
        self.group_size = group_size
        self.context_limit = int(config.max_position_embeddings)
        self.cpu_attention = False
        self.original_parameter_count = sum(p.numel() for p in self.model.parameters())
        self.original_parameter_bytes = sum(p.numel() * p.element_size() for p in self.model.parameters())
        self.original_dtype = self.model.transformer.wte.weight.dtype
        self.original_weight_bits = self.model.transformer.wte.weight.element_size() * 8
        if self.original_weight_bits not in (16, 32):
            raise ValueError("loaded GPT-2 weights must use floating-point 16-bit or 32-bit storage")
        layer_parameters = []
        for block in self.model.transformer.h:
            if getattr(block.attn, "pruned_heads", set()):
                raise ValueError("pruned attention heads are unsupported")
            projections = ((block.attn, "c_attn"), (block.attn, "c_proj"),
                           (block.mlp, "c_fc"), (block.mlp, "c_proj"))
            layer_parameters.append(sum(getattr(parent, name).weight.numel() for parent, name in projections))
            for parent, name in projections:
                original = getattr(parent, name)
                setattr(parent, name, MorphConv1D(original, group_size))
        self.layer_parameters = tuple(layer_parameters)
        self.model.requires_grad_(False)

    @classmethod
    def from_local(cls, path: Path | str, device: str = "auto", dtype: str = "float32",
                   group_size: int = 64) -> "GPT2Executor":
        """Load local safetensors/tokenizer files. Never downloads or executes repo code."""
        from transformers import AutoTokenizer, GPT2LMHeadModel
        path = Path(path)
        if not path.is_dir():
            raise FileNotFoundError(f"local model directory does not exist: {path}")
        dtypes = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
        if dtype not in dtypes:
            raise ValueError("dtype must be float32, float16, or bfloat16")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        target = torch.device(device)
        if target.type not in ("cpu", "cuda") or (target.type == "cuda" and not torch.cuda.is_available()):
            raise ValueError("choose an available CPU or CUDA device")
        tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True, trust_remote_code=False)
        model = GPT2LMHeadModel.from_pretrained(str(path), local_files_only=True,
                                              use_safetensors=True, trust_remote_code=False,
                                              dtype=dtypes[dtype], attn_implementation="eager")
        return cls(model.to(target), tokenizer, group_size)

    @property
    def device(self) -> torch.device:
        return self.model.transformer.wte.weight.device

    @property
    def dtype(self) -> torch.dtype:
        return self.original_dtype

    @property
    def quantized_layers(self) -> tuple[int, ...]:
        return tuple(index for index, block in enumerate(self.model.transformer.h)
                     if block.attn.c_attn.quantized)

    @property
    def layers(self) -> nn.ModuleList:
        return self.model.transformer.h

    def tokenize(self, prompt: str) -> list[int]:
        return list(self.tokenizer.encode(prompt, add_special_tokens=False))

    def decode(self, tokens: list[int]) -> str:
        return self.tokenizer.decode(tokens, skip_special_tokens=True,
                                     clean_up_tokenization_spaces=False)

    def set_quantized_layers(self, layer_ids: tuple[int, ...] | list[int]) -> None:
        selected = set(layer_ids)
        if any(not isinstance(i, int) or isinstance(i, bool) or not 0 <= i < len(self.layers) for i in selected):
            raise ValueError("quantized layer IDs must be valid integer block indices")
        for index, block in enumerate(self.layers):
            for projection in (block.attn.c_attn, block.attn.c_proj, block.mlp.c_fc, block.mlp.c_proj):
                projection.set_quantized(index in selected)

    def set_cpu_attention(self, enabled: bool) -> None:
        self.cpu_attention = bool(enabled)

    def apply_plan(self, plan: Any, request_id: str) -> None:
        self.set_quantized_layers(plan.quantized_layers)
        self.set_cpu_attention(request_id in plan.cpu_request_ids)

    def model_spec(self, sensitivity: tuple[float, ...] = ()) -> ModelSpec:
        return ModelSpec(name="gpt2-pretrained-reference", layers=len(self.layers),
                         hidden_size=int(self.config.n_embd), kv_heads=int(self.config.n_head),
                         head_dim=int(self.config.n_embd) // int(self.config.n_head),
                         parameters=self.original_parameter_count,
                         weight_bits=self.original_weight_bits, kv_bits=self.original_weight_bits,
                         group_size=self.group_size, sensitivity=sensitivity,
                         layer_parameters=self.layer_parameters, reference_dequantization=True)

    def memory_report(self) -> dict[str, int]:
        # parameters() removes the embedding/output-head tie; count each tensor once.
        parameters = sum(p.numel() * p.element_size() for p in self.model.parameters())
        buffers = sum(p.numel() * p.element_size() for p in self.model.buffers())
        projections = [m for m in self.model.modules() if isinstance(m, MorphConv1D)]
        masters = sum(m.master_bytes for m in projections)
        scratch = max((math.prod(m.shape) * torch.empty((), dtype=self.original_dtype).element_size()
                       for m in projections if m.quantized), default=0)
        return {"resident_model_bytes": parameters + buffers,
                "resident_parameter_bytes": parameters, "resident_buffer_bytes": buffers,
                "host_master_bytes": masters, "max_dequantization_scratch_bytes": scratch,
                "original_parameter_bytes": self.original_parameter_bytes,
                "original_parameter_count": self.original_parameter_count}

    def _validate_cache(self, cache: Cache | None, batch: int) -> int:
        if cache is None:
            return 0
        if len(cache) != len(self.layers):
            raise ValueError("cache must contain one KV pair per block")
        offset = int(cache[0][0].shape[2])
        for key, value in cache:
            if (key.shape != value.shape or key.ndim != 4 or key.shape[0] != batch
                    or key.shape[1] != self.config.n_head or key.shape[2] != offset
                    or key.shape[3] != self.config.n_embd // self.config.n_head):
                raise ValueError("inconsistent GPT-2 KV cache")
        return offset

    def forward(self, tokens: torch.Tensor, cache: Cache | None = None) -> tuple[torch.Tensor, Cache]:
        if tokens.ndim != 2 or tokens.shape[0] < 1 or tokens.shape[1] < 1:
            raise ValueError("tokens must have nonempty [batch, sequence] shape")
        if tokens.dtype not in (torch.int32, torch.int64):
            raise ValueError("token IDs must use an integer dtype")
        batch, length = tokens.shape
        offset = self._validate_cache(cache, batch)
        if offset + length > self.context_limit:
            raise ValueError("GPT-2 context capacity exceeded")
        tokens = tokens.to(self.device)
        positions = torch.arange(offset, offset + length, device=self.device)
        hidden = self.model.transformer.wte(tokens) + self.model.transformer.wpe(positions)[None]
        # Dropout is disabled: the loaded model is always in eval mode.
        next_cache = []
        heads = int(self.config.n_head)
        head_dim = int(self.config.n_embd) // heads
        for index, block in enumerate(self.layers):
            normalized = block.ln_1(hidden)
            q, new_k, new_v = block.attn.c_attn(normalized).split(int(self.config.n_embd), dim=-1)
            q, new_k, new_v = (part.view(batch, length, heads, head_dim).transpose(1, 2)
                               for part in (q, new_k, new_v))
            # Initial prefill remains on the projection device. CPU KV becomes
            # persistent immediately after that layer, before the next decode.
            attention_device = torch.device("cpu") if self.cpu_attention and cache is not None else self.device
            q, new_k, new_v = q.to(attention_device), new_k.to(attention_device), new_v.to(attention_device)
            if cache is None:
                key, value = new_k, new_v
            else:
                key = torch.cat((cache[index][0].to(attention_device), new_k), dim=2)
                value = torch.cat((cache[index][1].to(attention_device), new_v), dim=2)
            scores = q @ key.transpose(-1, -2)
            if self.config.scale_attn_weights:
                scores = scores / torch.full([], math.sqrt(head_dim), dtype=scores.dtype, device=attention_device)
            if self.config.scale_attn_by_inverse_layer_idx:
                scores = scores / float(index + 1)
            query_positions = torch.arange(length, device=attention_device)[:, None] + offset
            key_positions = torch.arange(key.shape[2], device=attention_device)[None, :]
            scores = scores.masked_fill(key_positions > query_positions, torch.finfo(scores.dtype).min)
            weights = torch.softmax(scores, dim=-1).to(value.dtype)
            attended = (weights @ value).transpose(1, 2).contiguous().view(batch, length, -1).to(self.device)
            hidden = hidden + block.attn.c_proj(attended)
            hidden = hidden + block.mlp.c_proj(block.mlp.act(block.mlp.c_fc(block.ln_2(hidden))))
            storage_device = torch.device("cpu") if self.cpu_attention else self.device
            next_cache.append((key.to(storage_device), value.to(storage_device)))
        hidden = self.model.transformer.ln_f(hidden)
        return self.model.lm_head(hidden), next_cache

    @torch.inference_mode()
    def generate(self, prompt: str, max_tokens: int = 32, callback: Any = None,
                 stop_on_eos: bool = True) -> dict:
        ids = self.tokenize(prompt)
        if not ids:
            raise ValueError("prompt must contain at least one token")
        if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens < 1:
            raise ValueError("max_tokens must be a positive integer")
        if len(ids) + max_tokens > self.context_limit:
            raise ValueError("prompt plus output exceeds GPT-2 context capacity")
        eos = getattr(self.tokenizer, "eos_token_id", None)
        _synchronize(self.device)
        started = time.perf_counter()
        logits, cache = self(torch.tensor([ids], device=self.device))
        output = [int(logits[0, -1].argmax())]
        _synchronize(self.device)
        first = time.perf_counter()
        token_ms = []
        finish_reason = "length"
        for index in range(1, max_tokens):
            if stop_on_eos and eos is not None and output[-1] == eos:
                finish_reason = "stop"
                break
            _synchronize(self.device)
            before = time.perf_counter()
            if callback is not None:
                updated = callback(self, cache, index)
                if updated is not None:
                    cache = updated
            logits, cache = self(torch.tensor([[output[-1]]], device=self.device), cache)
            output.append(int(logits[0, -1].argmax()))
            _synchronize(self.device)
            token_ms.append((time.perf_counter() - before) * 1000)
        if stop_on_eos and eos is not None and output[-1] == eos:
            finish_reason = "stop"
        _synchronize(self.device)
        return {"text": self.decode(output), "token_ids": output, "prompt_token_ids": ids,
                "prompt_tokens": len(ids), "completion_tokens": len(output),
                "finish_reason": finish_reason, "ttft_ms": (first - started) * 1000,
                "decode_token_ms": token_ms, "elapsed_ms": (time.perf_counter() - started) * 1000,
                "cache_tokens": cache[0][0].shape[2], "cache_device": str(cache[0][0].device)}
