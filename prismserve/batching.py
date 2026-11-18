"""Selective batching for the GPT-2 reference executor.

Every tick flattens the current token chunks across requests, executes each
transformer projection once, and performs ragged attention independently. This
is real shared matrix computation; attention remains a synchronous request loop.
The language-model head processes only each request's final current token.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .pretrained import Cache, GPT2Executor


@dataclass
class BatchInput:
    request_id: str
    token_ids: list[int]
    cache: Cache | None = None
    cpu_attention: bool = False
    prefill: bool = False


@dataclass
class BatchOutput:
    request_id: str
    next_token_id: int
    logits: torch.Tensor
    cache: Cache


@dataclass
class BatchResult:
    outputs: list[BatchOutput]
    linear_tokens: int
    projection_calls: int


def _validate(executor: GPT2Executor, inputs: list[BatchInput]) -> list[int]:
    seen = set()
    offsets = []
    for request in inputs:
        if not isinstance(request.request_id, str) or not request.request_id:
            raise ValueError("batch request IDs must be nonempty strings")
        if request.request_id in seen:
            raise ValueError("batch request IDs must be unique")
        seen.add(request.request_id)
        if not isinstance(request.token_ids, list) or not request.token_ids:
            raise ValueError("each request must supply a nonempty list of token IDs")
        if any(not isinstance(token, int) or isinstance(token, bool)
               or not 0 <= token < executor.config.vocab_size for token in request.token_ids):
            raise ValueError("token IDs must be integers within the model vocabulary")
        if not isinstance(request.cpu_attention, bool) or not isinstance(request.prefill, bool):
            raise ValueError("cpu_attention and prefill must be boolean")
        try:
            offset = executor._validate_cache(request.cache, batch=1)
        except (AttributeError, IndexError, TypeError, ValueError) as error:
            raise ValueError(f"invalid KV cache for request {request.request_id}") from error
        if offset + len(request.token_ids) > executor.context_limit:
            raise ValueError("request chunk exceeds GPT-2 context capacity")
        if request.cache is not None:
            for key, value in request.cache:
                if key.dtype != executor.dtype or value.dtype != executor.dtype:
                    raise ValueError("cache dtype must match model projection dtype")
        offsets.append(offset)
    return offsets


@torch.inference_mode()
def execute_batch(executor: GPT2Executor, inputs: list[BatchInput]) -> BatchResult:
    """Execute one shared tick without mutating input caches or executor policy.

    ``projection_calls`` counts the four Conv1D calls per transformer layer. The
    single final LM-head call is deliberately separate. Results' logits have
    shape [1,vocab] and contain only the last current-token prediction.
    """
    if not inputs:
        return BatchResult([], 0, 0)
    offsets = _validate(executor, inputs)
    spans = []
    flat_tokens, flat_positions = [], []
    for request, offset in zip(inputs, offsets):
        start = len(flat_tokens)
        flat_tokens.extend(request.token_ids)
        flat_positions.extend(range(offset, offset + len(request.token_ids)))
        spans.append((start, len(flat_tokens)))
    ids = torch.tensor(flat_tokens, device=executor.device, dtype=torch.long)
    positions = torch.tensor(flat_positions, device=executor.device, dtype=torch.long)
    transformer = executor.model.transformer
    hidden = transformer.wte(ids) + transformer.wpe(positions)
    heads = int(executor.config.n_head)
    width = int(executor.config.n_embd)
    head_dim = width // heads
    caches: list[Cache] = [[] for _ in inputs]
    calls = 0

    for layer_index, block in enumerate(executor.layers):
        projected = block.attn.c_attn(block.ln_1(hidden))
        calls += 1
        q_all, k_all, v_all = projected.split(width, dim=-1)
        attended_chunks = []
        for request_index, (request, offset, (start, end)) in enumerate(zip(inputs, offsets, spans)):
            length = end - start
            query, key_new, value_new = (
                part[start:end].view(1, length, heads, head_dim).transpose(1, 2)
                for part in (q_all, k_all, v_all)
            )
            # Every prefill chunk stays on the primary device, including chunks
            # with existing context. Decode placement is selected per request.
            offloaded_decode = request.cpu_attention and not request.prefill and request.cache is not None
            attention_device = torch.device("cpu") if offloaded_decode else executor.device
            query, key_new, value_new = (part.to(attention_device) for part in (query, key_new, value_new))
            if request.cache is None:
                # A view would retain the shared QKV allocation and other
                # requests' storage after those requests leave the batch.
                key = key_new.clone(memory_format=torch.contiguous_format)
                value = value_new.clone(memory_format=torch.contiguous_format)
            else:
                past_key, past_value = request.cache[layer_index]
                key = torch.cat((past_key.to(attention_device), key_new), dim=2)
                value = torch.cat((past_value.to(attention_device), value_new), dim=2)
            scores = query @ key.transpose(-1, -2)
            if executor.config.scale_attn_weights:
                scores = scores / torch.full([], math.sqrt(head_dim), dtype=scores.dtype, device=attention_device)
            if executor.config.scale_attn_by_inverse_layer_idx:
                scores = scores / float(layer_index + 1)
            query_positions = torch.arange(length, device=attention_device)[:, None] + offset
            key_positions = torch.arange(key.shape[2], device=attention_device)[None, :]
            scores = scores.masked_fill(key_positions > query_positions, torch.finfo(scores.dtype).min)
            weights = torch.softmax(scores, dim=-1).to(value.dtype)
            attended = (weights @ value).transpose(1, 2).contiguous().view(length, width)
            attended_chunks.append(attended.to(executor.device))
            storage_device = torch.device("cpu") if request.cpu_attention else executor.device
            caches[request_index].append((key.to(storage_device), value.to(storage_device)))
        hidden = hidden + block.attn.c_proj(torch.cat(attended_chunks, dim=0))
        calls += 1
        hidden = hidden + block.mlp.c_proj(block.mlp.act(block.mlp.c_fc(block.ln_2(hidden))))
        calls += 2

    # Gathering before final norm/head prevents a [all_prefill_tokens,vocab]
    # logits allocation. Each request emits one prediction per serving tick.
    last_hidden = torch.stack([hidden[end - 1] for _, end in spans], dim=0)
    logits = executor.model.lm_head(transformer.ln_f(last_hidden))
    selected = logits.argmax(dim=-1).cpu().tolist()
    outputs = [BatchOutput(request.request_id, int(selected[index]), logits[index:index + 1], caches[index])
               for index, request in enumerate(inputs)]
    return BatchResult(outputs, len(flat_tokens), calls)
