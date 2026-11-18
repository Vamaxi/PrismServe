"""Small, real, offline decoder for validating PrismServe mechanisms.

This byte-level model is a systems test vehicle, not a pretrained LLM. CUDA
attention offload is synchronous; NEO's optimized overlap is not reproduced.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
import math
import time
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .quantization import MorphLinear


TRAIN_TEXT = """A serving system receives requests and generates one token at a time.
The prompt is processed during prefill. The key and value cache stores attention history.
Memory capacity limits the number of active requests. Quantized weights can free memory.
A controller should observe the hardware before choosing a placement and precision.
CPU attention can keep the key and value cache in host memory while projections run on a GPU.
Transferring an entire cache every token wastes bandwidth. Transfer queries and new keys instead.
The smallest latency is not always the largest throughput. Measure both under the same workload.
Quality must be evaluated on held out examples. A prediction is not a measured quality guarantee.
Hardware varies in compute speed, memory size, memory bandwidth, and transfer bandwidth.
Restore full precision when pressure falls. Preserve the cache across precision transitions.
An experiment records its seed, configuration, hardware, assumptions, and limitations.
Scientific claims require evidence. A simulation explores assumptions; a measurement tests hardware.
The quick brown fox jumps over the lazy dog. Numbers include zero one two three four five.
"""
HELDOUT_TEXT = """The runtime selects a plan using memory capacity and measured bandwidth.
A request waits in a queue before its first output token. Later tokens use cached attention.
Offloading is useful only when the host has enough bandwidth and free memory.
Changing weights may change predictions even when the key and value tensors are preserved.
Compare a fixed precision baseline, a morphing policy, an offload policy, and a joint controller.
The results of this tiny model do not establish the accuracy of a larger language model.
"""


def encode(text: str) -> list[int]:
    return list(text.encode("utf-8"))


def decode(tokens: list[int]) -> str:
    return bytes(tokens).decode("utf-8", errors="replace")


@dataclass
class TinyConfig:
    vocab_size: int = 256
    hidden_size: int = 64
    layers: int = 2
    heads: int = 4
    max_seq_len: int = 256
    group_size: int = 64

    def __post_init__(self) -> None:
        if self.vocab_size != 256 or self.hidden_size <= 0 or self.heads <= 0:
            raise ValueError("byte model requires vocab_size=256 and positive dimensions")
        if self.hidden_size % self.heads or self.layers <= 0 or self.max_seq_len < 2:
            raise ValueError("invalid decoder configuration")


KV = tuple[torch.Tensor, torch.Tensor]
Cache = list[KV]


class DecoderLayer(nn.Module):
    def __init__(self, config: TinyConfig):
        super().__init__()
        self.config = config
        width = config.hidden_size
        self.norm1 = nn.LayerNorm(width)
        self.norm2 = nn.LayerNorm(width)
        self.qkv = MorphLinear(width, 3 * width, group_size=config.group_size)
        self.proj = MorphLinear(width, width, group_size=config.group_size)
        self.fc1 = MorphLinear(width, 4 * width, group_size=config.group_size)
        self.fc2 = MorphLinear(4 * width, width, group_size=config.group_size)

    def set_quantized(self, enabled: bool) -> None:
        for linear in (self.qkv, self.proj, self.fc1, self.fc2):
            linear.set_quantized(enabled)

    def forward(self, x: torch.Tensor, past: KV | None, cpu_attention: bool) -> tuple[torch.Tensor, KV]:
        batch, length, width = x.shape
        qkv = self.qkv(self.norm1(x)).view(batch, length, 3, self.config.heads, width // self.config.heads)
        q, k, v = (qkv[:, :, i].transpose(1, 2) for i in range(3))
        old_length = 0 if past is None else past[0].shape[2]
        # Prefill stays on the model device; decode attention can run on CPU.
        attention_device = torch.device("cpu") if cpu_attention and past is not None else x.device
        q, k, v = q.to(attention_device), k.to(attention_device), v.to(attention_device)
        if past is not None:
            k = torch.cat((past[0].to(attention_device), k), dim=2)
            v = torch.cat((past[1].to(attention_device), v), dim=2)
        scores = q @ k.transpose(-1, -2) / math.sqrt(width // self.config.heads)
        positions = torch.arange(length, device=attention_device)[:, None] + old_length
        keys = torch.arange(k.shape[2], device=attention_device)[None, :]
        scores = scores.masked_fill(keys > positions, float("-inf"))
        attended = torch.softmax(scores, dim=-1) @ v
        attended = attended.transpose(1, 2).contiguous().view(batch, length, width).to(x.device)
        x = x + self.proj(attended)
        x = x + self.fc2(F.gelu(self.fc1(self.norm2(x))))
        cache_device = torch.device("cpu") if cpu_attention else x.device
        return x, (k.to(cache_device), v.to(cache_device))


class TinyDecoder(nn.Module):
    def __init__(self, config: TinyConfig | None = None):
        super().__init__()
        self.config = config or TinyConfig()
        self.embedding = nn.Embedding(self.config.vocab_size, self.config.hidden_size)
        self.positions = nn.Embedding(self.config.max_seq_len, self.config.hidden_size)
        self.layers = nn.ModuleList(DecoderLayer(self.config) for _ in range(self.config.layers))
        self.norm = nn.LayerNorm(self.config.hidden_size)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self.cpu_attention = False

    @property
    def device(self) -> torch.device:
        return self.embedding.weight.device

    @property
    def quantized_layers(self) -> tuple[int, ...]:
        return tuple(i for i, layer in enumerate(self.layers) if layer.qkv.quantized)

    def set_quantized_layers(self, layer_ids: tuple[int, ...] | list[int]) -> None:
        selected = set(layer_ids)
        if any(i < 0 or i >= len(self.layers) for i in selected):
            raise ValueError("invalid layer index")
        for index, layer in enumerate(self.layers):
            layer.set_quantized(index in selected)

    def set_cpu_attention(self, enabled: bool) -> None:
        self.cpu_attention = bool(enabled)

    def apply_plan(self, plan: Any, request_id: str) -> None:
        self.set_quantized_layers(plan.quantized_layers)
        self.set_cpu_attention(request_id in plan.cpu_request_ids)

    def forward(self, tokens: torch.Tensor, cache: Cache | None = None) -> tuple[torch.Tensor, Cache]:
        if tokens.ndim != 2 or tokens.shape[1] < 1:
            raise ValueError("tokens must have nonempty [batch, sequence] shape")
        if cache is not None and len(cache) != len(self.layers):
            raise ValueError("cache must contain one entry per decoder layer")
        offset = 0 if cache is None else cache[0][0].shape[2]
        if offset + tokens.shape[1] > self.config.max_seq_len:
            raise ValueError("context capacity exceeded")
        tokens = tokens.to(self.device)
        positions = torch.arange(offset, offset + tokens.shape[1], device=self.device)
        x = self.embedding(tokens) + self.positions(positions)[None]
        output_cache = []
        for index, layer in enumerate(self.layers):
            x, kv = layer(x, None if cache is None else cache[index], self.cpu_attention)
            output_cache.append(kv)
        return self.lm_head(self.norm(x)), output_cache

    def memory_report(self) -> dict[str, int]:
        resident = sum(t.numel() * t.element_size() for t in self.parameters())
        resident += sum(t.numel() * t.element_size() for t in self.buffers())
        masters = sum(m.master_bytes for m in self.modules() if isinstance(m, MorphLinear))
        scratch = max((m.in_features * m.out_features * 4 for m in self.modules()
                       if isinstance(m, MorphLinear) and m.quantized), default=0)
        return {"resident_model_bytes": resident, "host_master_bytes": masters,
                "max_dequantization_scratch_bytes": scratch}


def resolve_device(device: str = "auto") -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(device)
    if resolved.type not in ("cpu", "cuda"):
        raise ValueError("reference backend supports CPU and CUDA")
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    return resolved


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def save_checkpoint(model: TinyDecoder, path: Path, metadata: dict | None = None) -> None:
    if model.quantized_layers:
        raise ValueError("restore full precision before saving a training checkpoint")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": asdict(model.config),
                "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                "metadata": metadata or {}}, path)


def load_model(checkpoint: Path | None = None, device: str = "auto", seed: int = 7) -> TinyDecoder:
    torch.manual_seed(seed)
    if checkpoint is None:
        model = TinyDecoder()
    else:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model = TinyDecoder(TinyConfig(**payload["config"]))
        model.load_state_dict(payload["state_dict"], strict=True)
    return model.to(resolve_device(device)).eval()


@torch.inference_mode()
def generate(model: TinyDecoder, prompt: str, max_tokens: int = 32,
             callback: Any = None) -> dict:
    ids = encode(prompt)
    if not ids:
        raise ValueError("prompt must not be empty")
    if max_tokens < 1 or len(ids) + max_tokens > model.config.max_seq_len:
        raise ValueError("prompt plus output exceeds context capacity")
    synchronize(model.device)
    started = time.perf_counter()
    logits, cache = model(torch.tensor([ids], device=model.device))
    output = [int(logits[0, -1].argmax())]
    synchronize(model.device)
    first = time.perf_counter()
    token_ms = []
    for index in range(1, max_tokens):
        synchronize(model.device)
        before = time.perf_counter()
        if callback is not None:
            cache = callback(model, cache, index) or cache
        logits, cache = model(torch.tensor([[output[-1]]], device=model.device), cache)
        output.append(int(logits[0, -1].argmax()))
        synchronize(model.device)
        token_ms.append((time.perf_counter() - before) * 1000)
    synchronize(model.device)
    return {"text": decode(output), "token_ids": output, "prompt_tokens": len(ids),
            "completion_tokens": len(output), "ttft_ms": (first - started) * 1000,
            "decode_token_ms": token_ms, "elapsed_ms": (time.perf_counter() - started) * 1000,
            "cache_tokens": cache[0][0].shape[2],
            "cache_device": str(cache[0][0].device)}


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")


def train_tiny(output_dir: Path, steps: int = 100, seed: int = 7) -> dict:
    if steps < 1:
        raise ValueError("steps must be positive")
    torch.set_num_threads(min(4, torch.get_num_threads()))
    torch.manual_seed(seed)
    model = TinyDecoder().to(resolve_device()).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002)
    source = torch.tensor(encode(TRAIN_TEXT), dtype=torch.long)
    generator = torch.Generator().manual_seed(seed)
    history = []
    synchronize(model.device)
    started = time.perf_counter()
    for step in range(steps):
        starts = torch.randint(0, len(source) - 97, (8,), generator=generator)
        batch = torch.stack([source[int(i): int(i) + 97] for i in starts]).to(model.device)
        logits, _ = model(batch[:, :-1])
        loss = F.cross_entropy(logits.reshape(-1, 256), batch[:, 1:].reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 0 or (step + 1) % 10 == 0 or step + 1 == steps:
            history.append({"step": step + 1, "train_nll": float(loss.detach())})
    synchronize(model.device)
    result = {"measurement_kind": "real_tiny_model_training", "seed": seed, "steps": steps,
              "device": str(model.device), "elapsed_s": time.perf_counter() - started,
              "parameters": sum(p.numel() for p in model.parameters()), "history": history,
              "training_data": "authored in torch_backend.TRAIN_TEXT; disjoint heldout text",
              "limitation": "Tiny byte model trained from scratch; not a pretrained LLM."}
    save_checkpoint(model.eval(), output_dir / "tiny_checkpoint.pt", {"steps": steps, "seed": seed})
    _write_json(output_dir / "training.json", result)
    return result


@torch.inference_mode()
def run_quality(output_dir: Path, checkpoint: Path | None = None) -> dict:
    torch.set_num_threads(min(4, torch.get_num_threads()))
    model = load_model(checkpoint)
    heldout = torch.tensor(encode(HELDOUT_TEXT), device=model.device)
    batches = [heldout[i: i + 97][None] for i in range(0, len(heldout) - 97, 96)]

    def evaluate() -> tuple[float, torch.Tensor]:
        total_nll, count, all_logits = 0.0, 0, []
        for batch in batches:
            logits, _ = model(batch[:, :-1])
            total_nll += float(F.cross_entropy(logits.reshape(-1, 256), batch[:, 1:].reshape(-1), reduction="sum"))
            count += batch.numel() - 1
            all_logits.append(logits.float().reshape(-1, 256))
        return total_nll / count, torch.cat(all_logits)

    baseline_nll, baseline_logits = evaluate()
    rows = []
    for name, selected in (("fp32", ()), ("int4_one_layer", (0,)),
                            ("int4_all_layers", tuple(range(model.config.layers)))):
        model.set_quantized_layers(selected)
        nll, logits = evaluate()
        kl = float(F.kl_div(F.log_softmax(logits, dim=-1),
                          F.softmax(baseline_logits, dim=-1), reduction="batchmean"))
        rows.append({"variant": name, "quantized_layers": list(selected), "nll": nll,
                     "perplexity": math.exp(nll), "relative_perplexity_change_pct":
                         (math.exp(nll - baseline_nll) - 1) * 100,
                     "mean_logit_kl": max(0.0, kl), **model.memory_report()})
    model.set_quantized_layers(())
    restored_nll, restored_logits = evaluate()
    result = {"measurement_kind": "real_tiny_model_quality", "device": str(model.device),
              "checkpoint": str(checkpoint) if checkpoint else None,
              "trained": checkpoint is not None, "evaluation_tokens": sum(b.numel() - 1 for b in batches),
              "variants": rows, "restore_max_abs_logit_error": float((baseline_logits - restored_logits).abs().max()),
              "restore_nll": restored_nll,
              "quality_scope": "Held-out authored bytes; not MMLU, GSM8K, WikiText or LLM accuracy.",
              "quality_bound": "No accuracy guarantee follows from the controller sensitivity proxy."}
    _write_json(output_dir / "quality.json", result)
    return result


def demo(output_dir: Path, steps: int = 60) -> dict:
    training = train_tiny(output_dir, steps)
    checkpoint = output_dir / "tiny_checkpoint.pt"
    quality = run_quality(output_dir, checkpoint)
    rows = []
    for device in (["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]):
        model = load_model(checkpoint, device=device)
        for name, selected, offload in (("fp32", (), False), ("int4", (0,), False),
                                        ("cpu_attention", (), True), ("hybrid", (0,), True)):
            model.set_quantized_layers(selected)
            model.set_cpu_attention(offload)
            generate(model, "The serving system", max_tokens=3)  # warmup excluded
            runs = [generate(model, "The serving system", max_tokens=16) for _ in range(3)]
            decode_times = [ms for run in runs for ms in run["decode_token_ms"]]
            rows.append({"device": device, "variant": name, "cpu_offload_effective": offload and device == "cuda",
                         "ttft_ms_mean": sum(run["ttft_ms"] for run in runs) / len(runs),
                         "tbt_ms_mean": sum(decode_times) / len(decode_times),
                         "output_token_ids": runs[0]["token_ids"], "cache_device": runs[0]["cache_device"],
                         **model.memory_report()})
    result = {"measurement_kind": "real_tiny_model_microbenchmark", "training": training,
              "quality_file": "quality.json", "quality": quality, "measurements": rows,
              "notes": ["Synchronous reference kernels with per-matrix INT4 dequantization scratch.",
                        "CUDA measurements synchronize the device; CPU placement is a no-op on CPU.",
                        "Cache precision stays FP32. No optimized asymmetric GPU/CPU overlap.",
                        "Timing samples are small and are not large-model performance evidence."]}
    _write_json(output_dir / "real_backend.json", result)
    return result
