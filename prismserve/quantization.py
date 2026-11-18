"""Portable weight-only INT4 reference implementation (no fused INT4 kernel).

The packed representation is real; forward temporarily dequantizes one matrix.
CPU master weights allow exact restoration and count against host memory.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class PackedInt4:
    data: torch.Tensor
    scales: torch.Tensor
    shape: tuple[int, ...]
    group_size: int

    @property
    def nbytes(self) -> int:
        return self.data.numel() + self.scales.numel() * self.scales.element_size()


def pack_int4(weight: torch.Tensor, group_size: int = 64) -> PackedInt4:
    """Symmetric per-group [-7,7] quantization, two signed values per byte."""
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    if not weight.is_floating_point() or not torch.isfinite(weight).all():
        raise ValueError("weights must be finite floating point values")
    values = weight.detach().float().reshape(-1)
    if not values.numel():
        raise ValueError("cannot quantize an empty tensor")
    groups = math.ceil(values.numel() / group_size)
    padded = F.pad(values, (0, groups * group_size - values.numel()))
    grouped = padded.view(groups, group_size)
    scales = grouped.abs().amax(dim=1) / 7.0
    scales = torch.where(scales > 0, scales, torch.ones_like(scales))
    quantized = (grouped / scales[:, None]).round().clamp(-7, 7)
    encoded = (quantized.reshape(-1).to(torch.int16) + 8).to(torch.uint8)
    if encoded.numel() % 2:
        encoded = F.pad(encoded, (0, 1), value=8)
    data = encoded[0::2] | (encoded[1::2] << 4)
    return PackedInt4(data.contiguous(), scales.contiguous(), tuple(weight.shape), group_size)


def unpack_int4(packed: PackedInt4, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    low = (packed.data & 15).to(torch.int16) - 8
    high = (packed.data >> 4).to(torch.int16) - 8
    flat = torch.stack((low, high), dim=1).reshape(-1)
    count = packed.scales.numel() * packed.group_size
    values = flat[:count].float().view(-1, packed.group_size) * packed.scales[:, None]
    return values.reshape(-1)[: math.prod(packed.shape)].reshape(packed.shape).to(dtype)


class MorphLinear(nn.Module):
    """A linear projection that releases its device FP weight when quantized."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True, group_size: int = 64):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        nn.init.normal_(self.weight, std=0.02)
        self.register_buffer("packed_data", torch.empty(0, dtype=torch.uint8), persistent=False)
        self.register_buffer("packed_scales", torch.empty(0), persistent=False)
        self._host_master: torch.Tensor | None = None

    @property
    def quantized(self) -> bool:
        return self.weight is None

    def set_quantized(self, enabled: bool) -> None:
        if enabled == self.quantized:
            return
        if enabled:
            device = self.weight.device
            self._host_master = self.weight.detach().cpu().clone()
            packed = pack_int4(self._host_master, self.group_size)
            self.packed_data = packed.data.to(device)
            self.packed_scales = packed.scales.to(device)
            self.register_parameter("weight", None)
        else:
            if self._host_master is None:
                raise RuntimeError("FP master unavailable")
            device = self.packed_data.device
            self.weight = nn.Parameter(self._host_master.to(device).clone(), requires_grad=False)
            self.packed_data = torch.empty(0, dtype=torch.uint8, device=device)
            self.packed_scales = torch.empty(0, device=device)

    @property
    def resident_weight_bytes(self) -> int:
        if self.quantized:
            return self.packed_data.numel() + self.packed_scales.element_size() * self.packed_scales.numel()
        return self.weight.numel() * self.weight.element_size()

    @property
    def master_bytes(self) -> int:
        return 0 if self._host_master is None else self._host_master.numel() * self._host_master.element_size()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.quantized:
            packed = PackedInt4(self.packed_data, self.packed_scales,
                                (self.out_features, self.in_features), self.group_size)
            weight = unpack_int4(packed, x.dtype)
        else:
            weight = self.weight
        return F.linear(x, weight, self.bias)
