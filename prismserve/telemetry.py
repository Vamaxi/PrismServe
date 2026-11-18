"""Hardware inventory and bounded microbenchmarks, with no invented accelerator.

Copy bandwidth counts bytes read + written for local memory and payload bytes
for host/device transfers. Measurements are process-local, not STREAM or peak
specification bandwidth; roofline profiles should be recalibrated per backend.
"""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import platform
import time

from .types import HardwareProfile


def sample_resources(include_process: bool = True) -> dict:
    """Sample live pressure; allocator reservation is distinct from live tensors.

    Read-only telemetry; availability may change after this sample. Rates are
    obtained separately from profile_hardware, never guessed from utilization.
    """
    import psutil
    memory = psutil.virtual_memory()
    result = {"monotonic_s": time.monotonic(), "host_available_bytes": memory.available,
              "host_total_bytes": memory.total, "host_used_percent": memory.percent,
              "process_rss_bytes": psutil.Process().memory_info().rss if include_process else None,
              "cuda_available": False}
    try:
        import torch
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            result.update(cuda_available=True, device_free_bytes=free, device_total_bytes=total,
                          device_allocated_bytes=torch.cuda.memory_allocated(),
                          device_reserved_bytes=torch.cuda.memory_reserved())
    except (ImportError, RuntimeError, OSError):
        pass
    return result


def detect_hardware() -> HardwareProfile:
    import psutil
    available = int(psutil.virtual_memory().available)
    cpu_name = platform.processor() or platform.machine() or "host CPU"
    # Inventory fallback rates are explicitly uncalibrated, never described as
    # measurements. profile_hardware replaces them with timed values.
    kwargs = dict(name=cpu_name, device_memory_bytes=available,
                  host_memory_bytes=available, device_bandwidth_gbps=10.0,
                  host_bandwidth_gbps=10.0, link_bandwidth_gbps=0.0,
                  device_tflops=0.05, host_tflops=0.05, device_kind="cpu",
                  source="inventory_uncalibrated_defaults", fixed_overhead_ms=0.1)
    try:
        import torch
        if torch.cuda.is_available():
            free, _ = torch.cuda.mem_get_info()
            kwargs.update(name=torch.cuda.get_device_name(0), device_memory_bytes=free,
                          device_kind="cuda", device_bandwidth_gbps=100.0,
                          link_bandwidth_gbps=4.0, device_tflops=1.0)
    except (ImportError, RuntimeError, OSError):
        pass
    return HardwareProfile(**kwargs)


def _host_rates() -> tuple[float, float]:
    import numpy as np
    # 16 MiB buffers avoid exhausting laptops; available caches can inflate the
    # result, so these are called copy microbenchmarks rather than DRAM peaks.
    source = np.ones(4 * 1024 * 1024, dtype=np.float32)
    target = np.empty_like(source)
    np.copyto(target, source)
    samples = []
    for _ in range(5):
        start = time.perf_counter_ns()
        np.copyto(target, source)
        elapsed = max(1, time.perf_counter_ns() - start)
        samples.append(2 * source.nbytes / elapsed)
    bandwidth = max(0.001, float(np.median(samples)))
    rng = np.random.default_rng(0)
    matrix = rng.standard_normal((256, 256), dtype=np.float32)
    matrix @ matrix
    samples = []
    for _ in range(3):
        start = time.perf_counter_ns()
        result = matrix @ matrix
        elapsed = max(1, time.perf_counter_ns() - start)
        samples.append(2 * 256 ** 3 / elapsed / 1000)
    assert result.shape == matrix.shape
    return bandwidth, max(1e-6, float(np.median(samples)))


def profile_hardware(output: str | Path | None = None) -> HardwareProfile:
    """Measure local-copy, matmul, and, when present, CUDA/PCIe rates.

    Returns available capacities at inventory time. Other processes can change
    availability immediately; execution must still handle allocation failures.
    CPU-only profiles have link_bandwidth_gbps=0 and never permit offloading.
    """
    profile = detect_hardware()
    values = asdict(profile)
    host_bw, host_compute = _host_rates()
    values.update(host_bandwidth_gbps=host_bw, host_tflops=host_compute,
                  source="measured_local_microbenchmarks")
    if profile.device_kind == "cpu":
        values.update(device_bandwidth_gbps=host_bw, device_tflops=host_compute)
    else:
        import torch
        device = torch.device("cuda")
        n = 4 * 1024 * 1024
        source = torch.ones(n, dtype=torch.float32, device=device)
        target = torch.empty_like(source)
        host = torch.empty(n, dtype=torch.float32, pin_memory=True)
        target.copy_(source)
        torch.cuda.synchronize()

        def timed(operation, repeats=10):
            begin = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            begin.record()
            for _ in range(repeats):
                operation()
            end.record()
            end.synchronize()
            return max(1e-6, begin.elapsed_time(end) / repeats)

        local_ms = timed(lambda: target.copy_(source))
        to_host_ms = timed(lambda: host.copy_(source, non_blocking=True))
        to_device_ms = timed(lambda: target.copy_(host, non_blocking=True))
        values.update(device_bandwidth_gbps=2 * source.numel() * 4 / (local_ms * 1e6),
                      link_bandwidth_gbps=source.numel() * 4 / (max(to_host_ms, to_device_ms) * 1e6))
        a = torch.randn((1024, 1024), device=device)
        b = torch.empty_like(a)
        torch.mm(a, a, out=b)
        mm_ms = timed(lambda: torch.mm(a, a, out=b), repeats=5)
        values["device_tflops"] = 2 * 1024 ** 3 / (mm_ms * 1e9)
        del source, target, host, a, b
    result = HardwareProfile(**values)
    if output is not None:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(result), indent=2) + "\n", encoding="utf-8")
    return result
