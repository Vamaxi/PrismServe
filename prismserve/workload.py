"""Seeded synthetic traces and strict, explicit CSV replay (all times in seconds)."""
from __future__ import annotations

import csv
import random
from pathlib import Path

from .types import RequestSpec


def synthetic_trace(seed: int = 7, requests: int = 96, duration_s: float = 20,
                    kind: str = "bursty", prompt_min: int = 256, prompt_max: int = 1536,
                    output_min: int = 24, output_max: int = 64) -> list[RequestSpec]:
    if requests <= 0 or duration_s <= 0:
        raise ValueError("requests and duration_s must be positive")
    if not 0 < prompt_min <= prompt_max or not 0 < output_min <= output_max:
        raise ValueError("token bounds must be positive and ordered")
    if kind not in {"bursty", "steady"}:
        raise ValueError("kind must be bursty or steady")
    rng = random.Random(seed)
    times = []
    for _ in range(requests):
        if kind == "bursty" and rng.random() < .65:
            t = rng.uniform(.35, .55) * duration_s
        else:
            t = rng.uniform(0, duration_s)
        times.append(t)
    times.sort()
    return [RequestSpec(f"r{i:05d}", t, rng.randint(prompt_min, prompt_max),
                        rng.randint(output_min, output_max)) for i, t in enumerate(times)]


def write_trace(requests: list[RequestSpec], path: Path) -> None:
    from dataclasses import asdict
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["request_id", "arrival_s", "prompt_tokens", "output_tokens"])
        writer.writeheader()
        writer.writerows(asdict(r) for r in requests)


def read_trace(path: Path) -> list[RequestSpec]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"request_id", "arrival_s", "prompt_tokens", "output_tokens"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"CSV requires columns {sorted(required)}; convert dataset columns explicitly")
        result = [RequestSpec(row["request_id"], float(row["arrival_s"]), int(row["prompt_tokens"]),
                              int(row["output_tokens"])) for row in reader]
    if not result:
        raise ValueError("trace is empty")
    if len({r.request_id for r in result}) != len(result):
        raise ValueError("request IDs must be unique")
    return sorted(result, key=lambda r: (r.arrival_s, r.request_id))
