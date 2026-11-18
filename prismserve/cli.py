"""Lazy imports keep analytical experiments independent of the PyTorch runtime."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import platform
import sys


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="prismserve", description="PrismServe research prototype")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor", help="Show Python, dependencies and actual hardware")
    profile = commands.add_parser("profile", help="Measure bounded host/device/link microbenchmarks")
    profile.add_argument("--output", type=Path, default=Path("results/local_profile.json"))
    sim = commands.add_parser("simulate", help="Run analytical policy comparisons")
    sim.add_argument("--config", type=Path, default=Path("configs/experiment.json"))
    sim.add_argument("--output", type=Path, default=Path("results/simulation"))
    sim.add_argument("--trace", type=Path, help="Normalized request CSV; config must contain one seed")
    train = commands.add_parser("train", help="Train a tiny byte decoder on the included authored corpus")
    train.add_argument("--output", type=Path, default=Path("results/tiny"))
    train.add_argument("--steps", type=int, default=100)
    train.add_argument("--seed", type=int, default=7)
    quality = commands.add_parser("quality", help="Measure held-out tiny-model NLL/PPL and logit KL")
    quality.add_argument("--checkpoint", type=Path)
    quality.add_argument("--output", type=Path, default=Path("results/quality"))
    demo_cmd = commands.add_parser("demo", help="Train and run real reversible INT4 and CPU KV demo")
    demo_cmd.add_argument("--output", type=Path, default=Path("results/demo"))
    demo_cmd.add_argument("--steps", type=int, default=60)
    adapt = commands.add_parser("adapt", help="Run real decoder with the common controller under declared quotas")
    adapt.add_argument("--checkpoint", type=Path)
    adapt.add_argument("--output", type=Path, default=Path("results/adaptive"))
    adapt.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    serve = commands.add_parser("serve", help="Start local tiny-model completion API")
    serve.add_argument("--checkpoint", type=Path)
    serve.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    report = commands.add_parser("report", help="Regenerate tables/figures from a completed simulation")
    report.add_argument("--results", type=Path, default=Path("results/simulation"))
    prepare = commands.add_parser("pretrained-prepare", help="Prepare the pinned GPT-2 checkpoint from cache or explicit download")
    prepare.add_argument("--output", type=Path, default=Path("models/gpt2"))
    prepare.add_argument("--download", action="store_true", help="Allow downloading missing public model files")
    prepare.add_argument("--dataset", action="store_true", help="Also fetch bounded WikiText-2 validation/test prefixes")
    pg = commands.add_parser("pretrained-generate", help="Generate actual pretrained GPT-2 text")
    pg.add_argument("--model-path", type=Path, default=Path("models/gpt2"))
    pg.add_argument("--prompt", default="Artificial intelligence can help scientists")
    pg.add_argument("--max-tokens", type=int, default=32)
    pg.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    pg.add_argument("--quantized-layers", default="", help="Comma-separated decoder layer indices")
    pg.add_argument("--cpu-attention", action="store_true")
    pb = commands.add_parser("pretrained-benchmark", help="Measured pretrained quality, correctness and CPU/GPU benchmarks")
    pb.add_argument("--model-path", type=Path, default=Path("models/gpt2"))
    pb.add_argument("--data-path", type=Path, default=Path("data/wikitext2"))
    pb.add_argument("--output", type=Path, default=Path("results/pretrained"))
    pb.add_argument("--quality-tokens", type=int, default=4096)
    pb.add_argument("--repeats", type=int, default=3)
    pb.add_argument("--max-tokens", type=int, default=32)
    pb.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    pa = commands.add_parser("pretrained-adapt", help="Real GPT-2 generation driven by the common controller")
    pa.add_argument("--model-path", type=Path, default=Path("models/gpt2"))
    pa.add_argument("--output", type=Path, default=Path("results/pretrained/adaptive.json"))
    pa.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ps = commands.add_parser("pretrained-serve", help="Serve the actual pretrained GPT-2 model locally")
    ps.add_argument("--model-path", type=Path, default=Path("models/gpt2"))
    ps.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ps.add_argument("--host", default="127.0.0.1")
    ps.add_argument("--port", type=int, default=8000)
    lc = commands.add_parser("live-calibrate", help="Measure actual multi-request batch costs and transition overhead")
    lc.add_argument("--model-path", type=Path, default=Path("models/gpt2"))
    lc.add_argument("--output", type=Path, default=Path("results/live/calibration.json"))
    lc.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    lc.add_argument("--repeats", type=int, default=3)
    lb = commands.add_parser("live-benchmark", help="Replay real steady/bursty requests through continuous batching")
    lb.add_argument("--config", type=Path, default=Path("configs/live_experiment.json"))
    lb.add_argument("--output", type=Path, default=Path("results/live"))
    lb.add_argument("--calibration", type=Path)
    lb.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ls = commands.add_parser("live-serve", help="Serve concurrent requests through one continuously batched worker")
    ls.add_argument("--model-path", type=Path, default=Path("models/gpt2"))
    ls.add_argument("--calibration", type=Path, default=Path("results/live/calibration.json"))
    ls.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ls.add_argument("--policy", choices=["full", "morph_only", "offload_only", "naive", "joint", "joint_uncalibrated", "full_v03", "joint_v03", "joint_no_gate", "joint_no_reuse"], default="joint")
    ls.add_argument("--device-budget-mib", type=float, default=0)
    ls.add_argument("--max-batch-size", type=int, default=8)
    ls.add_argument("--host", default="127.0.0.1")
    ls.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)
    try:
        if args.command == "live-calibrate":
            from .live_experiments import calibrate_live
            result = calibrate_live(args.model_path, args.output, args.device, args.repeats)
        elif args.command == "live-benchmark":
            from .live_experiments import run_live_experiments
            result = run_live_experiments(args.config, args.output, args.device, args.calibration)
        elif args.command == "live-serve":
            import uvicorn
            from .batch_service import create_batch_app
            uvicorn.run(create_batch_app(args.model_path, args.calibration, args.device, args.policy,
                                        args.device_budget_mib, args.max_batch_size), host=args.host, port=args.port)
            return
        elif args.command == "pretrained-prepare":
            from .pretrained_assets import prepare_model, prepare_wikitext
            result = prepare_model(args.output, args.download)
            if args.dataset:
                result["dataset"] = prepare_wikitext()
        elif args.command == "pretrained-generate":
            from .pretrained import GPT2Executor
            import torch
            torch.set_num_threads(min(4, torch.get_num_threads()))
            executor = GPT2Executor.from_local(args.model_path, args.device)
            selected = tuple(int(x.strip()) for x in args.quantized_layers.split(",") if x.strip())
            executor.set_quantized_layers(selected)
            executor.set_cpu_attention(args.cpu_attention)
            result = executor.generate(args.prompt, args.max_tokens)
            result["memory"] = executor.memory_report()
        elif args.command == "pretrained-benchmark":
            from .pretrained_experiments import benchmark_pretrained
            result = benchmark_pretrained(args.model_path, args.data_path, args.output, args.device,
                                          args.quality_tokens, args.repeats, args.max_tokens)
        elif args.command == "pretrained-adapt":
            from .pretrained_experiments import adaptive_pretrained
            result = adaptive_pretrained(args.model_path, args.output, args.device)
        elif args.command == "pretrained-serve":
            import uvicorn
            from .pretrained_service import create_pretrained_app
            uvicorn.run(create_pretrained_app(args.model_path, args.device), host=args.host, port=args.port)
            return
        elif args.command == "doctor":
            from .telemetry import detect_hardware
            result = {"python": sys.version, "platform": platform.platform(),
                      "dependencies": {n: importlib.util.find_spec(n) is not None for n in
                                       ["numpy", "psutil", "torch", "fastapi", "uvicorn", "matplotlib"]},
                      "hardware": asdict(detect_hardware())}
        elif args.command == "profile":
            from .telemetry import profile_hardware
            result = asdict(profile_hardware(args.output))
        elif args.command == "simulate":
            from .experiments import run_experiments
            result = run_experiments(args.config, args.output, args.trace)
        elif args.command == "report":
            from .reporting import build_report
            result = {"report": str(build_report(args.results).resolve())}
        elif args.command == "adapt":
            from .runtime import adaptive_demo
            result = adaptive_demo(args.output, args.checkpoint, args.device)
        elif args.command == "serve":
            import uvicorn
            from .service import create_app
            uvicorn.run(create_app(args.checkpoint, args.device), host=args.host, port=args.port)
            return
        else:
            from .torch_backend import demo, run_quality, train_tiny
            if args.command == "train":
                result = train_tiny(args.output, args.steps, args.seed)
            elif args.command == "quality":
                result = run_quality(args.output, args.checkpoint)
            else:
                result = demo(args.output, args.steps)
        print(json.dumps(result, indent=2, allow_nan=False))
    except (ValueError, FileNotFoundError, ImportError) as exc:
        parser.exit(2, f"prismserve: {exc}\n")
