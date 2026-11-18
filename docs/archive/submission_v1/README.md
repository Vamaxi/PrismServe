# PrismServe

**A hardware-profile-driven research prototype for jointly controlling weight precision and CPU attention placement.**

DA4022, Hardware Aware Deep Learning, IIT Guwahati. Inspired by **MorphServe (MLSys 2026)** and **NEO (MLSys 2025)**. Built independently from the four supplied PDFs.

The project asks when one resource-aware controller can coordinate reversible layer quantization and persistent host KV attention across heterogeneous hardware. It includes real tensor execution, actual concurrent serving, analytical experiments, baseline comparisons, tests, and submission materials.

**Submission version 1.0 adds the final capacity-frontier study and demonstration.** The 216-run grid compares FP32, quantization only, CPU attention only and joint control. It records rejections, latency, throughput, memory and actual precision-history quality. See [capacity results](results/capacity_frontier/report.md), [protocol](docs/capacity_frontier.md), and [demo/viva guide](docs/demo_and_viva.md).

**Version 0.5 adds reliability and backend-portability validation.** A separate study repeats GPU workloads with fresh calibration and hardware telemetry, compares four predictors on identical actual executions, runs the same serving controller with CPU/CUDA-specific calibration, and expands history-aware quality audits. See the [M10 protocol](docs/validation_study.md) and [M10 measured report](results/validation_study_final/report.md). An initial battery/AC diagnostic session is retained separately.

**Version 0.4 reduces controller overhead.** Compiled reservations, cached timing lookup and stable decision reuse retain fresh resource checks every iteration. Feedback separates prefill/decode phases, and a conservative gate compares adaptive plans against feasible full precision. [Controller design and reproduction](docs/controller_efficiency.md) and [new measured comparison](results/controller/comparison.md) include fresh v0.3 controls and gate/reuse ablations.

**Version 0.3 adds real continuous batching and measured control.** Multiple requests keep independent KV histories while sharing each layer's matrix multiplications. The controller uses actual batch/transition timings plus online feedback. [Live serving](docs/live_serving.md) documents the new path; [live results](results/live/report.md) contain actual wall-clock traffic experiments.

**Version 0.2 adds actual pretrained GPT-2 execution and serving.** The original checkpoint and BPE tokenizer run through the project's precision/CPU-attention executor. Numerical checks compare it against Hugging Face's original forward pass. [Pretrained results](results/pretrained/report.md) include real CPU/GPU latency and WikiText-2 subset quality.

## Start here

| Artifact | Purpose |
|---|---|
| [Final report](docs/final_report.md) | Research question, methods, actual results, limitations |
| [Presentation](docs/presentation.md) | Slides and speaker notes; generated PPTX/PDF in `deliverables/` |
| [Research review](docs/research_review.md) | Detailed paper comparison and proposal corrections |
| [Design](docs/design.md) | Control objective, equations, memory and timing model |
| [Experiments](docs/experiments.md) | E1–E5 rescope, controls, metrics, reproduction |
| [Real backend](docs/backend.md) | Decoder, quantization, checkpoint and API details |
| [Pretrained GPT-2](docs/pretrained.md) | Local checkpoint, real execution, serving and evaluation |
| [Pretrained results](results/pretrained/report.md) | Measured GPT-2 storage/latency/quality comparisons |
| [Continuous serving](docs/live_serving.md) | Shared model worker, concurrent HTTP, measured controller |
| [Real traffic results](results/live/report.md) | Actual steady/bursty replays, baselines, quality audit and overhead |
| [Controller efficiency](results/controller/comparison.md) | Fresh old/new comparison, overhead, throughput, ablations and quality |
| [Reliability and portability](results/validation_study_final/report.md) | AC-conditioned five-round study, hardware telemetry, matched prediction errors, CPU/CUDA serving and expanded quality |
| [Capacity frontier](results/capacity_frontier/report.md) | Sampled quota/context limits, independent mechanism controls, rejections and quality at active quantization points |
| [Demo and viva](docs/demo_and_viva.md) | One-command live demonstration, architecture walkthrough and viva answers |
| [Simulation results](results/simulation/report.md) | Generated tables, plots, raw traces and per-request metrics |
| [Load curves](results/load_sweep/report.md) | Offered-load sensitivity and throughput-versus-latency plots |
| [Validation](results/validation.json) | Recorded test outcomes |

**Evidence boundaries:** actual CPU/CUDA experiments include a 124,439,808-parameter pretrained GPT-2 model and the original tiny decoder. GPT-2 is a base text-completion model, with a 1024-token context, rather than an instruction-tuned chat model. Llama-8B-shaped results remain explicitly analytical simulations. Reference INT4 uses packed storage with floating-point dequantize-and-matmul, and CPU/GPU execution is synchronous. There is no claim of original-paper reproduction or universal speedup.

## Setup

Python 3.10+. The simulator needs NumPy and psutil; real execution needs PyTorch, and pretrained execution adds Transformers 4.57.x plus safetensors. Run from this folder. GPT-2 weights and evaluation prefixes are already prepared on this machine. A new machine needs the public model assets; no API key, gated access or external serving codebase is required.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[all]"
python -m prismserve doctor
```

For CUDA, install a PyTorch build appropriate to your driver using the [official installer](https://pytorch.org/get-started/locally/) before installing the package. CPU fallback is supported. The environment used for the saved run already had PyTorch 2.5.1+cu121 and an RTX 4050 Laptop GPU; see `results/environment.json` for versions. Core simulation alone: `python -m pip install -e .`.

## Run the project

For the final guided demonstration, from this folder:

```powershell
python scripts/demo_project.py
# Saved validated evidence only, without loading the model:
python scripts/demo_project.py --evidence-only
```

The default demo performs actual capacity replays and a temporary concurrent HTTP check. It labels saved evidence separately from fresh execution and stops its temporary server automatically. The prepared local CUDA model and calibration are required for live execution.

For concurrent pretrained serving on this machine:

```powershell
# Start the real continuously batched server using this machine's saved timings
python -m prismserve live-serve --calibration results/capacity_frontier/calibration_cuda.json

# On different hardware, calibrate into a new folder and serve with that file
# python -m prismserve live-calibrate --output results/local_live/calibration.json

# In another terminal, call the running server with four concurrent clients
python scripts/demo_concurrent.py

# In another terminal: four concurrent real HTTP clients, temporary test server
python scripts/smoke_batch_api.py --calibration results/capacity_frontier/calibration_cuda.json --output results/local_api_smoke

# Reproduce the final frontier study, sequentially
python scripts/run_capacity_frontier.py
python scripts/check_capacity_frontier.py --check-current
python scripts/update_frontier_report.py
```

The complete study includes fresh calibration and representative quantized-trajectory audits. The AC-conditioned configuration requires the laptop to remain plugged in; condition failures retain diagnostics and refuse the measurement. Completed runs resume automatically with the same source/config; a different source/config requires a new output folder. Run `python scripts/validate.py` before the study and keep other GPU model jobs stopped. [M11 protocol](docs/capacity_frontier.md) specifies its warm-start semantics and pilot refinement. M10 evidence remains in its archived folder; reproduce it with `run_validation_study.py --output results/m10_reproduction` and check its archived results without `--check-current`.

Historical v0.3 evidence is preserved in `results/live/`; its configuration selects legacy scheduling. Reproduce it into a separate output folder using `configs/live_experiment.json`. Run GPU tests and timing experiments sequentially. The efficient controller is the default for `live-serve`; `--policy joint_v03` selects the earlier implementation for comparison.

The live API uses `prismserve-gpt2-batched` as its model ID. Several clients can call `/v1/completions` concurrently; one worker forms shared model batches and controls global precision. Default memory availability comes from live telemetry. `--device-budget-mib 700` applies an explicit admission/reservation quota; it does not fill physical VRAM. This reference implementation batches dense projections and loops over ragged attention. It does not implement fused paged attention or asynchronous NEO kernels.

The earlier serialized pretrained path remains available:

```powershell
# Already prepared at models/gpt2; no training required
python -m prismserve pretrained-generate --prompt "Artificial intelligence can help scientists understand"

# Serve the actual pretrained model, then open http://127.0.0.1:8000/docs
python -m prismserve pretrained-serve

# Reproduce measured comparisons using prepared WikiText-2 prefixes
python -m prismserve pretrained-benchmark --output results/pretrained
python -m prismserve pretrained-adapt --output results/pretrained/adaptive.json

# Test an actual HTTP request, automatically stopping its temporary server
python scripts/smoke_api.py --pretrained
```

On another machine, run `python -m prismserve pretrained-prepare --download --dataset` to prepare the pinned public checkpoint and small validation/test prefixes. Model weights are omitted from the source ZIP to keep it small; their source revision and hashes are retained. The normal generation/serving paths use only local files and never download implicitly.

The original small-model and analytical paths remain available:

```powershell
# Actual local hardware inventory and bounded microbenchmarks
python -m prismserve doctor
python -m prismserve profile --output results/local_profile.json

# Fast analytical sanity run, then the full 63-run comparison
python -m prismserve simulate --config configs/smoke.json --output results/smoke
python -m prismserve simulate --config configs/experiment.json --output results/simulation
python scripts/load_sweep.py --output results/load_sweep

# Real tiny-model training, quality and CPU/CUDA execution (offline)
python -m prismserve demo --output results/demo --steps 80
python -m prismserve quality --checkpoint results/demo/tiny_checkpoint.pt --output results/quality

# Same controller applied to real generation under diagnostic capacity quotas
python -m prismserve adapt --checkpoint results/demo/tiny_checkpoint.pt --output results/adaptive
python -m prismserve adapt --checkpoint results/demo/tiny_checkpoint.pt --output results/adaptive_cpu --device cpu

# Numerical, resource accounting, policy, simulator and API verification
python scripts/validate.py
python scripts/smoke_api.py
```

The full sweep can take several minutes. Token-level virtual results are deterministic for a fixed config/trace; measured timings vary with system load. `adapt` uses declared quotas to exercise cache migration without filling physical GPU memory. On CPU, it correctly disables CPU self-offloading. With conservative reference-kernel scratch accounting, the tiny adaptive run may retain full precision; static mechanism demonstrations still verify INT4 transitions.

Individual training: `python -m prismserve train --output results/tiny --steps 100 --seed 7`. If no checkpoint is provided, quality/API commands explicitly use a random tiny model. UTF-8 bytes are tokens; maximum context is 256 bytes/tokens, including requested output.

## Local APIs

The pretrained API uses model ID `prismserve-gpt2`:

```powershell
python -m prismserve pretrained-serve
```

```powershell
$requestBody = @{
  model = "prismserve-gpt2"
  prompt = "Artificial intelligence can help scientists understand"
  max_tokens = 32
  quantized_layers = @(1, 7, 9)
  cpu_attention = $false
} | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/v1/completions -ContentType application/json -Body $requestBody
```

These three layer IDs come from validation-only calibration in the saved run. Request setup time, including precision changes, is included in HTTP TTFT. It is not included in the benchmark's warm steady-state TTFT, where it is reported separately. GPT-2 uses its original subword tokenizer, with a maximum prompt-plus-output budget of 1024 tokens. Arbitrary Llama/Qwen checkpoints and chat messages are outside this GPT-2 adapter's architecture scope.

The original tiny API is also supported:

```powershell
python -m prismserve serve --checkpoint results/demo/tiny_checkpoint.pt
```

Open `http://127.0.0.1:8000/docs` for the interactive API. For PowerShell:

```powershell
$requestBody = @{
  model = "prismserve-tiny"
  prompt = "The serving system"
  max_tokens = 16
  quantized_layers = @(0)
  cpu_attention = $true
} | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/v1/completions -ContentType application/json -Body $requestBody
```

The earlier tiny/pretrained services expose `/health`, `/v1/models`, `/v1/completions`, `/metrics`, with serialized requests and explicit precision/placement controls. The new `live-serve` API accepts concurrent requests and uses the measured controller automatically. Streaming, chat, fused paged attention and overlapped CPU/GPU kernels remain outside this reference implementation.

## Policies and experiments

| Policy | Behavior |
|---|---|
| `full` | Full precision, primary-device attention |
| `static_int4` | All morphable layers packed; reports risk without an adaptive quality constraint |
| `morph_only` | Dynamic precision; primary-device attention |
| `offload_only` | Full precision; adaptive host attention placement |
| `naive` | Maximize CPU placement, then choose minimum feasible quantization |
| `joint` | Joint candidate search with capacity, quality proxy and transition costs |
| `joint_no_morph`, `joint_no_offload`, `joint_no_hysteresis` | Mechanism/cooldown ablations |

These are independent mechanism-inspired analytical baselines. They are not executions of vLLM, MorphServe or NEO. NEO already uses adaptive load-aware scheduling. MorphServe expands KV capacity by reclaiming weight memory and does not compress existing KV tensors. The project preserves live KV; it never silently evicts attention history.

Configurations specify memory in bytes, bandwidth in decimal GB/s, compute in TFLOP/s, arrivals in seconds and latency objectives in milliseconds. `configs/experiment.json` fixes nonmorphable Llama-shaped parameters and varies effective device, host and link resources. Every candidate accounts for host recovery copies, padded packed weights/scales, reserved KV, workspace and transition peaks. The simulator models ideal partial overlap; measured reference execution does not realize that overlap. A configured sensitivity budget is a proxy, not a downstream accuracy guarantee.

CSV replay accepts explicit columns:

```csv
request_id,arrival_s,prompt_tokens,output_tokens
r0,0.0,256,32
r1,0.15,1024,64
```

Use `--trace path/to/trace.csv` with a config containing one seed. Convert public Azure/BurstGPT trace columns to this schema deliberately; the runner does not guess timestamp units or claim those datasets were downloaded. Per-run JSON contains token timestamps, transitions, completions and rejections. Rejected requests count as SLO failures. TTFT is arrival to first output; TBT is an actual inter-output gap. Reported throughput includes queue drain. Bootstrap intervals are exploratory across three seeds.

## Repository structure

```text
prismserve/
  types.py          validated profile/model/request/plan contracts
  telemetry.py      bounded measured hardware microbenchmarks
  estimator.py      capacity, transfer, transition and roofline estimates
  policy.py         joint search, baseline arms and cooldown
  simulator.py      arrivals, chunked prefill, continuous batching and metrics
  workload.py       seeded traces and strict CSV replay
  experiments.py    reproducible sweeps, manifests and bootstrap intervals
  reporting.py      generated result tables and plots
  quantization.py   real grouped packed INT4 and FP recovery
  torch_backend.py  actual tiny decoder, training and quality measurements
  runtime.py        shared controller -> actual token-boundary operations
  service.py        bounded local completion API
  pretrained.py     real GPT-2 forward/cache/packed precision executor
  pretrained_assets.py explicit pinned model and public evaluation data preparation
  pretrained_experiments.py actual parity, calibration, quality, timing and adaptive runs
  pretrained_service.py local GPT-2 completion API
  batching.py       shared dense projections and independent ragged attention/KV
  measured_policy.py calibrated timing, monotone workspace bounds and EWMA
  live_engine.py    continuous admission, prefill/decode scheduling and Futures
  live_experiments.py actual traffic replays, calibration and policy quality audits
  efficient_policy.py compiled reservations, phase feedback and conservative gate
  validation_study.py instrumented repeats, matched probes and expanded quality
  study_monitor.py   read-only GPU driver and host-resource sampling
  frontier.py        quota/context sweep, outcome accounting and quality selection
  batch_service.py  async concurrent HTTP handlers sharing the model worker
  cli.py            command line entry points
configs/            smoke and research experiments
tests/              unittest suite, including CUDA when available
scripts/            validation, real HTTP smoke test and document export
docs/               research review, design, report and presentation sources
results/            measured outputs and labeled analytical experiment artifacts
artifacts/source_text/  page-delimited source PDF extracts and hashes
deliverables/       exported report PDF and presentation PPTX/PDF
```

## Submission and extension

To regenerate current submission documents: run `python scripts/check_capacity_frontier.py`, `python scripts/update_frontier_report.py`, then `python scripts/build_submission.py` (export dependencies: `python -m pip install -e ".[submission]"`). Historical report sections have their own update scripts. Follow [freeze steps](docs/demo_and_viva.md) to record `deliverables/release.json`, package and verify the source ZIP. GPT-2 weights remain local and are excluded from the ZIP. The original PDFs are untouched; `python scripts/extract_sources.py` regenerates page extracts and their provenance hashes.

The defensible conclusion is that a common capability interface and joint controller can be implemented and evaluated, including real pretrained GPT-2 execution. Demonstrating reliable large-model gains still requires optimized low-bit kernels, asynchronous batching/host attention, component calibration, physical multi-machine experiments and broader task-quality evaluation. The original ±5%, 20%, 40% and 1.5% targets are not represented as achieved.
