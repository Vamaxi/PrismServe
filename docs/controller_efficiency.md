# Controller efficiency: v0.4

The previous real-serving experiment showed that joint control spent roughly 7–16% of total wall time sampling resources and selecting batches. Increasing concurrency did not compensate for that cost. This milestone reduces the work needed to make a decision and makes precision/placement changes more conservative.

## What changes

`EfficientController` compiles the existing safe memory formulas and calibration families once. A bounded cache stores reservations and nearest-shape timing lookups. Current device/host capacity is compared against the reservation on **every iteration**, including reused decisions. Cached feasibility is never trusted. The engine retains immutable model-storage accounting until precision actually changes and avoids scanning all model tensors or converting caches to their existing device every tick.

The engine refreshes its candidate search at least every four iterations. New request IDs, a changed active set, a prefill/decode phase change, or a failed resource check force a refresh. A stable decision can be reused for the intervening three iterations. Pending requests that were previously considered can consequently wait up to three additional execution iterations before admission is reconsidered. This is a configurable latency/overhead tradeoff, not a deadline guarantee.

Full-precision admission checks remain cheap and cover every possible FIFO prefix. Adaptive search examines a bounded set of prefix counts: maximum incoming, half, maximum feasible full-precision admission, its next larger prefix, zero, and one when empty. Existing requests continue; the engine does not discard their caches. The one-request and zero-admission cases preserve progress checks despite pruning.

Online feedback is separated into prefill, mixed and decode phases for each precision/offload family. A slow prefill no longer directly changes the decode correction. Feedback observes only the executed action; it does not estimate unobserved actions from that observation.

Before choosing quantization or CPU attention over a feasible full-precision reference, a conservative gate compares predicted cost per serviced request, transition amortization and heuristic uncertainty margins. The alternative must clear a 10% minimum gain after those margins. The reference can have a smaller batch. If no full-precision reference fits, the gate allows an adaptive feasible plan. These uncertainty margins are engineering heuristics, **not statistical confidence intervals** or a quality guarantee.

All model computation, packed INT4 storage, per-request KV ownership, numerical quality surrogate, temporary-copy reservations and actual batch primitive are retained. Physical limits are sampled every tick; process RSS collection is omitted on the optimized scheduling path. Actual CUDA allocation and free memory remain sampled.

## Comparison arms

| Policy | Scheduling and feedback |
|---|---|
| `full_v03` | Previous exhaustive admission and original measured controller; full precision |
| `joint_v03` | Previous exhaustive admission and original measured joint controller |
| `full` | Efficient admission/resource path; full precision |
| `joint` | Efficient joint search, phase feedback, reuse and conservative gate |
| `joint_no_gate` | Efficient joint controller with the conservative gate disabled |
| `joint_no_reuse` | Efficient joint controller with decision reuse disabled |

The ablations isolate the gate and reuse within the new implementation. They do not individually isolate compilation, admission pruning or phase feedback. Older v0.3 results retain their exact archived implementation under `results/live/`; fresh comparators in `results/controller/` reduce dependence on earlier thermal conditions. GPU policies execute sequentially in shuffled order. Two seeds on one laptop are exploratory evidence.

The configuration retains the earlier workload: 12 requests, 384 prompt tokens, 24 generated tokens, a 2-second arrival window, maximum batch eight, 64-token prefill chunks and a total prefill budget of 128. Three scenarios use steady/bursty arrivals and enforced device quotas of 1400/700 MiB. A separate audit scores 64 held-out continuation tokens per policy. Validation prefixes choose quantization candidates; test prefixes supply replay inputs and audit targets.

## Reproduce

Run from the project root with the local model assets prepared:

```powershell
python scripts/validate.py
python -m prismserve live-benchmark --config configs/controller_experiment.json --output results/controller --calibration results/live/calibration.json --device cuda
python scripts/smoke_batch_api.py --output results/controller
python scripts/check_controller_results.py
python scripts/update_controller_report.py
```

Regenerate calibration with `live-calibrate` when hardware, device or backend changes. The saved calibration belongs to this laptop's CUDA executor. Do not run GPU tests or other model workloads concurrently with timing experiments. The benchmark writes a source snapshot and hashes; avoid editing runtime modules while it runs. Original v0.3 replay configuration explicitly selects `scheduler_strategy: legacy`.

The server uses the efficient controller by default:

```powershell
python -m prismserve live-serve --calibration results/controller/calibration.json --policy joint
# In a second terminal:
python scripts/demo_concurrent.py
```

The response is non-streaming. Recorded TTFT measures engine token readiness from scheduled arrival; clients receive the full completion later. `search_admission_wall_fraction` includes telemetry, planning and admission relative to complete workload wall time. New fields split telemetry/planning and record plan reuse. Prediction errors compare actual synchronized execution with the estimate, separate from transitions and planning. An overhead target below 5% is evaluated from measurements, not assumed.

## Validation and evidence

Resource parity tests compare 100 randomized plans with the previous conservative formulas. Cache tests shrink live capacity after a successful cached reservation and verify refusal. Engine tests verify a fresh resource sample on every tick, decision reuse, and failure before execution when capacity shrinks. Phase tests check isolation between prefill and decode; gate tests cover uncertain gains and capacity necessity. Existing numerical, batching, HTTP, cancellation and cache-ownership tests continue to apply.

All 106 tests pass, including CPU-primary combined host-capacity parity. The completed study has 36 serving runs, 432 completed requests, no rejections, six independent quality audits and four-client real HTTP verification. All six optimized joint runs meet the exploratory control-overhead target (worst 2.82%). Throughput and prediction accuracy do not improve consistently. The joint quality audit quantizes five early prefill iterations despite restoring FP32 before output emission; historical KV can therefore differ even with 0% INT4 output exposure.

Raw events, separate quality trajectories, all policy outcomes and source provenance are retained in `results/controller/`. `comparison.md` summarizes measured overhead, throughput, errors and quality. This milestone does not supply fused INT4 kernels, asynchronous CPU attention, larger-model results or physical multi-machine portability evidence.
