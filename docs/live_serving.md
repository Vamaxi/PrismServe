# Real continuous-batching serving

Version 0.4 uses an efficient controller by default. It compiles reservations, reuses stable plans with fresh resource checks, separates phase feedback and gates adaptive choices against full precision. See [controller efficiency](controller_efficiency.md) for the algorithm, current commands and fresh old/new comparison. The recorded M8 results below belong to the preserved v0.3 experiment.

The live serving path accepts concurrent HTTP requests and executes them through one background model worker. Each request owns its KV history and output timestamps. Dense projection/MLP operations share a flattened token batch; attention operates on the corresponding ragged per-request histories. Requests can enter between decode iterations, and prefill is chunked within a token budget. This is actual multi-request GPT-2 execution, distinct from both the earlier serialized completion endpoint and the analytical trace simulator.

`prismserve.batch_service.create_batch_app` exposes `/health`, `/v1/models`, `/v1/completions` and `/metrics`. The model ID is `prismserve-gpt2-batched`. HTTP completion handlers await concurrent Futures using `asyncio.wrap_future`, allowing other handlers to submit while a shared execution tick is running. A single worker owns global layer precision and request placement; clients cannot override these states independently.

Run from the project root:

```powershell
python -m prismserve live-serve --calibration results/live/calibration.json
# In another terminal, submit concurrent requests to the running server:
python scripts/demo_concurrent.py
# Or launch an automatically cleaned-up test server with four actual clients:
python scripts/smoke_batch_api.py
```

Use `python -m prismserve live-calibrate --output results/live/calibration.json` when changing execution hardware. `python -m prismserve live-benchmark --calibration results/live/calibration.json` reproduces the configured real traffic comparison. The direct serialized `pretrained-serve` command remains available for the earlier reference endpoint.

Example completion body:

```json
{"model":"prismserve-gpt2-batched","prompt":"The purpose of an operating system is","max_tokens":32}
```

The API is a bounded base-model completion service, without chat messages or streaming. Tokenized prompt plus output reservation must fit the model context. Invalid requests return 400/422, a full queue or unavailable memory budget returns 429, and a closed or failed worker returns 503. The application lifecycle starts the worker and closes it on shutdown, resolving pending work rather than leaving HTTP handlers waiting indefinitely.

## Measured control and resource limits

The controller consumes measured component timings and online feedback rather than substituting the analytical simulator's token durations. Local timing calibration supplies initial estimates; online exponentially weighted updates track measured batch behavior. The timing file and hardware profile identify the calibration environment. Calibration does not make the model exact: batch shape, context lengths, quantization, transfers and competing work can alter prediction error.

Calibration records actual synchronized batch duration and incremental peak workspace for a grid of batch sizes, prompt/decode mixtures, context lengths, CPU placements and quantized-layer sets. Latency uses the nearest compatible measured shape and conservative upward scaling. Memory uses a monotone calibrated scratch bound plus temporary copies of fully reserved KV, protecting admission against later cache growth. Missing timing families are rejected rather than assigned a peak-hardware duration. An exponentially weighted ratio corrects the selected family's execution estimate using actual execution-only duration; transition costs are measured separately. This is deterministic feedback, not reinforcement learning. The explicit `joint_uncalibrated` ablation uses analytical predictions for decisions while still executing and timing every real request.

Memory quotas constrain admitted resident weights, cache reservations and transitions. A declared device-memory quota is an allocation budget enforced by the serving engine, not a claim that physical VRAM was exhausted. Actual tensor memory and resource samples remain separate observations. Live KV is neither evicted nor approximated merely to make a request fit. Precision may affect later cached values; restoring weights does not recompute earlier cache history.

Dense layers are batched, while attention remains a synchronous ragged loop in this reference backend. Multiple active requests and shared matrix execution are real; optimized paged attention, NEO's asynchronous pipeline, multi-GPU execution and universal speedup are not established by this implementation.

## Live experiment method

`configs/live_experiment.json` defines paired real traffic runs. The baseline `serial_full` uses the same batch primitive with batch size one. Other arms are full precision, morph-only, offload-only, naive stacking, joint measured control and uncalibrated joint control. Two arrival seeds and three steady/bursty quota scenarios give 42 greedy-serving runs. The configured workload uses 12 requests per run, 384-token prompts, 24 output tokens, a two-second arrival interval, maximum batch size eight, 64-token prefill chunks and a 128-token total prefill budget. Scenarios impose either 1400 MiB or 700 MiB device budgets. These counts describe the configured protocol; final artifacts must establish which requests complete or fail.

`linear_tokens` and `projection_calls` in iteration events identify shared execution. For 12 GPT-2 blocks, a tick invokes the four dense projections 48 times altogether, rather than 48 times for each request. Per-request causal attention slices and independently owned KV buffers preserve request isolation. Cached, chunked and CPU-placement parity tests compare this primitive with the original single-request executor. A real-model smoke comparison has maximum logit error approximately 3.05×10⁻⁴ within its declared 0.002 tolerance.

The quality audit is separate teacher-forced real execution on held-out WikiText-2 continuations. Every policy receives the same reference targets, and token NLL is evaluated under the precision and KV history actually used in that audit. Its policy trajectory is distinct from a greedy serving run, because teacher-forced tokens change history and timing. Report audit token counts and precision exposure together; an audit that never quantizes cannot establish quality during a different run's quantized outputs.

## Reading live results

Request results retain TTFT, every inter-token gap, output timestamps, admission time, queue wait, precision history and cache-placement history. Aggregate metrics include peak active concurrency, maximum execution batch size, completed and pending counts, and worker status. A maximum active count alone does not establish shared execution; inspect execution batch size and projection-call evidence as well.

TTFT here measures engine token readiness. The non-streaming HTTP endpoint returns the completed response, so HTTP elapsed time is separate from internal first-token time. The steady trace is repeated unchanged under two policy orders; burst timestamps differ by seed. There is substantial run-to-run timing variation, so the small experiment does not support universal speedups or confidence in every mean.

Compare traffic replays using the same prompts, timestamp sequence, output limits and resource quotas. Include failures, rejections, drain time and goodput in the comparison. Policy overhead is measured wall time and belongs in observed request latency. Dynamic precision also needs a separate quality audit using the exact model, token trace and data protocol; an additive sensitivity budget is not a task-accuracy guarantee. Numerical outcome claims should come from the final measured live experiment artifacts and their manifests.

## Recorded result

The final suite completes 42 serving runs (504 requests, 12,096 greedy output tokens) and seven separate 64-target quality audits. All 97 tests pass, and four real concurrent HTTP clients share a batch of four. Full-precision batching averages about 90 output tokens/s against 46 for the serialized control in the roomy burst scenario. The measured joint controller averages about 75 there; it does not beat batched full precision. Under 700 MiB, joint control reaches four active requests versus two, but its throughput is lower. Full tables and individual runs are in [live results](../results/live/report.md).

The joint quality audit quantizes 21.88% of its output steps and raises audit perplexity by about 0.779% over the FP32 audit. This is a small 64-target diagnostic with its own trajectory, not proof of the proposal's task-accuracy bound. `python scripts/check_live_results.py` verifies pairing, token accounting, source/data hashes, measured quotas and the saved API check without rerunning model inference.

An initial diagnostic attempt failed because temporary workspace grew after admission. It remains in `results/live_initial/` and is excluded from final comparisons. The corrected implementation reserves future concatenation space and has explicit regression coverage.
