# Reproducible evaluation plan

The current validation extension is documented in [M10 protocol](validation_study.md). It adds 60 instrumented GPU traffic repeats, matched-batch predictor comparisons, matched CPU/CUDA serving with separate calibration, and eight longer history-aware quality arms. Results belong to `results/validation_study/`; historical experiments below retain their original measurements and assumptions.

Two kinds of evidence answer different questions. **Measured tiny-model experiments** establish that packing, caching, restoration, CPU attention, and quality measurement execute. **Analytical simulation** explores capacity/performance tradeoffs at LLM scale. Every figure and table must identify which kind it shows.

## Hypotheses and ways to falsify them

H1: A common controller can choose different precision/attention placements for profiles that differ in device capacity, host bandwidth, or link bandwidth, without product-name-specific rules. Falsify this claim if placement remains insensitive to the corresponding bottleneck or if portability requires editing policy logic.

H2: Joint control can improve the latency/quality/capacity tradeoff over either control alone in some memory-limited regimes. This is conditional, not a universal-win claim. Falsify the stronger claim if gains vanish after charging host backups, migration, quality penalties, and transition peaks. Include regimes where full precision or one control wins.

H3: Hysteresis reduces unnecessary switching under oscillating load. Count changes and migration cost as well as TTFT/TBT; fewer changes can still harm responsiveness during a burst.

H4: Reversible packed storage and persistent host attention preserve the intended state transitions in a real decoder. Test causal/cached numerical parity and exact restoration of stored weights. Mixed-precision logits need not equal full precision, and restoring weights does not erase approximation already present in a live KV cache.

## Revised E1–E5

| Experiment | Implemented evidence | Interpretation and remaining work |
|---|---|---|
| E1: independent mechanism baselines | Full precision, static INT4, morph-only, offload-only and naive/joint policies under one simulator; real tiny tensor checks | These are simplified baselines inspired by papers, not the original MorphServe, NEO, vLLM, or ±5% reproductions. |
| E2: resource sensitivity | Replay the same trace against profiles with independently varied memory, host bandwidth and link bandwidth | Synthetic rates measure sensitivity of the analytical model. Multi-machine measurements remain future work. |
| E3: burst response | Timestamped burst traffic, queue drain, per-iteration placement/precision logs, TTFT/TBT, throughput and SLO accounting | Compare paired workloads/seeds. Report overload and rejection honestly. |
| E4: approximation quality | Held-out authored text NLL/perplexity, logit KL, reversible-weight checks on the tiny model | Results describe this model/corpus. MMLU, GSM8K and WikiText-2 are not claimed. An additive controller budget is not an accuracy guarantee. |
| E5: ablations and cost | Disable precision, disable offloading, disable hysteresis; wall-clock policy search timing; storage accounting | Search timing measured on this host is separate from analytical service time. A mixed measured/simulated overhead ratio is only an estimate. |

## Run sequence

From the repository root, in an environment with the optional backend dependencies installed:

```powershell
python -m prismserve doctor
python -m prismserve profile --output results/local_profile.json
python -m unittest discover -s tests -v
python -m prismserve simulate --config configs/experiment.json --output results/simulation
python -m prismserve train --output results/tiny --steps 100
python -m prismserve quality --checkpoint results/tiny/tiny_checkpoint.pt --output results/quality
python -m prismserve demo --output results/demo --steps 60
python -m prismserve report --results results/simulation
python scripts/load_sweep.py --output results/load_sweep
```

The README and CLI `--help` are authoritative for supported flags. Training steps are deliberately small for a local course demo. The initial checkpoint and corpus, seed, torch/device versions, and exact outputs should accompany a result used in a submission.

The submitted full experiment has 63 runs (three profiles, seven policies, three seeds). The additional offered-load sweep has 25 runs: five policies, five arrival-rate factors and one seed. It preserves token lengths and scales all timestamps together. Its confidence scope is exploratory; it includes a lower-load regime where TTFT meets the target while TBT still fails for many requests.

## Comparison controls

Use identical request timestamps, prompt lengths, maximum output lengths, model geometry, memory safety fraction, workspace convention, and SLOs for each policy in a paired comparison. Policies that deliberately exceed a quality budget, such as an aggressive static INT4 comparator, must be marked accordingly rather than ranked as quality-equivalent.

Keep policy names precise. `morph_only` is our reduced precision-control baseline; `offload_only` is our reduced placement-control baseline. Their performance is not evidence about the full systems in the papers. Avoid attributing an analytical gain against `full` to a gain against vLLM.

Run multiple workload seeds and report their individual results plus a distribution. Percentiles over one short trace do not supply a confidence interval. Separate request-level variability from variability across independent runs. Record configuration and provenance with each result. Preserve unfavourable settings and explain bottlenecks.

## Recommended regime matrix

| Regime | What changes | Expected diagnostic, not a promised result |
|---|---|---|
| Comfortable device memory | Raise device capacity | Precision/placement changes should become less necessary. |
| Memory-limited device | Lower device capacity with adequate host | Quantization or host KV can enlarge admitted concurrency. |
| Slow host | Lower host bandwidth only | Large CPU attention fractions should become expensive. |
| Slow link | Lower effective transfer bandwidth only | Migration and repeated placement changes should become expensive. |
| Host capacity limit | Reduce host budget | Host backups and host KV may make offloading infeasible. |
| Already low-bit weights | Reduce starting weight width | Additional weight morphing offers less headroom. |
| Oscillating arrivals | Alternate quiet and busy intervals | Compare change counts, tail latency, and hysteresis delay. |
| Low load | Spread arrivals | Quantization/placement overhead may outweigh capacity benefits. |

## Metrics to publish

Publish request count, completions, rejections, unfinished requests, output-token throughput, P50/P95/P99 TTFT, pooled TBT percentiles, and the precise request SLO rule. For single-token outputs, TBT/TPOT are missing, not zero. Publish device/host peaks, quantized-layer exposure, CPU-placement exposure, number of transitions, transition-time estimates, and policy search-time measurements. Report quality separately as NLL, perplexity and KL with the evaluated checkpoint/corpus named.

For tail latencies, include both absolute values and relative changes. If a baseline rejects every request, a speedup ratio is undefined. If goodput is zero, retain zero; do not omit that point. If policies complete different amounts of work, throughput and completion/rejection counts must be considered together.

## Threats to validity

1. A roofline time model and idealized overlap can underestimate real CPU, PCIe and launch overhead. Neither measured copy bandwidth nor peak TFLOPS fully calibrates inference.
2. Conservatively reserving maximum output length changes admission compared with paged engines that allocate on demand. Trace output lengths act as configured maxima; they are not predictions of a live model's future EOS.
3. A fixed sensitivity order and additive risk budget miss interactions among quantized layers, prompts, and historical KV values.
4. Tiny-model quality and reference dequantization costs do not transfer quantitatively to Llama-scale pretrained models with AWQ kernels.
5. Synthetic hardware profiles test a common interface and model response, not deployment across physical hardware vendors.
6. The local laptop's GPU power/thermal state and competing host processes affect measured timings. Record versions, warm up, synchronize CUDA timings, and avoid overinterpreting a single short run.
7. The standalone HTTP server serializes requests. Its latency is not a measurement of the continuous-batching simulator.

## Claims supported at submission

Use the generated results to fill in actual outcomes, including negative results. The safe conclusion is whether the implementation executes correctly and whether the specified model predicts useful regimes for joint control. The broader claim of hardware-agnostic, SLO-preserving LLM serving requires pretrained-model, calibrated-kernel and multi-machine validation beyond this course prototype.
