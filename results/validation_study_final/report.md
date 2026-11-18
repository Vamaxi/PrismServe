# M10: instrumented reliability and backend portability study

60 instrumented GPU runs complete 720 requests, plus 12 matched CPU/CUDA runs (48 requests). There are zero rejections. 54 prediction shapes supply 270 matched executions. Eight quality arms score 2048 held-out targets each. All 112 tests pass.

The first complete study changed power source during measurement and is retained separately in `results/validation_study/`. It is diagnostic, not the primary controlled comparison. The new study requires AC power before execution and at every recorded 500 ms host sample, including calibration; a failed condition refuses the affected measurement. Observed clock settings still vary.

| Original steady joint subgroup | Runs | Mean token/s | Mean observed SM MHz |
|---|---:|---:|---:|
| AC | 2 | 121.17 | 2006 |
| battery | 3 | 72.48 | 1244 |

This subgroup split is observational and also differs in round/order. It exposes a power/clock confound; it does not isolate a causal effect. Primary throughput comparisons below use the separate AC-conditioned session.

An intermediate AC attempt stopped when a constrained joint worker reported no feasible plan for admitted work. It remains incomplete in `results/validation_study_ac/` and is excluded from successful-run aggregates. Available host RAM was about 0.8–1.0 GiB during that attempt; the same trace later completed with more than 2 GiB available. Its exact original cause remains unproven. The final source adds failure-only state/reservation snapshots and saves failed replay records; it does not bypass the safety check. Completion and zero-rejection statements below refer to the final accepted series, not every diagnostic attempt.

## Repeated GPU comparison

| Scenario | Old joint token/s | New joint token/s | Paired change % | Exploratory 95% interval | Old/new control % |
|---|---:|---:|---:|---|---|
| steady_roomy | 121.19 | 123.73 | +2.1 | [+1.3, +2.7] | 12.78 / 4.62 |
| bursty_roomy | 134.51 | 138.74 | +3.2 | [+1.8, +5.2] | 7.49 / 2.77 |
| bursty_constrained | 93.11 | 113.37 | +21.9 | [+18.8, +24.8] | 17.92 / 4.48 |

Optimized joint control stays below 5% of workload wall time in 15/15 runs; its worst observed fraction is 4.97%. This is a new session, not a re-labeling of M9's six-run overhead result. The fraction depends on execution speed as well as absolute controller cost. All request latency includes telemetry, planning, transitions and drain.

Optimized full precision has higher mean throughput than optimized joint control in 3/3 scenarios. Improvements over the older joint controller do not establish an advantage over optimized full precision. Decision reuse also trades admission responsiveness for lower control cost: compare TTFT as well as throughput. Raw action histories distinguish control-overhead savings from gains due to actual quantization/offload. Optimized joint control selects FP32 and device attention for every emitted token in this final session; its measured gains therefore concern control cost/admission, not active quantization or CPU offload.

| Scenario | Policy | Mean token/s | Median token/s | SD | CV % | Control % | P95 TTFT ms | P95 TBT ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| steady_roomy | full_v03 | 122.80 | 122.99 | 0.69 | 0.6 | 9.02 | 99.6 | 15.8 |
| steady_roomy | joint_v03 | 121.19 | 121.25 | 0.51 | 0.4 | 12.78 | 111.9 | 17.7 |
| steady_roomy | full | 124.33 | 124.13 | 0.36 | 0.3 | 3.95 | 164.7 | 15.7 |
| steady_roomy | joint | 123.73 | 124.02 | 1.41 | 1.1 | 4.62 | 149.3 | 15.7 |
| bursty_roomy | full_v03 | 138.47 | 137.06 | 3.04 | 2.2 | 5.10 | 777.5 | 32.9 |
| bursty_roomy | joint_v03 | 134.51 | 134.96 | 4.54 | 3.4 | 7.49 | 829.0 | 33.6 |
| bursty_roomy | full | 141.66 | 140.87 | 4.57 | 3.2 | 2.17 | 750.3 | 31.5 |
| bursty_roomy | joint | 138.74 | 137.95 | 2.90 | 2.1 | 2.77 | 786.7 | 31.9 |
| bursty_constrained | full_v03 | 107.51 | 105.17 | 6.29 | 5.9 | 9.94 | 1487.9 | 14.3 |
| bursty_constrained | joint_v03 | 93.11 | 90.81 | 6.64 | 7.1 | 17.92 | 1820.0 | 25.8 |
| bursty_constrained | full | 115.52 | 111.89 | 7.01 | 6.1 | 3.49 | 1349.0 | 13.3 |
| bursty_constrained | joint | 113.37 | 112.26 | 5.37 | 4.7 | 4.48 | 1395.2 | 13.5 |

| Scenario | Policy | INT4 outputs % | CPU-attention outputs % | Max active | SLO failure % |
|---|---|---:|---:|---:|---:|
| steady_roomy | full_v03 | 0.00 | 0.00 | 3 | 0.00 |
| steady_roomy | joint_v03 | 0.00 | 0.00 | 3 | 0.00 |
| steady_roomy | full | 0.00 | 0.00 | 3 | 0.00 |
| steady_roomy | joint | 0.00 | 0.00 | 3 | 0.00 |
| bursty_roomy | full_v03 | 0.00 | 0.00 | 8 | 0.00 |
| bursty_roomy | joint_v03 | 0.00 | 0.00 | 8 | 0.00 |
| bursty_roomy | full | 0.00 | 0.00 | 8 | 0.00 |
| bursty_roomy | joint | 0.00 | 0.00 | 8 | 0.00 |
| bursty_constrained | full_v03 | 0.00 | 0.00 | 2 | 0.00 |
| bursty_constrained | joint_v03 | 0.00 | 22.15 | 4 | 6.67 |
| bursty_constrained | full | 0.00 | 0.00 | 2 | 0.00 |
| bursty_constrained | joint | 0.00 | 0.00 | 2 | 0.00 |

Observed SM clocks span 1185–2565 MHz; GPU temperatures span 59–64 °C. AC-state observations are ['True']. Driver power/thermal reason flags are retained per sample. These observations do not prove that a particular performance change was caused by throttling; clock settings and external OS activity were not controlled.

![Repeated real throughput](figures/repeated_throughput.png)

![Observed GPU clocks](figures/clock_throughput.png)

## Prediction on identical actual executions

On identical executed batches, overall MAE is 71.88 ms with the old static table, 10.31 ms with the fresh static table, 4.55 ms with legacy feedback and 5.54 ms with phase feedback. These forecasts share actual execution durations and are made before observing each result. They therefore diagnose calibration/feedback without conflating different serving trajectories. Feedback remains dependent on probe order, action-family granularity and unobserved shapes; the best predictor here need not produce the best admission policy.

| Predictor | Phase | Executions | MAE ms | Bias ms | MAPE % |
|---|---|---:|---:|---:|---:|
| fresh_legacy_ewma | all | 270 | 4.55 | +2.01 | 25.3 |
| fresh_legacy_ewma | decode | 120 | 4.31 | +1.46 | 25.9 |
| fresh_legacy_ewma | mixed | 90 | 4.78 | +4.15 | 25.7 |
| fresh_legacy_ewma | prefill | 60 | 4.69 | -0.10 | 23.3 |
| fresh_phase_ewma | all | 270 | 5.54 | +3.60 | 30.8 |
| fresh_phase_ewma | decode | 120 | 5.86 | +2.82 | 35.9 |
| fresh_phase_ewma | mixed | 90 | 5.08 | +4.61 | 26.7 |
| fresh_phase_ewma | prefill | 60 | 5.61 | +3.63 | 26.8 |
| fresh_static | all | 270 | 10.31 | +10.20 | 59.8 |
| fresh_static | decode | 120 | 10.01 | +9.81 | 66.3 |
| fresh_static | mixed | 90 | 11.92 | +11.92 | 64.1 |
| fresh_static | prefill | 60 | 8.50 | +8.41 | 40.1 |
| old_static | all | 270 | 71.88 | +71.88 | 384.7 |
| old_static | decode | 120 | 46.14 | +46.14 | 274.3 |
| old_static | mixed | 90 | 95.81 | +95.81 | 504.8 |
| old_static | prefill | 60 | 87.47 | +87.47 | 425.4 |

![Matched prediction error](figures/matched_prediction_mae.png)

## CPU/CUDA execution portability

| Backend | Policy | Mean token/s | P95 TTFT ms | P95 TBT ms | Control % | Max batch | INT4 output % |
|---|---|---:|---:|---:|---:|---:|---:|
| cpu | full | 31.71 | 472.6 | 61.4 | 1.17 | 3 | 0.00 |
| cpu | joint | 31.97 | 468.0 | 60.0 | 1.35 | 3 | 0.00 |
| cuda | full | 71.23 | 51.8 | 16.4 | 1.98 | 4 | 0.00 |
| cuda | joint | 69.85 | 53.2 | 19.2 | 2.55 | 4 | 0.00 |

All 12 matched backend runs complete. CPU/CUDA full-precision greedy outputs match in 96/96 token positions. CPU-primary execution selects no CPU-offload requests, as required by its capability contract. Both use the same controller/batching source and separate calibration. This establishes execution portability across two backends on one laptop, not physical multi-machine or GPU-vendor portability. CPU's combined host quota and CUDA's separate memory pools are not equivalent resources.

## Expanded history-aware quality audit

| Audit arm | Targets | Perplexity | Relative PPL % | INT4 output % | Quantized prefill iterations |
|---|---:|---:|---:|---:|---:|
| dynamic_full | 2048 | 36.451 | +0.000 | 0.00 | 0 |
| dynamic_joint | 2048 | 36.451 | +0.000 | 0.00 | 0 |
| dynamic_joint_no_gate | 2048 | 36.451 | +0.000 | 0.00 | 0 |
| fixed_fp32 | 2048 | 36.451 | +0.000 | 0.00 | 0 |
| fixed_static_int4 | 2048 | 37.648 | +3.284 | 100.00 | 8 |
| fixed_early_prefill_int4_restore | 2048 | 36.772 | +0.881 | 0.00 | 4 |
| fixed_decode_int4 | 2048 | 36.976 | +1.440 | 99.22 | 0 |
| fixed_cpu_attention | 2048 | 36.451 | +0.000 | 0.00 | 0 |

Each arm scores the same 2048 distinct held-out positions in 16 disjoint prompt/continuation windows; eight arms repeat those targets. Early-prefill quantization restores FP32 before output emission but retains affected KV. Controlled schedules isolate this history effect and bypass admission; their times are not serving benchmarks. Dynamic teacher forcing has its own policy trajectory. These subset perplexities are not downstream-task accuracy or proof of a quality bound. FP32 dynamic/fixed token NLL and CPU-attention comparisons pass the 0.002 tolerance; the maximum observed token-NLL discrepancy across both checks is 4.58e-05. All three dynamic audits remain unquantized in this session. Their equal perplexities do not certify adaptive quantized outputs; the controlled schedules supply the separate precision-history evidence.

![Controlled precision-history audit](figures/quality_history.png)

## Evidence limits

Five rounds and a single laptop remain a limited sample. Paired-change intervals bootstrap paired percent changes, not independent machines; they are exploratory, not population confidence guarantees. Steady traces repeat unchanged; burst arrivals vary by seed. Fresh calibration and standardized warmup change conditions relative to M9, so cross-session absolute rates are not controlled speedup evidence. TTFT measures engine token readiness in a non-streaming service. Reference INT4 still unpacks to floating point, and CPU attention remains synchronous. Broader hardware/model/task validation remains open.

Protocol: [validation_study.md](../../docs/validation_study.md). Raw timing, telemetry, sources and target histories are retained alongside this report.
