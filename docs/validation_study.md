# M10: reliability, prediction diagnosis and backend portability

The v0.4 study reduced controller overhead but showed unstable throughput and larger joint prediction errors. This follow-up keeps the serving policy unchanged and improves the experiment protocol. It asks whether those outcomes persist with more repeats and fresh calibration, whether phase feedback helps on identical executed work, and whether the same serving controller runs correctly on CPU and CUDA.

## Protocol

The GPU study retains GPT-2, the original workload lengths, memory quotas and three traffic scenarios. It compares original full/joint scheduling (`full_v03`, `joint_v03`) with efficient full/joint scheduling (`full`, `joint`). Five rounds use seeds 7, 19, 31, 43 and 59. Steady arrivals remain identical; burst timestamps vary by seed. Policy order rotates through a seeded cyclic arrangement, covering each position once in each complete block of four rounds. The fifth round adds a partial block. Scenario order is shuffled between rounds. Inputs are paired within each round.

Every replay receives three identical warmup passes, including the pass inside `replay`. GPU timing, calibration, tests and CPU inference run sequentially. A fresh 75-shape CUDA calibration replaces the older session's costs. Models and data are local; no new downloads are required.

A persistent `nvidia-smi` process samples the GPU at 500 ms: temperature, SM/memory clocks, power draw, utilization, memory use, P-state, and driver-reported power/thermal clock-limit reasons. A host thread samples process RSS, available RAM and CPU utilization. Measurement timestamps align samples to the actual timed workload, excluding external warmup. Machine records retain battery/AC state before and after each run. Unsupported values remain null. Raw CSV and parsed observations are saved.

This instrumentation observes conditions; it does not lock clocks, change laptop power settings or establish thermal causality. Instrumented runs include the sampler's possible interference equally across arms. Driver flags are reported observations, not proof that a particular latency change was caused by throttling. Paired bootstrap intervals over five rounds are exploratory and remain conditional on one laptop and one session.

The first complete session began on battery and later changed to AC, with a clear corresponding clock/throughput confound. Its 60 GPU runs and complete downstream audits remain in `results/validation_study/` as diagnostic evidence. The primary replication uses `configs/validation_study_ac.json` and `results/validation_study_final/`: AC must be present before execution and at every recorded 500 ms host sample, including calibration and quality probes. Condition failures retain raw telemetry and refuse acceptance. CPU clocks, GPU clocks and ambient conditions remain observed variables. The initial diagnostic and AC replication each retain their own frozen source snapshot; the replication adds sampled power-condition enforcement, without changing the serving algorithm.

An intermediate AC series in `results/validation_study_ac/` aborted during constrained joint serving and remains excluded from successful-run aggregates. Available host RAM was substantially lower during the aborted attempt than in a successful same-trace reproduction, but the precise cause was not recoverable from its old worker state. The final implementation saves failure-only snapshots of active work, sampled capacities and current reservation estimates before clearing failed requests. Replay writes a failed partial record and propagates the error; the complete-study checker refuses to silently aggregate such a series. The existing shrinking-capacity regression also verifies these diagnostic fields. This adds observability and retains the original fail-before-unsafe-execution behavior; it does not prove immunity to competing-process resource pressure.

## Matched prediction diagnosis

Comparing predictor MAE across adaptive serving policies also compares different actions, batch sizes and contexts. The new diagnostic forecasts each **identical actual execution** with four predictors:

| Predictor | Calibration | Feedback |
|---|---|---|
| `old_static` | Archived earlier CUDA timings | None |
| `fresh_static` | Current-session CUDA timings | None |
| `fresh_legacy_ewma` | Current-session CUDA timings | Original precision/CPU family EWMA |
| `fresh_phase_ewma` | Current-session CUDA timings | Separate prefill/mixed/decode EWMA |

All predictors forecast before execution; feedback updates afterward. Fifty-four shapes span FP32 and two quantization sets, batch sizes two/four, contexts 96/320, chunks 32/96, all three execution phases, and supported CPU-attention assignments. These differ from the calibration's contexts/chunks. Five scored repetitions per shape follow an unscored warmup. Shapes are shuffled using a fixed seed; caches are independently prepared for each shape and not grown between its repeated observations. Validation text supplies probe input. Transition/setup time is excluded from execution-only prediction error.

The comparison reports MAE, signed bias and absolute percentage error overall and per phase. Its paired executions remove the different-policy-trajectory confound. It remains a short prequential test with feedback dependent on shape order; it does not establish accuracy on every future workload.

## CPU/CUDA portability

Both backends execute the same `EfficientController`, `ContinuousEngine` and pretrained batch primitive. Each has separately measured calibration: 75 CUDA shapes, 33 CPU shapes. CPU profiles use measured host compute/bandwidth, CPU inventory naming and zero discrete-device link bandwidth. The contract automatically excludes offload on a CPU-primary executor; the policy does not inspect a processor product name.

The matched portability workload is deliberately smaller: four requests, 128 prompt tokens, eight output tokens, a one-second arrival window, maximum batch four, three seeds and full/joint policies on each backend. The device reservation quota is 1400 MiB; the host reservation quota is 2048 MiB. CPU feasibility uses the combined primary-model and host-backup reservation; CUDA uses separate device and host pools. These are different resource semantics, so equal quota numbers do not imply identical capacity.

This is **backend portability on one physical laptop**, not independent machines, GPU vendors or optimized-kernel portability. CPU/CUDA FP32 output agreement and observed policy actions supplement feasibility and completion checks. Source identity, paired traces and matching token budgets are verified from raw records.

## Expanded quality audits

Sixteen disjoint 256-token windows provide 128 prompt tokens and 128 held-out continuation targets each: **2048 distinct target positions per arm**. Windows never wrap or overlap; calibration uses the validation split, while audit windows come from the stored test prefix. Eight arms reuse these targets, totaling 16,384 likelihood evaluations, not 16,384 independent targets.

Three dynamic teacher-forced replays evaluate `full`, `joint` and `joint_no_gate` under a 700 MiB admission quota. Complete precision/placement histories include quantized prefill iterations, quantized output exposure and CPU attention. These audit trajectories are separate from greedy serving trajectories.

Five controlled schedules run through the actual batch primitive, in groups of four:

| Schedule | Precision and placement history |
|---|---|
| `fp32` | FP32 throughout; device attention |
| `static_int4` | Validation-selected three-block INT4 set throughout |
| `early_prefill_int4_restore` | First 64 prompt tokens quantized, then FP32 restored before first output |
| `decode_int4` | FP32 prefill/first output; quantization for subsequent decode |
| `cpu_attention` | FP32; KV and attention moved to CPU after first output |

Controlled schedules diagnose precision and cache history; they bypass admission control and do not constitute serving-throughput or quota evidence. Early-prefill quantization can affect later likelihoods despite 0% INT4 output exposure because historical KV remains. The checker compares FP32 likelihoods across the dynamic/fixed protocols and the CPU-attention schedule, using a 0.002 maximum absolute token-NLL tolerance. The small single-prefix audit measures perplexity, not downstream-task accuracy or a guaranteed quality bound.

## Run and inspect

```powershell
# Run unit/CUDA correctness checks first; avoid concurrent GPU model work.
python scripts/validate.py

# Full sequential study; completed runs/stages are reused on resume.
python -u scripts/run_validation_study.py
python scripts/check_validation_study.py --check-current
python scripts/update_validation_report.py

# A smaller full protocol pilot:
python scripts/run_validation_study.py --config configs/validation_study_smoke.json --output results/validation_study_smoke
python scripts/check_validation_study.py --output results/validation_study_smoke

# Individual stages; useful when running on another machine:
python scripts/run_validation_study.py --config configs/validation_study.json --stage cpu_portability --output results/another_cpu
```

Available stages are `gpu`, `prediction`, `cuda_portability`, `quality`, `cpu_portability`. A CPU-only stage is a partial study and is recorded as such. The complete-study checker requires all stages. Use a new output folder for a different source or configuration. A source snapshot is frozen **before** measurements; source changes during a stage cause refusal rather than retrospective relabeling. Do not edit runtime modules during a study. Saved calibration, source/data hashes, input traces, instrumentation, quality targets and complete raw events accompany results.

The primary new evidence belongs to `results/validation_study_final/`; the mixed-power diagnostic is separate. On hardware without battery/AC telemetry, select `configs/validation_study.json` explicitly and report its conditions. Earlier M8/M9 measurements retain their own source snapshots and numerical outputs. Checkers verify historical snapshots without requiring them to match a later working tree.
