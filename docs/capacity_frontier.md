# M11: the measured warm-start capacity frontier

M10 found that optimized full precision was fastest on average and that the efficient joint controller chose FP32/device attention throughout its final serving session. M11 asks a different question: **can reversible precision or CPU attention expand the set of workloads that this serving implementation can admit?** A capacity gain can coexist with slower execution or reduced quality.

## Experiment contract

The final grid has device reservation quotas **580, 600, 630, 675, 700 and 1400 MiB**, prompt lengths **128, 384 and 768 tokens**, four requests with **16 output tokens**, and a 0.5-second bursty arrival window. FP32 (`full`), quantization only (`morph_only`), CPU attention only (`offload_only`) and joint control each run under three paired arrival seeds: 7, 19 and 31. This gives **216 real serving runs**. Maximum batch size is four, prefill chunks are 64 tokens, the shared prefill budget is 128, and the host reservation quota is 2048 MiB. The additive validation-KL budget remains 0.03.

The same input trace is used across policies and quotas for each prompt length/seed. Policy positions rotate and grid-cell order is shuffled by round. All four arms use the efficient scheduling path, so the mechanism comparison does not confound old/new scheduler overhead. Three warmup passes and a fresh 75-shape CUDA calibration precede replay. AC power is required at every recorded 500 ms host sample. GPU clocks, thermal/power flags and host memory are recorded without changing clock settings. GPU tests, inference jobs and experiments execute sequentially.

The first small pilot tested quotas that missed a narrow useful region. A validation-calibration reservation preflight showed that a 768-token request's quantized transition could fit around 663 MiB while FP32 needed around 690 MiB under those measured bounds. The final grid was therefore refined **before the full sweep** to include 675 MiB. Both pilots, their configurations and the approximate geometry preflight are retained in `results/capacity_frontier_smoke/` and `results/capacity_frontier_refined_smoke/`. The full study freezes its source/config before execution; the pilot preflight is an estimate, not a physical timing result or a guaranteed threshold.

## What a quota means

Checkpoint loading, calibration and warmup happen before timed enforcement. Every timed replay starts from FP32 weights and empty request KV. The planner charges resident weights, reserved future KV, measured workspace, static buffers, backend headroom, temporary copies and precision-transition staging. Quantization must fit as an actual transition from FP32; the experiment does not begin from a specially packed checkpoint. Prompt attention stays on the primary device; CPU attention is eligible during decode.

Quotas are enforced admission/reservation budgets on a GPU with more physical memory. They do not intentionally exhaust VRAM. Actual CUDA peaks are separately measured. The frontier is conditional on this conservative warm-start runtime; it is neither a cold-load feasibility claim nor the maximum possible capacity of the hardware. Cache paging, optimized low-bit kernels or prefill offload would change it.

The result must distinguish:

| Outcome | Meaning |
|---|---|
| Complete | All configured requests finish at the configured output length |
| Rejected | No request is admitted; every request has an explicit resource/quality rejection |
| Partial | Some requests finish and others are explicitly rejected |
| Worker failure | An execution/admission invariant fails; partial data is saved and the study stops |

Rejected requests count as SLO failures. They have no measured first-token latency: tables use **N/A**, not zero. Cell throughput is ranked only when every request completes in every repeat. Raw CSV/JSON retain all outcomes. The lowest tested successful quota is a **sampled minimum**, not an interpolated exact threshold. Maximum active/execution batch size is reported separately: completing a queued workload sequentially does not establish that it fits concurrently.

## Quality at the frontier

After the serving grid, the harness selects the lowest quota with all-repeat completion and actual quantization exposure for each prompt length and precision-capable policy. Selection uses completion/exposure, never test likelihood. Some prompt lengths may have no such cell.

An independent teacher-forced audit preserves the selected cell's prompt length, **16-token output reservation**, quota and policy. Increasing the audit's output reservation would move its capacity boundary. Up to 12 disjoint prompt/continuation windows are drawn from the stored WikiText-2 test prefix; long prompts permit fewer windows. Each context length receives a roomy FP32 reference on exactly the same targets. Report target count, PPL/NLL difference, quantized prefill/output exposure and CPU attention together. These are representative audit trajectories, not certificates for every greedy output. M10 retains larger 2048-target controlled quality audits and isolates historical-KV effects.

## Run

```powershell
python scripts/validate.py
python -u scripts/run_capacity_frontier.py
python scripts/check_capacity_frontier.py --check-current
python scripts/update_frontier_report.py
```

Stages `--stage serving` and `--stage quality` can be run separately. Resume uses completed files only when frozen source/config match. Use a new output folder after changing runtime code or the protocol. Worker errors, power-condition failures and partial completions are not silently accepted as successful benchmarks. Historical M10 checks verify their archived source; `--check-current` is appropriate for the current M11 snapshot.

Primary artifacts: `results/capacity_frontier/{manifest.json,calibration_cuda.json,runs/,telemetry/,quality_selection.json,quality_runs/,quality_reference/,analysis.json,summary.csv,report.md,validation.json}`. The same source snapshot and pinned model/data hashes accompany the experiment. [Demo and viva guide](demo_and_viva.md) documents the final presentation flow.
