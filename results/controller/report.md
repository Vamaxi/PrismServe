# Real continuous-batching experiments

**Actual pretrained GPT-2 tensor execution. All reported times are measured wall-clock observations.**

Same prompt/arrival/output-budget traces across policies. Dense projections batch all current tokens. Ragged attention loops per request. Quotas are enforced admission budgets, not physical OOM events.

| Scenario | Policy | P95 TTFT ms | P95 TBT ms | Output token/s | SLO failure % | Max active | INT4 output % | CPU offload output % |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| bursty_constrained | full | 6845.58 | 55.47 | 32.76 | 70.83 | 2 | 0.00 | 0.00 |
| bursty_constrained | full_v03 | 7906.69 | 58.93 | 28.84 | 70.83 | 2 | 0.00 | 0.00 |
| bursty_constrained | joint | 8506.38 | 148.40 | 26.88 | 100.00 | 4 | 23.96 | 40.28 |
| bursty_constrained | joint_no_gate | 8719.91 | 151.61 | 26.21 | 100.00 | 4 | 17.36 | 42.01 |
| bursty_constrained | joint_no_reuse | 8382.70 | 154.92 | 26.96 | 100.00 | 4 | 11.98 | 37.85 |
| bursty_constrained | joint_v03 | 9702.29 | 129.23 | 24.34 | 100.00 | 4 | 12.15 | 14.76 |
| bursty_roomy | full | 5500.40 | 144.36 | 37.06 | 100.00 | 8 | 0.00 | 0.00 |
| bursty_roomy | full_v03 | 5787.64 | 130.73 | 35.97 | 95.83 | 8 | 0.00 | 0.00 |
| bursty_roomy | joint | 5674.97 | 154.09 | 37.02 | 100.00 | 8 | 3.12 | 6.60 |
| bursty_roomy | joint_no_gate | 6618.72 | 174.21 | 33.34 | 100.00 | 8 | 11.81 | 10.59 |
| bursty_roomy | joint_no_reuse | 5215.66 | 137.99 | 38.81 | 100.00 | 8 | 0.00 | 1.04 |
| bursty_roomy | joint_v03 | 6565.77 | 201.50 | 32.41 | 100.00 | 8 | 6.77 | 9.20 |
| steady_roomy | full | 2610.54 | 120.38 | 53.03 | 70.83 | 8 | 0.00 | 0.00 |
| steady_roomy | full_v03 | 5338.29 | 157.50 | 31.94 | 95.83 | 8 | 0.00 | 0.00 |
| steady_roomy | joint | 4064.02 | 147.87 | 41.11 | 100.00 | 8 | 2.78 | 4.34 |
| steady_roomy | joint_no_gate | 4899.62 | 197.42 | 32.65 | 100.00 | 8 | 21.88 | 16.32 |
| steady_roomy | joint_no_reuse | 4524.32 | 158.61 | 36.61 | 100.00 | 8 | 0.00 | 3.99 |
| steady_roomy | joint_v03 | 2510.27 | 134.86 | 67.04 | 70.83 | 8 | 36.28 | 6.94 |

## Quality audit

A separate teacher-forced replay uses the same held-out continuations for each policy and records token NLL under the actual selected precision and historical KV. It has its own policy trajectory. See quality.json and quality_runs/; these results do not label greedy generation as teacher-forced work.

| Policy | Audit targets | Audit perplexity | Relative PPL % | INT4 output % |
|---|---:|---:|---:|---:|
| full_v03 | 64 | 68.284 | 0.000 | 0.00 |
| joint_v03 | 64 | 68.403 | 0.174 | 21.88 |
| full | 64 | 68.284 | 0.000 | 0.00 |
| joint | 64 | 67.770 | -0.753 | 0.00 |
| joint_no_gate | 64 | 68.344 | 0.088 | 9.38 |
| joint_no_reuse | 64 | 67.730 | -0.811 | 0.00 |

## Interpretation limits

Do not assume a controller win: inspect the full baseline, admission, overhead, transition, and failure records. The measured nearest-shape model and EWMA feedback are estimates; real elapsed timings remain authoritative. Quality uses an additive validation-KL surrogate and a small separate audit, not a guaranteed task bound.

TTFT starts at scheduled arrival, including delayed submission and queueing. All token gaps include scheduling, precision transitions and migrations. Throughput includes the complete trace and queue drain. Rejected requests count as SLO failures. Memory includes independently reported live KV, reserved budgets and CUDA peak allocations.

![Actual output token/s](figures/output_tokens_per_s.png)

![Measured P95 TTFT ms](figures/ttft_p95_ms.png)
