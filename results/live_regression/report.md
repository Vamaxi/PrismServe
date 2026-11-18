# Real continuous-batching experiments

**Actual pretrained GPT-2 tensor execution. All reported times are measured wall-clock observations.**

Same prompt/arrival/output-budget traces across policies. Dense projections batch all current tokens. Ragged attention loops per request. Quotas are enforced admission budgets, not physical OOM events.

| Scenario | Policy | P95 TTFT ms | P95 TBT ms | Output token/s | SLO failure % | Max active | INT4 output % | CPU offload output % |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| bursty_constrained | full | 2605.66 | 23.84 | 73.05 | 25.00 | 2 | 0.00 | 0.00 |
| bursty_constrained | joint | 3679.92 | 77.74 | 57.67 | 41.67 | 4 | 2.78 | 40.28 |
| bursty_constrained | morph_only | 2860.51 | 24.23 | 68.96 | 41.67 | 2 | 4.86 | 0.00 |

## Quality audit

A separate teacher-forced replay uses the same held-out continuations for each policy and records token NLL under the actual selected precision and historical KV. It has its own policy trajectory. See quality.json and quality_runs/; these results do not label greedy generation as teacher-forced work.

| Policy | Audit targets | Audit perplexity | Relative PPL % | INT4 output % |
|---|---:|---:|---:|---:|

## Interpretation limits

Do not assume a controller win: inspect the full baseline, admission, overhead, transition, and failure records. The measured nearest-shape model and EWMA feedback are estimates; real elapsed timings remain authoritative. Quality uses an additive validation-KL surrogate and a small separate audit, not a guaranteed task bound.

TTFT starts at scheduled arrival, including delayed submission and queueing. All token gaps include scheduling, precision transitions and migrations. Throughput includes the complete trace and queue drain. Rejected requests count as SLO failures. Memory includes independently reported live KV, reserved budgets and CUDA peak allocations.

![Actual output token/s](figures/output_tokens_per_s.png)

![Measured P95 TTFT ms](figures/ttft_p95_ms.png)
