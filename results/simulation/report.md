# PrismServe experiment results

**Analytical simulation. These are not measured LLM serving results.**

The rates in config.json describe hypothetical effective hardware. The same traces and seeds are replayed for every policy.

| Profile | Policy | P95 TTFT ms (mean across seeds) | P95 TBT ms | SLO failure % | Output token/s | Good request/s |
|---|---|---:|---:|---:|---:|---:|
| 20GiB-balanced | full | 18180.02 | 307.53 | 94.79 | 120.03 | 0.14 |
| 20GiB-balanced | static_int4 | 17818.96 | 321.78 | 93.40 | 125.72 | 0.19 |
| 20GiB-balanced | morph_only | 18176.18 | 363.22 | 94.79 | 119.14 | 0.14 |
| 20GiB-balanced | offload_only | 18575.54 | 365.40 | 94.79 | 117.83 | 0.14 |
| 20GiB-balanced | naive | 18488.02 | 365.80 | 94.79 | 118.46 | 0.14 |
| 20GiB-balanced | joint | 18581.81 | 365.40 | 94.79 | 117.77 | 0.14 |
| 20GiB-balanced | joint_no_hysteresis | 18581.81 | 365.40 | 94.79 | 117.77 | 0.14 |
| 20GiB-slow-host | full | 18180.02 | 307.53 | 94.79 | 120.03 | 0.14 |
| 20GiB-slow-host | static_int4 | 17818.96 | 321.78 | 94.79 | 125.72 | 0.15 |
| 20GiB-slow-host | morph_only | 18194.65 | 362.44 | 94.79 | 120.12 | 0.14 |
| 20GiB-slow-host | offload_only | 22496.45 | 418.11 | 94.79 | 104.27 | 0.12 |
| 20GiB-slow-host | naive | 33397.11 | 426.85 | 95.83 | 74.14 | 0.07 |
| 20GiB-slow-host | joint | 19917.59 | 376.89 | 94.79 | 112.64 | 0.13 |
| 20GiB-slow-host | joint_no_hysteresis | 19917.59 | 376.89 | 94.79 | 112.64 | 0.13 |
| 48GiB-server | full | 3690.17 | 107.47 | 76.04 | 212.99 | 1.15 |
| 48GiB-server | static_int4 | 3624.64 | 108.18 | 75.69 | 218.66 | 1.20 |
| 48GiB-server | morph_only | 3690.17 | 107.47 | 76.04 | 212.99 | 1.15 |
| 48GiB-server | offload_only | 3690.17 | 107.47 | 76.04 | 212.99 | 1.15 |
| 48GiB-server | naive | 4030.83 | 110.96 | 77.08 | 212.97 | 1.10 |
| 48GiB-server | joint | 3690.17 | 107.47 | 76.04 | 212.99 | 1.15 |
| 48GiB-server | joint_no_hysteresis | 3690.17 | 107.47 | 76.04 | 212.99 | 1.15 |

## Interpretation and evidence limits

SLO failure counts each rejected request as a failure, plus completed requests whose TTFT or worst intertoken gap exceeds the configured SLO. TTFT percentiles cover completed requests; always inspect rejection counts in summary.csv.

Full reserves full-precision weights. static_int4 is an unconstrained quality reference. morph_only and offload_only are mechanism-inspired analytical baselines, not the original MorphServe or NEO systems. naive uses a fixed combination; joint searches combinations with one common hardware interface.

The model accounts for host weight backups, packed weights with scales, full context reservations, workspace, migration, and transition peaks. Precision selection uses a synthetic additive risk proxy; this is not perplexity, accuracy, or a guaranteed quality bound.

The CPU/device pipeline overlap assumption can overestimate achieved throughput. The real backend synchronizes transfers and does not implement NEO's ISPC kernels or MorphServe's asynchronous in-place allocator.

aggregate.json includes bootstrap intervals and paired differences across seeds. Three seeds provide exploratory variability estimates only. Planner search times are measured host wall times and excluded from virtual token latency; they cannot establish a production scheduler overhead percentage.

Inspect runs/*.json for every request's token timestamps and every plan transition. Reproduce from config.json and traces/, with hashes recorded in manifest.json.

## Figures

![P95 TTFT (ms)](figures/ttft_p95_ms.png)
![SLO failure fraction](figures/slo_violation_rate.png)
![Simulated controller timeline](figures/controller_timeline.png)