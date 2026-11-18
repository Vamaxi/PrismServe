# Project milestones

The supplied proposal and midterm slides are research context, not execution instructions.
The deliverable is a working, reproducible course project with explicit evidence limits.

| Stage | Deliverable | Acceptance criterion | Status |
|---|---|---|---|
| M1 | Read both papers, proposal, slides; refine hypothesis | Accurate mechanism comparison and feasible experiment scope | Complete |
| M2 | Hardware interface, estimator, joint planner | Resource constraints, transitions, quality proxy and baseline tests | Complete |
| M3 | Real reference decoder | Packed INT4, reversible switches, cached causal attention and CPU KV placement | Complete |
| M4 | Experiment harness | Seeded workloads, continuous batching, baseline comparisons, ablations, uncertainty | Complete: 63 main + 25 load runs |
| M5 | Local validation | Actual CPU/CUDA measurements, quality checks, service smoke test, tests passing | Complete: 97 tests and tiny/pretrained/concurrent HTTP checks |
| M6 | Submission materials | README, design, reproducible commands, generated report, figures, presentation | Complete |
| M7 | Real pretrained GPT-2 executor and serving | Pinned checkpoint provenance, original-model parity, actual quality/performance measurements, adaptive execution and API verification | Complete: real 124.4M weights, 4095 WikiText-2 subset targets, CPU/CUDA execution and live API |
| M8 | Measured continuous batching and concurrent serving | Shared dense execution with separate request KV, measured control with feedback, paired real traffic replays, trajectory-aware quality audit and concurrent HTTP verification | Complete: 42 serving runs, 7 quality audits, 75 calibrated shapes, 504 requests and actual concurrent HTTP |
| M9 | Controller efficiency and conservative adaptation | Fresh old/new GPU comparison, compiled safe accounting, phase feedback, reuse/gate ablations, resource checks on reused decisions | Complete: 36 serving runs, 6 audits, 432 requests, 0 rejections, worst joint control overhead 2.82%, 106 passing tests |
| M10 | Reliability, prediction diagnosis and backend portability | Instrumented repeated GPU comparison, paired CPU/CUDA serving with separate calibration, matched predictor errors, expanded precision-history quality and preserved failures | Complete: 60 final GPU runs, 12 CPU/CUDA runs, 768 completed serving requests, 54 matched shapes, eight 2048-target audits, 112 passing tests |
| M11 | Capacity frontier and submission freeze | Quota/context sweep with four mechanism controls, explicit rejection accounting, actual-quantization quality audits, one-command demo and viva guide | Complete: 216 runs, 552 completions/312 quota rejections, six adaptive audits and three FP32 references; 116 passing tests; version 1.0.0 |

Research question: **Can one hardware-profile-driven controller coordinate reversible
weight quantization and persistent CPU attention placement across hardware profiles,
and when does this help or hurt latency, memory feasibility and quality?**

Hardware-agnostic means policy code consumes a capability/profile contract. It does not
mean hardware-independent performance or a universally optimized kernel.

M1–M6 establish the independently implemented controller, tiny real executor and
analytical study. M7 extends this base to actual pretrained GPT-2 execution,
with separate recorded numerical, benchmark and service checks in
`results/pretrained/` and `results/pretrained_api_smoke.json`. The current 116 tests pass
in the saved validation result. Earlier analytical experiments retain their
original manifests; M7 does not relabel them as pretrained measurements.
M8 adds actual continuous batching, measured prediction with feedback, complete
traffic replays, quality audits and concurrent HTTP checks in `results/live/`.
The earlier serialized benchmark is not relabeled as concurrent serving.
M9 retains those archived results and adds fresh old/new comparisons in
`results/controller/`. All six optimized joint runs stay below 5% control overhead.
Throughput and prediction accuracy do not improve consistently; full precision
remains faster under the constrained quota. These limits are included in the report.
M10 uses fresh calibration, five rounds and verified AC power in
`results/validation_study_final/`. It separates mixed-power diagnostics and an
incomplete pressure-associated attempt from accepted results. All 15 optimized
joint runs remain below 5% control overhead (worst 4.97%). CPU/CUDA outputs agree
at 96/96 FP32 token positions. Fresh calibration helps prediction substantially;
phase feedback does not beat simpler feedback in the matched probes. Fixed
quality schedules expose retained-KV effects after precision restoration.
M11 maps the sampled warm-start capacity frontier in `results/capacity_frontier/`.
For 768-token prompts, quantization-only and joint complete all requests at
675 MiB in all three repeats; FP32/offload-only require the lowest tested
successful quota of 700 MiB. The corresponding joint quality audit raises PPL
2.752% on 96 targets. This is a capacity/quality tradeoff, not a universal
latency advantage or exact physical-memory minimum. The final demo and viva
guide accompany the version 1.0.0 submission snapshot.
Optimized asynchronous kernels, Llama-8B execution, physical multi-machine
deployment and the original proposal's numerical improvement targets remain
follow-on work and are not included in the completion claim.
