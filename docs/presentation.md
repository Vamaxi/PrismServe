---
marp: true
theme: default
paginate: true
---

# PrismServe

## Resource-calibrated joint precision and CPU-attention control

DA4022 · Hardware Aware Deep Learning · IIT Guwahati

Siddharth Khatod · Akash Iyer · Vamaxi Maheshwari · Kanika Bansal · Yashvardhan Kant

**Evidence:** real pretrained GPT-2 + small correctness decoder + separately labeled analytical simulation

<!-- Speaker notes: We built an independent project inspired by MorphServe and NEO. We do not claim their reported performance or a production LLM deployment. -->

---

# 1. Research question

Can one policy coordinate weight precision and host attention placement using a hardware resource interface?

- Device memory limits weights plus KV capacity.
- Host memory supplies capacity but CPU bandwidth and transfer costs limit its usefulness.
- Quantization reclaims capacity but changes model outputs.
- The best choice depends on hardware balance and current requests.

<!-- Speaker notes: Hardware agnostic means the policy consumes capacities and rates, not that optimized execution kernels are identical on every platform. -->

---

# 2. What the papers actually do

| MorphServe, MLSys 2026 | NEO, MLSys 2025 |
|---|---|
| Switch selected weight layers at runtime | Offload selected decoding attention plus KV |
| Reclaim weight memory as KV capacity | Keep historical KV on the host |
| Host stores recovery/quantized variants | Exchange current Q/K/V and attention output |
| Runtime pressure triggers and hysteresis | Adaptive load-aware batching using profiles |

MorphServe does not quantize live KV. NEO's scheduler is already adaptive.

<!-- Speaker notes: Cite MorphServe Sections 4.1–4.4 and NEO Sections 3.1–3.2. The contribution is studying the shared control space, not correcting a supposedly static NEO scheduler. -->

---

# 3. Revised scope

- Independent transparent controller and experiment harness.
- Actual pretrained GPT-2, cached decode, packed INT4 and CPU attention.
- Small byte decoder for fast numerical and restoration checks.
- Synthetic LLM-scale profiles and trace simulation.
- Explicit capacity, host backups, transitions and approximation-risk accounting.

Original proposal speedup and task-accuracy targets remain research targets.

<!-- Speaker notes: Local hardware is an RTX 4050 Laptop GPU with 6 GB VRAM, much smaller than the papers' main LLM testbeds. No original-system ±5% reproduction is claimed. -->

---

# 4. System structure

```text
resource profile + model geometry + current request state
                            |
                      resource estimator
                            |
                      joint plan search
                            |
            quantized layers + CPU-attention requests
                    /                     \
        analytical trace execution      GPT-2 / tiny execution
                    \                     /
                   labeled metrics and reports
```

<!-- Speaker notes: The simulator and real backend are separate paths. We can validate real operations without representing analytical service times as GPU measurements. -->

---

# 5. Policy and feasibility

- Enumerate sensitivity-ordered layer prefixes × offload fractions.
- Check device memory, host memory and transient movement.
- Charge host attention, link traffic and plan changes.
- Use a quality-risk surrogate and cooldown.
- Reduce admission when no candidate fits.

The search is bounded and deterministic. Its surrogate is not an accuracy proof.

<!-- Speaker notes: Contrast shared optimization with always-offload-first stacking. Explain why restore transitions can fail even when final full precision looks desirable. -->

---

# 6. Correct memory and transfer accounting

`KV bytes/token = 2 × layers × KV heads × head dimension × element bytes`

- Reserve prompt plus maximum requested output.
- Count scale metadata and CPU recovery weights.
- Preserve live KV; no attention-history eviction.
- Charge a full-cache copy when changing placement.
- Steady CPU attention transfers new tensors, not all old KV.

<!-- Speaker notes: GQA uses KV heads. Reserving maximum output is conservative. The real backend unpacks weights for matmul, so scratch and latency differ from optimized 4-bit kernels. -->

---

# 7. Experiments and controls

Compare full precision, static INT4, morph-only, offload-only, naive stacking and joint control.

- Identical timestamped workloads and model geometry.
- Independent device capacity, host bandwidth and link changes.
- Precision/offload/hysteresis ablations.
- Negative regimes and rejection accounting retained.

<!-- Speaker notes: These are our simplified controls inspired by the papers, not measurements of vLLM, MorphServe or NEO. Use more than one trace seed when making a general comparison. -->

---

# 8. Results: analytical simulation

63 runs · three workload seeds · synthetic hardware ·8B-shaped model

| Profile | Full P95 TTFT | Joint | Naive |
|---|---:|---:|---:|
|20 GiB balanced|18.180 s|18.582 s|18.488 s|
|20 GiB slow host/link|18.180 s|19.918 s|33.397 s|
|48 GiB server|3.690 s|3.690 s|4.031 s|

Slow host: joint has **40.36% lower TTFT than naive**, but **9.56% higher TTFT than full precision**. All 96 requests complete; zero rejections.

Joint SLO failures: 94.79%, 94.79%, 76.04%. Targets are not met. Hysteresis ablation has no effect in this workload.

<!-- Speaker notes: Exact means across per-run P95 values from results/simulation/summary.csv, final explicit-matrix-count config. The joint policy changes placement with hardware balance and avoids slow-host overoffloading, but it does not beat full precision in these settings. Static INT4 has a different proxy-quality budget. All figures are analytical simulation, not measured LLM serving. See results/simulation/figures/ttft_p95_ms.png for individual seeds. -->

---

# 9. Results: real tiny-model execution

149,248 parameters · 80 training steps · seed 7 · 480 held-out byte targets

| Variant | Resident bytes | Perplexity | CUDA mean TBT |
|---|---:|---:|---:|
| FP32 | 596,992 | 13.8275 | 4.174 ms |
| Layer 0 INT4 | 428,032 | 13.8193 | 6.624 ms |
| Both layers INT4 | 259,072 | 13.8189 | — |

Both-layer packing saves **56.6% resident storage**; CPU masters use 393,216 bytes. Fresh-inference restoration logit error: **0.0**.

Actual adaptive execution: CPU KV for 12 decode steps, then CUDA again. No adaptive quantization selected; reference unpacking is costly.

<!-- Speaker notes: Source results/demo/{training.json,quality.json,real_backend.json}. The tiny negative perplexity change is corpus-specific, not general quality improvement. INT4 CUDA timing is slower because this reference uses unpack-and-matmul. Adaptive artifact is results/adaptive/adaptive_runtime.json using the same demo checkpoint; declared quota changes at decode 8 and 20 move real KV without filling physical VRAM. -->

---

# 10. Results: actual pretrained GPT-2

124.4M parameters · FP32 · blocks 1, 7, 9 selected using validation only

| CUDA variant | Resident MiB | Mean TBT | Test-subset PPL |
|---|---:|---:|---:|
| FP32 |486.70|21.09 ms|36.550|
| Three blocks INT4 |417.09|25.95 ms|37.822|
| Three-block hybrid |417.09|27.27 ms|37.822|
| All 12 blocks INT4 |208.26|37.44 ms|45.997|

WikiText-2 test prefix: 4095 targets · context 256 · stride 128

Three-block PPL increases **3.48%**; all-block PPL increases **25.85%**. Reference INT4 saves storage and is slower. No ≤1.5% quality claim.

<!-- Speaker notes: Exact results/pretrained/{benchmark.json,quality.json,calibration.json}. Three prompts, three repeats, 32 output tokens, EOS stopping disabled, setup excluded. Validation first 128 tokens ranks one-block logitKL; blocks 1, 7, 9 chosen without test labels.4096 test input tokens yield 4095 targets scored exactly once.324 MiB host masters and 9 MiB unpack scratch accompany resident storage. CPU-attention quality rows are fresh primary-device prefill; decode parity is separate. This is real 124M GPT-2, not simulated 8B execution. -->

---

# 11. Pretrained correctness, adaptation and service

- Original Hugging Face forward error: **0.0**.
- Cached decode error: 9.16×10⁻⁵; CPU attention: 4.58×10⁻⁵.
- Common controller: actual CPU KV for 12 outputs, then CUDA again.
- Offload/restore application: 3.864/1.502 ms; actual generation 769.094 ms.
- Actual pretrained HTTP completion: block 0 INT4 + CPU KV verified.
- **97 tests pass**, without skips; checkpoint/data hashes retained.

Quota changes are diagnostic; physical VRAM is not filled. GPT-2 completes text; it is not an instruction-tuned chatbot.

<!-- Speaker notes: Original-model declared tolerance 0.002 absolute, results/pretrained/parity.json. Adaptive indices 8–19 use CPU KV; index 20 restores CUDA. Adaptive TTFT 22.575 ms, 31 planner decisions, no adaptive quantization chosen by conservative reference costs. HTTP smoke returns 8 tokens; setup 71.93 ms and cold service TTFT 1017.91 ms are not comparable to warm benchmark TTFT 20.36 ms. See results/pretrained_api_smoke.json and results/validation.json. Model is pinned 607a30d783dfa663caf39e06633721c8d4cfcd7e with safetensors hash; dataset responses retained. -->

---

<!-- LIVE_SLIDES_BEGIN -->
# 12. Real continuous-batching traffic

42 actual traffic runs · 7 policies · 2 seeds · identical prompts/arrivals

| Scenario | Serial token/s | Batched full token/s | Joint token/s |
|---|---:|---:|---:|
| steady_roomy | 36.6 | 62.5 | 53.0 |
| bursty_roomy | 46.3 | 90.0 | 75.2 |
| bursty_constrained | 52.2 | 72.8 | 62.8 |

True shared dense execution:48 projections per iteration across 12 blocks; separate KV per request.

1400/700 MiB are enforced admission budgets. All latency includes controller and transition costs.

<!-- Speaker notes: Means across two workload seeds. Same optimized batch primitive for serial and batched controls. See results/live/report.md for all arms, per-run percentiles, failures, overhead and memory. Increased concurrency need not improve throughput. All tensor execution is real; no virtual token time. -->

---

# 13. Live controller, quality and validation

- 75 measured shapes: batch sizes 1/4/8, mixed prefill/decode, precision and placement.
- Nearest timing lookup plus EWMA feedback; measured precision transitions.
- Reserve future KV and temporary copies before admission.
- Four concurrent HTTP clients share an actual batch of 4.
- 97 passing tests, including memory-growth and cancellation regressions.
- Separate teacher-forced audit: 64 held-out targets per policy; report precision exposure.

Reference attention remains synchronous and ragged. Quality and speedup guarantees remain unproven.

<!-- Speaker notes: The initial diagnostic suite exposed insufficient future workspace reservation and is retained separately in results/live_initial. Final results use corrected memory bounds and actual batching warmup. Calibration and resource estimates do not make a production allocator or fused paged-attention kernel. Keep audit trajectories separate from greedy output histories. -->
<!-- LIVE_SLIDES_END -->

---

<!-- CONTROLLER_SLIDES_BEGIN -->
# 14. Controller efficiency: measured follow-up

36 real traffic runs, fresh old-controller controls, gate/reuse ablations.

| Scenario | Old control % | New control % | Old token/s | New token/s |
|---|---:|---:|---:|---:|
| steady_roomy | 8.1 | 2.3 | 67.0 | 41.1 |
| bursty_roomy | 6.5 | 2.1 | 32.4 | 37.0 |
| bursty_constrained | 16.9 | 2.4 | 24.3 | 26.9 |

All six optimized joint runs meet the exploratory <5% control-overhead target (worst 2.82%).

<!-- Speaker notes: Compare actual full-precision arms as well as joint. Means of two seeds do not establish a reliable speedup. Complete latency, per-seed variability and prediction errors are in results/controller/comparison.md. -->

---

# 15. Faster decisions with fresh resource checks

- Compile safe reservation formulas and cache shape lookups.
- Reuse stable plans; sample capacity and check reservations each tick.
- Separate prefill, mixed and decode feedback.
- Conservative full-precision reference; capacity necessity bypasses the gate.
- 106 tests; four-client real HTTP check passes.

Reference kernels remain synchronous. Short audits do not guarantee task quality.

<!-- Speaker notes: Reuse can delay reconsidering existing pending arrivals by three iterations. Heuristic uncertainty is not a confidence interval. Compilation, pruning and phase feedback are combined; only reuse and gate have separate ablations. Every reused event records at least one fresh resource check. -->

---

<!-- CONTROLLER_SLIDES_END -->

<!-- VALIDATION_STUDY_SLIDES_BEGIN -->
# 16. Reliability: five instrumented rounds

60 real GPU runs; fresh calibration, three warmups, balanced order.

| Scenario | Old joint token/s | New joint token/s | New control % |
|---|---:|---:|---:|
| steady_roomy | 121.2 | 123.7 | 4.6 |
| bursty_roomy | 134.5 | 138.7 | 2.8 |
| bursty_constrained | 93.1 | 113.4 | 4.5 |

Clock/temperature/power logs retained; clock settings are observed, not fixed.

<!-- Speaker notes: The full report includes paired exploratory intervals and every repeat. Five repeats on a laptop do not establish universal wins. Clocks and driver flags do not prove a thermal explanation. M9 and M10 are separate sessions with different calibration and warmup. -->

---

# 17. Portability, prediction and quality evidence

- Same serving controller on CPU/CUDA: 12 paired-trace runs complete.
- FP32 greedy output agreement: 96/96 token positions.
- 54 matched shapes compare four predictors before observation.
- Eight quality arms, 2048 held-out targets each.
- Early quantized prefill can affect later FP32 outputs through KV history.
- Controlled early-prefill audit: +0.88% perplexity.
- 112 tests pass; all source/data snapshots verified.

One laptop with two execution backends; physical multi-machine validation remains.

<!-- Speaker notes: Controlled quality schedules are diagnostic, bypass admission, and are not serving-speed measurements. Targets repeat across arms. Matched probes remove the trajectory confound from comparing predictor errors. Consult the raw phase errors before attributing effects to phase feedback. -->

---

<!-- VALIDATION_STUDY_SLIDES_END -->

<!-- FRONTIER_SLIDES_BEGIN -->
# 18. Where adaptation becomes necessary

216 real quota/context replays; FP32, quantization only, offload only and joint.

| Prompt tokens | FP32 | Quantization only | Offload only | Joint |
|---:|---:|---:|---:|---:|
| 128 | 600 | 600 | 600 | 600 |
| 384 | 630 | 630 | 630 | 630 |
| 768 | 700 | 675 | 700 | 675 |

Lowest tested complete quotas (MiB), not exact physical limits.

<!-- Speaker notes: Include transition staging and future KV in admission. Cold loading and warmup precede the timed quota. The complete grid and every rejection are retained. Joint control is not assumed to beat either mechanism alone. -->

---

# 19. Capacity gain has a quality cost

| Prompt | Quota MiB | Policy completing all requests | Paired repeats with FP32 rejecting all |
|---:|---:|---|---:|
| 768 | 675 | joint | 3/3 |
| 768 | 675 | morph_only | 3/3 |

- 6 trajectory-aware quality audits, plus paired FP32 references.
- Audit prompt/output lengths match the selected frontier cell.
- Report quantization exposure and target counts with perplexity.
- Rejected requests are SLO failures; their latency is N/A.

<!-- Speaker notes: Read the actual audit PPL values from the report. The long-context audit is small and supplements M10; it does not prove task accuracy. Reference quantization can expand capacity while slowing execution. Explain what the common hardware contract does and what remains kernel/device-specific. -->

---

<!-- FRONTIER_SLIDES_END -->

# 20. What remains uncertain

- Ideal overlap and roofline estimates require component calibration.
- Additive layer risk misses interactions and prompt dependence.
- Reference unpack-and-matmul is not optimized INT4 execution.
- Synchronized host attention is not NEO's pipeline.
- Larger models, downstream tasks and physical multi-machine validation remain.

<!-- Speaker notes: Define the boundary of the evidence candidly. The project is useful because the decisions, assumptions and results can be inspected and extended. -->

---

# 21. Answer and next step

A common resource interface and joint policy drive actual pretrained GPT-2 generation and placement changes.

Whether they provide reliable LLM-scale SLO/quality gains depends on calibrated kernels, transition costs and workload regimes.

Next: optimize low-bit kernels and asynchronous batching; calibrate component costs; evaluate larger models across physical machines.

References: *MorphServe*, MLSys 2026; *NEO*, MLSys 2025; official GPT-2 and WikiText model/dataset cards. See `research_review.md`, `pretrained.md`, and `references.bib`.

<!-- Speaker notes: Explain the measured pretrained storage/quality/time tradeoff and the negative simulation regimes. Avoid claiming universal portability, original-paper reproduction, or guaranteed gains. The v0.1 simulator measurements retain their original source manifests; the pretrained extension has a separate v0.2 record. -->
