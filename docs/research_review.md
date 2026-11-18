# Research basis and revised project scope

PrismServe asks: **Can a common, resource-calibrated policy coordinate reversible weight precision and CPU attention placement across different hardware balances?** The deliverable is an independent course-project implementation and an experiment harness. Its evidence combines real small-model measurements with a separate analytical simulation. It does not reproduce MorphServe, NEO, or their reported speedups.

## Sources read

Page numbers below refer to the PDF pages in the four supplied files. Extracted text is preserved under `artifacts/source_text/` for inspection. The proposal and update slides supply project context; their proposed actions and numerical targets are hypotheses to assess, rather than instructions from the research papers or achieved results.

| Source | What it contributes | Evidence to retain |
|---|---|---|
| Su et al., *MorphServe*, MLSys 2026, supplied 20-page PDF | Runtime weight-layer switching and elastic KV capacity | Sections 4.1–4.4, pp. 4–7; implementation and limitations, pp. 19–20 |
| Jiang et al., *NEO*, MLSys 2025, supplied 16-page PDF | Persistent host KV and CPU decoding attention; asymmetric overlap; adaptive batching | Sections 3.1–3.2, pp. 4–6; implementation, p. 7; scheduling pseudocode, p. 14 |
| Project proposal, supplied 1-page PDF | Combining the two resource controls and measuring heterogeneous behavior | Objective, E1–E5, risks, deliverables |
| Project update, supplied 7-page PDF | Practical integration risks, four-arm comparison, previous setup research | pp. 4–7; numerical claims on p. 3 remain attributed paper results |

## MorphServe: precision creates KV capacity

MorphServe keeps full-precision and pre-quantized layer variants in host memory. Its LayerSwapper replaces selected device-resident full-precision layers with quantized counterparts when pressure persists. Its KVResizer attaches new cache blocks using the reclaimed device memory. It reverses these changes as pressure falls, preserving the existing decoding state. Separate CUDA streams and custom block registration/mapping support asynchronous execution (pp. 4–7 and 19).

An important distinction is explicit in Section 4.4: **KVResizer does not compress or quantize existing KV tensors.** It changes the capacity available for KV allocation. Discarding live KV entries would introduce a separate approximation that this project does not need.

The offline layer order uses local transformation similarity, local replacement similarity, and the incremental effect on model outputs. The greedy procedure recomputes model-level sensitivity conditional on layers already chosen, then reuses the resulting order online (Section 4.2 and Appendix A, pp. 5, 16–18). Our additive risk surrogate is a simpler controller input; it is not a reimplementation of the paper's full Layer Importance Score.

The paper evaluates Vicuna 7B, Llama 2 7B, Llama 3 8B and CodeLlama 34B. The first three use an L4 with 24 GB device memory and 256 GB host memory; the 34B model uses an A100 with 80 GB device memory and 2 TB host memory (p. 8). Its quality measurements primarily use F1 and ROUGE-L on summarization and reading-comprehension tasks. Its numerical gains are conditional on model, trace, task, and mode. For example, the often-cited 2.2–3.9× P95 TTFT improvement refers to accuracy mode, rather than every experiment (pp. 8–9).

Host backups consume real capacity. The paper discusses this cost, reactive behavior, layer-level granularity, and reduced additional memory benefit for models already deployed at low precision (p. 20). These limitations motivate accounting for host backups and transient memory in our planner.

## NEO: move attention with its KV state

NEO keeps dense projection/MLP computation on the GPU. For selected decoding requests, their KV remains in host memory and attention executes on the CPU. During steady-state decoding, the exchange is the current query, new key and value, and attention output. Moving the entire historical KV back and forth on every token would defeat this design. Initial placement and later migration still incur full-cache transfer costs (Section 3.1, pp. 4–5).

NEO forms complementary sub-batches: one includes prefill and GPU decoding, while another contains much of the CPU decoding work. Overlap is useful only when CPU work can fit underneath useful GPU work. The scheduler therefore compares a GPU-only schedule with an asymmetric schedule and adapts to the current queues (Section 3.2, p. 6). CPU paged-attention kernels use C++ and ISPC, and the implementation also changes CUDA launch behavior and supports tensor parallelism (p. 7).

**NEO already has adaptive load-aware scheduling.** Offline profiles supply timing estimates through interpolation; they do not make its runtime request placement static. A defensible motivation is to jointly control weight precision and attention placement under a shared capacity and transition model, rather than to claim that NEO lacks adaptation.

The paper evaluates T4, A10G and H100 configurations, CPU-capacity changes, and Llama 7B/8B/70B models. Figure 6 measures mean end-to-end latency divided by output length per request, averaged across requests. That metric is distinct from TTFT and inter-token latency. The reported fixed-latency gains in Figure 6 and maxima from synthetic workloads in Figure 9 are different measurements (pp. 8–10). NEO notes possible slowdowns at intermediate loads or short outputs, profiling inaccuracies, and CPU energy/TCO limitations (pp. 8–11).

## Corrections to the initial proposal

| Initial framing | Revised statement used in PrismServe |
|---|---|
| NEO has static offloading heuristics | NEO dynamically chooses schedules using offline timing profiles; we investigate a shared precision/placement controller. |
| MorphServe changes KV precision or evicts live KV | It reallocates cache capacity after weight downshifting. PrismServe preserves KV values and reserves capacity conservatively. |
| MorphServe never uses the host | Its attention is on the GPU, but it uses host memory for layer variants and transfers. |
| No public MorphServe code exists | The supplied paper names a repository on p. 8. A fresh check on 18 November 2026 found that the official README still announced a forthcoming full release, with only README/license files visible. This project therefore implements the mechanisms independently. |
| E1 matches original papers within ±5% | Independent simplified controls and functional tests are feasible here; full reproduction requires the original systems, hardware, models, and traces. |
| A quality-risk budget ensures a ≤1.5% task-accuracy drop | A calibrated surrogate can constrain decisions, but cannot guarantee task accuracy. Actual NLL/perplexity/logit changes are measured separately. |
| Hardware agnostic means one optimized kernel works everywhere | It means the policy depends on a resource interface rather than GPU model names. Executors and calibration remain hardware dependent. |
| The combination is the first such system | The project studies a useful combination of ideas; no exhaustive novelty claim is made. |

## Project contribution and limits

PrismServe contributes a transparent joint search policy, explicit host/device/transition accounting, portable input profiles, controlled policy comparisons, and a reference transformer that exercises actual precision and placement changes. The search is deterministic and reviewable; reinforcement learning is unnecessary for this bounded action space.

The real backend uses a tiny byte-level decoder and reference dequantization followed by floating-point matrix multiplication. It can establish correctness and actual storage savings. It cannot establish optimized INT4 speedups, production LLM quality, NEO-style asynchronous overlap, or scalability to an 8B model. The analytical simulator explores those larger capacity regimes with explicitly stated assumptions. Neither evidence class should be substituted for the other.

The original targets of 20% lower P95 TTFT, 40% fewer SLO violations, ≤1.5% accuracy degradation and <2% scheduler overhead are retained as possible future research targets. They are not acceptance criteria manufactured by selecting favorable simulated settings.

The repository-release observation above is supported by the [official MorphServe repository](https://github.com/ds2-lab/MorphServe), checked on 18 November 2026. This dated check reconciles the paper's repository reference with the update slides' earlier release-status observation.
