# PrismServe — Project Summary

We built **PrismServe**, a working research prototype that investigates:

> **Can one hardware-agnostic controller coordinate quantization and CPU offloading to serve language models under memory constraints?**

## 1. Research and approach

- Studied **MorphServe's runtime precision changes** and **NEO's CPU attention and KV placement**.
- Designed an independent implementation driven by hardware capabilities, available memory and measured execution costs.
- Hardware-agnostic refers to the controller interface; actual validation covers **CPU and CUDA on one laptop**.

## 2. Real pretrained-model execution

- Runs **GPT-2 with approximately 124 million parameters** using its original tokenizer and pretrained weights.
- Supports **reversible, layer-wise packed INT4 quantization** of selected block projections.
- Can move selected decode requests' **KV cache and attention computation to CPU**.
- Checked numerical correctness against the original model implementation, cached decoding and CPU/device placement.

## 3. Concurrent serving

- Multiple requests share dense model computation through **continuous batching**.
- Each request retains its own KV history and output timestamps.
- Added **chunked prefill, request queues and a concurrent HTTP API**.
- One model worker owns global precision and per-request placement.

## 4. Adaptive controller

- Measures hardware performance and monitors available memory.
- Chooses **layer precision, attention placement and request admission**.
- Accounts for future KV growth, workspace, host backups and transition-memory peaks.
- Reduces scheduling work using cached calculations and stable decision reuse, with fresh resource checks every iteration.
- Records diagnostic information when active execution becomes infeasible.

## 5. Experiments and verification

- Compared **FP32, quantization-only, offload-only and joint control**.
- Evaluated CPU/CUDA execution, prediction error, latency, throughput, memory and quality.
- Ran a final **216-experiment capacity sweep** across memory quotas and prompt lengths.
- Passed **116 tests**, including numerical, resource-accounting, batching, API and experiment-protocol checks.
- Preserved negative results and labeled unsuccessful or diagnostic attempts separately.

## 6. Measured capacity benefit and quality tradeoff

For four requests with **768-token prompts**, **16 output tokens** and a **675 MiB serving quota**:

| Approach | Completed | Rejected |
|---|---:|---:|
| FP32 | 0 | 4 |
| Quantization only | 4 | 0 |
| CPU attention only | 0 | 4 |
| Joint control | 4 | 0 |

These completion outcomes hold across **three paired repeats**. FP32 and offload-only need a lowest tested successful quota of **700 MiB**, while quantization-only and joint control succeed at **675 MiB**.

The corresponding joint quality audit shows **2.752% higher perplexity on 96 held-out targets**. This demonstrates a measured **capacity/quality tradeoff**. Joint control does not improve the sampled capacity boundary over quantization alone. The quota is an enforced warm-start serving budget, not the GPU's total physical memory or an exact hardware minimum.

## 7. Submission and documentation

- **Version 1.0.0** source freeze, provenance records and validated experiment artifacts.
- **16-page concise report** and **22-slide presentation**.
- **78-page detailed report** with **47 figures, 74 tables**, architecture diagrams, flowcharts and complete tables for 454 main-suite run records.
- Editable Word report and a **47-page graph and diagram atlas**.
- A **one-command live demo**, architecture walkthrough and viva guide.

## Conclusion

PrismServe demonstrates a common controller working on CPU and CUDA on one laptop, with real pretrained-model execution and capacity benefits in constrained cases. **Optimized FP32 remains a strong—and often faster—choice when sufficient memory is available.** The reference INT4 implementation unpacks weights for floating-point computation, CPU attention is synchronous, and broader hardware/model/task validation remains future work.

## Explore or run

- [Full README: setup, architecture and commands](README.md)
- [Detailed technical report](deliverables/PrismServe_Detailed_Report.pdf)
- [Measured capacity results](results/capacity_frontier/report.md)
- [Demo and viva guide](docs/demo_and_viva.md)

```powershell
# Read saved validated evidence; no pretrained weights required
python scripts/demo_project.py --evidence-only

# Live capacity and concurrent HTTP demo on the prepared CUDA environment
python scripts/demo_project.py
```
