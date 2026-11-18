# Real reference backend

`prismserve.torch_backend` implements a two-layer, 64-wide causal decoder with
learned positions and a 256-value UTF-8 byte tokenizer. It can train from scratch
on locally authored text without downloads. This model validates numerical and
memory mechanisms; its text quality is not evidence about pretrained LLMs.

`train_tiny(output_dir, steps=100, seed=7)` writes `tiny_checkpoint.pt` and
`training.json`. `run_quality(output_dir, checkpoint=None)` writes held-out
perplexity, logit KL, actual storage bytes and restoration error to `quality.json`.
`demo(output_dir, steps=60)` trains, evaluates and benchmarks CPU plus CUDA when
available, writing `real_backend.json`. Checkpoints use `torch.load(weights_only=True)`.

`TinyDecoder.set_quantized_layers((0,))` packs all linear matrices inside the
selected layer to actual 4-bit nibbles with FP32 scales, releases their resident
FP32 parameters, and retains FP32 master copies on the CPU for exact restoration.
The embedding and output head stay FP32. `memory_report()` separates resident
model storage, host master backups and the largest per-matrix dequantization
scratch. The forward path dequantizes a matrix for a normal floating-point
matmul: there is no fused INT4 kernel and no promise of INT4 speedup. CUDA's
allocator may retain freed blocks; tensor storage and allocator reservation differ.

`set_cpu_attention(True)` executes decode attention with the KV history on CPU
while keeping projections and MLP on the model device. Prefill runs on the model
device and its resulting cache is copied to the CPU. Subsequent decode steps move
only queries, new keys/values and attention output. Existing KV tensors are
preserved when precision changes. Restoring full precision does not recompute the
history created while a layer was quantized. New cache tensors are allocated by
concatenation, so temporary cache duplication remains a reference-backend cost.
The pipeline is synchronous. It does not reproduce NEO's asynchronous asymmetric
sub-batch overlap or MorphServe's preallocated in-place layer swapping.

`TinyDecoder.apply_plan(plan, request_id)` applies a shared controller `Plan` at a
token boundary. `generate(model, prompt, max_tokens=32, callback=None)` returns
generated token IDs and synchronized prefill/decode timings. Its optional
callback receives `(model, cache, output_index)` before each decode step. This is
the integration point for policy or pressure-replay experiments. Precision and
placement remain unchanged during a single forward call.

`prismserve.service.create_app(checkpoint=None, device='auto')` exposes `/health`,
`/v1/models`, `/v1/completions` and `/metrics`. A lock serializes requests and
precision transitions. Requests choose `quantized_layers` and `cpu_attention`
explicitly. This API is a bounded systems demo, without streaming, continuous
batching, production latency guarantees or a pretrained LLM. Use UTF-8 byte token
counts when checking the 256-token context capacity. An omitted checkpoint uses
deterministic random weights and `/health` reports this explicitly.
