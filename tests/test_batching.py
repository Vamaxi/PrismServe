"""Offline numerical and call-count evidence for true selective batching."""
import importlib.util
import os
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
import unittest

AVAILABLE = all(importlib.util.find_spec(name) is not None for name in ("torch", "transformers"))


class MockTokenizer:
    eos_token_id = 63
    def encode(self, text, **kwargs):
        return [ord(c) % 63 for c in text]
    def decode(self, ids, **kwargs):
        return " ".join(map(str, ids))


@unittest.skipUnless(AVAILABLE, "install the pretrained extra")
class BatchingTests(unittest.TestCase):
    def setUp(self):
        import torch
        from transformers import GPT2Config, GPT2LMHeadModel
        from prismserve.pretrained import GPT2Executor
        torch.manual_seed(73)
        torch.set_num_threads(min(2, torch.get_num_threads()))
        config = GPT2Config(vocab_size=64, n_positions=32, n_embd=32, n_layer=2,
                            n_head=4, resid_pdrop=0, embd_pdrop=0, attn_pdrop=0)
        config._attn_implementation = "eager"
        self.executor = GPT2Executor(GPT2LMHeadModel(config).eval(), MockTokenizer())

    def assert_matches_independent(self, inputs, result):
        import torch
        self.assertEqual(len(inputs), len(result.outputs))
        for request, output in zip(inputs, result.outputs):
            self.executor.set_cpu_attention(request.cpu_attention)
            with torch.inference_mode():
                reference, cache = self.executor(torch.tensor([request.token_ids], device=self.executor.device), request.cache)
            self.assertEqual(output.request_id, request.request_id)
            self.assertTrue(torch.allclose(output.logits, reference[:, -1], atol=2e-6, rtol=2e-5),
                            float((output.logits - reference[:, -1]).abs().max()))
            self.assertEqual(output.next_token_id, int(reference[0, -1].argmax()))
            for (actual_k, actual_v), (expected_k, expected_v) in zip(output.cache, cache):
                self.assertTrue(torch.allclose(actual_k, expected_k, atol=2e-6, rtol=2e-5))
                self.assertTrue(torch.allclose(actual_v, expected_v, atol=2e-6, rtol=2e-5))

    def test_unequal_initial_prefill_matches_independent(self):
        from prismserve.batching import BatchInput, execute_batch
        requests = [BatchInput("a", [1, 3, 5], prefill=True),
                    BatchInput("b", [7, 9, 11, 13, 15], prefill=True),
                    BatchInput("c", [22], prefill=True)]
        result = execute_batch(self.executor, requests)
        self.assert_matches_independent(requests, result)
        self.assertEqual(result.linear_tokens, 9)
        self.assertEqual(result.projection_calls, 8)

    def test_mixed_cached_chunk_prefill_decode_and_new_request(self):
        from prismserve.batching import BatchInput, execute_batch
        initial = execute_batch(self.executor, [BatchInput("long", [1, 2, 3], prefill=True),
                                                BatchInput("decode", [4, 5, 6, 7], prefill=True)])
        requests = [BatchInput("long", [8, 9], initial.outputs[0].cache, prefill=True),
                    BatchInput("decode", [10], initial.outputs[1].cache),
                    BatchInput("new", [11, 12, 13, 14], prefill=True)]
        result = execute_batch(self.executor, requests)
        self.assert_matches_independent(requests, result)
        self.assertEqual([r.cache[0][0].shape[2] for r in result.outputs], [5, 5, 4])

    def test_actual_projection_batch_shapes_and_call_count(self):
        from prismserve.batching import BatchInput, execute_batch
        shapes = []
        handles = []
        for block in self.executor.layers:
            for projection in (block.attn.c_attn, block.attn.c_proj, block.mlp.c_fc, block.mlp.c_proj):
                handles.append(projection.register_forward_pre_hook(lambda module, args: shapes.append(tuple(args[0].shape))))
        head_shapes = []
        handles.append(self.executor.model.lm_head.register_forward_pre_hook(lambda module, args: head_shapes.append(tuple(args[0].shape))))
        try:
            result = execute_batch(self.executor, [BatchInput("one", [1, 2]),
                                                   BatchInput("two", [3, 4, 5]),
                                                   BatchInput("three", [6])])
        finally:
            for handle in handles:
                handle.remove()
        self.assertEqual(result.projection_calls, 8)
        self.assertEqual(len(shapes), 8)
        self.assertTrue(all(len(shape) == 2 and shape[0] == 6 for shape in shapes))
        self.assertEqual(head_shapes, [(3, 32)])

    def test_request_isolation_and_cache_storage_ownership(self):
        import torch
        from prismserve.batching import BatchInput, execute_batch
        before = execute_batch(self.executor, [BatchInput("one", [1, 2, 3]), BatchInput("two", [4, 5])])
        after = execute_batch(self.executor, [BatchInput("one", [1, 2, 3]), BatchInput("two", [50, 51])])
        self.assertTrue(torch.equal(before.outputs[0].logits, after.outputs[0].logits))
        for output in before.outputs:
            for key, value in output.cache:
                self.assertEqual(key.untyped_storage().nbytes(), key.numel() * key.element_size())
                self.assertEqual(value.untyped_storage().nbytes(), value.numel() * value.element_size())
        self.assertNotEqual(before.outputs[0].cache[0][0].untyped_storage().data_ptr(),
                            before.outputs[1].cache[0][0].untyped_storage().data_ptr())

    def test_future_prefill_tokens_cannot_change_cached_prefix(self):
        import torch
        from prismserve.batching import BatchInput, execute_batch
        before = execute_batch(self.executor, [BatchInput("a", [1, 2, 3, 4], prefill=True),
                                               BatchInput("b", [5, 6], prefill=True)])
        after = execute_batch(self.executor, [BatchInput("a", [1, 2, 3, 40], prefill=True),
                                              BatchInput("b", [5, 6], prefill=True)])
        for (key, value), (changed_key, changed_value) in zip(before.outputs[0].cache, after.outputs[0].cache):
            self.assertTrue(torch.equal(key[:, :, :-1], changed_key[:, :, :-1]))
            self.assertTrue(torch.equal(value[:, :, :-1], changed_value[:, :, :-1]))
        self.assertTrue(torch.equal(before.outputs[1].logits, after.outputs[1].logits))

    def test_precision_change_preserves_input_caches_and_restore(self):
        import torch
        from prismserve.batching import BatchInput, execute_batch
        initial_inputs = [BatchInput("a", [1, 3, 5]), BatchInput("b", [7, 9])]
        initial = execute_batch(self.executor, initial_inputs)
        snapshot = [[(key.clone(), value.clone()) for key, value in output.cache] for output in initial.outputs]
        decode = [BatchInput(output.request_id, [output.next_token_id], output.cache) for output in initial.outputs]
        self.executor.set_quantized_layers((0,))
        changed = execute_batch(self.executor, decode)
        self.assert_matches_independent(decode, changed)
        for original, copies in zip(initial.outputs, snapshot):
            for (key, value), (old_key, old_value) in zip(original.cache, copies):
                self.assertTrue(torch.equal(key, old_key))
                self.assertTrue(torch.equal(value, old_value))
        self.executor.set_quantized_layers(())
        restored = execute_batch(self.executor, initial_inputs)
        for before, after in zip(initial.outputs, restored.outputs):
            self.assertTrue(torch.equal(before.logits, after.logits))

    def test_invalid_ids_context_and_cache_are_rejected(self):
        import torch
        from prismserve.batching import BatchInput, execute_batch
        bad = [[BatchInput("a", [1]), BatchInput("a", [2])], [BatchInput("", [1])],
               [BatchInput("a", [])], [BatchInput("a", [True])], [BatchInput("a", [-1])],
               [BatchInput("a", [64])], [BatchInput("a", [1] * 33)],
               [BatchInput("a", [1], cache=[])],
               [BatchInput("a", [1], cache=[(torch.zeros(1), torch.zeros(1))] * 2)]]
        for requests in bad:
            with self.subTest(requests=requests):
                with self.assertRaises(ValueError):
                    execute_batch(self.executor, requests)
        self.assertEqual(execute_batch(self.executor, []).projection_calls, 0)

    def test_executor_global_placement_is_not_mutated(self):
        from prismserve.batching import BatchInput, execute_batch
        self.executor.set_cpu_attention(True)
        execute_batch(self.executor, [BatchInput("device", [1, 2], cpu_attention=False)])
        self.assertTrue(self.executor.cpu_attention)

    def test_cuda_mixed_request_placement_and_prefill_chunk(self):
        import torch
        from prismserve.batching import BatchInput, execute_batch
        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")
        self.executor.cuda()
        initial = execute_batch(self.executor, [BatchInput("gpu", [1, 3], prefill=True),
                                                BatchInput("cpu", [7, 9, 11], cpu_attention=True, prefill=True)])
        self.assertEqual(initial.outputs[0].cache[0][0].device.type, "cuda")
        self.assertEqual(initial.outputs[1].cache[0][0].device.type, "cpu")
        requests = [BatchInput("gpu", [5], initial.outputs[0].cache),
                    BatchInput("cpu", [13], initial.outputs[1].cache, cpu_attention=True)]
        result = execute_batch(self.executor, requests)
        self.assert_matches_independent(requests, result)
        # A CPU-resident request can next perform a prefill chunk on primary CUDA.
        chunks = [BatchInput("cpu", [15, 17], result.outputs[1].cache, cpu_attention=True, prefill=True)]
        chunk_result = execute_batch(self.executor, chunks)
        self.assertEqual(chunk_result.outputs[0].cache[0][0].device.type, "cpu")
        self.assertEqual(chunk_result.outputs[0].cache[0][0].shape[2], 6)


if __name__ == "__main__":
    unittest.main()
