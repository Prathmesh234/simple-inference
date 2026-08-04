"""CPU correctness tests for the Section 16 paged KV-cache reference path.

Run:
    uv run python -m unittest serving.test_paged_kv_cache -v
"""

from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F

from config import ModelConfig, RopeScalingConfig
from iterations.inference_01_contiguous_eager.engine import (
    InferenceEngine as ContiguousInferenceEngine,
)
from model.llama import LlamaModel
from serving.block_allocator import BlockAllocator
from serving.engine import InferenceEngine
from serving.paged_decoder import PagedDecodeBuffers, PagedDecoder
from serving.paged_kv_cache import PagedKVCache, paged_attention_forward
from serving.radix_cache import RadixCache


class BlockAllocatorTest(unittest.TestCase):
    def test_reservation_allocation_and_reuse(self) -> None:
        allocator = BlockAllocator(num_blocks=4)
        allocator.reserve(request_id=1, block_count=2)
        allocator.reserve(request_id=2, block_count=1)
        self.assertFalse(allocator.can_reserve(request_id=3, block_count=2))

        first = allocator.allocate(1)
        second = allocator.allocate(1)
        self.assertEqual((first, second), (0, 1))
        allocator.incref(2, first)
        self.assertEqual(allocator.ref_count(first), 2)

        allocator.release_request(1)
        self.assertEqual(allocator.ref_count(first), 1)
        self.assertEqual(allocator.free_blocks, 3)
        allocator.release_request(2)
        self.assertEqual(allocator.free_blocks, 4)


class PagedKVCacheTest(unittest.TestCase):
    def setUp(self) -> None:
        self.cache = PagedKVCache(
            n_layers=2,
            num_blocks=4,
            block_size=2,
            n_heads_kv=1,
            head_dim=2,
            dtype=torch.float32,
            device="cpu",
        )

    def test_append_across_blocks_and_gather_logical_order(self) -> None:
        self.cache.reserve_request(request_id=10, max_tokens=5)
        k = torch.tensor([[[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]])
        v = k + 100
        self.cache.append(layer_idx=0, request_id=10, start_pos=0, k=k, v=v)

        # Positions 0..2 occupy two physical pages, but gather restores them
        # to one logical sequence ordered by absolute token position.
        self.assertEqual(len(self.cache.block_tables[10]), 2)
        got_k, got_v = self.cache.gather(layer_idx=0, request_id=10)
        torch.testing.assert_close(got_k, k)
        torch.testing.assert_close(got_v, v)

        tail_k = torch.tensor([[[4.0, 40.0], [5.0, 50.0]]])
        self.cache.append(layer_idx=0, request_id=10, start_pos=3, k=tail_k, v=tail_k + 100)
        got_k, _ = self.cache.gather(layer_idx=0, request_id=10)
        torch.testing.assert_close(got_k, torch.cat((k, tail_k), dim=1))

    def test_release_returns_blocks_and_drops_table(self) -> None:
        self.cache.reserve_request(request_id=10, max_tokens=4)
        values = torch.ones(1, 3, 2)
        self.cache.append(layer_idx=0, request_id=10, start_pos=0, k=values, v=values)
        self.assertEqual(self.cache.allocator.free_blocks, 2)

        self.cache.release_request(10)
        self.assertNotIn(10, self.cache.block_tables)
        self.assertEqual(self.cache.allocator.free_blocks, 4)

        self.cache.reserve_request(request_id=11, max_tokens=2)
        self.assertEqual(self.cache.allocator.free_blocks, 4)
        self.cache.append(layer_idx=0, request_id=11, start_pos=0, k=values[:, :2], v=values[:, :2])
        self.assertEqual(self.cache.block_tables[11], [0])


class RadixCacheTest(unittest.TestCase):
    def test_match_shares_complete_pages_and_cache_clear_is_refcount_safe(self) -> None:
        cache = PagedKVCache(
            n_layers=1,
            num_blocks=6,
            block_size=2,
            n_heads_kv=1,
            head_dim=2,
            dtype=torch.float32,
            device="cpu",
        )
        radix = RadixCache(cache, max_blocks=4)
        cache.reserve_request(request_id=1, max_tokens=4)
        values = torch.arange(8, dtype=torch.float32).view(1, 4, 2)
        cache.append(0, 1, 0, values, values)
        radix.insert([1, 2, 3, 4], cache.block_tables[1])

        cache.release_request(1)
        self.assertEqual(cache.allocator.free_blocks, 4)

        match = radix.match([1, 2, 3, 4, 5])
        self.assertEqual(match.token_count, 4)
        self.assertEqual(len(match.block_ids), 2)
        cache.reserve_request(
            request_id=2,
            max_tokens=6,
            shared_block_ids=match.block_ids,
            cached_tokens=match.token_count,
        )

        radix.clear()
        self.assertEqual(cache.allocator.free_blocks, 4)
        cache.release_request(2)
        self.assertEqual(cache.allocator.free_blocks, 6)

    def test_newer_path_evicts_old_prefix_and_forces_a_miss(self) -> None:
        cache = PagedKVCache(
            n_layers=1,
            num_blocks=4,
            block_size=2,
            n_heads_kv=1,
            head_dim=2,
            dtype=torch.float32,
            device="cpu",
        )
        radix = RadixCache(cache, max_blocks=2)

        old_tokens = [1, 2, 3, 4]
        cache.reserve_request(request_id=1, max_tokens=4)
        old_values = torch.arange(8, dtype=torch.float32).view(1, 4, 2)
        cache.append(0, 1, 0, old_values, old_values)
        radix.insert(old_tokens, cache.block_tables[1])
        cache.release_request(1)

        new_tokens = [9, 9, 8, 8]
        cache.reserve_request(request_id=2, max_tokens=4)
        new_values = old_values + 100
        cache.append(0, 2, 0, new_values, new_values)
        radix.insert(new_tokens, cache.block_tables[2])
        cache.release_request(2)

        self.assertEqual(radix.match(old_tokens + [5]).token_count, 0)
        self.assertEqual(radix.match(new_tokens + [7]).token_count, 4)
        self.assertEqual(radix.stats()["evictions"], 2)
        self.assertEqual(cache.allocator.free_blocks, 2)


class PagedAttentionTest(unittest.TestCase):
    def test_masked_gathered_attention_matches_per_request_sdpa(self) -> None:
        torch.manual_seed(0)
        q = torch.randn(2, 2, 1, 2)
        k = torch.randn(2, 1, 3, 2)
        v = torch.randn(2, 1, 3, 2)
        positions = torch.tensor([[1], [2]])
        lengths = torch.tensor([2, 3])

        actual = paged_attention_forward(q, k, v, positions, lengths, num_kv_groups=2)
        expected = torch.cat(
            [
                F.scaled_dot_product_attention(
                    q[row:row + 1], k[row:row + 1, :, :length], v[row:row + 1, :, :length],
                    enable_gqa=True,
                )
                for row, length in enumerate(lengths.tolist())
            ],
            dim=0,
        )
        torch.testing.assert_close(actual, expected)


class PagedEngineTest(unittest.TestCase):
    @staticmethod
    def _tiny_model() -> LlamaModel:
        cfg = ModelConfig(
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            vocab_size=32,
            num_attention_heads=2,
            num_key_value_heads=1,
            rms_norm_eps=1e-5,
            rope_theta=10_000.0,
            max_position_embeddings=16,
            rope_scaling=RopeScalingConfig(
                rope_type="default",
                factor=1.0,
                low_freq_factor=1.0,
                high_freq_factor=4.0,
                original_max_position_embeddings=16,
            ),
            torch_dtype="float32",
        )
        torch.manual_seed(1)
        model = LlamaModel(cfg, torch.device("cpu"))
        for parameter in model.parameters():
            torch.nn.init.normal_(parameter, mean=0.0, std=0.02)
        return model.eval()

    def test_paged_engine_matches_contiguous_engine_and_releases_pages(self) -> None:
        model = self._tiny_model()
        common = dict(
            model=model,
            max_running=2,
            max_seq_len=8,
            temperature=0.0,
            warmup=False,
            use_cuda_graphs=False,
        )
        paged = InferenceEngine(
            **common,
            block_size=2,
            num_kv_blocks=8,
        )
        contiguous = ContiguousInferenceEngine(
            model=model,
            max_running=2,
            max_seq_len=8,
            temperature=0.0,
            warmup=False,
        )
        prompts = ([1, 4, 6], [1, 5])

        paged_reqs = [paged.add_request(list(prompt), max_new_tokens=3) for prompt in prompts]
        contiguous_reqs = [
            contiguous.add_request(list(prompt), max_new_tokens=3) for prompt in prompts
        ]
        paged.run()
        contiguous.run()

        self.assertEqual(
            [request.generated for request in paged_reqs],
            [request.generated for request in contiguous_reqs],
        )
        self.assertEqual(paged.kv.allocator.free_blocks, paged.kv.num_blocks)

    def test_capacity_blocked_request_waits_until_pages_are_released(self) -> None:
        engine = InferenceEngine(
            model=self._tiny_model(),
            max_running=2,
            max_seq_len=4,
            block_size=2,
            num_kv_blocks=2,
            eos_id=-1,
            temperature=0.0,
            warmup=False,
            use_cuda_graphs=False,
        )
        first = engine.add_request([1, 4, 6], max_new_tokens=1)
        second = engine.add_request([1, 5, 7], max_new_tokens=1)

        engine.step()
        self.assertEqual(first.state.name, "FINISHED")
        self.assertEqual(second.state.name, "WAITING")
        self.assertEqual(list(engine.scheduler.waiting), [second])

        engine.step()
        self.assertNotIn(second, engine.scheduler.waiting)
        self.assertEqual(second.state.name, "FINISHED")

    def test_prefix_cache_reuses_pages_and_matches_uncached_output(self) -> None:
        model = self._tiny_model()
        common = dict(
            model=model,
            max_running=1,
            max_seq_len=8,
            block_size=2,
            num_kv_blocks=8,
            eos_id=-1,
            temperature=0.0,
            warmup=False,
            use_cuda_graphs=False,
        )
        cached = InferenceEngine(
            **common,
            use_prefix_cache=True,
            prefix_cache_blocks=4,
        )
        baseline = InferenceEngine(**common)

        cached.add_request([1, 4, 6, 8, 9], max_new_tokens=2)
        cached.run()
        reused = cached.add_request([1, 4, 6, 8, 10], max_new_tokens=2)
        cached.run()

        expected = baseline.add_request([1, 4, 6, 8, 10], max_new_tokens=2)
        baseline.run()

        self.assertEqual(reused.cached_prefix_len, 4)
        self.assertEqual(reused.generated, expected.generated)
        self.assertEqual(cached.prefix_cache.stats()["matched_tokens"], 4)

    def test_graph_metadata_updates_in_place_and_pads_with_scratch(self) -> None:
        model = self._tiny_model()
        cache = PagedKVCache(
            n_layers=1,
            num_blocks=4,
            block_size=2,
            n_heads_kv=1,
            head_dim=4,
            num_scratch_blocks=1,
            dtype=torch.float32,
            device="cpu",
        )
        cache.reserve_request(request_id=10, max_tokens=4)
        values = torch.ones(1, 3, 4)
        cache.append(0, 10, 0, values, values)
        decoder = PagedDecoder(
            model,
            cache,
            max_running=2,
            max_seq_len=8,
            n_heads_q=2,
            n_heads_kv=1,
            head_dim=4,
            use_triton=True,
            use_cuda_graphs=True,
        )
        bufs = PagedDecodeBuffers(
            batch_size=2,
            max_blocks=decoder.max_blocks,
            head_dim=4,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )
        table_ptr = bufs.block_tables.data_ptr()
        decoder._fill(
            bufs,
            batch_size=2,
            request_ids=[10],
            positions=[3],
            last_tokens=[7],
        )

        self.assertEqual(bufs.block_tables.data_ptr(), table_ptr)
        self.assertEqual(bufs.block_tables[0].tolist(), [0, 1, -1, -1])
        self.assertEqual(bufs.block_tables[1].tolist(), [4, -1, -1, -1])
        self.assertEqual(bufs.positions.tolist(), [3, 0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
