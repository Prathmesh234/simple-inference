"""Correctness smoke test for the paged CUDA-graph milestone."""

from __future__ import annotations

import unittest

import torch

from config import ModelConfig, RopeScalingConfig
from model.llama import LlamaModel

from .engine import InferenceEngine
from .paged_cuda_graph import PagedDecodeBuffers, PagedGraphDecoder


def tiny_model(device: torch.device) -> LlamaModel:
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
    model = LlamaModel(cfg, device)
    for parameter in model.parameters():
        torch.nn.init.normal_(parameter, mean=0.0, std=0.02)
    return model.eval()


def build_engine(
    device: torch.device,
    *,
    use_cuda_graphs: bool = False,
    warmup: bool = False,
) -> InferenceEngine:
    return InferenceEngine(
        model=tiny_model(device),
        max_running=4,
        max_seq_len=16,
        block_size=2,
        num_kv_blocks=32,
        eos_id=-1,
        temperature=0.0,
        warmup=warmup,
        use_cuda_graphs=use_cuda_graphs,
        use_paged_attention=True,
    )


class EngineSmokeTest(unittest.TestCase):
    def test_graph_metadata_buffer_keeps_its_address(self) -> None:
        buffers = PagedDecodeBuffers(
            B=2,
            max_blocks=4,
            head_dim=4,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )
        address = buffers.block_tables.data_ptr()
        buffers.block_tables.copy_(torch.arange(8).view(2, 4))

        self.assertEqual(buffers.block_tables.data_ptr(), address)
        self.assertEqual(PagedGraphDecoder._make_buckets(7), [1, 2, 4, 7])

    def test_eager_fallback_finishes_and_releases_pages(self) -> None:
        engine = build_engine(torch.device("cpu"))
        requests = [
            engine.add_request([1, 4, 6], max_new_tokens=4),
            engine.add_request([1, 5], max_new_tokens=4),
        ]

        finished = engine.run()

        self.assertEqual(set(finished), {request.id for request in requests})
        self.assertEqual([len(request.generated) for request in requests], [4, 4])
        self.assertEqual(engine.kv.allocator.free_blocks, engine.kv.num_blocks)


if __name__ == "__main__":
    unittest.main(verbosity=2)
