"""Teacher-forced MAL on stock DFlashDraftModel."""

from __future__ import annotations

import json
import os
import tempfile
import unittest

import torch
from transformers import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

from specforge.modeling.auto import AutoDraftModel, AutoDraftModelConfig
from specforge.modeling.draft.dflash import DFlashDraftModel, target_input_embeddings


TINY = {
    "architectures": ["DFlashDraftModel"],
    "model_type": "qwen3",
    "block_size": 4,
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "num_hidden_layers": 1,
    "num_target_layers": 4,
    "head_dim": 16,
    "max_position_embeddings": 512,
    "vocab_size": 256,
    "layer_types": ["full_attention"],
    "dflash_config": {
        "mask_token_id": 0,
        "target_layer_ids": [1],
    },
}


def _tiny_stock():
    fd, path = tempfile.mkstemp(suffix=".json")
    with os.fdopen(fd, "w") as handle:
        json.dump(TINY, handle)
    try:
        config = AutoDraftModelConfig.from_file(path)
        return AutoDraftModel.from_config(config)
    finally:
        os.unlink(path)


def _tiny_target():
    config = Qwen3Config(
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_hidden_layers=2,
        head_dim=16,
        max_position_embeddings=128,
        vocab_size=256,
        tie_word_embeddings=False,
    )
    config._attn_implementation = "sdpa"
    return Qwen3ForCausalLM(config).eval()


class DFlashAcceptanceTest(unittest.TestCase):
    def test_stock_acceptance_along_sequence_does_not_use_kv_cache(self):
        torch.manual_seed(4)
        target = _tiny_target()
        model = _tiny_stock()
        model.eval()
        self.assertIsInstance(model, DFlashDraftModel)
        sequence_ids = torch.randint(1, 256, (1, 14))
        lengths = model.acceptance_along_sequence(
            target, sequence_ids, prompt_len=6, temperature=0.0
        )
        self.assertTrue(lengths)
        self.assertTrue(all(length >= 1 for length in lengths))
        self.assertEqual(sum(lengths), sequence_ids.shape[1] - 6)

    def test_stock_acceptance_masks_future_block_tokens(self):
        torch.manual_seed(5)
        target = _tiny_target()
        model = _tiny_stock()
        model.eval()
        sequence_ids = torch.randint(1, 256, (1, 14))
        captured = []
        original = model._teacher_force_block

        def wrapped(**kwargs):
            captured.append(kwargs["noise_embedding"].detach().clone())
            return original(**kwargs)

        model._teacher_force_block = wrapped
        model.acceptance_along_sequence(
            target, sequence_ids, prompt_len=6, temperature=0.0
        )
        self.assertTrue(captured)
        embed = target_input_embeddings(target)
        expected_ids = torch.full(
            (1, TINY["block_size"]),
            model.mask_token_id,
            dtype=sequence_ids.dtype,
        )
        expected_ids[:, 0] = sequence_ids[:, 6]
        torch.testing.assert_close(captured[0], embed(expected_ids))
