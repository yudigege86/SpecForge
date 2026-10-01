"""DFlash2-linear: local conv/selector on the linear-context backbone."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from types import SimpleNamespace

import torch
from transformers import Qwen3Config

from specforge.modeling.auto import AutoDraftModel, AutoDraftModelConfig
from specforge.modeling.draft.dflash2 import CandidateSelector, DFlashGroupedConv
from specforge.modeling.draft.dflash2_linear import (
    DFlash2LinearDecoderLayer,
    DFlash2LinearDraftModel,
)
from specforge.modeling.draft.dflash_linear import DFlashLinearDraftModel


LINEAR_TINY = {
    "architectures": ["DFlashLinearDraftModel"],
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
        "linear_context": {
            "variant": "gdn",
            "injection": "gated_residual",
            "num_heads": 2,
            "key_dim": 16,
            "value_dim": 16,
            "normalize_qk": True,
            "backend": "naive",
        },
    },
}

DFLASH2_FIELDS = {
    "conv_kernel_size": 2,
    "conv_group_size": 16,
    "selector_rank": 4,
    "selector_top_k": 4,
}


def _payload(architecture: str, **dflash_overrides) -> dict:
    payload = json.loads(json.dumps(LINEAR_TINY))
    payload["architectures"] = [architecture]
    if architecture == "DFlash2LinearDraftModel":
        payload["dflash_config"].update(DFLASH2_FIELDS)
    payload["dflash_config"].update(dflash_overrides)
    return payload


def _from_payload(payload: dict):
    fd, path = tempfile.mkstemp(suffix=".json")
    with os.fdopen(fd, "w") as handle:
        json.dump(payload, handle)
    try:
        config = AutoDraftModelConfig.from_file(path)
        return AutoDraftModel.from_config(config)
    finally:
        os.unlink(path)


def _tiny_config(**dflash_overrides) -> Qwen3Config:
    method_config = {
        "mask_token_id": 31,
        "target_layer_ids": [1],
        "conv_group_size": 4,
        "conv_kernel_size": 2,
        "block_size": 4,
        "selector_rank": 4,
        "selector_top_k": 3,
        "linear_context": {
            "variant": "gdn",
            "injection": "gated_residual",
            "num_heads": 2,
            "key_dim": 4,
            "value_dim": 4,
            "normalize_qk": True,
            "backend": "naive",
        },
        **dflash_overrides,
    }
    return Qwen3Config(
        architectures=["DFlash2LinearDraftModel"],
        hidden_size=16,
        intermediate_size=32,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_hidden_layers=1,
        num_target_layers=4,
        head_dim=4,
        max_position_embeddings=64,
        vocab_size=32,
        layer_types=["full_attention"],
        dflash_config=method_config,
    )


def _forward(model, *, batch=2, seq_len=16, anchors=2, block=4, hidden=64):
    noise = torch.randn(batch, anchors * block, hidden)
    target = torch.randn(batch, seq_len, hidden)
    if batch == 2:
        anchor_positions = torch.tensor([[2, 8], [4, 12]])
    else:
        anchor_positions = torch.tensor([[5]])
        if anchors != 1:
            raise ValueError("single-batch helper only supports one anchor")
    position_ids = (anchor_positions.unsqueeze(-1) + torch.arange(block)).view(
        batch, -1
    )
    return model(
        position_ids=position_ids,
        noise_embedding=noise,
        target_hidden=target,
        anchor_positions=anchor_positions,
    )


class DFlash2LinearArchitectureTest(unittest.TestCase):
    def test_builds_local_conv_and_selector_modules(self):
        model = DFlash2LinearDraftModel(_tiny_config())
        layer = model.layers[0]
        self.assertIsInstance(layer, DFlash2LinearDecoderLayer)
        self.assertIsInstance(layer.attention_conv, DFlashGroupedConv)
        self.assertIsInstance(layer.mlp_conv, DFlashGroupedConv)
        self.assertIsInstance(model.candidate_selector, CandidateSelector)
        keys = set(model.state_dict())
        self.assertIn("layers.0.attention_conv.base_kernel", keys)
        self.assertIn("layers.0.mlp_conv.kernel_projection.weight", keys)
        self.assertIn("candidate_selector.successor_codebook", keys)

    def test_post_init_keeps_conv_and_selector_identity(self):
        model = DFlash2LinearDraftModel(_tiny_config())
        for conv in (model.layers[0].attention_conv, model.layers[0].mlp_conv):
            torch.testing.assert_close(
                conv.kernel_projection.weight,
                torch.zeros_like(conv.kernel_projection.weight),
            )
        torch.testing.assert_close(
            model.candidate_selector.successor_codebook,
            torch.zeros_like(model.candidate_selector.successor_codebook),
        )

    def test_rejects_missing_dflash2_fields(self):
        for key in DFLASH2_FIELDS:
            with self.subTest(key=key):
                with self.assertRaisesRegex(ValueError, key):
                    DFlash2LinearDraftModel(_tiny_config(**{key: None}))

    def test_rejects_conv_kernel_larger_than_block(self):
        with self.assertRaisesRegex(ValueError, "must not exceed block_size"):
            DFlash2LinearDraftModel(_tiny_config(conv_kernel_size=5))

    def test_init_forward_matches_linear_on_shared_weights(self):
        linear = _from_payload(_payload("DFlashLinearDraftModel"))
        d2 = _from_payload(_payload("DFlash2LinearDraftModel"))
        self.assertIsInstance(linear, DFlashLinearDraftModel)
        self.assertIsInstance(d2, DFlash2LinearDraftModel)
        extra = set(d2.state_dict()) - set(linear.state_dict())
        self.assertTrue(any("attention_conv" in key for key in extra))
        self.assertTrue(any("mlp_conv" in key for key in extra))
        self.assertTrue(any(key.startswith("candidate_selector.") for key in extra))
        self.assertFalse(set(linear.state_dict()) - set(d2.state_dict()))

        merged = d2.state_dict()
        merged.update(linear.state_dict())
        d2.load_state_dict(merged)
        torch.testing.assert_close(
            d2.layers[0].attention_conv.kernel_projection.weight,
            torch.zeros_like(d2.layers[0].attention_conv.kernel_projection.weight),
        )
        torch.testing.assert_close(
            d2.candidate_selector.successor_codebook,
            torch.zeros_like(d2.candidate_selector.successor_codebook),
        )

        linear.eval()
        d2.eval()
        torch.manual_seed(0)
        with torch.no_grad():
            expected = _forward(linear)
        torch.manual_seed(0)
        with torch.no_grad():
            actual = _forward(d2)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_conv_wraps_packed_local_tokens_not_the_prefix_scan(self):
        model = _from_payload(_payload("DFlash2LinearDraftModel"))
        model.eval()
        layer = model.layers[0]
        prepared_shapes = []
        original_prepare = layer.attention_conv.prepare

        def capture_prepare(hidden_states):
            prepared_shapes.append(tuple(hidden_states.shape))
            return original_prepare(hidden_states)

        layer.attention_conv.prepare = capture_prepare
        with torch.no_grad():
            _forward(model)
        self.assertEqual(prepared_shapes, [(2, 8, 64)])
        self.assertFalse(hasattr(layer.context_scan, "prepare"))

    def test_resume_contract_renames_dflash2_keys(self):
        from specforge.algorithms.dflash_linear.providers import resume_contract

        draft = DFlash2LinearDraftModel(_tiny_config())
        training = SimpleNamespace(
            block_size=4,
            mask_token_id=31,
            attention_backend="sdpa",
            num_anchors=8,
            loss_decay_gamma=None,
            loss_type="dflash",
            dpace_alpha=0.5,
            lk_loss_type=None,
            kl_scale=1.0,
            kl_decay=1.0,
            selector_loss_alpha=1.0,
            selector_warmup_ratio=0.0005,
            selector_ramp_ratio=0.0005,
            selector_stop_gradient=False,
        )
        contract = resume_contract(None, draft, training)
        self.assertEqual(contract["dflash_linear_conv_kernel_size"], 2)
        self.assertEqual(contract["dflash_linear_conv_group_size"], 4)
        self.assertEqual(contract["dflash_linear_selector_rank"], 4)
        self.assertEqual(contract["dflash_linear_selector_top_k"], 3)
        self.assertEqual(contract["dflash_linear_selector_loss_alpha"], 1.0)
        self.assertFalse(contract["dflash_linear_selector_stop_gradient"])
        self.assertNotIn("dflash2_selector_rank", contract)
        self.assertEqual(contract["dflash_linear_variant"], "gdn")


if __name__ == "__main__":
    unittest.main(verbosity=2)
