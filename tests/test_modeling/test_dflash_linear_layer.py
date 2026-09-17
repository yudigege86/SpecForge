"""Context-first GDN layer tests for linear-context DFlash."""

from __future__ import annotations

import json
import os
import tempfile
import unittest

import torch

from specforge.modeling.auto import AutoDraftModel, AutoDraftModelConfig
from specforge.modeling.draft.dflash_linear import DFlashLinearDecoderLayer


TINY = {
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


def _tiny_model(**linear_context):
    payload = json.loads(json.dumps(TINY))
    payload["dflash_config"]["linear_context"].update(linear_context)
    fd, path = tempfile.mkstemp(suffix=".json")
    with os.fdopen(fd, "w") as handle:
        json.dump(payload, handle)
    try:
        config = AutoDraftModelConfig.from_file(path)
        return AutoDraftModel.from_config(config)
    finally:
        os.unlink(path)


class DFlashLinearLayerTest(unittest.TestCase):
    def test_from_config_builds_linear_decoder_layers(self):
        model = _tiny_model()
        self.assertIsInstance(model.layers[0], DFlashLinearDecoderLayer)
        self.assertEqual(model.layers[0].settings["variant"], "gdn")
        self.assertEqual(model.layers[0].settings["backend"], "naive")
        self.assertEqual(model.layers[0].horizon_embed.num_embeddings, TINY["block_size"])
        self.assertEqual(model.layers[0].horizon_embed.embedding_dim, TINY["hidden_size"])
        from specforge.modeling.draft.linear_context import LOG_DECAY_BIAS_INIT

        torch.testing.assert_close(
            model.layers[0].context_scan.log_decay_bias.detach().cpu(),
            torch.full_like(
                model.layers[0].context_scan.log_decay_bias.detach().cpu(),
                LOG_DECAY_BIAS_INIT,
            ),
        )

    def test_forward_shape_is_packed_blocks_not_context_plus_block(self):
        model = _tiny_model()
        model.eval()
        batch, seq_len, anchors, block = 2, 16, 2, 4
        noise = torch.randn(batch, anchors * block, 64)
        target = torch.randn(batch, seq_len, 64)
        anchor_positions = torch.tensor([[2, 8], [4, 12]])
        offsets = torch.arange(block).view(1, 1, -1)
        position_ids = (anchor_positions.unsqueeze(-1) + offsets).view(batch, -1)
        out = model(
            position_ids=position_ids,
            noise_embedding=noise,
            target_hidden=target,
            anchor_positions=anchor_positions,
        )
        self.assertEqual(tuple(out.shape), (batch, anchors * block, 64))

    def test_blocks_are_independent(self):
        model = _tiny_model()
        model.eval()
        torch.manual_seed(0)
        batch, seq_len, anchors, block = 1, 16, 2, 4
        noise = torch.randn(batch, anchors * block, 64)
        target = torch.randn(batch, seq_len, 64)
        anchor_positions = torch.tensor([[3, 10]])
        offsets = torch.arange(block).view(1, 1, -1)
        position_ids = (anchor_positions.unsqueeze(-1) + offsets).view(batch, -1)
        keep = torch.tensor([[True, True]])
        with torch.no_grad():
            baseline = model(
                position_ids=position_ids,
                noise_embedding=noise,
                target_hidden=target,
                anchor_positions=anchor_positions,
                block_keep_mask=keep,
            )
            leaked = noise.clone()
            leaked[:, block:] = torch.randn_like(leaked[:, block:])
            perturbed = model(
                position_ids=position_ids,
                noise_embedding=leaked,
                target_hidden=target,
                anchor_positions=anchor_positions,
                block_keep_mask=keep,
            )
        torch.testing.assert_close(
            perturbed[:, :block],
            baseline[:, :block],
            atol=0,
            rtol=0,
        )
        self.assertFalse(torch.equal(perturbed[:, block:], baseline[:, block:]))

    def test_future_target_features_do_not_change_prefix_block(self):
        model = _tiny_model()
        model.eval()
        torch.manual_seed(1)
        noise = torch.randn(1, 4, 64)
        target = torch.randn(1, 16, 64)
        anchors = torch.tensor([[5]])
        position_ids = (anchors.unsqueeze(-1) + torch.arange(4)).view(1, -1)
        with torch.no_grad():
            baseline = model(
                position_ids=position_ids,
                noise_embedding=noise,
                target_hidden=target,
                anchor_positions=anchors,
            )
            leaked = target.clone()
            leaked[:, 5:] = torch.randn_like(leaked[:, 5:])
            perturbed = model(
                position_ids=position_ids,
                noise_embedding=noise,
                target_hidden=leaked,
                anchor_positions=anchors,
            )
        torch.testing.assert_close(perturbed, baseline, atol=0, rtol=0)

    def test_spec_generate_is_deferred(self):
        model = _tiny_model()
        with self.assertRaises(NotImplementedError):
            model.spec_generate()

    def test_kda_uses_channelwise_decay_and_forwards(self):
        model = _tiny_model(variant="kda")
        layer = model.layers[0]
        self.assertEqual(layer.settings["variant"], "kda")
        self.assertEqual(layer.context_scan.variant, "kda")
        self.assertEqual(
            layer.context_scan.g_proj.out_features,
            layer.settings["num_heads"] * layer.settings["key_dim"],
        )
        self._assert_forward_shape(model)

    def test_independent_injection_skips_the_gate(self):
        model = _tiny_model(injection="independent")
        layer = model.layers[0]
        self.assertIsNone(layer.inject_gate)
        self.assertIsNone(layer.self_attn.q_r_proj)
        self.assertIsNotNone(layer.context_residual)
        self._assert_forward_shape(model)

    def test_qkv_conditioning_adds_retrieved_projection_bias(self):
        model = _tiny_model(injection="qkv_conditioning")
        layer = model.layers[0]
        self.assertIsNone(layer.inject_gate)
        self.assertIsNotNone(layer.self_attn.q_r_proj)
        self._assert_forward_shape(model)

    def test_context_residual_can_be_disabled(self):
        model = _tiny_model(context_residual=False)
        self.assertIsNone(model.layers[0].context_residual)
        self.assertIsNotNone(model.layers[0].inject_gate)
        self._assert_forward_shape(model)

    def test_unknown_injection_is_rejected(self):
        with self.assertRaises(ValueError):
            _tiny_model(injection="not_a_mode")

    def test_horizon_embedding_makes_identical_masks_retrieve_differently(self):
        model = _tiny_model()
        layer = model.layers[0]
        block = TINY["block_size"]
        hidden = TINY["hidden_size"]
        layer.horizon_embed.weight.data.copy_(
            torch.arange(block, dtype=torch.float32).unsqueeze(1).expand(-1, hidden)
        )
        normalized = torch.ones(1, 1, block, hidden)
        prefix_state = torch.randn(
            1,
            1,
            int(layer.settings["num_heads"]),
            int(layer.settings["key_dim"]),
            int(layer.settings["value_dim"]),
        )
        retrieved = layer._context_read(normalized, prefix_state)
        self.assertFalse(torch.allclose(retrieved[0, 0, 0], retrieved[0, 0, 1]))
        self.assertFalse(torch.allclose(retrieved[0, 0, 0], retrieved[0, 0, -1]))

        layer.horizon_embed.weight.data.zero_()
        collapsed = layer._context_read(normalized, prefix_state)
        for offset in range(1, block):
            torch.testing.assert_close(
                collapsed[0, 0, offset],
                collapsed[0, 0, 0],
                atol=0,
                rtol=0,
            )

    def test_online_wrapper_loss_backward_is_finite(self):
        from torch import nn

        from specforge.algorithms.dflash_linear.model import OnlineDFlashLinearModel

        draft = _tiny_model()
        hidden = draft.config.hidden_size
        vocab = draft.config.vocab_size
        wrapper = OnlineDFlashLinearModel(
            draft_model=draft,
            target_lm_head=nn.Linear(hidden, vocab, bias=False),
            target_embed_tokens=nn.Embedding(vocab, hidden),
            mask_token_id=0,
            block_size=TINY["block_size"],
            attention_backend="sdpa",
            num_anchors=2,
        )
        seq_len = 16
        torch.manual_seed(0)
        input_ids = torch.randint(1, vocab, (1, seq_len))
        hidden_states = torch.randn(1, seq_len, hidden)
        loss_mask = torch.ones(1, seq_len)
        loss, _accuracy, _metrics = wrapper(
            input_ids=input_ids,
            hidden_states=hidden_states,
            loss_mask=loss_mask,
        )
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        grads = [
            parameter.grad.abs().sum()
            for parameter in draft.parameters()
            if parameter.grad is not None
        ]
        self.assertTrue(grads)
        self.assertGreater(sum(grads).item(), 0)

    def test_resume_contract_persists_linear_settings(self):
        from types import SimpleNamespace

        from specforge.algorithms.dflash_linear.providers import resume_contract

        draft = _tiny_model(variant="kda", injection="independent")
        training = SimpleNamespace(
            block_size=4,
            mask_token_id=0,
            attention_backend="sdpa",
            num_anchors=8,
            loss_decay_gamma=None,
            loss_type="dflash",
            dpace_alpha=0.5,
            lk_loss_type=None,
            kl_scale=1.0,
            kl_decay=1.0,
        )
        contract = resume_contract(None, draft, training)
        self.assertEqual(contract["dflash_linear_variant"], "kda")
        self.assertEqual(contract["dflash_linear_injection"], "independent")
        self.assertTrue(contract["dflash_linear_context_residual"])
        self.assertEqual(contract["dflash_linear_backend"], "naive")
        self.assertEqual(contract["dflash_linear_block_size"], 4)

    def _assert_forward_shape(self, model):
        model.eval()
        noise = torch.randn(1, 4, 64)
        target = torch.randn(1, 16, 64)
        anchors = torch.tensor([[5]])
        position_ids = (anchors.unsqueeze(-1) + torch.arange(4)).view(1, -1)
        out = model(
            position_ids=position_ids,
            noise_embedding=noise,
            target_hidden=target,
            anchor_positions=anchors,
        )
        self.assertEqual(tuple(out.shape), (1, 4, 64))
