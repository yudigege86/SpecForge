"""CPU tests for config-gated EAGLE-4 draft modules."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest import mock

import torch
import torch._dynamo
from transformers import LlamaConfig

try:
    from specforge.algorithms.eagle3.model import OnlineEagle3Model
except ImportError:
    OnlineEagle3Model = None

from specforge.modeling._mask_utils import _make_causal_mask
from specforge.modeling.auto import AutoDraftModel
from specforge.modeling.draft.eagle4 import (
    LlamaForCausalLMEagle4,
    candidate_logits,
    cutoff_retrieval_loss,
)
from specforge.modeling.draft.eagle4_moe import (
    Eagle4MoE,
    _expert_mlp,
    _grouped_mm_fn,
    relu2,
    resolve_act_fn,
)
from specforge.modeling.draft.eagle4_structure import (
    EAGLE4_ARCH,
    draft_sliding_window,
    draft_uses_rope,
    parse_efficient_lm_head_config,
    retrieval_positive_count,
)
from specforge.modeling.draft.flex_attention import generate_eagle3_mask
from specforge.modeling.draft.llama3_eagle import (
    LlamaDecoderLayer,
    LlamaForCausalLMEagle3,
    NoRotaryEmbedding,
)

torch._dynamo.config.disable = True


def _config(**overrides) -> LlamaConfig:
    config = LlamaConfig(
        vocab_size=overrides.get("vocab_size", 16),
        hidden_size=overrides.get("hidden_size", 8),
        intermediate_size=overrides.get("intermediate_size", 16),
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=32,
        pad_token_id=0,
        rms_norm_eps=1e-5,
        hidden_act="silu",
        initializer_range=0.02,
    )
    config.draft_vocab_size = overrides.get("draft_vocab_size", config.vocab_size)
    config.architectures = [EAGLE4_ARCH]
    for key, value in overrides.items():
        setattr(config, key, value)
    if "architectures" not in overrides:
        config.architectures = [EAGLE4_ARCH]
    return config


def _inputs(batch=2, seq=4, hidden=8):
    generator = torch.Generator().manual_seed(0)
    hidden_states = torch.randn(batch, seq, hidden * 3, generator=generator)
    inputs_embeds = torch.randn(batch, seq, hidden, generator=generator)
    return hidden_states, inputs_embeds


class Eagle4StructureTest(unittest.TestCase):
    def test_sliding_window_requires_eagle4_architecture(self):
        other = LlamaConfig()
        other.sliding_window = 8
        other.architectures = ["LlamaForCausalLMEagle3"]
        self.assertIsNone(draft_sliding_window(other))
        enabled = _config(sliding_window=8)
        self.assertEqual(8, draft_sliding_window(enabled))
        self.assertIsNone(draft_sliding_window(_config()))

    def test_no_rope_flags(self):
        self.assertTrue(draft_uses_rope(_config()))
        self.assertFalse(draft_uses_rope(_config(use_rope=False)))
        self.assertFalse(draft_uses_rope(_config(position_embedding_type="none")))
        self.assertFalse(draft_uses_rope(_config(partial_rotary_factor=0)))

    def test_efficient_head_config_validation(self):
        self.assertIsNone(parse_efficient_lm_head_config(_config()))
        parsed = parse_efficient_lm_head_config(
            _config(
                efficient_lm_head_config={
                    "type": "low_rank",
                    "rank": 4,
                    "num_candidates": 3,
                }
            )
        )
        self.assertEqual(
            {"type": "low_rank", "rank": 4, "num_candidates": 3},
            parsed,
        )
        with self.assertRaisesRegex(ValueError, "low_rank"):
            parse_efficient_lm_head_config(
                _config(efficient_lm_head_config={"type": "full", "rank": 4})
            )

    def test_retrieval_positive_count(self):
        self.assertEqual(32, retrieval_positive_count(128, 151936))
        self.assertEqual(2, retrieval_positive_count(8, 16))
        self.assertEqual(1, retrieval_positive_count(4, 16))
        with self.assertRaisesRegex(ValueError, "num_candidates >= 4"):
            retrieval_positive_count(3, 16)


class Eagle4FusionTest(unittest.TestCase):
    def test_eagle3_qkv_width_stays_2h(self):
        config = LlamaConfig(
            vocab_size=16,
            draft_vocab_size=8,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=16,
            pad_token_id=0,
        )
        model = LlamaForCausalLMEagle3(config, attention_backend="sdpa")
        self.assertEqual(16, model.midlayer.self_attn.q_proj.in_features)

    def test_attn_norm_switches_qkv_width_and_eh_proj(self):
        off = LlamaForCausalLMEagle4(_config(), attention_backend="sdpa")
        self.assertEqual(16, off.midlayer.self_attn.q_proj.in_features)
        self.assertIsNone(off.midlayer.eh_proj)

        on = LlamaForCausalLMEagle4(_config(attn_norm=True), attention_backend="sdpa")
        self.assertEqual(8, on.midlayer.self_attn.q_proj.in_features)
        self.assertEqual(16, on.midlayer.eh_proj.in_features)
        self.assertEqual(8, on.midlayer.eh_proj.out_features)
        hidden, embeds = _inputs()
        out = on(hidden_states=hidden, inputs_embeds=embeds, ttt_length=1)
        self.assertEqual((2, 4, 8), tuple(out.shape))

    def test_fc_norm_and_norm_output_flags(self):
        model = LlamaForCausalLMEagle4(
            _config(fc_norm=True, norm_output=True),
            attention_backend="sdpa",
        )
        self.assertEqual(3, len(model.fc_norm))
        raw = torch.randn(1, 3, 8)
        torch.testing.assert_close(model.next_depth_hidden(raw), model.norm(raw))
        logits = model.compute_logits(raw)
        torch.testing.assert_close(logits, model.lm_head(model.norm(raw)))

    def test_no_rope_is_identity_rotation(self):
        model = LlamaForCausalLMEagle4(_config(use_rope=False), attention_backend="sdpa")
        rotary = model.midlayer.self_attn.rotary_emb
        self.assertIsInstance(rotary, NoRotaryEmbedding)
        query = torch.randn(1, 2, 4, 4)
        cos, sin = rotary(query, seq_len=4)
        self.assertTrue(torch.allclose(cos, torch.ones_like(cos)))
        self.assertTrue(torch.allclose(sin, torch.zeros_like(sin)))
        hidden, embeds = _inputs()
        out = model(hidden_states=hidden, inputs_embeds=embeds)
        self.assertTrue(torch.isfinite(out).all())

    def test_sliding_window_mask_hides_keys_behind_the_window(self):
        mask = _make_causal_mask(
            (1, 6),
            torch.float32,
            torch.device("cpu"),
            window=3,
        )
        self.assertEqual(0.0, mask[0, 0, 5, 4].item())
        self.assertEqual(0.0, mask[0, 0, 5, 5].item())
        self.assertLess(mask[0, 0, 5, 1].item(), 0.0)
        self.assertLess(mask[0, 0, 5, 2].item(), 0.0)

        model = LlamaForCausalLMEagle4(
            _config(sliding_window=3), attention_backend="sdpa"
        )
        hidden = torch.ones(1, 6, 8)
        attn = model.prepare_decoder_attention_mask(
            torch.ones(1, 6), hidden, 1, 6, 0
        )
        self.assertLess(attn[0, 0, 5, 1].item(), 0.0)
        self.assertEqual(0.0, attn[0, 0, 5, 4].item())

    def test_flex_mask_windows_context_and_keeps_suffix_diagonal(self):
        mask_mod = generate_eagle3_mask(
            seq_lengths=torch.tensor([8]),
            Q_LEN=8,
            KV_LEN=16,
            lck=1,
            sliding_window=3,
        )
        b = torch.tensor(0)
        h = torch.tensor(0)
        self.assertFalse(bool(mask_mod(b, h, torch.tensor(5), torch.tensor(1))))
        self.assertTrue(bool(mask_mod(b, h, torch.tensor(5), torch.tensor(4))))
        self.assertTrue(bool(mask_mod(b, h, torch.tensor(5), torch.tensor(13))))

    def test_fa_and_usp_reject_a_sliding_window(self):
        config = _config(sliding_window=4)
        for backend in ("fa", "usp"):
            with self.subTest(backend=backend):
                with self.assertRaisesRegex(ValueError, "flex_attention"):
                    LlamaDecoderLayer(config, attention_backend=backend)


class Eagle4MoETest(unittest.TestCase):
    def _moe_config(self, **overrides):
        return _config(
            enable_moe=True,
            n_routed_experts=2,
            num_experts_per_tok=1,
            moe_intermediate_size=8,
            enable_post_mlp_norm=True,
            **overrides,
        )

    def test_moe_replaces_dense_mlp_and_post_mlp_norm_is_next_depth(self):
        model = LlamaForCausalLMEagle4(self._moe_config(), attention_backend="sdpa")
        self.assertIsInstance(model.midlayer.mlp, Eagle4MoE)
        self.assertIsNotNone(model.midlayer.post_mlp_layernorm)
        raw = torch.randn(2, 3, 8)
        torch.testing.assert_close(
            model.next_depth_hidden(raw),
            model.midlayer.post_mlp_layernorm(raw),
        )
        hidden, embeds = _inputs()
        out = model(hidden_states=hidden, inputs_embeds=embeds)
        self.assertEqual((2, 4, 8), tuple(out.shape))
        self.assertTrue(torch.isfinite(out).all())

    def test_router_bias_updates_after_a_training_step(self):
        model = LlamaForCausalLMEagle4(self._moe_config(), attention_backend="sdpa")
        model.train()
        hidden, embeds = _inputs()
        model(hidden_states=hidden, inputs_embeds=embeds)
        router = model.midlayer.mlp.gate
        self.assertGreater(int(router.load.sum().item()), 0)
        router.load.copy_(torch.tensor([8, 0], device=router.load.device))
        before = router.e_score_correction_bias.clone()
        imbalance = model.update_moe_router_bias()
        self.assertGreater(imbalance, 1.0)
        self.assertFalse(torch.equal(before, router.e_score_correction_bias))
        self.assertEqual(0, int(router.load.sum().item()))

    def test_enable_moe_requires_experts(self):
        with self.assertRaisesRegex(ValueError, "n_routed_experts"):
            LlamaForCausalLMEagle4(_config(enable_moe=True), attention_backend="sdpa")

    def test_relu2_resolves_when_act2fn_omits_it(self):
        missing = mock.MagicMock()
        missing.__getitem__.side_effect = KeyError("relu2")
        with mock.patch("specforge.modeling.draft.eagle4_moe.ACT2FN", missing):
            self.assertIs(resolve_act_fn("relu2"), relu2)
        x = torch.tensor([-1.0, 2.0])
        torch.testing.assert_close(relu2(x), torch.tensor([0.0, 4.0]))

    def test_latent_relu2_moe_constructs_and_runs(self):
        model = LlamaForCausalLMEagle4(
            self._moe_config(
                moe_architecture="nemotron_latent_relu2",
                hidden_act="relu2",
                moe_latent_size=4,
                moe_shared_expert_intermediate_size=8,
                n_shared_experts=1,
            ),
            attention_backend="sdpa",
        )
        mlp = model.midlayer.mlp
        self.assertTrue(mlp.latent)
        self.assertIsNotNone(mlp.shared_experts)
        hidden, embeds = _inputs()
        out = model(hidden_states=hidden, inputs_embeds=embeds)
        self.assertEqual((2, 4, 8), tuple(out.shape))
        self.assertTrue(torch.isfinite(out).all())

    def test_grouped_mm_is_skipped_on_hip(self):
        h = mock.Mock()
        h.is_cuda = True
        h.device.type = "cuda"
        with mock.patch(
            "specforge.modeling.draft.eagle4_moe.F.grouped_mm",
            mock.Mock(),
        ), mock.patch(
            "specforge.modeling.draft.eagle4_moe.torch.version.hip",
            "6.4",
        ):
            self.assertIsNone(_grouped_mm_fn(h))

    def test_expert_mlp_falls_back_when_grouped_mm_raises(self):
        h = torch.randn(4, 4)
        idx = torch.tensor([[0], [1], [0], [1]])
        weights = torch.ones(4, 1)
        up = torch.randn(2, 8, 4)
        down = torch.randn(2, 4, 8)
        expected = _expert_mlp(h, idx, weights, up, down, True, relu2)
        grouped = mock.Mock(side_effect=RuntimeError("no grouped_mm"))
        with mock.patch(
            "specforge.modeling.draft.eagle4_moe._grouped_mm_fn",
            return_value=grouped,
        ):
            out = _expert_mlp(h, idx, weights, up, down, True, relu2)
        torch.testing.assert_close(out, expected)
        grouped.assert_called()


class Eagle4CandidateHeadTest(unittest.TestCase):
    def _head_config(self, **overrides):
        return _config(
            efficient_lm_head_config={
                "type": "low_rank",
                "rank": 4,
                "num_candidates": 3,
            },
            **overrides,
        )

    def test_pruned_vocab_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "full-vocab"):
            LlamaForCausalLMEagle4(
                self._head_config(draft_vocab_size=8, vocab_size=16),
                attention_backend="sdpa",
            )

    def test_candidate_shapes_and_exact_rescore(self):
        model = LlamaForCausalLMEagle4(self._head_config(), attention_backend="sdpa")
        self.assertEqual((4, 8), tuple(model.efficient_lm_head_down_proj.weight.shape))
        self.assertEqual((16, 4), tuple(model.efficient_lm_head.weight.shape))
        self.assertEqual((16, 8), tuple(model.lm_head.weight.shape))
        self.assertTrue(bool(model.t2d.all().item()))
        self.assertEqual((16,), tuple(model.d2t.shape))

        hidden = torch.randn(2, 3, 8)
        logits = candidate_logits(
            hidden,
            model.efficient_lm_head_down_proj.weight,
            model.efficient_lm_head.weight,
            model.lm_head.weight,
            3,
        )
        self.assertEqual((2, 3, 16), tuple(logits.shape))
        approx = torch.nn.functional.linear(
            torch.nn.functional.linear(
                hidden, model.efficient_lm_head_down_proj.weight
            ),
            model.efficient_lm_head.weight,
        )
        indices = approx.topk(3, dim=-1).indices
        dense = torch.nn.functional.linear(hidden, model.lm_head.weight)
        torch.testing.assert_close(logits.gather(-1, indices), dense.gather(-1, indices))
        fill = torch.finfo(logits.dtype).min
        blocked = torch.ones_like(logits, dtype=torch.bool).scatter_(-1, indices, False)
        self.assertTrue(torch.all(logits[blocked] == fill))
        self.assertEqual(3, int((~blocked)[0, 0].sum().item()))

    def test_compute_logits_matches_dense_on_chosen_rows(self):
        model = LlamaForCausalLMEagle4(self._head_config(), attention_backend="sdpa")
        raw = torch.randn(1, 2, 8)
        logits = model.compute_logits(raw)
        normed = model.norm(raw)
        dense = model.lm_head(normed)
        approx = torch.nn.functional.linear(
            torch.nn.functional.linear(
                normed, model.efficient_lm_head_down_proj.weight
            ),
            model.efficient_lm_head.weight,
        )
        indices = approx.topk(3, dim=-1).indices
        torch.testing.assert_close(logits.gather(-1, indices), dense.gather(-1, indices))
        fill = torch.finfo(logits.dtype).min
        blocked = torch.ones_like(logits, dtype=torch.bool).scatter_(-1, indices, False)
        self.assertTrue(torch.all(logits[blocked] == fill))

    def test_factors_receive_gradients(self):
        model = LlamaForCausalLMEagle4(self._head_config(), attention_backend="sdpa")
        raw = torch.randn(2, 3, 8, requires_grad=True)
        logits = model.compute_logits(raw)
        logits[torch.isfinite(logits)].sum().backward()
        self.assertIsNotNone(model.efficient_lm_head.weight.grad)
        self.assertIsNotNone(model.efficient_lm_head_down_proj.weight.grad)

    def test_svd_init_and_export_roundtrip(self):
        model = LlamaForCausalLMEagle4(self._head_config(), attention_backend="sdpa")
        model.init_efficient_head_from_dense()
        self.assertEqual((16, 4), tuple(model.efficient_lm_head.weight.shape))
        state = model.state_dict()
        self.assertIn("lm_head.weight", state)
        self.assertIn("efficient_lm_head.weight", state)
        self.assertIn("efficient_lm_head_down_proj.weight", state)
        self.assertIn("t2d", state)
        self.assertIn("d2t", state)

        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            with open(os.path.join(directory, "config.json"), encoding="utf-8") as handle:
                payload = json.load(handle)
            self.assertEqual([EAGLE4_ARCH], payload["architectures"])
            self.assertEqual("low_rank", payload["efficient_lm_head_config"]["type"])
            self.assertEqual(4, payload["efficient_lm_head_config"]["rank"])
            reloaded = AutoDraftModel.from_pretrained(
                directory, local_files_only=True
            )
            self.assertIsInstance(reloaded, LlamaForCausalLMEagle4)
            torch.testing.assert_close(
                reloaded.efficient_lm_head.weight,
                model.efficient_lm_head.weight,
            )
            torch.testing.assert_close(reloaded.lm_head.weight, model.lm_head.weight)

    def test_candidate_recall_is_one_when_target_is_the_argmax(self):
        model = LlamaForCausalLMEagle4(self._head_config(), attention_backend="sdpa")
        with torch.no_grad():
            model.lm_head.weight.zero_()
            model.lm_head.weight[3].fill_(1.0)
            model.efficient_lm_head_down_proj.weight.copy_(
                torch.eye(4, 8)[:4]
            )
            model.efficient_lm_head.weight.zero_()
            model.efficient_lm_head.weight[3, 0] = 10.0
        hidden = torch.zeros(1, 2, 8)
        hidden[..., 0] = 1.0
        target = torch.full((1, 2), 3, dtype=torch.long)
        recall = model.candidate_recall(hidden, target)
        self.assertGreater(float(recall), 0.99)

    def test_stage1_compute_logits_is_the_dense_head(self):
        model = LlamaForCausalLMEagle4(_config(), attention_backend="sdpa")
        self.assertIsNone(model.efficient_lm_head)
        raw = torch.randn(2, 3, 8)
        torch.testing.assert_close(
            model.compute_logits(raw), model.lm_head(model.norm(raw))
        )

    def test_only_candidate_factors_are_trainable(self):
        model = LlamaForCausalLMEagle4(self._head_config(), attention_backend="sdpa")
        model.freeze_non_candidate_parameters()
        trainable = {
            name for name, param in model.named_parameters() if param.requires_grad
        }
        self.assertEqual(
            {"efficient_lm_head.weight", "efficient_lm_head_down_proj.weight"},
            trainable,
        )


class Eagle4RetrievalLossTest(unittest.TestCase):
    def test_cutoff_is_the_nth_non_positive_and_in_set_swap_still_has_loss(self):
        # C=8, m=2, τ is the 7th non-positive. Token 0 is the only retrieved
        # positive and sits at ρ_C, so it is in S but still below τ.
        vocab = 16
        candidate_count = 8
        dense = torch.zeros(1, vocab)
        dense[0, 0] = 4.0
        dense[0, 1] = 3.0
        ranking = torch.tensor(
            [2, 3, 4, 5, 6, 7, 8, 0, 9, 10, 11, 12, 13, 14, 15, 1]
        )
        values = torch.arange(vocab, 0, -1, dtype=torch.float32)
        scores = torch.zeros(1, vocab, requires_grad=True)
        with torch.no_grad():
            scores[0, ranking] = values
        loss, metrics = cutoff_retrieval_loss(dense, scores, candidate_count)
        self.assertEqual(8, int(metrics["cutoff_index"][0].item()))
        self.assertEqual([0, 1], metrics["positives"][0].tolist())
        self.assertGreater(float(loss.detach()), 0.0)
        retrieved = scores.detach().topk(candidate_count, dim=-1).indices[0].tolist()
        self.assertIn(0, retrieved)
        self.assertNotIn(1, retrieved)

    def test_positive_above_cutoff_has_zero_gradient(self):
        vocab = 16
        dense = torch.zeros(1, vocab)
        dense[0, 0] = 5.0
        dense[0, 1] = 4.0
        scores = torch.arange(vocab, 0, -1, dtype=torch.float32).unsqueeze(0)
        scores = scores.detach().clone().requires_grad_(True)
        loss, metrics = cutoff_retrieval_loss(dense, scores, 8)
        self.assertEqual(8, int(metrics["cutoff_index"][0].item()))
        self.assertEqual(0.0, float(loss.detach()))
        loss.backward()
        self.assertTrue(torch.all(scores.grad == 0))

    def test_positive_below_cutoff_trains_factors_and_pushes_tau_down(self):
        model = LlamaForCausalLMEagle4(
            _config(
                efficient_lm_head_config={
                    "type": "low_rank",
                    "rank": 4,
                    "num_candidates": 8,
                }
            ),
            attention_backend="sdpa",
        )
        model.freeze_non_candidate_parameters()
        hidden = torch.zeros(1, 1, 8)
        hidden[..., 0] = 1.0
        with torch.no_grad():
            model.norm.weight.fill_(1.0)
            model.lm_head.weight.zero_()
            model.lm_head.weight[0, 0] = 8.0
            model.lm_head.weight[1, 0] = 7.0
            model.efficient_lm_head_down_proj.weight.zero_()
            model.efficient_lm_head_down_proj.weight[0, 0] = 1.0
            model.efficient_lm_head.weight.zero_()
            ranking = torch.tensor(
                [2, 3, 4, 5, 6, 7, 8, 0, 9, 10, 11, 12, 13, 14, 15, 1]
            )
            model.efficient_lm_head.weight[ranking, 0] = torch.arange(
                16, 0, -1, dtype=torch.float32
            )
        hidden_n = model.norm(hidden)
        scores_before = torch.nn.functional.linear(
            torch.nn.functional.linear(
                hidden_n, model.efficient_lm_head_down_proj.weight
            ),
            model.efficient_lm_head.weight,
        ).detach()
        loss, metrics = model.retrieval_loss(hidden)
        tau = int(metrics["cutoff_index"][0].item())
        self.assertEqual(8, tau)
        self.assertGreater(float(loss.detach()), 0.0)
        loss.backward()
        self.assertIsNotNone(model.efficient_lm_head.weight.grad)
        self.assertIsNotNone(model.efficient_lm_head_down_proj.weight.grad)
        self.assertIsNone(model.lm_head.weight.grad)
        self.assertGreater(model.efficient_lm_head.weight.grad[tau, 0].item(), 0.0)

        lr = 0.5
        with torch.no_grad():
            model.efficient_lm_head.weight -= lr * model.efficient_lm_head.weight.grad
            model.efficient_lm_head_down_proj.weight -= (
                lr * model.efficient_lm_head_down_proj.weight.grad
            )
        scores_after = torch.nn.functional.linear(
            torch.nn.functional.linear(
                model.norm(hidden), model.efficient_lm_head_down_proj.weight
            ),
            model.efficient_lm_head.weight,
        ).detach()
        self.assertLess(scores_after[0, 0, tau].item(), scores_before[0, 0, tau].item())

    def test_candidate_head_requires_a_dense_checkpoint(self):
        from types import SimpleNamespace

        from specforge.algorithms.model_providers import build_eagle4_draft

        cfg = SimpleNamespace(
            model=SimpleNamespace(
                vocab_mapping_path="",
                draft_checkpoint_path=None,
            )
        )
        with self.assertRaisesRegex(ValueError, "draft_checkpoint_path"):
            build_eagle4_draft(
                cfg,
                _config(
                    efficient_lm_head_config={
                        "type": "low_rank",
                        "rank": 4,
                        "num_candidates": 8,
                    }
                ),
            )

    def test_candidate_head_accepts_identity_mapping_and_rejects_shrink(self):
        from types import SimpleNamespace

        from specforge.algorithms.model_providers import build_eagle4_draft

        config = _config(
            efficient_lm_head_config={
                "type": "low_rank",
                "rank": 4,
                "num_candidates": 8,
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            identity = os.path.join(tmp, "identity.pt")
            torch.save(
                {
                    "d2t": torch.zeros(16, dtype=torch.int64),
                    "t2d": torch.ones(16, dtype=torch.bool),
                },
                identity,
            )
            shrinking = os.path.join(tmp, "shrink.pt")
            t2d = torch.ones(16, dtype=torch.bool)
            t2d[-1] = False
            torch.save(
                {
                    "d2t": torch.zeros(16, dtype=torch.int64),
                    "t2d": t2d,
                },
                shrinking,
            )

            def _cfg(mapping):
                return SimpleNamespace(
                    model=SimpleNamespace(
                        vocab_mapping_path=mapping,
                        draft_checkpoint_path=os.path.join(tmp, "ckpt"),
                        torch_dtype="float32",
                        load_target_embedding=False,
                        cache_dir=tmp,
                        trust_remote_code=False,
                    ),
                    training=SimpleNamespace(
                        attention_backend="sdpa",
                        strategy="eagle4",
                    ),
                )

            report = SimpleNamespace(
                missing_keys={"efficient_lm_head.weight"}
            )
            with mock.patch(
                "specforge.algorithms.model_providers._warm_start",
                return_value=report,
            ), mock.patch(
                "specforge.algorithms.model_providers._device",
                return_value=torch.device("cpu"),
            ):
                model = build_eagle4_draft(_cfg(identity), config)
                self.assertTrue(bool(model.t2d.all().item()))
                trainable = {
                    name
                    for name, param in model.named_parameters()
                    if param.requires_grad
                }
                self.assertEqual(
                    {
                        "efficient_lm_head.weight",
                        "efficient_lm_head_down_proj.weight",
                    },
                    trainable,
                )
                with self.assertRaisesRegex(ValueError, "identity"):
                    build_eagle4_draft(_cfg(shrinking), config)


@unittest.skipUnless(
    OnlineEagle3Model is not None, "OnlineEagle3Model imports require triton"
)
class Eagle4TTTHookTest(unittest.TestCase):
    def test_ttt_uses_next_depth_hidden_and_identity_vocab(self):
        model = LlamaForCausalLMEagle4(
            _config(
                efficient_lm_head_config={
                    "type": "low_rank",
                    "rank": 4,
                    "num_candidates": 4,
                },
                norm_output=True,
            ),
            attention_backend="sdpa",
        ).eval()
        self.assertTrue(bool(model.t2d.all().item()))
        wrapper = OnlineEagle3Model(model, length=1, attention_backend="sdpa")
        batch, seq, hidden, vocab = 1, 4, 8, 16
        with mock.patch.object(
            model, "next_depth_hidden", wraps=model.next_depth_hidden
        ) as next_depth:
            plosses, *_ = wrapper(
                input_ids=torch.ones(batch, seq, dtype=torch.long),
                attention_mask=torch.ones(batch, seq),
                target=torch.randn(batch, seq, vocab),
                loss_mask=torch.ones(batch, seq, 1),
                hidden_states=torch.randn(batch, seq, hidden * 3),
            )
        self.assertTrue(next_depth.called)
        self.assertEqual(1, len(plosses))
        self.assertTrue(torch.isfinite(plosses[0]))
        self.assertIsNotNone(wrapper.last_candidate_recall)
        self.assertIsNotNone(wrapper.last_retrieval_hit_1)
        self.assertEqual("retrieval_loss", wrapper.objective_metric_name)


if __name__ == "__main__":
    unittest.main()
