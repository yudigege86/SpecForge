import inspect
import unittest

from scripts.eval_eagle4_acc import (
    evaluate_heads,
    evaluate_trained_head,
    load_trained_draft,
    relative_drop,
    summarize_comparison,
)

try:
    import torch
    from transformers import LlamaConfig

    from scripts.eval_eagle4_acc import load_svd_draft
    from specforge.modeling.draft.eagle4 import LlamaForCausalLMEagle4
    from specforge.modeling.draft.eagle4_structure import EAGLE4_ARCH

    TORCH_OK = True
except ImportError:
    TORCH_OK = False


class EvalEagle4AccHelpersTest(unittest.TestCase):
    def test_relative_drop_is_signed_fraction(self):
        self.assertAlmostEqual(-0.25, relative_drop(4.0, 3.0))
        self.assertEqual(0.0, relative_drop(0.0, 1.0))

    def test_summarize_comparison_reports_svd_drop_against_dense(self):
        payload = summarize_comparison(
            {
                "eval/simulated_acc_len": 4.0,
                "eval/avg_acc": 0.8,
                "eval/per_position_acc": [0.8, 0.5],
            },
            [
                {
                    "rank": 128,
                    "num_candidates": 128,
                    "eval/simulated_acc_len": 3.0,
                    "eval/avg_acc": 0.6,
                    "eval/candidate_recall": 0.9,
                }
            ],
        )
        run = payload["svd"][0]
        self.assertEqual(4.0, payload["dense"]["eval/simulated_acc_len"])
        self.assertAlmostEqual(-1.0, run["delta_simulated_acc_len"])
        self.assertAlmostEqual(-0.25, run["relative_simulated_acc_len"])
        self.assertAlmostEqual(-0.2, run["delta_avg_acc"])
        self.assertAlmostEqual(0.9, run["eval/candidate_recall"])

    def test_evaluate_heads_imports_feature_listing_locally(self):
        source = inspect.getsource(evaluate_heads)
        self.assertIn("from specforge.runtime.data_plane.offline_reader import list_feature_files", source)
        self.assertIn("if not files:", source)

    def test_trained_loader_keeps_checkpoint_factors(self):
        source = inspect.getsource(load_trained_draft)
        self.assertNotIn("init_efficient_head_from_dense", source)
        self.assertIn('strategy="eagle4"', source)
        self.assertIn('key.startswith("efficient_lm_head")', source)

    def test_trained_eval_compares_against_a_dense_baseline(self):
        source = inspect.getsource(evaluate_trained_head)
        self.assertIn("summarize_comparison", source)
        self.assertIn("model.num_candidates = int(k)", source)


@unittest.skipUnless(TORCH_OK, "torch and transformers are required")
class EvalEagle4SvdInitTest(unittest.TestCase):
    def test_svd_init_from_dense_keeps_exact_rescore_and_high_recall(self):
        config = LlamaConfig(
            vocab_size=16,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=32,
            pad_token_id=0,
            rms_norm_eps=1e-5,
            hidden_act="silu",
        )
        config.architectures = [EAGLE4_ARCH]
        config.draft_vocab_size = 16
        dense = LlamaForCausalLMEagle4(config, attention_backend="sdpa")
        torch.nn.init.normal_(dense.lm_head.weight, mean=0.0, std=0.5)
        svd = load_svd_draft(
            dense,
            rank=4,
            num_candidates=8,
            attention_backend="sdpa",
            dtype=torch.float32,
        )
        hidden = torch.randn(2, 3, 8)
        dense_logits = dense.compute_logits(hidden)
        svd_logits = svd.compute_logits(hidden)
        approx = torch.nn.functional.linear(
            torch.nn.functional.linear(
                svd.norm(hidden), svd.efficient_lm_head_down_proj.weight
            ),
            svd.efficient_lm_head.weight,
        )
        indices = approx.topk(8, dim=-1).indices
        torch.testing.assert_close(
            svd_logits.gather(-1, indices),
            dense_logits.gather(-1, indices),
            atol=1e-4,
            rtol=1e-4,
        )
        target = dense_logits.argmax(-1)
        recall = float(svd.candidate_recall(hidden, target))
        self.assertGreaterEqual(recall, 0.99)
