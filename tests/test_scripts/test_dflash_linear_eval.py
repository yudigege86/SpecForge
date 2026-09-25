"""CPU tests for SPEED-Bench MAL eval helpers."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_SPEC = importlib.util.spec_from_file_location(
    "dflash_linear_eval",
    ROOT / "scripts" / "eval" / "dflash_linear_eval.py",
)
_EVAL = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_EVAL)

PLACEHOLDER = _EVAL.PLACEHOLDER
aggregate_mal_report = _EVAL.aggregate_mal_report
assert_no_placeholders = _EVAL.assert_no_placeholders
clip_completion_at_eos = _EVAL.clip_completion_at_eos
load_eval_rows = _EVAL.load_eval_rows
render_prompt = _EVAL.render_prompt
select_rows = _EVAL.select_rows
turns_have_placeholder = _EVAL.turns_have_placeholder
turns_to_messages = _EVAL.turns_to_messages

try:
    import torch
except ImportError:
    torch = None


class SpeedBenchEvalHelpersTest(unittest.TestCase):
    def test_placeholder_detection_and_refusal(self):
        rows = [
            {"question_id": "ok", "turns": ["Write a function."]},
            {
                "question_id": "bad",
                "turns": [f"{PLACEHOLDER}\n\n{{question}}"],
            },
        ]
        self.assertFalse(turns_have_placeholder(rows[0]["turns"]))
        self.assertTrue(turns_have_placeholder(rows[1]["turns"]))
        with self.assertRaises(SystemExit):
            assert_no_placeholders(rows, source="fixture")

    def test_turns_to_messages_are_user_only(self):
        messages = turns_to_messages(["first", "follow up"])
        self.assertEqual(
            messages,
            [
                {"role": "user", "content": "first"},
                {"role": "user", "content": "follow up"},
            ],
        )

    def test_render_prompt_skips_thinking_kwargs(self):
        class Tokenizer:
            def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
                del tokenize, add_generation_prompt
                return " | ".join(item["content"] for item in messages) + " <gen>"

        rendered = render_prompt(Tokenizer(), turns_to_messages(["hello"]))
        self.assertEqual(rendered, "hello <gen>")

    def test_render_prompt_enables_thinking_by_default(self):
        class Tokenizer:
            def apply_chat_template(
                self,
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            ):
                del tokenize, add_generation_prompt, messages
                return f"think={enable_thinking}"

        rendered = render_prompt(Tokenizer(), turns_to_messages(["hello"]))
        self.assertEqual(rendered, "think=True")
        rendered_off = render_prompt(
            Tokenizer(), turns_to_messages(["hello"]), enable_thinking=False
        )
        self.assertEqual(rendered_off, "think=False")

    def test_load_eval_rows_rejects_placeholders(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "qualitative.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "question_id": "x",
                        "category": "coding",
                        "turns": [PLACEHOLDER],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(SystemExit):
                load_eval_rows(str(path))

    def test_select_rows_is_stratified(self):
        rows = []
        for category in ("coding", "math", "writing"):
            for index in range(4):
                rows.append(
                    {
                        "question_id": f"{category}-{index}",
                        "category": category,
                        "turns": [f"{category} {index}"],
                    }
                )
        selected = select_rows(rows, n=6, categories=None)
        self.assertEqual(len(selected), 6)
        counts = {}
        for row in selected:
            counts[row["category"]] = counts.get(row["category"], 0) + 1
        self.assertEqual(counts, {"coding": 2, "math": 2, "writing": 2})

    @unittest.skipIf(torch is None, "torch is not installed")
    def test_clip_completion_at_eos_keeps_first_stop(self):
        prompt_len = 3
        sequence = torch.tensor([[1, 2, 3, 4, 9, 5, 9]])
        clipped, finished = clip_completion_at_eos(sequence, prompt_len, [9])
        self.assertTrue(finished)
        self.assertEqual(clipped.tolist(), [[1, 2, 3, 4, 9]])

    def test_aggregate_mal_report_groups_by_category(self):
        rows = [
            {
                "category": "coding",
                "completion_tokens": 8,
                "spec_accept_length": 3.0,
                "accepts": [4, 2],
            },
            {
                "category": "coding",
                "completion_tokens": 4,
                "spec_accept_length": 1.0,
                "accepts": [1],
            },
            {
                "category": "math",
                "completion_tokens": 6,
                "spec_accept_length": 2.0,
                "accepts": [2],
            },
        ]
        report = aggregate_mal_report(rows, block_accepts=[4, 2, 1, 2])
        self.assertEqual(report["n"], 3)
        self.assertAlmostEqual(report["spec_accept_length_mean"], 2.0)
        self.assertEqual(report["per_category"]["coding"]["n"], 2)
        self.assertAlmostEqual(
            report["per_category"]["coding"]["spec_accept_length_mean"], 2.0
        )
        self.assertEqual(report["per_category"]["math"]["n"], 1)
        self.assertIn("unknown", report["per_sub_category"])
        self.assertEqual(report["per_sub_category"]["unknown"]["n"], 3)


class HumanEvalMtBenchPrepareTest(unittest.TestCase):
    def test_humaneval_row_is_single_coding_turn(self):
        row = _EVAL.humaneval_example_to_row(
            {
                "task_id": "HumanEval/0",
                "prompt": 'def foo():\n    """hi"""\n',
                "entry_point": "foo",
            }
        )
        self.assertEqual(row["question_id"], "HumanEval/0")
        self.assertEqual(row["category"], "coding")
        self.assertEqual(row["protocol"], "concat_user")
        self.assertEqual(row["source"], "humaneval")
        self.assertIn("def foo", row["turns"][0])
        self.assertIn("passes the tests", row["turns"][0])
        self.assertIn("```python", row["turns"][0])
        self.assertEqual(_EVAL.score_units_for_row(row), 1)
        self.assertEqual(
            _EVAL.score_units_for_row(row, mt_bench_turns="all"), 1
        )

    def test_mtbench_row_scores_two_interleaved_turns(self):
        row = _EVAL.mtbench_example_to_row(
            {
                "question_id": 81,
                "category": "writing",
                "turns": ["first question", "follow up"],
            }
        )
        self.assertEqual(row["protocol"], "mt_bench")
        self.assertEqual(_EVAL.score_units_for_row(row), 1)
        self.assertEqual(
            _EVAL.score_units_for_row(row, mt_bench_turns="all"), 2
        )
        first = _EVAL.prompt_messages_for_turn(row, [])
        self.assertEqual(first, [{"role": "user", "content": "first question"}])
        second = _EVAL.prompt_messages_for_turn(row, ["answer one"])
        self.assertEqual(
            second,
            [
                {"role": "user", "content": "first question"},
                {"role": "assistant", "content": "answer one"},
                {"role": "user", "content": "follow up"},
            ],
        )

    def test_speedbench_stays_concat_user(self):
        row = {
            "question_id": "x",
            "category": "coding",
            "turns": ["one", "two"],
        }
        self.assertEqual(_EVAL.eval_protocol(row), "concat_user")
        self.assertEqual(
            _EVAL.prompt_messages_for_turn(row, []),
            [
                {"role": "user", "content": "one"},
                {"role": "user", "content": "two"},
            ],
        )


class SglangMalHelpersTest(unittest.TestCase):
    def test_sglang_is_ready_waits_until_dflash(self):
        ready, algorithm = _EVAL.sglang_is_ready({}, require_dflash=True)
        self.assertFalse(ready)
        self.assertEqual(algorithm, "")
        ready, algorithm = _EVAL.sglang_is_ready(
            {"speculative_algorithm": "DFLASH"},
            require_dflash=True,
        )
        self.assertTrue(ready)
        self.assertEqual(algorithm, "DFLASH")
        with self.assertRaises(SystemExit):
            _EVAL.sglang_is_ready(
                {"speculative_algorithm": "EAGLE"},
                require_dflash=True,
            )

    def test_card_accept_length_is_completion_over_verify(self):
        report = aggregate_mal_report(
            [
                {
                    "question_id": "a",
                    "category": "coding",
                    "completion_tokens": 16,
                    "spec_accept_length": 4.0,
                    "spec_verify_ct": 2,
                    "accepts": [8, 8],
                },
                {
                    "question_id": "b",
                    "category": "math",
                    "completion_tokens": 9,
                    "spec_accept_length": 3.0,
                    "spec_verify_ct": 3,
                    "accepts": [3, 3, 3],
                },
            ],
            block_accepts=[8, 8, 3, 3, 3],
        )
        self.assertAlmostEqual(report["card_accept_length_mean"], 5.5)
        self.assertAlmostEqual(report["token_weighted_accept_length"], 5.0)

    def test_cli_defaults_match_zlab_card(self):
        args = _EVAL.build_parser().parse_args(
            [
                "mal",
                "--target",
                "t",
                "--draft",
                "d",
                "--eval-jsonl",
                "e.jsonl",
                "--out",
                "o.json",
            ]
        )
        self.assertTrue(args.enable_thinking)
        self.assertEqual(args.max_new_tokens, 4096)
        self.assertEqual(args.mt_bench_turns, "first")
        self.assertEqual(args.feature_offset, "auto")
        self.assertEqual(args.feature_source, "hf")
        args_sg = _EVAL.build_parser().parse_args(
            [
                "mal",
                "--target",
                "t",
                "--draft",
                "d",
                "--replay-json",
                "r.json",
                "--out",
                "o.json",
                "--feature-source",
                "sglang",
            ]
        )
        self.assertEqual(args_sg.feature_source, "sglang")
        args_off = _EVAL.build_parser().parse_args(
            [
                "sglang-mal",
                "--target",
                "t",
                "--eval-jsonl",
                "e.jsonl",
                "--out",
                "o.json",
                "--disable-thinking",
            ]
        )
        self.assertFalse(args_off.enable_thinking)

    def test_flatten_token_ids_reads_batch_encoding_keys(self):
        from collections import UserDict

        class Encoding(UserDict):
            pass

        self.assertEqual(
            _EVAL.flatten_token_ids(Encoding({"input_ids": [7, 8, 9]})),
            [7, 8, 9],
        )
        self.assertEqual(_EVAL.flatten_token_ids({"input_ids": [[1, 2, 3]]}), [1, 2, 3])
        self.assertEqual(_EVAL.flatten_token_ids([4, 5]), [4, 5])

    def test_normalize_aux_feature_keeps_seq_hidden(self):
        if torch is None:
            self.skipTest("torch is not installed")
        feat = torch.arange(12, dtype=torch.float32).view(3, 4)
        out = _EVAL.normalize_aux_feature(feat, 3)
        self.assertEqual(tuple(out.shape), (3, 4))
        self.assertTrue(torch.equal(out, feat.cpu()))
        packed = feat.unsqueeze(0)
        self.assertEqual(tuple(_EVAL.normalize_aux_feature(packed, 3).shape), (3, 4))
        layered = torch.arange(24, dtype=torch.float32).view(3, 2, 4)
        self.assertEqual(
            tuple(_EVAL.normalize_aux_feature(layered, 3).shape), (3, 8)
        )

    def test_split_prompt_and_completion_from_full_ids(self):
        prompt, completion = _EVAL.split_prompt_and_completion(
            [1, 2, 3, 4, 5],
            prompt_ids=[1, 2],
            prompt_tokens=2,
            completion_tokens=3,
        )
        self.assertEqual(prompt, [1, 2])
        self.assertEqual(completion, [3, 4, 5])

    def test_parse_sglang_generate_extracts_token_ids(self):
        parsed = _EVAL.parse_sglang_generate(
            {
                "text": "ok",
                "output_ids": [10, 11, 12, 13],
                "meta_info": {
                    "prompt_tokens": 2,
                    "completion_tokens": 2,
                    "spec_accept_length": 2.0,
                    "finish_reason": "stop",
                },
            },
            max_new_tokens=512,
            prompt_ids=[10, 11],
        )
        self.assertEqual(parsed["prompt_ids"], [10, 11])
        self.assertEqual(parsed["completion_ids"], [12, 13])

    def test_load_replay_trajectories_requires_token_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sglang.json"
            path.write_text(
                json.dumps(
                    {
                        "raw": [
                            {
                                "question_id": "HumanEval/0",
                                "category": "coding",
                                "prompt_ids": [1, 2],
                                "completion_ids": [3, 4, 5],
                                "spec_accept_length": 3.0,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            rows = _EVAL.load_replay_trajectories(str(path))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["sequence_ids"], [1, 2, 3, 4, 5])
            self.assertEqual(rows[0]["prompt_len"], 2)
            missing = Path(tmp) / "bad.json"
            missing.write_text(
                json.dumps({"raw": [{"question_id": "x", "category": "coding"}]}),
                encoding="utf-8",
            )
            with self.assertRaises(SystemExit):
                _EVAL.load_replay_trajectories(str(missing))

    def test_sampling_stops_at_eos_by_default(self):
        params = _EVAL.sglang_sampling_params(max_new_tokens=512, ignore_eos=False)
        self.assertEqual(params["temperature"], 0.0)
        self.assertFalse(params["ignore_eos"])
        self.assertEqual(params["max_new_tokens"], 512)

    def test_parse_sglang_generate_reads_spec_accept_length(self):
        parsed = _EVAL.parse_sglang_generate(
            {
                "text": "ok",
                "meta_info": {
                    "completion_tokens": 40,
                    "spec_accept_length": 2.5,
                    "spec_verify_ct": 16,
                    "finish_reason": {"type": "stop"},
                },
            },
            max_new_tokens=512,
        )
        self.assertEqual(parsed["completion_tokens"], 40)
        self.assertAlmostEqual(parsed["spec_accept_length"], 2.5)
        self.assertTrue(parsed["finished_on_eos"])
        self.assertFalse(parsed["hit_max_new_tokens"])

    def test_parse_sglang_generate_derives_mal_from_verify_count(self):
        parsed = _EVAL.parse_sglang_generate(
            {
                "meta_info": {
                    "completion_tokens": 12,
                    "spec_verify_ct": 4,
                    "finish_reason": "length",
                }
            },
            max_new_tokens=12,
        )
        self.assertAlmostEqual(parsed["spec_accept_length"], 3.0)
        self.assertTrue(parsed["hit_max_new_tokens"])
        self.assertFalse(parsed["finished_on_eos"])

    def test_compare_mal_reports_by_question_and_category(self):
        offline = aggregate_mal_report(
            [
                {
                    "question_id": "a",
                    "category": "coding",
                    "completion_tokens": 8,
                    "spec_accept_length": 3.0,
                    "accepts": [3],
                },
                {
                    "question_id": "b",
                    "category": "math",
                    "completion_tokens": 6,
                    "spec_accept_length": 2.0,
                    "accepts": [2],
                },
            ],
            block_accepts=[3, 2],
        )
        sglang = aggregate_mal_report(
            [
                {
                    "question_id": "a",
                    "category": "coding",
                    "completion_tokens": 8,
                    "spec_accept_length": 3.2,
                    "accepts": [],
                },
                {
                    "question_id": "b",
                    "category": "math",
                    "completion_tokens": 6,
                    "spec_accept_length": 1.8,
                    "accepts": [],
                },
            ],
            block_accepts=[3.2, 1.8],
        )
        compared = _EVAL.compare_mal_reports(offline, sglang)
        self.assertEqual(compared["n_matched"], 2)
        self.assertAlmostEqual(compared["offline_mean"], 2.5)
        self.assertAlmostEqual(compared["sglang_mean"], 2.5)
        self.assertAlmostEqual(compared["delta_mean"], 0.0)
        self.assertAlmostEqual(compared["per_category"]["coding"]["delta_mean"], 0.2)
        self.assertAlmostEqual(compared["per_category"]["math"]["delta_mean"], -0.2)


class DraftDispatchTest(unittest.TestCase):
    @unittest.skipIf(torch is None, "torch is not installed")
    def test_stock_and_linear_use_different_teacher_force_hooks(self):
        from specforge.modeling.draft.dflash import DFlashDraftModel
        from specforge.modeling.draft.dflash_linear import DFlashLinearDraftModel

        self.assertIsNot(
            DFlashLinearDraftModel._teacher_force_block,
            DFlashDraftModel._teacher_force_block,
        )
        self.assertIs(
            DFlashLinearDraftModel.acceptance_along_sequence,
            DFlashDraftModel.acceptance_along_sequence,
        )
