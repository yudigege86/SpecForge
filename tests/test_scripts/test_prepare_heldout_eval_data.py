import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.prepare_heldout_eval_data import (
    build_heldout_rows,
    conversation_fingerprint,
    exclusion_keys,
    is_heldout,
    normalize_eval_row,
    select_heldout,
    write_jsonl,
)


def _sharegpt(row_id: str, user: str, assistant: str) -> dict:
    return {
        "id": row_id,
        "conversations": [
            {"from": "human", "value": user},
            {"from": "gpt", "value": assistant},
        ],
    }


def _processed(row_id: str, user: str, assistant: str) -> dict:
    return {
        "id": row_id,
        "conversations": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
    }


class HeldoutEvalDataTest(unittest.TestCase):
    def test_fingerprint_is_stable_and_role_aware(self):
        row = _processed("1", "hi", "hello")
        first = conversation_fingerprint(row["conversations"])
        second = conversation_fingerprint(row["conversations"])
        self.assertEqual(first, second)
        other = conversation_fingerprint(
            _processed("1", "hi", "goodbye")["conversations"]
        )
        self.assertNotEqual(first, other)

    def test_raw_sharegpt_and_processed_rows_share_a_fingerprint(self):
        raw = normalize_eval_row(_sharegpt("a", "q", "a"))
        processed = _processed("a", "q", "a")
        self.assertEqual(
            conversation_fingerprint(raw["conversations"]),
            conversation_fingerprint(processed["conversations"]),
        )

    def test_exclusion_keys_cover_id_and_fingerprint(self):
        ids, fingerprints = exclusion_keys([_sharegpt("train-1", "q", "a")])
        self.assertEqual({"train-1"}, ids)
        self.assertEqual(
            {conversation_fingerprint(_processed("train-1", "q", "a")["conversations"])},
            fingerprints,
        )

    def test_same_id_or_same_text_is_not_heldout(self):
        train_ids, train_fps = exclusion_keys([_processed("1", "q", "a")])
        self.assertFalse(is_heldout(_processed("1", "other", "text"), train_ids, train_fps))
        self.assertFalse(is_heldout(_processed("2", "q", "a"), train_ids, train_fps))
        self.assertTrue(is_heldout(_processed("2", "other", "text"), train_ids, train_fps))

    def test_select_heldout_drops_training_rows(self):
        train_ids, train_fps = exclusion_keys([_processed("1", "q", "a")])
        selected = select_heldout(
            [
                _processed("1", "q", "a"),
                _processed("2", "held", "out"),
                _processed("3", "q", "a"),
            ],
            train_ids=train_ids,
            train_fingerprints=train_fps,
            num_samples=8,
            seed=0,
        )
        self.assertEqual(["2"], [row["id"] for row in selected])

    def test_build_heldout_rows_never_emits_train_conversations(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            train = root / "train.jsonl"
            source = root / "sharegpt.jsonl"
            write_jsonl(
                train,
                [_processed("1", "train user", "train assistant")],
            )
            write_jsonl(
                source,
                [
                    _sharegpt("1", "train user", "train assistant"),
                    _sharegpt("2", "eval user", "eval assistant"),
                    _sharegpt("3", "train user", "train assistant"),
                    _sharegpt("4", "another eval", "answer"),
                ],
            )
            rows, stats = build_heldout_rows(
                train_jsonl=train,
                num_samples=2,
                seed=0,
                sharegpt_jsonl=source,
                allow_ultrachat_fill=False,
            )
        self.assertEqual(2, stats["written"])
        self.assertEqual({"2", "4"}, {row["id"] for row in rows})
        train_ids, train_fps = exclusion_keys([_processed("1", "train user", "train assistant")])
        for row in rows:
            self.assertTrue(is_heldout(row, train_ids, train_fps))

    def test_build_heldout_rows_errors_when_sharegpt_cannot_fill(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            train = root / "train.jsonl"
            source = root / "sharegpt.jsonl"
            write_jsonl(train, [_processed("1", "q", "a")])
            write_jsonl(source, [_sharegpt("1", "q", "a")])
            with self.assertRaisesRegex(ValueError, "only 0 ShareGPT rows"):
                build_heldout_rows(
                    train_jsonl=train,
                    num_samples=2,
                    seed=0,
                    sharegpt_jsonl=source,
                    allow_ultrachat_fill=False,
                )
