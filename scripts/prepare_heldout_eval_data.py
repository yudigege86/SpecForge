"""Write an eval JSONL whose conversations are absent from a training JSONL.

The 1-epoch Qwen3-4B run trained on ``sharegpt_train.jsonl``. This command
excludes every training row by conversation id **and** by a fingerprint of the
processed messages, then writes a held-out file for offline capture.

It prefers leftover ShareGPT rows (same distribution). If those cannot fill
``--num-samples``, it backfills from UltraChat, which was not in the training
file at all.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any

try:
    from scripts.prepare_data import process_sharegpt_row, process_ultrachat_row
except ImportError:  # python scripts/prepare_heldout_eval_data.py
    from prepare_data import process_sharegpt_row, process_ultrachat_row

ProcessedRow = dict[str, Any]


def conversation_fingerprint(conversations: Iterable[Mapping[str, Any]]) -> str:
    payload = [
        {"role": str(message.get("role", "")), "content": str(message.get("content", ""))}
        for message in conversations
    ]
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def usable_chat(row: Mapping[str, Any]) -> bool:
    conversations = row.get("conversations") or []
    if not isinstance(conversations, list) or not conversations:
        return False
    roles = {str(message.get("role")) for message in conversations}
    return "user" in roles and "assistant" in roles


def normalize_eval_row(row: Mapping[str, Any]) -> ProcessedRow | None:
    conversations = row.get("conversations")
    if isinstance(conversations, list) and conversations:
        first = conversations[0]
        if isinstance(first, Mapping) and "from" in first:
            processed, _skipped = process_sharegpt_row(row)
            return processed if usable_chat(processed) else None
        if isinstance(first, Mapping) and "role" in first:
            if "id" not in row:
                return None
            processed = {
                "id": str(row["id"]),
                "conversations": [
                    {"role": str(message["role"]), "content": str(message["content"])}
                    for message in conversations
                    if "role" in message and "content" in message
                ],
            }
            return processed if usable_chat(processed) else None
    if "messages" in row:
        processed, _skipped = process_ultrachat_row(row)
        return processed if usable_chat(processed) else None
    return None


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def exclusion_keys(rows: Iterable[Mapping[str, Any]]) -> tuple[set[str], set[str]]:
    ids: set[str] = set()
    fingerprints: set[str] = set()
    for row in rows:
        processed = normalize_eval_row(row)
        if processed is None:
            if "id" in row:
                ids.add(str(row["id"]))
            conversations = row.get("conversations")
            if isinstance(conversations, list) and conversations:
                first = conversations[0]
                if isinstance(first, Mapping) and "role" in first:
                    fingerprints.add(conversation_fingerprint(conversations))
            continue
        ids.add(str(processed["id"]))
        fingerprints.add(conversation_fingerprint(processed["conversations"]))
    return ids, fingerprints


def is_heldout(
    row: Mapping[str, Any], train_ids: set[str], train_fingerprints: set[str]
) -> bool:
    if not usable_chat(row):
        return False
    if str(row["id"]) in train_ids:
        return False
    return conversation_fingerprint(row["conversations"]) not in train_fingerprints


def select_heldout(
    candidates: Iterable[Mapping[str, Any]],
    *,
    train_ids: set[str],
    train_fingerprints: set[str],
    num_samples: int,
    seed: int,
) -> list[ProcessedRow]:
    heldout = [
        dict(row)
        for row in candidates
        if is_heldout(row, train_ids, train_fingerprints)
    ]
    rng = random.Random(seed)
    rng.shuffle(heldout)
    return heldout[:num_samples]


def _processed_from_source(rows: Iterable[Mapping[str, Any]]) -> Iterator[ProcessedRow]:
    for row in rows:
        processed = normalize_eval_row(row)
        if processed is not None:
            yield processed


def load_sharegpt_source(path: Path | None) -> list[ProcessedRow]:
    if path is not None:
        return list(_processed_from_source(iter_jsonl(path)))
    try:
        from scripts.prepare_data import load_dataset_preset
    except ImportError:
        from prepare_data import load_dataset_preset

    dataset, _processor = load_dataset_preset("sharegpt")
    return list(_processed_from_source(dataset))


def load_ultrachat_source() -> Iterator[ProcessedRow]:
    try:
        from scripts.prepare_data import load_dataset_preset
    except ImportError:
        from prepare_data import load_dataset_preset

    dataset, _processor = load_dataset_preset("ultrachat")
    yield from _processed_from_source(dataset)


def build_heldout_rows(
    *,
    train_jsonl: Path,
    num_samples: int,
    seed: int,
    sharegpt_jsonl: Path | None = None,
    allow_ultrachat_fill: bool = True,
) -> tuple[list[ProcessedRow], dict[str, int]]:
    train_ids, train_fingerprints = exclusion_keys(iter_jsonl(train_jsonl))
    sharegpt_rows = load_sharegpt_source(sharegpt_jsonl)
    selected = select_heldout(
        sharegpt_rows,
        train_ids=train_ids,
        train_fingerprints=train_fingerprints,
        num_samples=num_samples,
        seed=seed,
    )
    stats = {
        "train_rows_excluded": len(train_ids),
        "sharegpt_heldout": len(selected),
        "ultrachat_heldout": 0,
    }
    if len(selected) < num_samples:
        if not allow_ultrachat_fill:
            raise ValueError(
                f"only {len(selected)} ShareGPT rows are absent from {train_jsonl}; "
                f"need {num_samples}. Pass a larger source or allow UltraChat fill."
            )
        needed = num_samples - len(selected)
        already = {str(row["id"]) for row in selected}
        already_fp = {conversation_fingerprint(row["conversations"]) for row in selected}
        fill: list[ProcessedRow] = []
        for row in load_ultrachat_source():
            if is_heldout(row, train_ids | already, train_fingerprints | already_fp):
                fill.append(row)
                already.add(str(row["id"]))
                already_fp.add(conversation_fingerprint(row["conversations"]))
            if len(fill) >= needed:
                break
        stats["ultrachat_heldout"] = len(fill)
        selected.extend(fill)
    if len(selected) < num_samples:
        raise ValueError(
            f"could only collect {len(selected)} held-out rows; need {num_samples}"
        )
    for row in selected:
        if not is_heldout(row, train_ids, train_fingerprints):
            raise RuntimeError(f"held-out row leaked training data: id={row['id']!r}")
    stats["written"] = len(selected)
    return selected, stats


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=build_heldout_rows.__doc__)
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--sharegpt-jsonl",
        type=Path,
        default=None,
        help="Optional local ShareGPT JSONL (raw or processed). Default: HF preset.",
    )
    parser.add_argument(
        "--no-ultrachat-fill",
        action="store_true",
        help="Fail instead of filling leftover slots from UltraChat.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.num_samples <= 0:
        raise SystemExit("--num-samples must be > 0")
    rows, stats = build_heldout_rows(
        train_jsonl=args.train_jsonl,
        num_samples=args.num_samples,
        seed=args.seed,
        sharegpt_jsonl=args.sharegpt_jsonl,
        allow_ultrachat_fill=not args.no_ultrachat_fill,
    )
    written = write_jsonl(args.output_jsonl, rows)
    print(json.dumps({"output": str(args.output_jsonl), "written": written, **stats}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
