#!/usr/bin/env python3
"""Offline teacher-forced MAL for DFlash-family drafts.

Accepts stock ``DFlashDraftModel`` and ``DFlashLinearDraftModel`` exports.
Datasets: SPEED-Bench Qualitative, HumanEval, and MT-Bench. Generation stops
at EOS; reports overall and per-category mean accept length.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.request
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Iterable, Optional

PLACEHOLDER = "FULL BENCHMARK DATA SHOULD BE FETCHED FROM THE SOURCE USING SPECDEC_BENCH"
SPEEDBENCH_REPO = "nvidia/SPEED-Bench"
PREPARE_SCRIPT_URL = (
    "https://raw.githubusercontent.com/NVIDIA-NeMo/Skills/"
    "refs/heads/main/nemo_skills/dataset/speed-bench/prepare.py"
)
HUMANEVAL_REPO = "openai/openai_humaneval"
MTBENCH_URL = (
    "https://raw.githubusercontent.com/lm-sys/FastChat/main/"
    "fastchat/llm_judge/data/mt_bench/question.jsonl"
)
REQUIRED_FIELDS = (
    "question_id",
    "category",
    "turns",
)
PREPARE_DATASETS = (
    "qualitative",
    "humaneval",
    "mt-bench",
)


def turns_have_placeholder(turns: Iterable[Any]) -> bool:
    return any(PLACEHOLDER in str(turn) for turn in turns or [])


def assert_no_placeholders(rows: list[dict[str, Any]], *, source: str) -> None:
    bad = [
        row.get("question_id")
        for row in rows
        if turns_have_placeholder(row.get("turns") or [])
    ]
    if bad:
        raise SystemExit(
            f"{source} still contains {len(bad)} placeholder SPEED-Bench rows "
            f"(example question_id={bad[0]!r}). Re-run prepare-speedbench."
        )


def turns_to_messages(turns: list[Any]) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for turn in turns:
        if isinstance(turn, dict):
            role = str(turn.get("role") or "user")
            content = str(turn.get("content") or turn.get("value") or "")
        else:
            role = "user"
            content = str(turn)
        if not content.strip():
            continue
        messages.append({"role": role, "content": content})
    if not messages:
        raise ValueError("turns produced no chat messages")
    if messages[-1]["role"] != "user":
        raise ValueError("last turn must be a user message")
    return messages


def eval_protocol(row: dict[str, Any]) -> str:
    protocol = str(row.get("protocol") or "").strip().lower()
    if protocol:
        return protocol
    source = str(row.get("source") or "").strip().lower()
    if source in {"mt-bench", "mtbench"}:
        return "mt_bench"
    return "concat_user"


def humaneval_user_content(prompt: str) -> str:
    # z-lab/dflash benchmark.py HumanEval wrap (model-card protocol).
    return (
        "Write a solution to the following problem and make sure that it "
        f"passes the tests:\n```python\n{prompt}\n```"
    )


def humaneval_example_to_row(example: dict[str, Any]) -> dict[str, Any]:
    task_id = example.get("task_id") or example.get("question_id")
    prompt = str(example.get("prompt") or "")
    if not task_id or not prompt.strip():
        raise ValueError("HumanEval example missing task_id or prompt")
    return {
        "question_id": str(task_id),
        "category": "coding",
        "sub_category": str(example.get("entry_point") or ""),
        "turns": [humaneval_user_content(prompt)],
        "source": "humaneval",
        "protocol": "concat_user",
        "entry_point": example.get("entry_point"),
    }


def mtbench_example_to_row(example: dict[str, Any]) -> dict[str, Any]:
    question_id = example.get("question_id")
    turns = list(example.get("turns") or [])
    if question_id is None or len(turns) < 1:
        raise ValueError("MT-Bench example missing question_id or turns")
    return {
        "question_id": str(question_id),
        "category": str(example.get("category") or "unknown"),
        "turns": [str(turn) for turn in turns],
        "source": "mt-bench",
        "protocol": "mt_bench",
        "multiturn": True,
    }


def prompt_messages_for_turn(
    row: dict[str, Any],
    assistant_replies: list[str],
) -> list[dict[str, str]]:
    """Chat messages to score for the next user turn.

    ``concat_user`` (SPEED-Bench Qualitative, HumanEval) concatenates every
    user turn into one prompt with no assistant replies. That is **not**
    z-lab's loop, which generates and appends an assistant message after
    each turn. ``mt_bench`` scores one user turn at a time; default
    ``mt_bench_turns=first`` matches the z-lab card (turn 1 only).
    """

    turns = turns_to_messages(row["turns"])
    if eval_protocol(row) != "mt_bench":
        return turns
    turn_i = len(assistant_replies)
    if turn_i >= len(turns):
        raise ValueError("no remaining MT-Bench turns")
    messages: list[dict[str, str]] = []
    for index, user in enumerate(turns):
        if index > turn_i:
            break
        messages.append(user)
        if index < turn_i:
            messages.append(
                {"role": "assistant", "content": assistant_replies[index]}
            )
    return messages


def score_units_for_row(
    row: dict[str, Any],
    *,
    mt_bench_turns: str = "first",
) -> int:
    if eval_protocol(row) == "mt_bench" and mt_bench_turns == "all":
        return max(len(turns_to_messages(row["turns"])), 1)
    return 1


def render_prompt(
    tokenizer,
    messages: list[dict[str, str]],
    *,
    enable_thinking: bool = True,
) -> str:
    kwargs: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": True,
        "enable_thinking": enable_thinking,
    }
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        try:
            return tokenizer.apply_chat_template(
                messages,
                chat_template_kwargs={"enable_thinking": enable_thinking},
                **kwargs,
            )
        except TypeError:
            return tokenizer.apply_chat_template(messages, **kwargs)


def render_prompt_ids(
    tokenizer,
    messages: list[dict[str, str]],
    *,
    enable_thinking: bool = True,
) -> list[int]:
    kwargs: dict[str, Any] = {
        "tokenize": True,
        "add_generation_prompt": True,
        "enable_thinking": enable_thinking,
    }
    try:
        ids = tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        try:
            ids = tokenizer.apply_chat_template(
                messages,
                chat_template_kwargs={"enable_thinking": enable_thinking},
                **kwargs,
            )
        except TypeError:
            ids = tokenizer.apply_chat_template(messages, **kwargs)
    return flatten_token_ids(ids)


def flatten_token_ids(ids: Any) -> list[int]:
    if hasattr(ids, "tolist") and not isinstance(ids, (list, tuple)):
        ids = ids.tolist()
    if isinstance(ids, Mapping) and "input_ids" in ids:
        ids = ids["input_ids"]
        if hasattr(ids, "tolist") and not isinstance(ids, (list, tuple)):
            ids = ids.tolist()
    if isinstance(ids, tuple):
        ids = list(ids)
    if isinstance(ids, list) and ids and isinstance(ids[0], (list, tuple)):
        ids = list(ids[0])
    if not isinstance(ids, list) or not ids or isinstance(ids[0], str):
        raise TypeError(f"apply_chat_template tokenize=True returned {type(ids)!r}")
    return [int(x) for x in ids]


def load_eval_rows(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if "turns" not in row and row.get("messages"):
                row["turns"] = [
                    message.get("content")
                    for message in row["messages"]
                    if message.get("role") == "user"
                ]
            missing = [key for key in REQUIRED_FIELDS if not row.get(key)]
            if missing:
                raise SystemExit(f"{path} row missing {missing}")
            rows.append(row)
    if not rows:
        raise SystemExit(f"no eval rows in {path}")
    assert_no_placeholders(rows, source=path)
    return rows


def select_rows(
    rows: list[dict[str, Any]],
    *,
    n: Optional[int],
    categories: Optional[list[str]],
) -> list[dict[str, Any]]:
    selected = rows
    if categories:
        wanted = {item.lower() for item in categories}
        selected = [
            row for row in selected if str(row.get("category", "")).lower() in wanted
        ]
        if not selected:
            raise SystemExit(f"no rows matched categories {categories}")
    if n is None or n >= len(selected):
        return selected
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        by_category[str(row.get("category") or "unknown")].append(row)
    names = list(by_category)
    out: list[dict[str, Any]] = []
    index = 0
    while len(out) < n:
        name = names[index % len(names)]
        bucket = by_category[name]
        slot = index // len(names)
        if slot < len(bucket):
            out.append(bucket[slot])
        index += 1
        if index > n * max(len(names), 1) + len(selected):
            break
    return out[:n]


def _stop_ids(tokenizer, ignore_eos: bool) -> Optional[list[int]]:
    if ignore_eos:
        return None
    ids: list[int] = []
    eos = getattr(tokenizer, "eos_token_id", None)
    if isinstance(eos, int):
        ids.append(eos)
    elif isinstance(eos, (list, tuple)):
        ids.extend(int(x) for x in eos if x is not None)
    return ids or None


def clip_completion_at_eos(
    sequence_ids,
    prompt_len: int,
    stop_ids: Optional[list[int]],
):
    """Keep prompt plus completion through the first EOS, inclusive."""

    if not stop_ids:
        return sequence_ids, False
    completion = sequence_ids[0, prompt_len:]
    matches = []
    for stop_id in stop_ids:
        hits = (completion == int(stop_id)).nonzero(as_tuple=True)[0]
        if hits.numel():
            matches.append(int(hits[0].item()))
    if not matches:
        return sequence_ids, False
    cut = min(matches)
    return sequence_ids[:, : prompt_len + cut + 1], True


def mal_stats(values: list[float]) -> dict[str, Optional[float]]:
    if not values:
        return {
            "spec_accept_length_mean": None,
            "spec_accept_length_p50": None,
        }
    return {
        "spec_accept_length_mean": float(statistics.mean(values)),
        "spec_accept_length_p50": float(statistics.median(values)),
    }


def _split_mal_stats(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    by_value: dict[str, list[float]] = defaultdict(list)
    block_by_value: dict[str, list[float]] = defaultdict(list)
    empty_by_value: dict[str, int] = defaultdict(int)
    n_by_value: dict[str, int] = defaultdict(int)
    for row in rows:
        name = row.get(key)
        if name is None or name == "":
            name = "unknown"
        name = str(name)
        n_by_value[name] += 1
        if int(row.get("completion_tokens") or 0) <= 0:
            empty_by_value[name] += 1
        if isinstance(row.get("spec_accept_length"), (int, float)):
            by_value[name].append(float(row["spec_accept_length"]))
        for length in row.get("accepts") or []:
            block_by_value[name].append(float(length))
    split: dict[str, dict[str, Any]] = {}
    for name in sorted(n_by_value):
        values = by_value.get(name) or []
        blocks = block_by_value.get(name) or []
        split[name] = {
            "n": n_by_value[name],
            "empty_outputs": empty_by_value.get(name, 0),
            **mal_stats(values),
            "block_accept_length_mean": (
                float(statistics.mean(blocks)) if blocks else None
            ),
        }
    return split


def _verify_count(row: dict[str, Any]) -> Optional[int]:
    verify = row.get("spec_verify_ct")
    try:
        if verify is not None:
            return int(verify)
    except (TypeError, ValueError):
        pass
    n_blocks = row.get("n_blocks")
    try:
        if n_blocks is not None:
            return int(n_blocks)
    except (TypeError, ValueError):
        pass
    accepts = row.get("accepts") or []
    return len(accepts) if accepts else None


def turn_card_accept_length(row: dict[str, Any]) -> Optional[float]:
    """z-lab card metric: completion_tokens / spec_verify_ct for one turn."""

    completion = int(row.get("completion_tokens") or 0)
    verify = _verify_count(row)
    if verify:
        return float(completion) / float(verify)
    value = row.get("spec_accept_length")
    if isinstance(value, (int, float)):
        return float(value)
    return None


def aggregate_mal_report(
    rows: list[dict[str, Any]],
    *,
    block_accepts: list[float],
    extra: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    per_prompt = [
        float(row["spec_accept_length"])
        for row in rows
        if isinstance(row.get("spec_accept_length"), (int, float))
    ]
    card_turns = [
        length
        for length in (turn_card_accept_length(row) for row in rows)
        if length is not None
    ]
    total_completion = sum(int(row.get("completion_tokens") or 0) for row in rows)
    total_verify = sum(_verify_count(row) or 0 for row in rows)
    report = {
        "n": len(rows),
        "empty_outputs": sum(1 for row in rows if int(row.get("completion_tokens") or 0) <= 0),
        **mal_stats(per_prompt),
        "card_accept_length_mean": (
            float(statistics.mean(card_turns)) if card_turns else None
        ),
        "token_weighted_accept_length": (
            float(total_completion) / float(total_verify) if total_verify else None
        ),
        "block_accept_length_mean": (
            float(statistics.mean(block_accepts)) if block_accepts else None
        ),
        "per_category": _split_mal_stats(rows, "category"),
        "per_sub_category": _split_mal_stats(rows, "sub_category"),
        "per_difficulty": _split_mal_stats(rows, "difficulty"),
        "per_multiturn": _split_mal_stats(rows, "multiturn"),
        "raw": rows,
    }
    if extra:
        report.update(extra)
    return report


def draft_report_metadata(draft) -> dict[str, Any]:
    config = getattr(draft, "config", None)
    architectures = list(getattr(config, "architectures", None) or [])
    method = dict(getattr(config, "dflash_config", None) or {})
    meta: dict[str, Any] = {
        "architectures": architectures,
        "block_size": int(getattr(draft, "block_size", 0) or 0),
        "target_layer_ids": list(getattr(draft, "target_layer_ids", None) or []),
        "mask_token_id": getattr(draft, "mask_token_id", None),
    }
    linear = method.get("linear_context")
    if linear:
        from specforge.modeling.draft.dflash_linear import resolve_linear_context_settings

        meta["linear_context"] = resolve_linear_context_settings(config)
    return meta


def generate_greedy(
    target,
    input_ids,
    *,
    tokenizer,
    max_new_tokens: int,
    ignore_eos: bool,
):
    stop_ids = _stop_ids(tokenizer, ignore_eos)
    use_cache = bool(getattr(generate_greedy, "_use_cache", True))
    gen_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "use_cache": use_cache,
        "attention_mask": input_ids.new_ones(input_ids.shape),
    }
    if stop_ids:
        gen_kwargs["eos_token_id"] = stop_ids
        gen_kwargs["pad_token_id"] = stop_ids[0]
    elif tokenizer.pad_token_id is not None:
        gen_kwargs["pad_token_id"] = tokenizer.pad_token_id
    try:
        sequence_ids = target.generate(input_ids, **gen_kwargs)
    except Exception as exc:
        message = str(exc)
        if use_cache and (
            "has_previous_state" in message or "DynamicCache" in message
        ):
            generate_greedy._use_cache = False
            gen_kwargs["use_cache"] = False
            sequence_ids = target.generate(input_ids, **gen_kwargs)
        else:
            raise
    prompt_len = int(input_ids.shape[1])
    sequence_ids, finished_on_eos = clip_completion_at_eos(
        sequence_ids, prompt_len, stop_ids
    )
    completion = int(sequence_ids.shape[1] - prompt_len)
    if not finished_on_eos and stop_ids and completion > 0:
        last = int(sequence_ids[0, -1].item())
        finished_on_eos = last in stop_ids
    hit_max = completion >= max_new_tokens and not finished_on_eos
    return sequence_ids, prompt_len, completion, finished_on_eos, hit_max


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response:
        dest.write_bytes(response.read())


def _rows_from_hf(config: str) -> list[dict[str, Any]]:
    from datasets import load_dataset

    dataset = load_dataset(SPEEDBENCH_REPO, config, split="test")
    rows = []
    for example in dataset:
        rows.append(
            {
                "question_id": example.get("question_id"),
                "category": example.get("category"),
                "sub_category": example.get("sub_category"),
                "turns": list(example.get("turns") or []),
                "source": example.get("source"),
                "src_id": example.get("src_id"),
                "difficulty": example.get("difficulty"),
                "multiturn": example.get("multiturn"),
            }
        )
    return rows


def _rows_from_humaneval() -> list[dict[str, Any]]:
    from datasets import load_dataset

    dataset = load_dataset(HUMANEVAL_REPO, split="test")
    rows = [humaneval_example_to_row(example) for example in dataset]
    if not rows:
        raise SystemExit(f"{HUMANEVAL_REPO} produced no rows")
    return rows


def _rows_from_mtbench() -> list[dict[str, Any]]:
    with tempfile.TemporaryDirectory(prefix="mtbench-prepare-") as tmp:
        dest = Path(tmp) / "mtbench.jsonl"
        print(f"downloading {MTBENCH_URL}", flush=True)
        _download(MTBENCH_URL, dest)
        rows: list[dict[str, Any]] = []
        for line in dest.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rows.append(mtbench_example_to_row(json.loads(line)))
    if not rows:
        raise SystemExit("MT-Bench prepare produced no rows")
    return rows


def _emit_prepared(out: Path, rows: list[dict[str, Any]], *, dataset: str) -> int:
    write_jsonl(out, rows)
    categories = sorted({str(row.get("category")) for row in rows})
    print(
        json.dumps(
            {
                "dataset": dataset,
                "out": str(out),
                "n": len(rows),
                "categories": categories,
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


def _rows_from_nvidia_prepare(config: str, workdir: Path) -> list[dict[str, Any]]:
    script = workdir / "prepare.py"
    _download(PREPARE_SCRIPT_URL, script)
    output_dir = workdir / "out"
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(script),
        "--config",
        config,
        "--output_dir",
        str(output_dir),
    ]
    subprocess.check_call(cmd)
    produced = output_dir / f"{config}.jsonl"
    if not produced.is_file():
        raise SystemExit(f"NVIDIA prepare.py did not write {produced}")
    rows: list[dict[str, Any]] = []
    with produced.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            example = json.loads(line)
            if example.get("turns"):
                turns = list(example["turns"])
            else:
                turns = [
                    message.get("content")
                    for message in example.get("messages") or []
                    if message.get("role") == "user"
                ]
            rows.append(
                {
                    "question_id": example.get("question_id"),
                    "category": example.get("category"),
                    "sub_category": example.get("sub_category"),
                    "turns": turns,
                    "source": example.get("source"),
                    "src_id": example.get("src_id"),
                    "difficulty": example.get("difficulty"),
                    "multiturn": example.get("multiturn"),
                }
            )
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def cmd_prepare_speedbench(args: argparse.Namespace) -> int:
    out = Path(args.out)
    config = getattr(args, "config", None) or "qualitative"
    print(f"loading {SPEEDBENCH_REPO} config={config}", flush=True)
    rows = _rows_from_hf(config)
    if any(turns_have_placeholder(row.get("turns") or []) for row in rows):
        print(
            f"placeholders present; running NVIDIA prepare.py from {PREPARE_SCRIPT_URL}",
            flush=True,
        )
        with tempfile.TemporaryDirectory(prefix="speedbench-prepare-") as tmp:
            rows = _rows_from_nvidia_prepare(config, Path(tmp))
    assert_no_placeholders(rows, source=f"{SPEEDBENCH_REPO}/{config}")
    return _emit_prepared(out, rows, dataset=f"speedbench-{config}")


def cmd_prepare(args: argparse.Namespace) -> int:
    dataset = str(args.dataset or "qualitative").lower()
    if dataset in {"qualitative", "speedbench", "speedbench-qualitative"}:
        args.config = getattr(args, "config", None) or "qualitative"
        return cmd_prepare_speedbench(args)
    out = Path(args.out)
    if dataset == "humaneval":
        print(f"loading {HUMANEVAL_REPO}", flush=True)
        rows = _rows_from_humaneval()
    elif dataset in {"mt-bench", "mtbench"}:
        rows = _rows_from_mtbench()
    else:
        raise SystemExit(
            f"unknown dataset {dataset!r}; choose one of {PREPARE_DATASETS}"
        )
    return _emit_prepared(out, rows, dataset=dataset)


def _pack_scored_row(
    row: dict[str, Any],
    *,
    question_id: Any,
    turn_index: Optional[int],
    completion: int,
    mean_accept: Optional[float],
    lengths: list[int],
    eos_stop: bool,
    hit_max: bool,
    extra: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    packed = {
        "question_id": question_id,
        "category": row.get("category"),
        "sub_category": row.get("sub_category"),
        "difficulty": row.get("difficulty"),
        "multiturn": row.get("multiturn"),
        "source": row.get("source"),
        "protocol": eval_protocol(row),
        "turn_index": turn_index,
        "completion_tokens": completion,
        "spec_accept_length": mean_accept,
        "spec_verify_ct": len(lengths),
        "n_blocks": len(lengths),
        "accepts": lengths,
        "finished_on_eos": eos_stop,
        "hit_max_new_tokens": hit_max,
    }
    if extra:
        packed.update(extra)
    return packed


def _decode_completion(tokenizer, sequence_ids, prompt_len: int) -> str:
    new_ids = sequence_ids[0, prompt_len:]
    return tokenizer.decode(new_ids, skip_special_tokens=True)


def cmd_mal(args: argparse.Namespace) -> int:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from specforge.modeling.auto import AutoDraftModel
    from specforge.modeling.draft.dflash import context_feature_offset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    if device.type != "cuda":
        raise SystemExit("MAL eval needs a GPU")
    replay_json = getattr(args, "replay_json", None)
    feature_source = str(getattr(args, "feature_source", "hf") or "hf").lower()
    if feature_source not in ("hf", "sglang"):
        raise SystemExit(f"unknown --feature-source {feature_source!r}")
    if feature_source == "sglang" and not replay_json:
        raise SystemExit("mal --feature-source sglang requires --replay-json")
    replay_features = None
    if replay_json:
        rows = load_replay_trajectories(
            replay_json,
            n=args.n,
            categories=args.categories,
        )
        print(
            f"replaying {len(rows)} frozen trajectories from {replay_json}",
            flush=True,
        )
        if feature_source == "sglang":
            layer_ids = draft_target_layer_ids(args.draft)
            replay_features = capture_sglang_aux_features(
                args.target,
                [row["sequence_ids"] for row in rows],
                layer_ids,
            )
    else:
        if not args.eval_jsonl:
            raise SystemExit("mal requires --eval-jsonl or --replay-json")
        rows = select_rows(
            load_eval_rows(args.eval_jsonl),
            n=args.n,
            categories=args.categories,
        )
        print(
            f"loading tokenizer {args.target}; n={len(rows)} max_new_tokens={args.max_new_tokens}",
            flush=True,
        )
    tokenizer = None
    if not replay_json:
        tokenizer = AutoTokenizer.from_pretrained(args.target, trust_remote_code=True)
    print(f"loading target {args.target}", flush=True)
    target = AutoModelForCausalLM.from_pretrained(
        args.target,
        trust_remote_code=True,
        torch_dtype=dtype,
    )
    target.to(device)
    target.eval()
    print(f"loading draft {args.draft}", flush=True)
    draft = AutoDraftModel.from_pretrained(args.draft, torch_dtype=dtype)
    draft.to(device)
    draft.eval()
    if not hasattr(draft, "acceptance_along_sequence"):
        raise SystemExit(
            f"{args.draft} does not implement acceptance_along_sequence"
        )
    draft_meta = draft_report_metadata(draft)
    print("draft", json.dumps(draft_meta, default=str), flush=True)
    raw_offset = getattr(args, "feature_offset", "auto")
    if raw_offset in (None, "auto"):
        feature_offset = context_feature_offset(target)
    else:
        feature_offset = int(raw_offset)
    print(
        f"feature_offset={feature_offset} (cli={raw_offset}) feature_source={feature_source} "
        f"enable_thinking={args.enable_thinking} max_new_tokens={args.max_new_tokens} "
        f"mt_bench_turns={args.mt_bench_turns} replay={bool(replay_json)}",
        flush=True,
    )
    if replay_features:
        expected_width = int(len(draft.target_layer_ids) * draft.config.hidden_size)
        widths = {int(feat.shape[-1]) for feat in replay_features}
        print(
            f"sglang_aux n={len(replay_features)} widths={sorted(widths)} "
            f"expected_width={expected_width}",
            flush=True,
        )
        if expected_width not in widths:
            raise SystemExit(
                f"SGLang aux width {sorted(widths)} != draft concat width {expected_width}"
            )
        with torch.inference_mode():
            first_ids = torch.tensor(
                [rows[0]["sequence_ids"]], dtype=torch.long, device=device
            )
            hf_out = target(
                input_ids=first_ids,
                output_hidden_states=True,
                use_cache=False,
            )
            from specforge.modeling.draft.dflash import extract_context_feature

            hf_feat = extract_context_feature(
                hf_out.hidden_states, draft.target_layer_ids, offset=feature_offset
            )[0]
            sg_feat = replay_features[0].to(device=device)
            seq = min(hf_feat.shape[0], sg_feat.shape[0])
            print(
                f"feature_cosine_token_mean seq0={token_cosine_mean(hf_feat[:seq], sg_feat[:seq])}",
                flush=True,
            )
    total_units = (
        len(rows)
        if replay_json
        else sum(
            score_units_for_row(row, mt_bench_turns=args.mt_bench_turns) for row in rows
        )
    )
    print(
        f"score_units={total_units} protocol_mix="
        f"{sorted({eval_protocol(row) for row in rows})}",
        flush=True,
    )

    scored: list[dict[str, Any]] = []
    block_accepts: list[float] = []
    finished_on_eos = 0
    hit_max_new_tokens = 0
    started = time.perf_counter()
    unit_index = 0

    def score_messages(messages: list[dict[str, str]]) -> dict[str, Any]:
        nonlocal finished_on_eos, hit_max_new_tokens
        prompt_ids = render_prompt_ids(
            tokenizer, messages, enable_thinking=args.enable_thinking
        )
        input_ids = torch.tensor(
            [prompt_ids], dtype=torch.long, device=device
        )
        with torch.inference_mode():
            sequence_ids, prompt_len, completion, eos_stop, hit_max = generate_greedy(
                target,
                input_ids,
                tokenizer=tokenizer,
                max_new_tokens=args.max_new_tokens,
                ignore_eos=args.ignore_eos,
            )
            lengths: list[int] = []
            if completion > 0:
                lengths = draft.acceptance_along_sequence(
                    target,
                    sequence_ids,
                    prompt_len=prompt_len,
                    temperature=0.0,
                    feature_offset=feature_offset,
                )
        if eos_stop:
            finished_on_eos += 1
        if hit_max:
            hit_max_new_tokens += 1
        mean_accept = float(statistics.mean(lengths)) if lengths else None
        if lengths:
            block_accepts.extend(float(x) for x in lengths)
        verify_ct = len(lengths)
        if completion > 0 and verify_ct:
            mean_accept = float(completion) / float(verify_ct)
        return {
            "completion": completion,
            "lengths": lengths,
            "mean_accept": mean_accept,
            "eos_stop": eos_stop,
            "hit_max": hit_max,
            "text": _decode_completion(tokenizer, sequence_ids, prompt_len),
        }

    def score_frozen(row: dict[str, Any], target_hidden=None) -> dict[str, Any]:
        nonlocal finished_on_eos, hit_max_new_tokens
        sequence_ids = torch.tensor(
            [row["sequence_ids"]], dtype=torch.long, device=device
        )
        prompt_len = int(row["prompt_len"])
        completion = int(sequence_ids.shape[1] - prompt_len)
        with torch.inference_mode():
            lengths: list[int] = []
            if completion > 0:
                kwargs = {
                    "prompt_len": prompt_len,
                    "temperature": 0.0,
                    "feature_offset": feature_offset,
                }
                if target_hidden is not None:
                    kwargs["target_hidden"] = target_hidden.to(
                        device=device, dtype=dtype
                    ).unsqueeze(0)
                lengths = draft.acceptance_along_sequence(
                    target, sequence_ids, **kwargs
                )
        eos_stop = bool(row.get("finished_on_eos"))
        hit_max = bool(row.get("hit_max_new_tokens"))
        if eos_stop:
            finished_on_eos += 1
        if hit_max:
            hit_max_new_tokens += 1
        mean_accept = float(statistics.mean(lengths)) if lengths else None
        if lengths:
            block_accepts.extend(float(x) for x in lengths)
        verify_ct = len(lengths)
        if completion > 0 and verify_ct:
            mean_accept = float(completion) / float(verify_ct)
        return {
            "completion": completion,
            "lengths": lengths,
            "mean_accept": mean_accept,
            "eos_stop": eos_stop,
            "hit_max": hit_max,
        }

    if replay_json:
        for row_i, row in enumerate(rows):
            hidden = replay_features[row_i] if replay_features is not None else None
            one = score_frozen(row, target_hidden=hidden)
            scored.append(
                _pack_scored_row(
                    row,
                    question_id=row.get("question_id"),
                    turn_index=row.get("turn_index"),
                    completion=one["completion"],
                    mean_accept=one["mean_accept"],
                    lengths=one["lengths"],
                    eos_stop=one["eos_stop"],
                    hit_max=one["hit_max"],
                    extra={
                        "sglang_spec_accept_length": row.get("spec_accept_length"),
                        "sglang_spec_verify_ct": row.get("spec_verify_ct"),
                        "feature_offset": feature_offset,
                        "feature_source": feature_source,
                    },
                )
            )
            unit_index += 1
            if unit_index % 8 == 0 or unit_index == 1:
                so_far = [
                    item["spec_accept_length"]
                    for item in scored
                    if item["spec_accept_length"] is not None
                ]
                running = statistics.mean(so_far) if so_far else None
                print(
                    f"replay {unit_index}/{total_units} running_mal={running} "
                    f"last={one['mean_accept']} new_tokens={one['completion']} "
                    f"sglang={row.get('spec_accept_length')} "
                    f"category={row.get('category')}",
                    flush=True,
                )
    else:
        for row in rows:
            n_turns = score_units_for_row(row, mt_bench_turns=args.mt_bench_turns)
            replies: list[str] = []
            for turn_i in range(n_turns):
                messages = prompt_messages_for_turn(row, replies)
                one = score_messages(messages)
                question_id = row.get("question_id")
                turn_index = turn_i + 1 if n_turns > 1 else None
                if n_turns > 1:
                    question_id = f"{question_id}/t{turn_index}"
                scored.append(
                    _pack_scored_row(
                        row,
                        question_id=question_id,
                        turn_index=turn_index,
                        completion=one["completion"],
                        mean_accept=one["mean_accept"],
                        lengths=one["lengths"],
                        eos_stop=one["eos_stop"],
                        hit_max=one["hit_max"],
                    )
                )
                replies.append(one["text"] or "")
                unit_index += 1
                if unit_index % 8 == 0 or unit_index == 1:
                    so_far = [
                        item["spec_accept_length"]
                        for item in scored
                        if item["spec_accept_length"] is not None
                    ]
                    running = statistics.mean(so_far) if so_far else None
                    print(
                        f"mal {unit_index}/{total_units} running_mal={running} "
                        f"last={one['mean_accept']} new_tokens={one['completion']} "
                        f"category={row.get('category')}",
                        flush=True,
                    )

    report = aggregate_mal_report(
        scored,
        block_accepts=block_accepts,
        extra={
            "label": "offline_replay_mal" if replay_json else "offline_mal",
            "backend": "offline_replay" if replay_json else "offline",
            "replay_json": replay_json,
            "target": args.target,
            "draft": args.draft,
            "eval_jsonl": args.eval_jsonl,
            "n_prompts": len(rows),
            "max_new_tokens": args.max_new_tokens,
            "enable_thinking": args.enable_thinking,
            "mt_bench_turns": args.mt_bench_turns,
            "ignore_eos": args.ignore_eos,
            "feature_source": feature_source,
            "finished_on_eos": finished_on_eos,
            "hit_max_new_tokens": hit_max_new_tokens,
            "elapsed_s": time.perf_counter() - started,
            **draft_meta,
        },
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    summary_path = Path(args.summary) if args.summary else out.with_name("summary.md")
    write_summary(summary_path, report)
    printable = {
        key: report[key]
        for key in (
            "n",
            "spec_accept_length_mean",
            "spec_accept_length_p50",
            "block_accept_length_mean",
            "empty_outputs",
            "finished_on_eos",
            "hit_max_new_tokens",
            "per_category",
            "elapsed_s",
            "architectures",
        )
        if key in report
    }
    print(json.dumps(printable, indent=2), flush=True)
    print(f"wrote {out}", flush=True)
    if report["spec_accept_length_mean"] is None:
        raise SystemExit("MAL eval produced no acceptance lengths")
    return 0


def sglang_sampling_params(
    *,
    max_new_tokens: int,
    ignore_eos: bool,
) -> dict[str, Any]:
    return {
        "temperature": 0.0,
        "max_new_tokens": int(max_new_tokens),
        "ignore_eos": bool(ignore_eos),
    }


def _finish_kind(finish: Any) -> str:
    if isinstance(finish, dict):
        finish = finish.get("type") or finish.get("reason") or ""
    return str(finish or "").lower()


def parse_sglang_generate(
    payload: Any,
    *,
    max_new_tokens: int,
    prompt_ids: Optional[list[int]] = None,
) -> dict[str, Any]:
    if isinstance(payload, list):
        if not payload:
            raise ValueError("SGLang generate returned an empty list")
        payload = payload[0]
    if not isinstance(payload, dict):
        raise ValueError(f"SGLang generate payload is {type(payload)!r}")
    meta = payload.get("meta_info") or payload.get("meta") or {}
    if not isinstance(meta, dict):
        meta = {}
    completion = int(meta.get("completion_tokens") or 0)
    verify = meta.get("spec_verify_ct")
    try:
        verify_ct = int(verify) if verify is not None else None
    except (TypeError, ValueError):
        verify_ct = None
    accept = meta.get("spec_accept_length")
    try:
        accept_f = float(accept) if accept is not None else None
    except (TypeError, ValueError):
        accept_f = None
    if accept_f is None and verify_ct:
        accept_f = float(completion) / float(verify_ct)
    kind = _finish_kind(meta.get("finish_reason"))
    finished_on_eos = kind in {"stop", "eos"}
    hit_max = kind in {"length", "max_new_tokens", "abort"} or (
        not finished_on_eos and completion >= int(max_new_tokens)
    )
    if finished_on_eos:
        hit_max = False
    raw_ids = extract_sglang_output_ids(payload)
    split_prompt = _as_int_list(prompt_ids)
    split_completion: Optional[list[int]] = None
    if raw_ids:
        prompt_tokens = meta.get("prompt_tokens")
        try:
            prompt_tokens_i = int(prompt_tokens) if prompt_tokens is not None else None
        except (TypeError, ValueError):
            prompt_tokens_i = None
        try:
            split_prompt, split_completion = split_prompt_and_completion(
                raw_ids,
                prompt_ids=split_prompt,
                prompt_tokens=prompt_tokens_i,
                completion_tokens=completion or None,
            )
        except ValueError:
            split_completion = None
    return {
        "text": payload.get("text"),
        "completion_tokens": completion,
        "spec_accept_length": accept_f,
        "spec_verify_ct": verify_ct,
        "finished_on_eos": finished_on_eos,
        "hit_max_new_tokens": hit_max,
        "finish_reason": meta.get("finish_reason"),
        "prompt_ids": split_prompt,
        "completion_ids": split_completion,
        "output_ids": raw_ids,
        "payload_keys": sorted(str(key) for key in payload.keys()),
    }


def _as_int_list(value: Any) -> Optional[list[int]]:
    if value is None:
        return None
    if isinstance(value, tuple):
        value = list(value)
    if not isinstance(value, list):
        return None
    if not value:
        return []
    first = value[0]
    if isinstance(first, (list, tuple)) and first:
        return [int(item[0]) for item in value]
    return [int(item) for item in value]


def extract_sglang_output_ids(payload: dict[str, Any]) -> Optional[list[int]]:
    meta = payload.get("meta_info") or payload.get("meta") or {}
    if not isinstance(meta, dict):
        meta = {}
    for candidate in (
        payload.get("output_ids"),
        payload.get("token_ids"),
        meta.get("output_ids"),
        meta.get("output_token_ids"),
        meta.get("completion_ids"),
        payload.get("output_token_logprobs"),
        meta.get("output_token_logprobs"),
    ):
        ids = _as_int_list(candidate)
        if ids:
            return ids
    return None


def split_prompt_and_completion(
    output_ids: list[int],
    *,
    prompt_ids: Optional[list[int]],
    prompt_tokens: Optional[int],
    completion_tokens: Optional[int],
) -> tuple[list[int], list[int]]:
    if prompt_ids and output_ids[: len(prompt_ids)] == prompt_ids:
        return list(prompt_ids), output_ids[len(prompt_ids) :]
    if prompt_tokens and len(output_ids) >= int(prompt_tokens):
        prompt = output_ids[: int(prompt_tokens)]
        completion = output_ids[int(prompt_tokens) :]
        if completion_tokens is None or len(completion) == int(completion_tokens):
            return prompt, completion
    if completion_tokens is not None and len(output_ids) == int(completion_tokens):
        if not prompt_ids:
            raise ValueError("completion-only output_ids require prompt_ids")
        return list(prompt_ids), output_ids
    if completion_tokens and len(output_ids) > int(completion_tokens):
        cut = len(output_ids) - int(completion_tokens)
        return output_ids[:cut], output_ids[cut:]
    if prompt_ids:
        return list(prompt_ids), output_ids
    raise ValueError("could not split prompt and completion token ids")


def draft_target_layer_ids(draft_path: str) -> list[int]:
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(draft_path, trust_remote_code=True)
    method = dict(getattr(config, "dflash_config", None) or {})
    layer_ids = method.get("target_layer_ids")
    if not layer_ids:
        raise SystemExit(f"{draft_path} has no dflash_config.target_layer_ids")
    return [int(x) for x in layer_ids]


def normalize_aux_feature(aux: Any, seq_len: int):
    import torch

    tensor = aux
    if isinstance(tensor, (list, tuple)):
        parts = []
        for item in tensor:
            item_t = item if torch.is_tensor(item) else torch.as_tensor(item)
            if item_t.dim() == 3:
                item_t = item_t.reshape(item_t.shape[-2], item_t.shape[-1])
            elif item_t.dim() == 1:
                item_t = item_t.unsqueeze(0)
            parts.append(item_t)
        tensor = torch.cat(parts, dim=-1)
    else:
        tensor = tensor if torch.is_tensor(tensor) else torch.as_tensor(tensor)
    if tensor.dim() == 3:
        if tensor.shape[0] == 1:
            tensor = tensor[0]
        elif int(tensor.shape[0]) == int(seq_len):
            tensor = tensor.reshape(seq_len, -1)
        else:
            raise ValueError(
                f"expected packed aux [1, L, H] or [L, *, *], got {tuple(tensor.shape)}"
            )
    if tensor.dim() != 2:
        raise ValueError(f"aux feature rank {tensor.dim()} shape={tuple(tensor.shape)}")
    if tensor.shape[0] != seq_len and tensor.shape[1] == seq_len:
        tensor = tensor.transpose(0, 1).contiguous()
    if int(tensor.shape[0]) != int(seq_len):
        raise ValueError(f"aux seq {tensor.shape[0]} != sequence length {seq_len}")
    return tensor.detach().to("cpu")


def _ensure_single_rank_dist() -> None:
    import os
    import socket

    import torch.distributed as dist

    from specforge.distributed import get_tp_group, init_distributed

    if get_tp_group() is not None:
        return
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    if "MASTER_PORT" not in os.environ:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            os.environ["MASTER_PORT"] = str(sock.getsockname()[1])
    if not dist.is_initialized():
        init_distributed(tp_size=1)


def _bind_dflash_capture_layers(capture, layer_ids: list[int]) -> None:
    try:
        capture.set_capture_layers(layer_ids, capture_method="dflash")
        return
    except Exception as exc:
        last = exc
    root = getattr(getattr(capture, "_backend", None), "model_runner", None)
    root = getattr(root, "model", None)
    queue = [root]
    seen: set[int] = set()
    while queue:
        obj = queue.pop(0)
        if obj is None:
            continue
        obj_id = id(obj)
        if obj_id in seen:
            continue
        seen.add(obj_id)
        setter = getattr(obj, "set_dflash_layers_to_capture", None)
        if callable(setter):
            setter(layer_ids)
            print(f"set_dflash_layers_to_capture on {type(obj).__name__}", flush=True)
            return
        for name in ("model", "language_model"):
            nxt = getattr(obj, name, None)
            if nxt is not None:
                queue.append(nxt)
    raise RuntimeError(
        f"could not set DFLASH capture layers {layer_ids}: {last}"
    )


def capture_sglang_aux_features(
    model_path: str,
    sequences: list[list[int]],
    layer_ids: list[int],
    *,
    mem_fraction_static: float = 0.85,
) -> list[Any]:
    import gc

    import torch

    from specforge.offline_capture import OfflineSGLangCapture

    _ensure_single_rank_dist()
    print(
        f"loading SGLang capture target {model_path} layers={layer_ids} "
        f"mem_fraction_static={mem_fraction_static}",
        flush=True,
    )
    capture = OfflineSGLangCapture.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        mem_fraction_static=mem_fraction_static,
        disable_radix_cache=True,
    )
    _bind_dflash_capture_layers(capture, list(layer_ids))
    features = []
    try:
        for index, ids in enumerate(sequences):
            aux_rows, _last = capture.capture_rows([list(map(int, ids))])
            if not aux_rows:
                raise RuntimeError("SGLang capture returned no aux rows")
            feat = normalize_aux_feature(aux_rows[0], len(ids))
            features.append(feat)
            print(
                f"sglang-capture {index + 1}/{len(sequences)} seq={len(ids)} "
                f"aux={tuple(feat.shape)}",
                flush=True,
            )
    finally:
        runner = getattr(getattr(capture, "_backend", None), "model_runner", None)
        if runner is not None and getattr(runner, "model", None) is not None:
            try:
                runner.model.to("cpu")
            except Exception:
                pass
            runner.model = None
        del capture
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return features


def token_cosine_mean(left, right) -> float:
    import torch

    a = left.float().reshape(-1, left.shape[-1])
    b = right.float().reshape(-1, right.shape[-1])
    seq = min(a.shape[0], b.shape[0])
    a = a[:seq]
    b = b[:seq]
    denom = a.norm(dim=-1).clamp_min(1e-12) * b.norm(dim=-1).clamp_min(1e-12)
    return float(((a * b).sum(dim=-1) / denom).mean().item())


def load_replay_trajectories(
    path: str,
    *,
    n: Optional[int] = None,
    categories: Optional[list[str]] = None,
) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    raw = payload.get("raw") if isinstance(payload, dict) else payload
    if not isinstance(raw, list) or not raw:
        raise SystemExit(f"{path} has no raw trajectories")
    rows: list[dict[str, Any]] = []
    for row in raw:
        if not isinstance(row, dict):
            raise SystemExit(f"{path} raw row is {type(row)!r}")
        prompt_ids = _as_int_list(row.get("prompt_ids"))
        completion_ids = _as_int_list(row.get("completion_ids"))
        sequence_ids = _as_int_list(row.get("sequence_ids"))
        prompt_len = row.get("prompt_len")
        if sequence_ids and prompt_len is not None:
            prompt_len_i = int(prompt_len)
            prompt_ids = sequence_ids[:prompt_len_i]
            completion_ids = sequence_ids[prompt_len_i:]
        if prompt_ids is None or completion_ids is None:
            raise SystemExit(
                f"{path} row {row.get('question_id')!r} missing prompt_ids/completion_ids"
            )
        packed = dict(row)
        packed["prompt_ids"] = prompt_ids
        packed["completion_ids"] = completion_ids
        packed["sequence_ids"] = prompt_ids + completion_ids
        packed["prompt_len"] = len(prompt_ids)
        packed["turns"] = packed.get("turns") or [packed.get("text") or "replay"]
        packed["category"] = packed.get("category") or "unknown"
        packed["question_id"] = packed.get("question_id")
        rows.append(packed)
    return select_rows(rows, n=n, categories=categories)


def sglang_is_ready(
    info: dict[str, Any] | None,
    *,
    require_dflash: bool = True,
) -> tuple[bool, str]:
    payload = info or {}
    algorithm = str(
        payload.get("speculative_algorithm") or payload.get("spec_algorithm") or ""
    ).upper()
    if require_dflash:
        if algorithm == "DFLASH":
            return True, algorithm
        if algorithm:
            raise SystemExit(
                f"SGLang speculative_algorithm={algorithm!r}, expected DFLASH"
            )
        return False, algorithm
    return True, algorithm


def wait_sglang(base: str, *, timeout: float, require_dflash: bool = True) -> dict[str, Any]:
    import urllib.error
    import urllib.request

    base = base.rstrip("/")
    started = time.time()
    last_err = "not attempted"
    info: dict[str, Any] = {}
    while time.time() - started <= timeout:
        try:
            urllib.request.urlopen(base + "/health", timeout=5)
            try:
                with urllib.request.urlopen(base + "/get_server_info", timeout=5) as resp:
                    info = json.loads(resp.read().decode())
            except Exception:
                info = {}
            ready, algorithm = sglang_is_ready(info, require_dflash=require_dflash)
            if not ready:
                last_err = f"waiting for DFLASH, got {algorithm or 'unknown'}"
                time.sleep(5)
                continue
            print(f"healthy {base} speculative_algorithm={algorithm or 'unknown'}", flush=True)
            return info
        except SystemExit:
            raise
        except Exception as exc:  # noqa: BLE001
            last_err = str(exc)
            time.sleep(5)
    raise SystemExit(f"SGLang health timeout after {timeout:.0f}s: {last_err}")


def compare_mal_reports(
    offline: dict[str, Any],
    sglang: dict[str, Any],
) -> dict[str, Any]:
    offline_raw = {
        str(row.get("question_id")): row
        for row in (offline.get("raw") or [])
        if row.get("question_id") is not None
    }
    matched = []
    missing_offline = []
    missing_sglang_ids = set(offline_raw)
    for row in sglang.get("raw") or []:
        qid = str(row.get("question_id"))
        missing_sglang_ids.discard(qid)
        other = offline_raw.get(qid)
        if other is None:
            missing_offline.append(qid)
            continue
        left = other.get("spec_accept_length")
        right = row.get("spec_accept_length")
        if not isinstance(left, (int, float)) or not isinstance(right, (int, float)):
            continue
        matched.append(
            {
                "question_id": qid,
                "category": row.get("category") or other.get("category"),
                "offline": float(left),
                "sglang": float(right),
                "delta": float(right) - float(left),
            }
        )
    by_category: dict[str, list[float]] = defaultdict(list)
    for item in matched:
        by_category[str(item.get("category") or "unknown")].append(item["delta"])

    def _mean(values: list[float]) -> Optional[float]:
        return float(statistics.mean(values)) if values else None

    deltas = [item["delta"] for item in matched]
    per_category = {}
    for category in sorted(set(offline.get("per_category") or {}) | set(sglang.get("per_category") or {})):
        off = (offline.get("per_category") or {}).get(category) or {}
        sg = (sglang.get("per_category") or {}).get(category) or {}
        per_category[category] = {
            "n_matched": len(by_category.get(category) or []),
            "offline": off.get("spec_accept_length_mean"),
            "sglang": sg.get("spec_accept_length_mean"),
            "delta_mean": _mean(by_category.get(category) or []),
        }
    return {
        "n_offline": offline.get("n"),
        "n_sglang": sglang.get("n"),
        "n_matched": len(matched),
        "offline_mean": offline.get("spec_accept_length_mean"),
        "sglang_mean": sglang.get("spec_accept_length_mean"),
        "abs_delta_mean": (
            float(statistics.mean(abs(x) for x in deltas)) if deltas else None
        ),
        "delta_mean": _mean(deltas),
        "delta_p50": float(statistics.median(deltas)) if deltas else None,
        "per_category": per_category,
        "missing_in_offline": missing_offline,
        "missing_in_sglang": sorted(missing_sglang_ids),
        "pairs": matched,
    }


def write_compare_summary(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# SPEED-Bench offline MAL vs SGLang DFLASH",
        "",
        f"n_matched: {report.get('n_matched')}",
        f"offline_mean: {report.get('offline_mean')}",
        f"sglang_mean: {report.get('sglang_mean')}",
        f"delta_mean (sglang-offline): {report.get('delta_mean')}",
        f"abs_delta_mean: {report.get('abs_delta_mean')}",
        f"delta_p50: {report.get('delta_p50')}",
        "",
        "## per category",
        "",
    ]
    for category, stats in (report.get("per_category") or {}).items():
        lines.append(
            f"- {category}: n={stats.get('n_matched')} "
            f"offline={stats.get('offline')} sglang={stats.get('sglang')} "
            f"delta={stats.get('delta_mean')}"
        )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def cmd_sglang_mal(args: argparse.Namespace) -> int:
    import urllib.error
    import urllib.request

    from transformers import AutoTokenizer

    rows = select_rows(
        load_eval_rows(args.eval_jsonl),
        n=args.n,
        categories=args.categories,
    )
    wait_sglang(args.base, timeout=args.wait_timeout, require_dflash=True)
    print(
        f"SGLang MAL n={len(rows)} max_new_tokens={args.max_new_tokens} "
        f"enable_thinking={args.enable_thinking} mt_bench_turns={args.mt_bench_turns} "
        f"base={args.base}",
        flush=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.target, trust_remote_code=True)
    url = args.base.rstrip("/") + "/generate"
    total_units = sum(
        score_units_for_row(row, mt_bench_turns=args.mt_bench_turns) for row in rows
    )
    scored: list[dict[str, Any]] = []
    block_accepts: list[float] = []
    finished_on_eos = 0
    hit_max_new_tokens = 0
    started = time.perf_counter()
    unit_index = 0

    def generate_one(messages: list[dict[str, str]]) -> dict[str, Any]:
        nonlocal finished_on_eos, hit_max_new_tokens
        text = render_prompt(
            tokenizer, messages, enable_thinking=args.enable_thinking
        )
        prompt_ids = render_prompt_ids(
            tokenizer, messages, enable_thinking=args.enable_thinking
        )
        payload = {
            "text": text,
            "input_ids": prompt_ids,
            "stream": False,
            "sampling_params": sglang_sampling_params(
                max_new_tokens=args.max_new_tokens,
                ignore_eos=args.ignore_eos,
            ),
        }
        def post(body_payload: dict[str, Any]) -> Any:
            req = urllib.request.Request(
                url,
                data=json.dumps(body_payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=args.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))

        try:
            body = post(payload)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            if exc.code == 400 and "input_ids" in payload:
                payload.pop("input_ids", None)
                try:
                    body = post(payload)
                except urllib.error.HTTPError as retry_exc:
                    retry_detail = retry_exc.read().decode("utf-8", errors="replace")
                    raise SystemExit(
                        f"SGLang generate HTTP {retry_exc.code}: {retry_detail[:2000]}"
                    ) from retry_exc
            else:
                raise SystemExit(
                    f"SGLang generate HTTP {exc.code}: {detail[:2000]}"
                ) from exc
        parsed = parse_sglang_generate(
            body,
            max_new_tokens=args.max_new_tokens,
            prompt_ids=prompt_ids,
        )
        if not parsed.get("completion_ids"):
            payload["return_logprob"] = True
            payload["logprob_start_len"] = 0
            body = post(payload)
            parsed = parse_sglang_generate(
                body,
                max_new_tokens=args.max_new_tokens,
                prompt_ids=prompt_ids,
            )
        if not parsed.get("completion_ids") or not parsed.get("prompt_ids"):
            raise SystemExit(
                "SGLang generate did not return token ids for replay; "
                f"keys={parsed.get('payload_keys')}"
            )
        if parsed["finished_on_eos"]:
            finished_on_eos += 1
        if parsed["hit_max_new_tokens"]:
            hit_max_new_tokens += 1
        mean_accept = parsed["spec_accept_length"]
        if mean_accept is not None:
            block_accepts.append(float(mean_accept))
        return parsed

    for row in rows:
        n_turns = score_units_for_row(row, mt_bench_turns=args.mt_bench_turns)
        replies: list[str] = []
        for turn_i in range(n_turns):
            parsed = generate_one(prompt_messages_for_turn(row, replies))
            question_id = row.get("question_id")
            turn_index = turn_i + 1 if n_turns > 1 else None
            if n_turns > 1:
                question_id = f"{question_id}/t{turn_index}"
            scored.append(
                _pack_scored_row(
                    row,
                    question_id=question_id,
                    turn_index=turn_index,
                    completion=parsed["completion_tokens"],
                    mean_accept=parsed["spec_accept_length"],
                    lengths=[],
                    eos_stop=parsed["finished_on_eos"],
                    hit_max=parsed["hit_max_new_tokens"],
                    extra={
                        "spec_verify_ct": parsed["spec_verify_ct"],
                        "n_blocks": parsed["spec_verify_ct"] or 0,
                        "prompt_ids": parsed["prompt_ids"],
                        "completion_ids": parsed["completion_ids"],
                        "prompt_len": len(parsed["prompt_ids"] or []),
                    },
                )
            )
            replies.append(str(parsed.get("text") or ""))
            unit_index += 1
            if unit_index % 8 == 0 or unit_index == 1:
                so_far = [
                    item["spec_accept_length"]
                    for item in scored
                    if item["spec_accept_length"] is not None
                ]
                running = statistics.mean(so_far) if so_far else None
                print(
                    f"sglang {unit_index}/{total_units} running_mal={running} "
                    f"last={parsed['spec_accept_length']} "
                    f"new_tokens={parsed['completion_tokens']} "
                    f"category={row.get('category')}",
                    flush=True,
                )

    report = aggregate_mal_report(
        scored,
        block_accepts=block_accepts,
        extra={
            "label": "sglang_mal",
            "backend": "sglang",
            "base": args.base,
            "target": args.target,
            "draft": args.draft,
            "eval_jsonl": args.eval_jsonl,
            "n_prompts": len(rows),
            "max_new_tokens": args.max_new_tokens,
            "enable_thinking": args.enable_thinking,
            "mt_bench_turns": args.mt_bench_turns,
            "ignore_eos": args.ignore_eos,
            "finished_on_eos": finished_on_eos,
            "hit_max_new_tokens": hit_max_new_tokens,
            "elapsed_s": time.perf_counter() - started,
        },
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    summary_path = Path(args.summary) if args.summary else out.with_name("summary.md")
    write_summary(summary_path, report)
    print(json.dumps(
        {
            key: report[key]
            for key in (
                "n",
                "spec_accept_length_mean",
                "spec_accept_length_p50",
                "block_accept_length_mean",
                "empty_outputs",
                "finished_on_eos",
                "hit_max_new_tokens",
                "per_category",
                "elapsed_s",
            )
            if key in report
        },
        indent=2,
    ), flush=True)
    print(f"wrote {out}", flush=True)
    if report["spec_accept_length_mean"] is None:
        raise SystemExit("SGLang MAL eval produced no acceptance lengths")
    if args.compare_json:
        compare = compare_mal_reports(
            json.loads(Path(args.compare_json).read_text(encoding="utf-8")),
            report,
        )
        compare_path = Path(args.compare_out) if args.compare_out else out.with_name(
            "compare_sglang.json"
        )
        compare_path.write_text(json.dumps(compare, indent=2), encoding="utf-8")
        write_compare_summary(compare_path.with_name("compare_sglang.md"), compare)
        print(
            json.dumps(
                {
                    key: compare[key]
                    for key in (
                        "n_matched",
                        "offline_mean",
                        "sglang_mean",
                        "delta_mean",
                        "abs_delta_mean",
                        "per_category",
                    )
                },
                indent=2,
            ),
            flush=True,
        )
        print(f"wrote {compare_path}", flush=True)
    return 0


def cmd_compare_mal(args: argparse.Namespace) -> int:
    offline = json.loads(Path(args.offline).read_text(encoding="utf-8"))
    sglang = json.loads(Path(args.sglang).read_text(encoding="utf-8"))
    report = compare_mal_reports(offline, sglang)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_compare_summary(
        Path(args.summary) if args.summary else out.with_name("compare_sglang.md"),
        report,
    )
    print(json.dumps(
        {
            key: report[key]
            for key in (
                "n_matched",
                "offline_mean",
                "sglang_mean",
                "delta_mean",
                "abs_delta_mean",
                "per_category",
            )
        },
        indent=2,
    ), flush=True)
    return 0


def write_summary(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# MAL report",
        "",
        f"backend: {report.get('backend') or 'offline'}",
        f"target: {report.get('target')}",
        f"draft: {report.get('draft')}",
        f"architectures: {report.get('architectures')}",
        f"block_size: {report.get('block_size')}",
        f"n: {report.get('n')}",
        f"spec_accept_length_mean: {report.get('spec_accept_length_mean')}",
        f"card_accept_length_mean: {report.get('card_accept_length_mean')}",
        f"token_weighted_accept_length: {report.get('token_weighted_accept_length')}",
        f"spec_accept_length_p50: {report.get('spec_accept_length_p50')}",
        f"block_accept_length_mean: {report.get('block_accept_length_mean')}",
        f"empty_outputs: {report.get('empty_outputs')}",
        f"finished_on_eos: {report.get('finished_on_eos')}",
        f"hit_max_new_tokens: {report.get('hit_max_new_tokens')}",
        f"ignore_eos: {report.get('ignore_eos')}",
        f"enable_thinking: {report.get('enable_thinking')}",
        f"mt_bench_turns: {report.get('mt_bench_turns')}",
        f"max_new_tokens: {report.get('max_new_tokens')}",
        f"elapsed_s: {report.get('elapsed_s')}",
        "",
        "## per category",
        "",
    ]
    for category, stats in (report.get("per_category") or {}).items():
        lines.append(
            f"- {category}: n={stats.get('n')} "
            f"mal={stats.get('spec_accept_length_mean')} "
            f"p50={stats.get('spec_accept_length_p50')} "
            f"block={stats.get('block_accept_length_mean')} "
            f"empty={stats.get('empty_outputs')}"
        )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def add_generation_flags(parser: argparse.ArgumentParser, *, sglang: bool = False) -> None:
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--enable-thinking",
        dest="enable_thinking",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--disable-thinking",
        dest="enable_thinking",
        action="store_false",
    )
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument(
        "--mt-bench-turns",
        choices=("first", "all"),
        default="first",
    )
    if sglang:
        parser.add_argument("--timeout", type=float, default=1800)
        parser.add_argument("--wait-timeout", type=float, default=1200)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    prepare = sub.add_parser("prepare-speedbench")
    prepare.add_argument("--out", required=True)
    prepare.add_argument("--config", default="qualitative")
    prepare.set_defaults(func=cmd_prepare_speedbench)

    prepare_any = sub.add_parser("prepare")
    prepare_any.add_argument("--out", required=True)
    prepare_any.add_argument(
        "--dataset",
        default="qualitative",
        choices=list(PREPARE_DATASETS),
    )
    prepare_any.add_argument("--config", default="qualitative")
    prepare_any.set_defaults(func=cmd_prepare)

    mal = sub.add_parser("mal")
    mal.add_argument("--target", required=True)
    mal.add_argument("--draft", required=True)
    mal.add_argument("--eval-jsonl", default="")
    mal.add_argument("--replay-json", default=None)
    mal.add_argument(
        "--feature-offset",
        default="auto",
        help="HF hidden_states index offset. auto: 1 for all targets.",
    )
    mal.add_argument(
        "--feature-source",
        choices=("hf", "sglang"),
        default="hf",
        help="Prefix features for teacher-force: HuggingFace hidden_states or SGLang DFLASH aux capture.",
    )
    mal.add_argument("--out", required=True)
    mal.add_argument("--summary", default=None)
    mal.add_argument("--n", type=int, default=None)
    mal.add_argument("--categories", nargs="*", default=None)
    add_generation_flags(mal)
    mal.set_defaults(func=cmd_mal)

    sglang = sub.add_parser("sglang-mal")
    sglang.add_argument("--target", required=True)
    sglang.add_argument("--draft", default="")
    sglang.add_argument("--eval-jsonl", required=True)
    sglang.add_argument("--out", required=True)
    sglang.add_argument("--summary", default=None)
    sglang.add_argument("--base", default="http://127.0.0.1:30000")
    sglang.add_argument("--n", type=int, default=None)
    sglang.add_argument("--categories", nargs="*", default=None)
    add_generation_flags(sglang, sglang=True)
    sglang.add_argument("--compare-json", default=None)
    sglang.add_argument("--compare-out", default=None)
    sglang.set_defaults(func=cmd_sglang_mal)

    compare = sub.add_parser("compare-mal")
    compare.add_argument("--offline", required=True)
    compare.add_argument("--sglang", required=True)
    compare.add_argument("--out", required=True)
    compare.add_argument("--summary", default=None)
    compare.set_defaults(func=cmd_compare_mal)
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "seed", None) is not None:
        try:
            import torch

            torch.manual_seed(args.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(args.seed)
        except Exception:
            pass
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
