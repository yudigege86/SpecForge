#!/usr/bin/env python3
"""Replay exported dflash_linear weights on SGLang capture features.

Uses the same packed-anchor training forward as ``OnlineDFlashLinearModel``.
Hidden states come from ``valid-links`` ``.ckpt`` files, not from a HuggingFace
target generate. Only the frozen target embedding and LM head are loaded.
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import statistics
import time
from pathlib import Path
from typing import Any, Optional

import torch

from specforge.algorithms.common.hidden_states_data import normalize_offline_sample
from specforge.data.loss_mask import has_consecutive_supervised_tokens
from specforge.algorithms.dflash_linear.model import OnlineDFlashLinearModel
from specforge.modeling.auto import AutoDraftModel
from specforge.modeling.draft.linear_context import (
    fla_available,
    resolve_scan_backend,
)
from specforge.modeling.target.target_utils import TargetEmbeddingsAndHead
from specforge.runtime.data_plane.offline_reader import list_feature_files
from specforge.training.strategies.base import _cpu_max_valid_anchors


def load_ckpt(path: str) -> dict[str, Any]:
    """Load one capture file without mmap (pickle-protocol ckpts on NFS)."""

    if path.endswith(".gz"):
        with gzip.open(path, "rb") as handle:
            return torch.load(io.BytesIO(handle.read()), weights_only=False)
    return torch.load(path, weights_only=False)


def _ratio(pair: tuple[torch.Tensor, torch.Tensor]) -> float:
    num, den = pair
    den_v = float(den.detach().float().reshape(()).item())
    num_v = float(num.detach().float().reshape(()).item())
    if den_v <= 0:
        return float("nan")
    return num_v / den_v


def _pair_to_floats(pair: tuple[torch.Tensor, torch.Tensor]) -> tuple[float, float]:
    num, den = pair
    return (
        float(num.detach().float().reshape(()).item()),
        float(den.detach().float().reshape(()).item()),
    )


def build_wrapper(
    draft,
    target_parts: TargetEmbeddingsAndHead,
    *,
    attention_backend: str,
    num_anchors: int,
    loss_decay_gamma: float,
) -> OnlineDFlashLinearModel:
    method = dict(getattr(draft.config, "dflash_config", None) or {})
    mask_token_id = int(method.get("mask_token_id", 248070))
    draft.mask_token_id = mask_token_id
    wrapper = OnlineDFlashLinearModel(
        draft_model=draft,
        target_lm_head=target_parts.lm_head,
        target_embed_tokens=target_parts.embed_tokens,
        mask_token_id=mask_token_id,
        block_size=int(draft.block_size),
        attention_backend=attention_backend,
        num_anchors=num_anchors,
        loss_decay_gamma=loss_decay_gamma,
        loss_type="dflash",
        lk_loss_type=None,
    )
    wrapper.eval()
    wrapper.requires_grad_(False)
    return wrapper


def replay_one(
    wrapper: OnlineDFlashLinearModel,
    raw: dict[str, Any],
    *,
    max_len: int,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, float]:
    sample = normalize_offline_sample(raw, max_len)
    input_ids = sample["input_ids"]
    loss_mask = sample["loss_mask"]
    hidden_states = sample["hidden_states"].to(dtype=dtype)
    max_valid_anchors = _cpu_max_valid_anchors(loss_mask)
    loss, accuracy, metrics = wrapper(
        input_ids=input_ids.to(device),
        hidden_states=hidden_states.to(device),
        loss_mask=loss_mask.to(device),
        max_valid_anchors=max_valid_anchors,
        collect_detailed_metrics=True,
    )
    ratios = metrics["ratio_metrics"]
    ce_num, ce_den = _pair_to_floats(ratios["ce_loss"])
    acc_num, acc_den = _pair_to_floats(ratios["acc"])
    eal_num, eal_den = _pair_to_floats(
        ratios["dflash/hard_label/expected_accepted_length"]
    )
    return {
        "seq_len": float(input_ids.shape[1]),
        "max_valid_anchors": float(max_valid_anchors or 0),
        "loss": float(loss.detach().float().item()),
        "acc": float(accuracy.detach().float().item()),
        "ce_loss": ce_num / ce_den if ce_den > 0 else float("nan"),
        "ce_num": ce_num,
        "ce_den": ce_den,
        "acc_num": acc_num,
        "acc_den": acc_den,
        "expected_accepted_length": eal_num / eal_den if eal_den > 0 else float("nan"),
        "eal_num": eal_num,
        "eal_den": eal_den,
    }


def mean_finite(values: list[float]) -> Optional[float]:
    finite = [v for v in values if v == v]
    if not finite:
        return None
    return float(statistics.mean(finite))


def cmd_replay(args: argparse.Namespace) -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    if device.type != "cuda":
        raise SystemExit("capture replay needs a GPU")
    if not fla_available("gdn"):
        raise SystemExit("capture replay requires FLA GDN")
    backend = resolve_scan_backend("auto", on_cuda=True, num_anchors=args.num_anchors)
    print(f"auto_backend={backend} device={device} dtype={dtype}", flush=True)
    if backend != "fla":
        raise SystemExit(f"expected FLA scan backend, got {backend!r}")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print(f"loading draft {args.draft}", flush=True)
    draft = AutoDraftModel.from_pretrained(args.draft, torch_dtype=dtype)
    draft.to(device)
    draft.eval()
    draft_cfg = json.loads(Path(args.draft, "config.json").read_text(encoding="utf-8"))
    print("architectures", draft_cfg.get("architectures"), flush=True)
    if "DFlashLinearDraftModel" not in (draft_cfg.get("architectures") or []):
        raise SystemExit("draft export is not DFlashLinearDraftModel")

    print(f"loading target embed/head {args.target}", flush=True)
    target_parts = TargetEmbeddingsAndHead.from_pretrained(
        args.target,
        embed_key=args.embedding_key,
        lm_head_key=args.lm_head_key,
        cache_dir=args.cache_dir,
        device=str(device),
        dtype=dtype,
        trust_remote_code=True,
    )
    wrapper = build_wrapper(
        draft,
        target_parts,
        attention_backend=args.attention_backend,
        num_anchors=args.num_anchors,
        loss_decay_gamma=args.loss_decay_gamma,
    )

    files = list_feature_files(args.hidden_states_path)
    if not files:
        raise SystemExit(f"no feature files under {args.hidden_states_path}")
    print(f"feature_files={len(files)} root={args.hidden_states_path}", flush=True)

    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    started = time.perf_counter()
    for path in files:
        if len(rows) >= args.n:
            break
        try:
            raw = load_ckpt(path)
            if "hidden_states" not in raw:
                skipped.append({"path": path, "reason": "missing hidden_states"})
                continue
            loss_mask = raw["loss_mask"]
            if loss_mask.dim() == 2:
                loss_mask = loss_mask[0]
            if not has_consecutive_supervised_tokens(loss_mask[: args.max_length]):
                skipped.append({"path": path, "reason": "no consecutive supervised tokens"})
                continue
            metrics = replay_one(
                wrapper,
                raw,
                max_len=args.max_length,
                device=device,
                dtype=dtype,
            )
        except Exception as exc:  # noqa: BLE001
            skipped.append({"path": path, "reason": str(exc)})
            print(f"SKIP {path}: {exc}", flush=True)
            continue
        metrics["path"] = path
        rows.append(metrics)
        print(
            f"[{len(rows)}/{args.n}] seq={int(metrics['seq_len'])} "
            f"anchors={int(metrics['max_valid_anchors'])} "
            f"ce={metrics['ce_loss']:.4f} acc={metrics['acc']:.4f} "
            f"eal={metrics['expected_accepted_length']:.4f} {Path(path).name}",
            flush=True,
        )

    if not rows:
        raise SystemExit("replay produced no successful samples")

    ce_num = sum(r["ce_num"] for r in rows)
    ce_den = sum(r["ce_den"] for r in rows)
    acc_num = sum(r["acc_num"] for r in rows)
    acc_den = sum(r["acc_den"] for r in rows)
    eal_num = sum(r["eal_num"] for r in rows)
    eal_den = sum(r["eal_den"] for r in rows)
    report = {
        "n": len(rows),
        "skipped": len(skipped),
        "hidden_states_path": args.hidden_states_path,
        "draft": args.draft,
        "target": args.target,
        "max_length": args.max_length,
        "num_anchors": args.num_anchors,
        "loss_decay_gamma": args.loss_decay_gamma,
        "attention_backend": args.attention_backend,
        "seed": args.seed,
        "elapsed_s": time.perf_counter() - started,
        "ce_loss_micro": ce_num / ce_den if ce_den else None,
        "acc_micro": acc_num / acc_den if acc_den else None,
        "expected_accepted_length_micro": eal_num / eal_den if eal_den else None,
        "ce_loss_mean": mean_finite([r["ce_loss"] for r in rows]),
        "acc_mean": mean_finite([r["acc"] for r in rows]),
        "expected_accepted_length_mean": mean_finite(
            [r["expected_accepted_length"] for r in rows]
        ),
        "training_reference": {
            "ce": 4.53,
            "acc": 0.212,
            "expected_accepted_length": 2.10,
        },
        "samples": rows,
        "skipped_files": skipped,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in (
        "n",
        "skipped",
        "ce_loss_micro",
        "acc_micro",
        "expected_accepted_length_micro",
        "ce_loss_mean",
        "acc_mean",
        "expected_accepted_length_mean",
        "elapsed_s",
        "training_reference",
    )}, indent=2), flush=True)
    print(f"wrote {out}", flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft", required=True)
    parser.add_argument("--target", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--hidden-states-path", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--n", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--num-anchors", type=int, default=512)
    parser.add_argument("--loss-decay-gamma", type=float, default=7.0)
    parser.add_argument("--attention-backend", default="sdpa")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--embedding-key",
        default="model.language_model.embed_tokens.weight",
    )
    parser.add_argument("--lm-head-key", default="lm_head.weight")
    parser.add_argument("--cache-dir", default=None)
    parser.set_defaults(func=cmd_replay)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
