#!/usr/bin/env python3
"""A/B SGLang capture hidden_states against a HuggingFace target forward.

Uses the same input_ids as each ``.ckpt``. Does not load the draft. Reports
length, norms, and cosine of the concat ``target_layer_ids`` feature plus a
per-layer alignment table (capture chunks vs HF ``hidden_states``).
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import statistics
import time
from pathlib import Path
from typing import Any

import torch

from specforge.modeling.draft.dflash import extract_context_feature
from specforge.runtime.data_plane.offline_reader import list_feature_files


def load_ckpt(path: str) -> dict[str, Any]:
    if path.endswith(".gz"):
        with gzip.open(path, "rb") as handle:
            return torch.load(io.BytesIO(handle.read()), weights_only=False)
    return torch.load(path, weights_only=False)


def _as_2d(hidden: torch.Tensor) -> torch.Tensor:
    if hidden.dim() == 3:
        if hidden.shape[0] != 1:
            raise ValueError(f"expected [seq, width] or [1, seq, width], got {tuple(hidden.shape)}")
        hidden = hidden.squeeze(0)
    if hidden.dim() != 2:
        raise ValueError(f"expected rank-2 hidden, got {tuple(hidden.shape)}")
    return hidden


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.reshape(-1).float()
    b = b.reshape(-1).float()
    denom = float(a.norm().item() * b.norm().item())
    if denom <= 0:
        return float("nan")
    return float(torch.dot(a, b).item() / denom)


def _token_cosine_mean(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float()
    b = b.float()
    an = a.norm(dim=-1).clamp_min(1e-12)
    bn = b.norm(dim=-1).clamp_min(1e-12)
    return float(((a * b).sum(dim=-1) / (an * bn)).mean().item())


def mean_finite(values: list[float]) -> float | None:
    finite = [v for v in values if v == v]
    if not finite:
        return None
    return float(statistics.mean(finite))


def cmd_ab(args: argparse.Namespace) -> int:
    from transformers import AutoModelForCausalLM

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    if device.type != "cuda":
        raise SystemExit("feature A/B needs a GPU")

    layer_ids = [int(x) for x in args.target_layer_ids.split(",") if x.strip()]
    print(f"loading target {args.target} dtype={dtype}", flush=True)
    target = AutoModelForCausalLM.from_pretrained(
        args.target,
        trust_remote_code=True,
        torch_dtype=dtype,
    )
    target.to(device)
    target.eval()

    files = list_feature_files(args.hidden_states_path)
    if not files:
        raise SystemExit(f"no feature files under {args.hidden_states_path}")
    print(f"feature_files={len(files)} n={args.n} layers={layer_ids}", flush=True)

    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    with torch.no_grad():
        for path in files:
            if len(rows) >= args.n:
                break
            raw = load_ckpt(path)
            if "input_ids" not in raw or "hidden_states" not in raw:
                print(f"SKIP {path}: missing keys {list(raw)}", flush=True)
                continue
            input_ids = raw["input_ids"]
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)
            input_ids = input_ids[:, : args.max_length].to(device)
            capture = _as_2d(raw["hidden_states"])[: input_ids.shape[1]].to(device)
            output = target(
                input_ids=input_ids,
                output_hidden_states=True,
                use_cache=False,
            )
            hs = output.hidden_states
            hf = extract_context_feature(hs, layer_ids)[0]
            seq = min(capture.shape[0], hf.shape[0])
            capture = capture[:seq]
            hf = hf[:seq]
            n_hs = len(hs)
            row: dict[str, Any] = {
                "path": path,
                "seq_len": seq,
                "n_hidden_states": n_hs,
                "capture_width": int(capture.shape[-1]),
                "hf_width": int(hf.shape[-1]),
                "capture_norm": float(capture.float().norm().item()),
                "hf_norm": float(hf.float().norm().item()),
                "cosine_flat": _cosine(capture, hf) if capture.shape == hf.shape else float("nan"),
                "cosine_token_mean": (
                    _token_cosine_mean(capture, hf) if capture.shape == hf.shape else float("nan")
                ),
            }
            hidden_size = int(hs[0].shape[-1])
            n_chunks = capture.shape[-1] // hidden_size if hidden_size else 0
            alignment = []
            if n_chunks > 0 and capture.shape[-1] == n_chunks * hidden_size:
                for chunk_i in range(n_chunks):
                    cap_chunk = capture[:, chunk_i * hidden_size : (chunk_i + 1) * hidden_size]
                    best = {"hf_index": None, "cosine": float("-inf")}
                    named = {}
                    for hf_i, tensor in enumerate(hs):
                        layer = _as_2d(tensor.detach())[:seq].to(device)
                        if layer.shape != cap_chunk.shape:
                            continue
                        cos = _token_cosine_mean(cap_chunk, layer)
                        named[str(hf_i)] = cos
                        if cos == cos and cos > best["cosine"]:
                            best = {"hf_index": hf_i, "cosine": cos}
                    alignment.append(
                        {
                            "capture_chunk": chunk_i,
                            "best_hf_index": best["hf_index"],
                            "best_cosine": best["cosine"],
                            "config_layer_plus_offset": (
                                layer_ids[chunk_i] + 1 if chunk_i < len(layer_ids) else None
                            ),
                        }
                    )
            row["layer_alignment"] = alignment
            rows.append(row)
            print(
                f"[{len(rows)}/{args.n}] seq={seq} cap_w={row['capture_width']} "
                f"hf_w={row['hf_width']} n_hs={n_hs} "
                f"cos_flat={row['cosine_flat']:.4f} cos_tok={row['cosine_token_mean']:.4f} "
                f"{Path(path).name}",
                flush=True,
            )
            if alignment:
                mapped = ",".join(
                    f"{a['capture_chunk']}->{a['best_hf_index']}({a['best_cosine']:.3f})"
                    for a in alignment
                )
                print(f"  align {mapped}", flush=True)

    if not rows:
        raise SystemExit("feature A/B produced no samples")
    report = {
        "n": len(rows),
        "target": args.target,
        "hidden_states_path": args.hidden_states_path,
        "target_layer_ids": layer_ids,
        "extract_offset": 1,
        "max_length": args.max_length,
        "elapsed_s": time.perf_counter() - started,
        "cosine_flat_mean": mean_finite([r["cosine_flat"] for r in rows]),
        "cosine_token_mean": mean_finite([r["cosine_token_mean"] for r in rows]),
        "capture_norm_mean": mean_finite([r["capture_norm"] for r in rows]),
        "hf_norm_mean": mean_finite([r["hf_norm"] for r in rows]),
        "width_match": all(r["capture_width"] == r["hf_width"] for r in rows),
        "samples": rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    summary = {k: report[k] for k in (
        "n",
        "target_layer_ids",
        "cosine_flat_mean",
        "cosine_token_mean",
        "capture_norm_mean",
        "hf_norm_mean",
        "width_match",
        "elapsed_s",
    )}
    print(json.dumps(summary, indent=2), flush=True)
    print(f"wrote {out}", flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--hidden-states-path", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--target-layer-ids", default="1,8,15,22,29")
    parser.set_defaults(func=cmd_ab)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
