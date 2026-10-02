#!/usr/bin/env python3
"""Export-time HTTP serve + ShareGPT eval for linear-context DFlash.

Stock SGLang ``--speculative-algorithm DFLASH`` loads ``DFlashDraftModel`` and
concatenates context KV. This server keeps ``DFlashLinearDraftModel`` and drafts
via ``spec_generate`` (GDN/KDA prefix state, dense B-token block).
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

import torch

ROLE_FROM = {
    "human": "user",
    "gpt": "assistant",
    "chatgpt": "assistant",
    "bing": "assistant",
    "bard": "assistant",
}


def load_sharegpt_prompts(
    path: str,
    tokenizer,
    limit: Optional[int],
    enable_thinking: bool = True,
) -> list[str]:
    prompts: list[str] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            messages: list[dict[str, str]] = []
            for message in row.get("conversations") or []:
                if "role" in message:
                    role = str(message["role"])
                    content = str(message.get("content") or message.get("value") or "")
                else:
                    role = ROLE_FROM.get(str(message.get("from", "")), "")
                    content = str(message.get("value") or message.get("content") or "")
                if role not in {"user", "assistant"} or not content:
                    continue
                messages.append({"role": role, "content": content})
            if messages and messages[-1]["role"] == "assistant":
                messages = messages[:-1]
            if not messages or messages[-1]["role"] != "user":
                continue
            try:
                rendered = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=enable_thinking,
                )
            except TypeError:
                rendered = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    chat_template_kwargs={"enable_thinking": enable_thinking},
                )
            prompts.append(rendered)
            if limit is not None and len(prompts) >= limit:
                break
    if not prompts:
        raise SystemExit(f"no usable ShareGPT prompts in {path}")
    return prompts


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


def _decode_new(tokenizer, input_ids: torch.Tensor, output_ids: torch.Tensor) -> str:
    new_ids = output_ids[0, input_ids.shape[1] :]
    return tokenizer.decode(new_ids, skip_special_tokens=True)


def vanilla_generate(
    target,
    tokenizer,
    text: str,
    *,
    max_new_tokens: int,
    temperature: float,
    ignore_eos: bool,
) -> dict[str, Any]:
    encoded = tokenizer(text, return_tensors="pt")
    input_ids = encoded["input_ids"].to(next(target.parameters()).device)
    gen_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 1e-5,
        "use_cache": True,
    }
    if temperature > 1e-5:
        gen_kwargs["temperature"] = temperature
    stop_ids = _stop_ids(tokenizer, ignore_eos)
    if stop_ids:
        gen_kwargs["eos_token_id"] = stop_ids
        gen_kwargs["pad_token_id"] = stop_ids[0]
    elif tokenizer.pad_token_id is not None:
        gen_kwargs["pad_token_id"] = tokenizer.pad_token_id
    started = time.perf_counter()
    output_ids = target.generate(input_ids, **gen_kwargs)
    e2e = time.perf_counter() - started
    completion = int(output_ids.shape[1] - input_ids.shape[1])
    text_out = _decode_new(tokenizer, input_ids, output_ids)
    return {
        "text": text_out,
        "completion_tokens": max(completion, 0),
        "e2e_s": e2e,
        "spec_accept_length": None,
    }


def spec_generate(
    draft,
    target,
    tokenizer,
    text: str,
    *,
    max_new_tokens: int,
    temperature: float,
    ignore_eos: bool,
) -> dict[str, Any]:
    encoded = tokenizer(text, return_tensors="pt")
    input_ids = encoded["input_ids"].to(next(target.parameters()).device)
    started = time.perf_counter()
    output_ids = draft.spec_generate(
        target,
        input_ids,
        max_new_tokens=max_new_tokens,
        stop_token_ids=_stop_ids(tokenizer, ignore_eos),
        temperature=temperature,
    )
    e2e = time.perf_counter() - started
    completion = int(output_ids.shape[1] - input_ids.shape[1])
    accepts = list(getattr(draft, "last_acceptance_lengths", None) or [])
    mean_accept = float(statistics.mean(accepts)) if accepts else None
    return {
        "text": _decode_new(tokenizer, input_ids, output_ids),
        "completion_tokens": max(completion, 0),
        "e2e_s": e2e,
        "spec_accept_length": mean_accept,
    }


class ServeState:
    def __init__(self, mode: str, target, tokenizer, draft=None):
        self.mode = mode
        self.target = target
        self.tokenizer = tokenizer
        self.draft = draft


def _handler(state: ServeState):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):  # noqa: A003
            print("[serve]", format % args, flush=True)

        def do_GET(self):  # noqa: N802
            if urlparse(self.path).path.rstrip("/") == "/health":
                self._send(200, {"ok": True, "mode": state.mode})
                return
            self._send(404, {"error": "not found"})

        def do_POST(self):  # noqa: N802
            if urlparse(self.path).path.rstrip("/") != "/generate":
                self._send(404, {"error": "not found"})
                return
            length = int(self.headers.get("Content-Length") or 0)
            try:
                payload = json.loads(self.rfile.read(length).decode() or "{}")
            except json.JSONDecodeError:
                self._send(400, {"error": "invalid json"})
                return
            sampling = payload.get("sampling_params") or {}
            text = str(payload.get("text") or "")
            max_new = int(sampling.get("max_new_tokens") or 64)
            temperature = float(sampling.get("temperature") or 0.0)
            ignore_eos = bool(sampling.get("ignore_eos", True))
            try:
                with torch.inference_mode():
                    if state.mode == "spec":
                        if state.draft is None:
                            raise RuntimeError("spec mode requires a draft")
                        result = spec_generate(
                            state.draft,
                            state.target,
                            state.tokenizer,
                            text,
                            max_new_tokens=max_new,
                            temperature=temperature,
                            ignore_eos=ignore_eos,
                        )
                    else:
                        result = vanilla_generate(
                            state.target,
                            state.tokenizer,
                            text,
                            max_new_tokens=max_new,
                            temperature=temperature,
                            ignore_eos=ignore_eos,
                        )
            except Exception as exc:  # noqa: BLE001
                traceback.print_exc()
                self._send(500, {"error": str(exc)})
                return
            meta = {
                "completion_tokens": result["completion_tokens"],
            }
            if result["spec_accept_length"] is not None:
                meta["spec_accept_length"] = result["spec_accept_length"]
            self._send(
                200,
                {
                    "text": result["text"],
                    "meta_info": meta,
                    "usage": {"completion_tokens": result["completion_tokens"]},
                },
            )

        def _send(self, code: int, body: dict[str, Any]) -> None:
            raw = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

    return Handler


def load_models(args: argparse.Namespace):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from specforge.modeling.auto import AutoDraftModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    print(f"loading tokenizer {args.target}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.target, trust_remote_code=True)
    print(f"loading target {args.target} dtype={dtype} device={device}", flush=True)
    target = AutoModelForCausalLM.from_pretrained(
        args.target,
        trust_remote_code=True,
        torch_dtype=dtype,
    )
    target.to(device)
    target.eval()
    draft = None
    if args.mode == "spec" or args.mode == "both":
        print(f"loading draft {args.draft}", flush=True)
        draft = AutoDraftModel.from_pretrained(args.draft, torch_dtype=dtype)
        draft.to(device)
        draft.eval()
    return target, tokenizer, draft, device


def cmd_serve(args: argparse.Namespace) -> int:
    if args.mode == "spec" and not args.draft:
        raise SystemExit("serve --mode spec requires --draft")
    target, tokenizer, draft, _device = load_models(args)
    state = ServeState(args.mode, target, tokenizer, draft)
    server = HTTPServer((args.host, args.port), _handler(state))
    print(
        f"serving mode={args.mode} on http://{args.host}:{args.port}/generate",
        flush=True,
    )
    server.serve_forever()
    return 0


def cmd_wait(args: argparse.Namespace) -> int:
    import urllib.request

    health = args.base.rstrip("/") + "/health"
    started = time.time()
    last_err = "not attempted"
    while time.time() - started <= args.timeout:
        try:
            urllib.request.urlopen(health, timeout=5)
            print(f"healthy {health}", flush=True)
            return 0
        except Exception as exc:  # noqa: BLE001
            last_err = str(exc)
            time.sleep(5)
    raise SystemExit(f"health timeout after {args.timeout:.0f}s: {last_err}")


def cmd_eval(args: argparse.Namespace) -> int:
    import urllib.error
    import urllib.request

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.target, trust_remote_code=True)
    prompts = load_sharegpt_prompts(
        args.eval_jsonl,
        tokenizer,
        args.n,
        enable_thinking=not args.disable_thinking,
    )
    ignore_eos = args.ignore_eos
    if ignore_eos is None:
        ignore_eos = False
    url = args.base.rstrip("/") + "/generate"

    def once(text: str) -> dict[str, Any]:
        payload = {
            "text": text,
            "sampling_params": {
                "max_new_tokens": args.max_new_tokens,
                "temperature": 0.0,
                "ignore_eos": ignore_eos,
            },
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        started = time.perf_counter()
        with urllib.request.urlopen(req, timeout=args.timeout) as resp:
            body = json.loads(resp.read().decode())
        e2e = time.perf_counter() - started
        meta = body.get("meta_info") or {}
        completion = meta.get("completion_tokens")
        if not isinstance(completion, int) or completion <= 0:
            usage = body.get("usage") or {}
            completion = usage.get("completion_tokens") or max(
                1, len(str(body.get("text") or "").split())
            )
        accept = meta.get("spec_accept_length")
        return {
            "e2e_s": e2e,
            "completion_tokens": int(completion),
            "tok_s": int(completion) / e2e if e2e > 0 else 0.0,
            "text_preview": str(body.get("text") or "")[:240],
            "spec": {"spec_accept_length": accept} if accept is not None else {},
            "meta_keys": sorted(meta.keys()),
        }

    print(
        f"warmup {args.warmup} prompts {len(prompts)} ignore_eos={ignore_eos}",
        flush=True,
    )
    for i in range(args.warmup):
        once(prompts[i % len(prompts)])
    rows = [once(text) for text in prompts]
    tok = [r["tok_s"] for r in rows]
    e2e = [r["e2e_s"] for r in rows]
    accepts = [
        float(r["spec"]["spec_accept_length"])
        for r in rows
        if isinstance((r.get("spec") or {}).get("spec_accept_length"), (int, float))
    ]
    report = {
        "label": args.label,
        "url": url,
        "n": len(rows),
        "e2e_s_mean": statistics.mean(e2e),
        "tok_s_mean": statistics.mean(tok),
        "spec_accept_length_mean": statistics.mean(accepts) if accepts else None,
        "spec_accept_length_p50": statistics.median(accepts) if accepts else None,
        "empty_outputs": sum(
            1 for r in rows if not str(r.get("text_preview") or "").strip()
        ),
        "enable_thinking": not args.disable_thinking,
        "ignore_eos": ignore_eos,
        "max_new_tokens": args.max_new_tokens,
        "warmup": args.warmup,
        "meta_keys": rows[0]["meta_keys"] if rows else [],
        "first_spec": rows[0]["spec"] if rows else {},
        "first_preview": rows[0]["text_preview"] if rows else "",
        "raw": rows,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "label",
                    "tok_s_mean",
                    "e2e_s_mean",
                    "spec_accept_length_mean",
                    "first_spec",
                )
            },
            indent=2,
        )
    )
    if not str(report["first_preview"]).strip():
        raise SystemExit("empty generate output")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    vanilla = json.loads(Path(args.vanilla).read_text(encoding="utf-8"))
    spec = json.loads(Path(args.spec).read_text(encoding="utf-8"))
    v_tok = float(vanilla["tok_s_mean"])
    s_tok = float(spec["tok_s_mean"])
    ratio = s_tok / v_tok if v_tok > 0 else 0.0
    accept = spec.get("spec_accept_length_mean")
    lines = [
        "# Linear-context DFlash export/serve eval",
        "",
        f"vanilla_tok_s: {v_tok:.2f}",
        f"spec_tok_s: {s_tok:.2f}",
        f"vanilla_e2e_s: {float(vanilla.get('e2e_s_mean') or 0):.3f}",
        f"spec_e2e_s: {float(spec.get('e2e_s_mean') or 0):.3f}",
        f"speedup: {ratio:.3f}x",
        f"spec_accept_length_mean: {accept}",
        f"spec_accept_length_p50: {spec.get('spec_accept_length_p50')}",
        f"requests: vanilla={vanilla.get('n')} spec={spec.get('n')}",
        f"empty_outputs: vanilla={vanilla.get('empty_outputs')} spec={spec.get('empty_outputs')}",
        "",
        "Serving stack: SpecForge spec_generate (DFlashLinearDraftModel).",
        "Stock SGLang DFLASH is not used; it rewrites drafts to DFlashDraftModel.",
        "",
    ]
    errors: list[str] = []
    if int(vanilla.get("n") or 0) != int(spec.get("n") or 0):
        errors.append("request counts differ")
    if vanilla.get("empty_outputs") or spec.get("empty_outputs"):
        errors.append("empty generate outputs")
    if accept is None:
        errors.append("spec generate did not report spec_accept_length")
    elif float(accept) < args.min_accept:
        errors.append(
            f"spec_accept_length_mean={accept:.3f} < {args.min_accept}"
        )
    if ratio < args.min_speedup:
        errors.append(
            f"draft did not speed generate: {s_tok:.2f} vs {v_tok:.2f} "
            f"({ratio:.3f}x < {args.min_speedup}x)"
        )
    if errors:
        lines.append("## gate failures")
        lines.extend(f"- {err}" for err in errors)
        lines.append("")
    Path(args.out).write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    if errors and args.fail_on_gates:
        raise SystemExit("eval gates failed")
    return 0


def cmd_mal(args: argparse.Namespace) -> int:
    args.mode = "spec"
    if not args.draft:
        raise SystemExit("mal requires --draft")
    target, tokenizer, draft, device = load_models(args)
    prompts = load_sharegpt_prompts(
        args.eval_jsonl,
        tokenizer,
        args.n,
        enable_thinking=not args.disable_thinking,
    )
    ignore_eos = False if args.ignore_eos is None else bool(args.ignore_eos)
    rows: list[dict[str, Any]] = []
    block_accepts: list[float] = []
    started_all = time.perf_counter()
    for index, text in enumerate(prompts):
        encoded = tokenizer(text, return_tensors="pt")
        input_ids = encoded["input_ids"].to(device)
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": False,
            "use_cache": True,
        }
        stop_ids = _stop_ids(tokenizer, ignore_eos)
        if stop_ids:
            gen_kwargs["eos_token_id"] = stop_ids
            gen_kwargs["pad_token_id"] = stop_ids[0]
        elif tokenizer.pad_token_id is not None:
            gen_kwargs["pad_token_id"] = tokenizer.pad_token_id
        with torch.inference_mode():
            sequence_ids = target.generate(input_ids, **gen_kwargs)
            prompt_len = int(input_ids.shape[1])
            completion = int(sequence_ids.shape[1] - prompt_len)
            if completion <= 0:
                rows.append(
                    {
                        "completion_tokens": 0,
                        "spec_accept_length": None,
                        "n_blocks": 0,
                        "text_preview": "",
                    }
                )
                continue
            lengths = draft.acceptance_along_sequence(
                target, sequence_ids, prompt_len=prompt_len, temperature=0.0
            )
        mean_accept = float(statistics.mean(lengths)) if lengths else None
        if lengths:
            block_accepts.extend(float(x) for x in lengths)
        rows.append(
            {
                "completion_tokens": completion,
                "spec_accept_length": mean_accept,
                "n_blocks": len(lengths),
                "accepts": lengths,
                "text_preview": _decode_new(tokenizer, input_ids, sequence_ids)[:240],
            }
        )
        if (index + 1) % 8 == 0 or index == 0:
            so_far = [r["spec_accept_length"] for r in rows if r["spec_accept_length"]]
            running = statistics.mean(so_far) if so_far else None
            print(
                f"mal {index + 1}/{len(prompts)} running_mal={running} "
                f"last={mean_accept} new_tokens={completion}",
                flush=True,
            )
    per_prompt = [
        r["spec_accept_length"]
        for r in rows
        if isinstance(r.get("spec_accept_length"), (int, float))
    ]
    report = {
        "label": "dflash_linear_mal",
        "n": len(rows),
        "spec_accept_length_mean": (
            statistics.mean(per_prompt) if per_prompt else None
        ),
        "spec_accept_length_p50": (
            statistics.median(per_prompt) if per_prompt else None
        ),
        "block_accept_length_mean": (
            statistics.mean(block_accepts) if block_accepts else None
        ),
        "empty_outputs": sum(1 for r in rows if r["completion_tokens"] <= 0),
        "enable_thinking": not args.disable_thinking,
        "ignore_eos": ignore_eos,
        "max_new_tokens": args.max_new_tokens,
        "elapsed_s": time.perf_counter() - started_all,
        "first_preview": rows[0]["text_preview"] if rows else "",
        "raw": rows,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "label",
                    "n",
                    "spec_accept_length_mean",
                    "spec_accept_length_p50",
                    "block_accept_length_mean",
                    "empty_outputs",
                    "elapsed_s",
                )
            },
            indent=2,
        )
    )
    if report["spec_accept_length_mean"] is None:
        raise SystemExit("MAL eval produced no acceptance lengths")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    serve = sub.add_parser("serve")
    serve.add_argument("--target", required=True)
    serve.add_argument("--draft", default=None)
    serve.add_argument("--mode", choices=("vanilla", "spec"), required=True)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=30000)
    serve.set_defaults(func=cmd_serve)

    wait = sub.add_parser("wait")
    wait.add_argument("--base", required=True)
    wait.add_argument("--timeout", type=float, default=900)
    wait.set_defaults(func=cmd_wait)

    ev = sub.add_parser("eval")
    ev.add_argument("--base", required=True)
    ev.add_argument("--label", required=True)
    ev.add_argument("--out", required=True)
    ev.add_argument("--target", required=True)
    ev.add_argument("--eval-jsonl", required=True)
    ev.add_argument("--n", type=int, default=256)
    ev.add_argument("--warmup", type=int, default=2)
    ev.add_argument("--max-new-tokens", type=int, default=64)
    ev.add_argument("--timeout", type=float, default=300)
    ev.add_argument("--disable-thinking", action="store_true")
    ev.add_argument("--ignore-eos", dest="ignore_eos", action="store_true")
    ev.add_argument("--no-ignore-eos", dest="ignore_eos", action="store_false")
    ev.set_defaults(ignore_eos=None, func=cmd_eval)

    cmp = sub.add_parser("compare")
    cmp.add_argument("--vanilla", required=True)
    cmp.add_argument("--spec", required=True)
    cmp.add_argument("--out", required=True)
    cmp.add_argument("--min-speedup", type=float, default=1.0)
    cmp.add_argument("--min-accept", type=float, default=1.05)
    cmp.add_argument("--fail-on-gates", action="store_true")
    cmp.set_defaults(func=cmd_compare)

    mal = sub.add_parser("mal")
    mal.add_argument("--target", required=True)
    mal.add_argument("--draft", required=True)
    mal.add_argument("--eval-jsonl", required=True)
    mal.add_argument("--out", required=True)
    mal.add_argument("--n", type=int, default=256)
    mal.add_argument("--max-new-tokens", type=int, default=64)
    mal.add_argument("--disable-thinking", action="store_true")
    mal.add_argument("--ignore-eos", dest="ignore_eos", action="store_true")
    mal.add_argument("--no-ignore-eos", dest="ignore_eos", action="store_false")
    mal.set_defaults(ignore_eos=None, func=cmd_mal)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
