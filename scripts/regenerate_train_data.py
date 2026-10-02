"""
This script will re-generate the dataset from target model,
which better aligns the draft model with the target model’s output distribution.

Usage:
1. Set up one or more SGLang servers for the target model.

python3 -m sglang.launch_server \
	--model Qwen/Qwen3.5-4B \
	--mem-fraction-static 0.8 \
	--tp 1 \
	--trust-remote-code \
    --cuda-graph-max-bs 32 \
	--host 0.0.0.0 \
	--port 30000 \
	--dtype bfloat16 \
    --reasoning-parser qwen3 \
    --attention-backend aiter \
    --disable-radix-cache \
    --context-length 4096


2. Regenerate the dataset using the `regenerate_train_data.py` script.
python scripts/regenerate_train_data.py \
    --model Qwen/Qwen3.5-4B \
    --concurrency 32 \
    --max-length 4096 \
    --server-address localhost:30000 localhost:30010 localhost:30020 localhost:30030 localhost:30040 localhost:30050 localhost:30060 localhost:30070 \
    --temperature 1.0 \
    --top-p 0.95 \
    --top-k 20 \
    --min-p 0.0 \
    --presence-penalty 1.5 \
    --sglang-repetition-penalty 1.0 \
    --input-file-path ./cache/dataset/perfectblend_train.jsonl \
    --output-file-path ./cache/dataset/perfectblend_train_regen.jsonl \
    --resume-by-id \
    --reasoning save
"""

import argparse
import json
import os
import random
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Set

_WRITE_LOCK = threading.Lock()

try:
    from tqdm import tqdm
except ModuleNotFoundError:  # pragma: no cover

    class tqdm:  # type: ignore[no-redef]
        def __init__(self, iterable=None, total=None, desc=None, initial=0):
            del total, desc, initial
            self.iterable = iterable

        def update(self, n=1):
            del n

        def __iter__(self):
            return iter(self.iterable or [])

try:
    from openai import OpenAI
except ModuleNotFoundError as exc:
    OpenAI = None
    _OPENAI_IMPORT_ERROR = exc
else:
    _OPENAI_IMPORT_ERROR = None

try:
    from scripts.conversation_validation import has_think_marker, validate_conversation
except ModuleNotFoundError:
    from conversation_validation import has_think_marker, validate_conversation

MAX_LENGTH_MARGIN = 8
MIN_NEW_TOKENS = 16


def validate_regen_input(data: Any) -> str | None:
    """Return why a ShareGPT row cannot be regenerated, or ``None``."""
    if not isinstance(data, dict):
        return "Expected a JSON object"

    return validate_conversation(
        data.get("conversations"),
        error_style="regeneration",
    )


def set_skipped(data: Any, error: str) -> Dict[str, Any]:
    if not isinstance(data, dict):
        return {"status": "skipped", "error": error, "data": data}
    data["status"] = "skipped"
    data["error"] = error
    return data


def count_lines(path: str) -> int:
    with open(path, encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def write_jsonl(handle, obj: Any) -> None:
    payload = json.dumps(obj, ensure_ascii=False) + "\n"
    with _WRITE_LOCK:
        handle.write(payload)
        handle.flush()


def drop_truncated_trailing_line(path: str) -> None:
    """Drop a partial last line so resume-by-id can reread complete JSONL."""
    if not os.path.exists(path):
        return
    size = os.path.getsize(path)
    if size == 0:
        return
    with open(path, "rb+") as handle:
        data = handle.read()
        if not data:
            return
        if not data.endswith(b"\n"):
            last_nl = data.rfind(b"\n")
            if last_nl == -1:
                handle.seek(0)
                handle.truncate(0)
                return
            data = data[: last_nl + 1]
            handle.seek(0)
            handle.write(data)
            handle.truncate()
        last_nl = data.rfind(b"\n")
        prev_nl = data.rfind(b"\n", 0, last_nl)
        last_line = data[prev_nl + 1 : last_nl]
        if not last_line.strip():
            return
        try:
            json.loads(last_line.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            keep_to = prev_nl + 1 if prev_nl != -1 else 0
            handle.seek(keep_to)
            handle.truncate()


def load_jsonl_ids(path: str) -> Set[str]:
    ids: Set[str] = set()
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return ids
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            row_id = row.get("id")
            if row_id is not None and str(row_id).strip():
                ids.add(str(row_id))
    return ids


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Re-generate training data using sglang model server"
    )

    # model related arguments
    model_group = parser.add_argument_group("model")
    model_group.add_argument("--model", type=str, required=True)
    model_group.add_argument(
        "--reasoning",
        choices=["none", "save", "disable"],
        default="none",
        help=(
            "Reasoning mode: 'none' for standard models, 'save' to store "
            "reasoning_content, or 'disable' to disable thinking via extra_body"
        ),
    )
    model_group.add_argument(
        "--is-gpt-oss",
        action="store_true",
        help="Whether the model is a GPT-OSS model",
    )
    model_group.add_argument(
        "--tokenizer-path",
        type=str,
        default=None,
        help="Tokenizer used for --max-length prompt accounting (defaults to --model)",
    )

    # sampling params
    sampling_params_group = parser.add_argument_group("sampling parameters")
    sampling_params_group.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for sglang model server",
    )
    sampling_params_group.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Nucleus sampling top_p",
    )
    sampling_params_group.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Top-k sampling value sent via extra_body",
    )
    sampling_params_group.add_argument(
        "--min-p",
        type=float,
        default=None,
        help="min_p sampling value sent via extra_body",
    )
    sampling_params_group.add_argument(
        "--presence-penalty",
        type=float,
        default=None,
        help="OpenAI presence_penalty",
    )
    sampling_params_group.add_argument(
        "--repetition-penalty",
        type=float,
        default=None,
        help="Mapped to presence_penalty in the OpenAI API",
    )
    sampling_params_group.add_argument(
        "--sglang-repetition-penalty",
        type=float,
        default=None,
        help="SGLang repetition_penalty sent via extra_body",
    )
    sampling_params_group.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="Maximum number of tokens (default: 4096)",
    )
    sampling_params_group.add_argument(
        "--max-length",
        type=int,
        default=None,
        help=(
            "Maximum prompt+completion tokens. When set, each turn uses "
            "max_tokens = max_length - prompt_tokens - margin."
        ),
    )

    # optimization
    optimization_group = parser.add_argument_group("optimization")
    optimization_group.add_argument(
        "--concurrency",
        type=int,
        default=64,
        help="The number of requests to send to a single server concurrently, the total number of concurrent requests is concurrency * number of server addresses",
    )

    # data related arguments
    data_group = parser.add_argument_group("data")
    data_group.add_argument(
        "--input-file-path", type=str, required=True, help="Path to the input file"
    )
    data_group.add_argument(
        "--output-file-path", type=str, required=True, help="Path to the output file"
    )
    data_group.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="The number of samples to regenerate, if not provided, all samples will be regenerated",
    )
    data_group.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing output file, skip already processed samples",
    )
    data_group.add_argument(
        "--resume-by-id",
        action="store_true",
        help="Resume by skipping ids already present in success/error/skipped files",
    )

    # sglang server
    server_group = parser.add_argument_group("sglang server")
    server_group.add_argument(
        "--server-address",
        type=str,
        nargs="+",
        help="Server address and port for sglang model server",
    )
    return parser.parse_args()


def get_random_reasoning_effort() -> str:
    """Get a random reasoning effort level for the model with weighted probabilities."""
    # usage example: https://huggingface.co/openai/gpt-oss-20b/discussions/28
    # Reasoning effort levels with weights: LOW(4), MEDIUM(4), HIGH(2)
    reasoning_efforts = [
        "low",
        "medium",
        "high",
    ]
    weights = [4, 4, 2]
    return random.choices(reasoning_efforts, weights=weights, k=1)[0]


def compute_context_length(conversations: List[Dict[str, Any]]) -> int:
    """
    This is a rough estimate of the context length measured in untokenized
    tokens.
    """
    length = 0
    for message in conversations:
        content = message.get("content")
        if isinstance(content, str):
            # {"role": "assistant", "content": "Hi, how can I help?"}
            length += len(content.split())
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    text = part.get("text")
                    if isinstance(text, str):
                        length += len(text.split())
    return length


def get_tokenizer(args):
    cached = getattr(args, "_tokenizer", None)
    if cached is not None:
        return cached
    from transformers import AutoTokenizer

    path = getattr(args, "tokenizer_path", None) or args.model
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    args._tokenizer = tokenizer
    return tokenizer


def _template_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    template_messages = []
    for message in messages:
        template_messages.append(
            {
                "role": message["role"],
                "content": message.get("content") or "",
            }
        )
    return template_messages


def count_prompt_tokens(args, messages: List[Dict[str, Any]]) -> int:
    counter = getattr(args, "prompt_token_counter", None)
    if counter is not None:
        return int(counter(messages))
    tokenizer = get_tokenizer(args)
    template_messages = _template_messages(messages)
    enable_thinking = getattr(args, "reasoning", "none") == "save"
    kwargs = {
        "tokenize": True,
        "add_generation_prompt": True,
        "enable_thinking": enable_thinking,
    }
    try:
        rendered = tokenizer.apply_chat_template(template_messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        rendered = tokenizer.apply_chat_template(template_messages, **kwargs)
    if hasattr(rendered, "input_ids"):
        rendered = rendered.input_ids
    if isinstance(rendered, dict):
        rendered = rendered["input_ids"]
    if rendered and isinstance(rendered[0], (list, tuple)):
        return len(rendered[0])
    return len(rendered)


def remaining_completion_tokens(args, messages: List[Dict[str, Any]]) -> int:
    prompt_tokens = count_prompt_tokens(args, messages)
    return int(args.max_length) - prompt_tokens - MAX_LENGTH_MARGIN


def build_query_kwargs(args, messages, max_tokens=None):
    effective_max_tokens = max_tokens if max_tokens is not None else args.max_tokens

    query_messages = messages
    if args.reasoning == "save":
        query_messages = []
        for message in messages:
            query_message = dict(message)
            if query_message.get("role") == "assistant":
                query_message.pop("reasoning_content", None)
            query_messages.append(query_message)

    query_kwargs = dict(
        model=args.model,
        messages=query_messages,
        max_tokens=effective_max_tokens,
        temperature=args.temperature,
        stream=False,
    )
    if args.top_p is not None:
        query_kwargs["top_p"] = args.top_p
    if args.repetition_penalty is not None:
        query_kwargs["presence_penalty"] = args.repetition_penalty
    presence_penalty = getattr(args, "presence_penalty", None)
    if presence_penalty is not None:
        query_kwargs["presence_penalty"] = presence_penalty
    extra_body = {}
    if args.top_k is not None:
        extra_body["top_k"] = args.top_k
    min_p = getattr(args, "min_p", None)
    if min_p is not None:
        extra_body["min_p"] = min_p
    sglang_repetition_penalty = getattr(args, "sglang_repetition_penalty", None)
    if sglang_repetition_penalty is not None:
        extra_body["repetition_penalty"] = sglang_repetition_penalty
    if args.reasoning == "disable":
        extra_body["chat_template_kwargs"] = {"enable_thinking": False}
    elif args.reasoning == "save":
        extra_body["chat_template_kwargs"] = {"enable_thinking": True}
    if extra_body:
        query_kwargs["extra_body"] = extra_body
    if args.is_gpt_oss:
        query_kwargs["reasoning_effort"] = get_random_reasoning_effort()
    return query_kwargs


def _has_assistant(messages: List[Dict[str, Any]]) -> bool:
    return any(message.get("role") == "assistant" for message in messages)


def call_sglang(
    args,
    server_address: str,
    data: List[Dict[str, Any]],
    max_tokens=None,
) -> str:
    """Send a batch of prompts to sglang /v1/completions."""
    if OpenAI is None:
        raise ModuleNotFoundError(
            "dataset regeneration requires the OpenAI client; install "
            "SpecForge's data extra with `pip install 'specforge[data]'`"
        ) from _OPENAI_IMPORT_ERROR
    client = OpenAI(base_url=f"http://{server_address}/v1", api_key="None")

    messages = data["conversations"]
    regenerated_messages = []

    # ignore data which starts with an assistant message
    if messages[0]["role"] == "assistant":
        data["status"] = "error"
        data["error"] = "Data starts with an assistant message"
        return data

    for message in messages:
        if message["role"] == "system":
            regenerated_messages.append(message)
        elif message["role"] == "assistant":
            continue
        elif message["role"] == "user":
            regenerated_messages.append(message)

            if max_tokens is not None:
                turn_max_tokens = max_tokens
            elif getattr(args, "max_length", None) is not None:
                turn_max_tokens = remaining_completion_tokens(
                    args, regenerated_messages
                )
                if turn_max_tokens < MIN_NEW_TOKENS:
                    regenerated_messages.pop()
                    if not _has_assistant(regenerated_messages):
                        return set_skipped(
                            data, "Prompt already fills max_length"
                        )
                    break
            else:
                turn_max_tokens = args.max_tokens

            query_kwargs = build_query_kwargs(
                args, regenerated_messages, turn_max_tokens
            )

            try:
                resp = client.chat.completions.create(**query_kwargs)
            except Exception as e:
                data["status"] = "error"
                data["error"] = str(e)
                return data
            response_text = resp.choices[0].message.content
            if args.reasoning == "disable" and (
                not isinstance(response_text, str)
                or not response_text.strip()
                or has_think_marker(response_text)
            ):
                return set_skipped(
                    data,
                    "Non-reasoning assistant response is empty or contains a thinking marker",
                )
            if not isinstance(response_text, str):
                response_text = "" if response_text is None else str(response_text)
            resp_msg = {
                "role": "assistant",
                "content": response_text,
            }
            if args.reasoning == "save":
                response_message = resp.choices[0].message
                reasoning_content = getattr(response_message, "reasoning_content", None)
                if reasoning_content is None:
                    model_extra = getattr(response_message, "model_extra", None)
                    if isinstance(model_extra, dict):
                        reasoning_content = model_extra.get("reasoning_content")
                if not isinstance(reasoning_content, str):
                    reasoning_content = (
                        "" if reasoning_content is None else str(reasoning_content)
                    )
                missing = not response_text.strip() or not reasoning_content.strip()
                if max_tokens is None and missing:
                    # --max-length already reserved decode tokens. Keep a
                    # truncated think/answer instead of dropping the row.
                    if getattr(args, "max_length", None) is None:
                        reason = (
                            "Reasoning generation requires non-empty assistant "
                            "content and reasoning_content"
                        )
                        finish_reason = getattr(
                            resp.choices[0], "finish_reason", None
                        )
                        if finish_reason:
                            reason = f"{reason} (finish_reason={finish_reason})"
                        data["status"] = "error"
                        data["error"] = reason
                        return data
                if max_tokens is None and (
                    has_think_marker(response_text)
                    or has_think_marker(reasoning_content)
                ):
                    return set_skipped(
                        data,
                        "Reasoning response contains a residual thinking marker",
                    )
                resp_msg["reasoning_content"] = reasoning_content
            regenerated_messages.append(resp_msg)
        else:
            data["status"] = "error"
            data["error"] = f"Invalid message role: {message['role']}"
            return data
    data["conversations"] = regenerated_messages
    data["status"] = "success"
    return data


def _record_result(
    regen_data,
    output_file_handle,
    error_file_handle,
    skipped_file_handle,
    stats,
):
    if regen_data["status"] == "error":
        write_jsonl(error_file_handle, regen_data)
        stats["error_samples"] += 1
        return
    if regen_data["status"] == "skipped":
        write_jsonl(skipped_file_handle, regen_data)
        stats["skipped_samples"] += 1
        return
    ctx_len = compute_context_length(regen_data.get("conversations", []))
    stats["context_token_sum"] += ctx_len
    if stats["context_token_min"] is None:
        stats["context_token_min"] = ctx_len
    else:
        stats["context_token_min"] = min(stats["context_token_min"], ctx_len)
    stats["context_token_max"] = max(stats["context_token_max"], ctx_len)
    write_jsonl(output_file_handle, regen_data)
    stats["success_samples"] += 1


def main():
    # Parse command line arguments
    args = parse_arguments()

    # Validate parameters
    if not (0.0 <= args.temperature <= 1.0):
        raise ValueError("Temperature must be between 0.0 and 1.0")

    if args.max_tokens <= 0:
        raise ValueError("Max tokens must be greater than 0")

    if args.max_length is not None and args.max_length <= MAX_LENGTH_MARGIN:
        raise ValueError("Max length must be greater than the context margin")

    if args.resume and args.resume_by_id:
        raise ValueError("Use either --resume or --resume-by-id, not both")

    print(f"Configuration:")
    print(f"  Model path: {args.model}")
    print(f"  Max tokens: {args.max_tokens}")
    print(f"  Max length: {args.max_length}")
    print(f"  Concurrency: {args.concurrency}")
    print(f"  Temperature: {args.temperature}")
    print(f"  API URL: {args.server_address}")
    print(f"  Input file: {args.input_file_path}")
    print(f"  Output file: {args.output_file_path}")
    print(f"  Resume mode: {args.resume}")
    print(f"  Resume by id: {args.resume_by_id}")
    print("-" * 50)
    total_lines = count_lines(args.input_file_path)

    skip_lines = 0
    processed_ids: Set[str] = set()
    error_file_path = args.output_file_path.replace(".jsonl", "_error.jsonl")
    skipped_file_path = args.output_file_path.replace(".jsonl", "_skipped.jsonl")

    if args.resume and os.path.exists(args.output_file_path):
        existing_success = count_lines(args.output_file_path)
        existing_error = 0
        if os.path.exists(error_file_path):
            existing_error = count_lines(error_file_path)
        existing_skipped = 0
        if os.path.exists(skipped_file_path):
            existing_skipped = count_lines(skipped_file_path)
        skip_lines = existing_success + existing_error + existing_skipped
        print(f"Resume mode enabled:")
        print(f"  Found {existing_success} successful samples in output file")
        print(f"  Found {existing_error} error samples in error file")
        print(f"  Found {existing_skipped} skipped samples in skipped file")
        print(f"  Skipping first {skip_lines} input samples")
        print("-" * 50)

        if skip_lines >= total_lines:
            print(f"All {total_lines} samples already processed. Nothing to do.")
            return

    if args.resume_by_id:
        for path in (args.output_file_path, skipped_file_path):
            drop_truncated_trailing_line(path)
            processed_ids |= load_jsonl_ids(path)
        drop_truncated_trailing_line(error_file_path)
        if os.path.exists(error_file_path) and os.path.getsize(error_file_path) > 0:
            prev_error_path = error_file_path.replace(".jsonl", ".prev.jsonl")
            with open(error_file_path, encoding="utf-8") as current:
                previous_errors = current.read()
            with open(prev_error_path, "a", encoding="utf-8") as previous:
                previous.write(previous_errors)
            with open(error_file_path, "w", encoding="utf-8"):
                pass
            print(f"Archived previous errors to {prev_error_path} for retry")
        remaining_ids = 0
        with open(args.input_file_path, encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    row = json.loads(stripped)
                except json.JSONDecodeError:
                    remaining_ids += 1
                    continue
                row_id = row.get("id") if isinstance(row, dict) else None
                if row_id is None or str(row_id) not in processed_ids:
                    remaining_ids += 1
        print("Resume-by-id mode enabled:")
        print(f"  Found {len(processed_ids)} already processed ids")
        print(f"  Remaining input rows: {remaining_ids}")
        print("-" * 50)
        if remaining_ids == 0:
            print(f"All {total_lines} samples already processed. Nothing to do.")
            return

    # test all server addresses
    valid_server_addresses = []
    for server_address in args.server_address:
        dummy_data = dict(
            conversations=[{"role": "user", "content": "Hello, how are you?"}]
        )
        result = call_sglang(
            args,
            server_address,
            dummy_data,
            max_tokens=1,
        )
        if result is not None and result.get("status") == "success":
            valid_server_addresses.append(server_address)
        else:
            print(f"Server {server_address} is not available")

    if len(valid_server_addresses) == 0:
        raise ValueError("No server address is available")
    print(
        f"Using {len(valid_server_addresses)} server addresses: {valid_server_addresses}"
    )
    print("-" * 50)

    if args.max_length is not None and getattr(args, "prompt_token_counter", None) is None:
        get_tokenizer(args)

    # Determine file open mode based on resume flag
    if args.resume_by_id:
        file_mode = "a"
    elif args.resume and skip_lines > 0:
        file_mode = "a"
    else:
        file_mode = "w"
    print(
        f"Regenerating dataset and saving the output to {args.output_file_path} and error log to {error_file_path}"
    )
    print(
        f"File open mode: {file_mode} ({'append' if file_mode == 'a' else 'overwrite'})"
    )
    print("-" * 50)
    stats = {
        "context_token_sum": 0,
        "context_token_min": None,
        "context_token_max": 0,
        "success_samples": 0,
        "error_samples": 0,
        "skipped_samples": 0,
    }
    submitted_samples = 0
    skipped_existing = 0
    pbar_initial = skip_lines if skip_lines > 0 else len(processed_ids)

    # Create progress bar
    with (
        open(args.input_file_path, "r") as input_file,
        open(args.output_file_path, file_mode) as output_file_handle,
        open(error_file_path, file_mode) as error_file_handle,
        open(skipped_file_path, file_mode, encoding="utf-8") as skipped_file_handle,
    ):
        executor = ThreadPoolExecutor(
            max_workers=args.concurrency * len(valid_server_addresses)
        )
        waiting_queue = {
            server_address: [] for server_address in valid_server_addresses
        }
        pbar = tqdm(total=total_lines, desc="Processing", initial=pbar_initial)
        start_server_index = 0

        if skip_lines > 0:
            print(f"Skipping {skip_lines} already processed samples...")
            for _ in range(skip_lines):
                next(input_file, None)
            print(f"Resuming from sample {skip_lines + 1}")

        for line in input_file:
            if args.num_samples is not None and submitted_samples >= args.num_samples:
                break

            data = json.loads(line.strip())
            row_id = data.get("id") if isinstance(data, dict) else None
            if (
                args.resume_by_id
                and row_id is not None
                and str(row_id) in processed_ids
            ):
                skipped_existing += 1
                continue

            invalid_reason = validate_regen_input(data)
            if invalid_reason is not None:
                write_jsonl(
                    skipped_file_handle, set_skipped(data, invalid_reason)
                )
                stats["skipped_samples"] += 1
                pbar.update(1)
                continue

            # find server address with the least waiting requests
            server_address = valid_server_addresses[start_server_index]
            start_server_index = (start_server_index + 1) % len(valid_server_addresses)

            # submit prompt to sglang
            while len(waiting_queue[server_address]) >= args.concurrency:
                finished_on_request = False
                # check if any future is done, if so, write the result to the output file
                for req_future in waiting_queue[server_address]:
                    if req_future.done():
                        regen_data = req_future.result()
                        _record_result(
                            regen_data,
                            output_file_handle,
                            error_file_handle,
                            skipped_file_handle,
                            stats,
                        )
                        waiting_queue[server_address].remove(req_future)
                        finished_on_request = True

                if finished_on_request:
                    break

            req_future = executor.submit(
                call_sglang,
                args,
                server_address,
                data,
            )
            waiting_queue[server_address].append(req_future)
            submitted_samples += 1
            pbar.update(1)

        # deal with all the remaining requests
        for server_address, waiting_queue_items in waiting_queue.items():
            for req_future in waiting_queue_items:
                regen_data = req_future.result()
                _record_result(
                    regen_data,
                    output_file_handle,
                    error_file_handle,
                    skipped_file_handle,
                    stats,
                )

    success_samples = stats["success_samples"]
    error_samples = stats["error_samples"]
    skipped_samples = stats["skipped_samples"]
    context_token_sum = stats["context_token_sum"]
    context_token_min = stats["context_token_min"]
    context_token_max = stats["context_token_max"]

    print(f"\nProcessing completed!")
    if success_samples > 0:
        avg_len = context_token_sum / success_samples
        print("Context length statistics (token count over conversations):")
        print(f"Number of successful examples: {success_samples}")
        print(f"Shortest context length: {context_token_min}")
        print(f"Longest context length: {context_token_max}")
        print(f"Average context length: {avg_len:.2f}")
    else:
        print("No successful examples to compute context length statistics.")

    total_processed = success_samples + error_samples + skipped_samples
    if skip_lines > 0 or skipped_existing > 0:
        previously = skip_lines if skip_lines > 0 else skipped_existing
        print(f"\nResume processing completed!")
        print(f"  Previously processed: {previously}")
        print(
            f"  Newly processed: {total_processed} "
            f"({success_samples} success, {error_samples} failed, "
            f"{skipped_samples} skipped)"
        )
        print(f"  Total: {previously + total_processed}")
    else:
        print(
            f"\nProcessing completed! {success_samples} samples regenerated, "
            f"{error_samples} samples failed, {skipped_samples} samples skipped."
        )


if __name__ == "__main__":
    main()
