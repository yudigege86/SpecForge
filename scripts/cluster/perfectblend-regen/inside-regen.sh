#!/bin/bash
# Inside the SGLang ROCm container: 8 local servers + regeneration client.
set -euo pipefail

cd /workspace/SpecForge

INPUT_FILE="${INPUT_FILE:?}"
OUTPUT_FILE="${OUTPUT_FILE:?}"
DONE_FILE="${DONE_FILE:?}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3.5-4B}"
CONCURRENCY="${CONCURRENCY:-32}"
MAX_LENGTH="${MAX_LENGTH:-4096}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-4096}"
RUN_DIR="${RUN_DIR:?}"
mkdir -p "${RUN_DIR}" "$(dirname "${OUTPUT_FILE}")"

count_lines() {
  local path="$1"
  if [[ -f "${path}" ]]; then
    awk 'END {print NR+0}' "${path}"
  else
    echo 0
  fi
}

remaining_rows() {
  python3 - "${INPUT_FILE}" "${OUTPUT_FILE}" "${skipped_file}" <<'PY'
import json
import os
import sys

def load_ids(path):
    ids = set()
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return ids
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            row_id = row.get("id")
            if row_id is not None:
                ids.add(str(row_id))
    return ids

input_path, success_path, skipped_path = sys.argv[1:4]
done = load_ids(success_path) | load_ids(skipped_path)
remaining = 0
with open(input_path, encoding="utf-8") as handle:
    for line in handle:
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            remaining += 1
            continue
        row_id = row.get("id")
        if row_id is None or str(row_id) not in done:
            remaining += 1
print(remaining)
PY
}

input_rows="$(count_lines "${INPUT_FILE}")"
error_file="${OUTPUT_FILE%.jsonl}_error.jsonl"
skipped_file="${OUTPUT_FILE%.jsonl}_skipped.jsonl"
success_rows="$(count_lines "${OUTPUT_FILE}")"
error_rows="$(count_lines "${error_file}")"
skipped_rows="$(count_lines "${skipped_file}")"
left="$(remaining_rows)"
echo "input_rows=${input_rows} success=${success_rows} error=${error_rows} skipped=${skipped_rows} remaining=${left}"

if [[ "${left}" -eq 0 && "${input_rows}" -gt 0 ]]; then
  echo "shard already complete; writing ${DONE_FILE}"
  {
    echo "input=${input_rows}"
    echo "success=${success_rows}"
    echo "error=${error_rows}"
    echo "skipped=${skipped_rows}"
    echo "remaining=0"
  } > "${DONE_FILE}"
  echo "PASS: already complete"
  exit 0
fi

echo "=== pip install openai tqdm without touching torch/sglang ==="
python3 -m pip install --no-cache-dir openai tqdm

export SGLANG_USE_AITER=1
export AITER_FLYDSL_FORCE=1
# Unified-attn Triton compile aborts on gfx950 during Qwen3.5 decode
# (llvm iota_range Begin <= End). Keep AITER, skip that kernel.
unset SGLANG_USE_AITER_UNIFIED_ATTN

PORTS=(30000 30010 30020 30030 30040 30050 30060 30070)
server_pids=()

cleanup() {
  local pid
  for pid in "${server_pids[@]:-}"; do
    kill -TERM "${pid}" 2>/dev/null || true
    pkill -TERM -P "${pid}" 2>/dev/null || true
  done
  sleep 2
  for pid in "${server_pids[@]:-}"; do
    kill -KILL "${pid}" 2>/dev/null || true
    pkill -KILL -P "${pid}" 2>/dev/null || true
  done
  # Do not `wait` on the GPU process trees: a bare wait can hold the
  # container (and the Slurm slot) for minutes after the shard is done.
}
trap cleanup EXIT

start_server() {
  local gpu="$1"
  local port="$2"
  echo "starting sglang gpu=${gpu} port=${port}"
  HIP_VISIBLE_DEVICES="${gpu}" CUDA_VISIBLE_DEVICES="${gpu}" \
    python3 -m sglang.launch_server \
      --model-path "${MODEL_PATH}" \
      --trust-remote-code \
      --tp-size 1 \
      --dtype bfloat16 \
      --mem-fraction-static 0.8 \
      --attention-backend aiter \
      --disable-radix-cache \
      --disable-cuda-graph \
      --reasoning-parser qwen3 \
      --context-length "${CONTEXT_LENGTH}" \
      --max-running-requests "${CONCURRENCY}" \
      --host 127.0.0.1 \
      --port "${port}" \
      > "${RUN_DIR}/server-${gpu}.log" 2>&1 &
  server_pids+=("$!")
}

wait_health() {
  local port="$1"
  local timeout_sec="$2"
  local pid="$3"
  python3 - "${port}" "${timeout_sec}" "${pid}" <<'PY'
import os
import sys
import time
import urllib.request

port = int(sys.argv[1])
timeout_sec = int(sys.argv[2])
pid = int(sys.argv[3])
url = f"http://127.0.0.1:{port}/health"
deadline = time.time() + timeout_sec
last_error = "no attempt"
while time.time() < deadline:
    if pid > 0:
        try:
            os.kill(pid, 0)
        except OSError:
            print(f"server pid {pid} died waiting for {url}", file=sys.stderr)
            raise SystemExit(1)
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            if 200 <= response.status < 300:
                print(f"healthy {port}")
                raise SystemExit(0)
            last_error = f"status={response.status}"
    except SystemExit:
        raise
    except Exception as exc:
        last_error = str(exc)
    time.sleep(5)
print(f"timeout waiting for {url}: {last_error}", file=sys.stderr)
raise SystemExit(1)
PY
}

start_server 0 "${PORTS[0]}"
wait_health "${PORTS[0]}" 1500 "${server_pids[-1]}"
for gpu in 1 2 3 4 5 6 7; do
  start_server "${gpu}" "${PORTS[${gpu}]}"
done
for gpu in 1 2 3 4 5 6 7; do
  wait_health "${PORTS[${gpu}]}" 900 "${server_pids[${gpu}]}"
done

addresses=()
for port in "${PORTS[@]}"; do
  addresses+=("127.0.0.1:${port}")
done

echo "=== regenerate shard ${SHARD_ID} ==="
python3 scripts/regenerate_train_data.py \
  --model "${MODEL_PATH}" \
  --reasoning save \
  --temperature 1.0 \
  --top-p 0.95 \
  --top-k 20 \
  --min-p 0.0 \
  --presence-penalty 1.5 \
  --sglang-repetition-penalty 1.0 \
  --max-length "${MAX_LENGTH}" \
  --concurrency "${CONCURRENCY}" \
  --server-address "${addresses[@]}" \
  --input-file-path "${INPUT_FILE}" \
  --output-file-path "${OUTPUT_FILE}" \
  --resume-by-id \
  2>&1 | tee "${RUN_DIR}/regen.log"

success_rows="$(count_lines "${OUTPUT_FILE}")"
error_rows="$(count_lines "${error_file}")"
skipped_rows="$(count_lines "${skipped_file}")"
left="$(remaining_rows)"
echo "input_rows=${input_rows} success=${success_rows} error=${error_rows} skipped=${skipped_rows} remaining=${left}"

if [[ "${left}" -ne 0 ]]; then
  echo "FAIL: shard ${SHARD_ID} still has ${left} unfinished input rows"
  exit 1
fi

{
  echo "input=${input_rows}"
  echo "success=${success_rows}"
  echo "error=${error_rows}"
  echo "skipped=${skipped_rows}"
  echo "remaining=0"
} > "${DONE_FILE}"
echo "PASS: shard ${SHARD_ID} complete"
