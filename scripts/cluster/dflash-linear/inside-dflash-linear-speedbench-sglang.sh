#!/bin/bash
# SPEED-Bench Qualitative MAL via stock SGLang DFLASH.
set -euo pipefail

cd /workspace/SpecForge

echo "=== pip install -e . --no-deps ==="
pip install -e . --no-deps
pip install datasets pandas tiktoken requests

echo "=== CPU SGLang MAL helper tests ==="
python3 -m unittest tests.test_scripts.test_dflash_linear_eval -v

TARGET_MODEL="${TARGET_MODEL:?TARGET_MODEL is required}"
DRAFT_HF="${DRAFT_HF:?DRAFT_HF is required}"
EVAL_DATASET="${EVAL_DATASET:-qualitative}"
EVAL_PY="${EVAL_PY:-/workspace/SpecForge/scripts/eval/dflash_linear_eval.py}"
EVAL_N="${EVAL_N:-}"
EVAL_CATEGORIES="${EVAL_CATEGORIES:-}"
MT_BENCH_TURNS="${MT_BENCH_TURNS:-first}"
FORCE_PREPARE="${FORCE_PREPARE:-0}"
REPLAY_OFFLINE="${REPLAY_OFFLINE:-0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-}"
ENABLE_THINKING="${ENABLE_THINKING:-}"
if [[ -z "${MAX_NEW_TOKENS}" ]]; then
  if [[ "${EVAL_DATASET}" == "qualitative" || "${EVAL_DATASET}" == "speedbench" || "${EVAL_DATASET}" == "speedbench-qualitative" ]]; then
    MAX_NEW_TOKENS=512
  else
    MAX_NEW_TOKENS=4096
  fi
fi
if [[ -z "${ENABLE_THINKING}" ]]; then
  if [[ "${EVAL_DATASET}" == "qualitative" || "${EVAL_DATASET}" == "speedbench" || "${EVAL_DATASET}" == "speedbench-qualitative" ]]; then
    ENABLE_THINKING=0
  else
    ENABLE_THINKING=1
  fi
fi
if [[ -z "${SERVER_PORT:-}" ]]; then
  SERVER_PORT=$((31000 + ${SLURM_JOB_ID:-0} % 1000))
fi
BASE="http://127.0.0.1:${SERVER_PORT}"
COMPARE_JSON="${COMPARE_JSON:-}"
BLOCK_SIZE="${BLOCK_SIZE:-16}"
MEM_FRACTION="${MEM_FRACTION:-0.92}"
SERVER_PID=""
JSONL_DIR="/shared_nfs/naqin/Linear-Context-DFlash/speedbench"
case "${EVAL_DATASET}" in
  qualitative|speedbench|speedbench-qualitative)
    EVAL_DATASET="qualitative"
    DEFAULT_JSONL="${JSONL_DIR}/qualitative.jsonl"
    ;;
  humaneval)
    DEFAULT_JSONL="${JSONL_DIR}/humaneval.jsonl"
    ;;
  mt-bench|mtbench)
    EVAL_DATASET="mt-bench"
    DEFAULT_JSONL="${JSONL_DIR}/mtbench.jsonl"
    ;;
  *)
    echo "unknown EVAL_DATASET=${EVAL_DATASET}"
    exit 1
    ;;
esac
EVAL_JSONL="${EVAL_JSONL:-${SPEEDBENCH_JSONL:-${DEFAULT_JSONL}}}"

mkdir -p "${RUN_DIR}" "$(dirname "${EVAL_JSONL}")"
test -f "${EVAL_PY}"
if [[ "${FORCE_PREPARE}" == "1" || ! -s "${EVAL_JSONL}" ]]; then
  echo "=== prepare ${EVAL_DATASET} ==="
  python3 "${EVAL_PY}" prepare --dataset "${EVAL_DATASET}" --out "${EVAL_JSONL}" \
    2>&1 | tee "${RUN_DIR}/prepare.log"
fi
test -s "${EVAL_JSONL}"

python3 - <<PY
import json
from pathlib import Path
draft = Path("${DRAFT_HF}")
cfg_path = draft / "config.json"
if cfg_path.is_file():
    cfg = json.loads(cfg_path.read_text())
    arch = (cfg.get("architectures") or ["unknown"])[0]
    print("architectures", cfg.get("architectures"))
    if arch == "DFlashLinearDraftModel":
        raise SystemExit(
            "stock SGLang DFLASH cannot load DFlashLinearDraftModel; "
            "use the offline mal CLI for linear drafts"
        )
else:
    print("draft_hf_id", "${DRAFT_HF}")
PY

python3 - <<'PY'
import sglang
print("sglang", getattr(sglang, "__version__", "?"))
PY

HELP="$(python3 -m sglang.launch_server --help 2>&1 || true)"
LAUNCH=(
  python3 -m sglang.launch_server
  --model-path "${TARGET_MODEL}"
  --speculative-algorithm DFLASH
  --speculative-draft-model-path "${DRAFT_HF}"
  --tp-size 1
  --dtype bfloat16
  --mem-fraction-static "${MEM_FRACTION}"
  --trust-remote-code
  --host 127.0.0.1
  --port "${SERVER_PORT}"
  --disable-cuda-graph
  --max-running-requests 1
)
if grep -q -- "--speculative-num-draft-tokens" <<<"${HELP}"; then
  LAUNCH+=(--speculative-num-draft-tokens "${BLOCK_SIZE}")
elif grep -q -- "--speculative-dflash-block-size" <<<"${HELP}"; then
  LAUNCH+=(--speculative-dflash-block-size "${BLOCK_SIZE}")
fi
if grep -q -- "--mamba-scheduler-strategy" <<<"${HELP}"; then
  LAUNCH+=(--mamba-scheduler-strategy extra_buffer)
fi
if grep -q -- "--disable-radix-cache" <<<"${HELP}"; then
  LAUNCH+=(--disable-radix-cache)
fi

stop_server() {
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
  SERVER_PID=""
}
trap stop_server EXIT

echo "=== ${LAUNCH[*]} ==="
printf '%s\n' "${LAUNCH[@]}" > "${RUN_DIR}/sglang.argv"
"${LAUNCH[@]}" >"${RUN_DIR}/sglang-server.log" 2>&1 &
SERVER_PID=$!
sleep 5
if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
  echo "FAIL: SGLang server exited immediately"
  tail -n 80 "${RUN_DIR}/sglang-server.log" || true
  exit 1
fi
echo "waiting for SGLang ready on ${BASE}"
ready=0
for _ in $(seq 1 180); do
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "FAIL: SGLang server died before ready"
    tail -n 120 "${RUN_DIR}/sglang-server.log" || true
    exit 1
  fi
  if grep -q "The server is fired up and ready to roll" "${RUN_DIR}/sglang-server.log"; then
    ready=1
    break
  fi
  sleep 5
done
if [[ "${ready}" != "1" ]]; then
  echo "FAIL: SGLang did not become ready"
  tail -n 120 "${RUN_DIR}/sglang-server.log" || true
  exit 1
fi

MAL_ARGS=(
  --target "${TARGET_MODEL}"
  --draft "${DRAFT_HF}"
  --eval-jsonl "${EVAL_JSONL}"
  --base "${BASE}"
  --out "${RUN_DIR}/sglang_mal.json"
  --summary "${RUN_DIR}/summary.md"
  --max-new-tokens "${MAX_NEW_TOKENS}"
  --mt-bench-turns "${MT_BENCH_TURNS}"
)
if [[ -n "${EVAL_N}" ]]; then
  MAL_ARGS+=(--n "${EVAL_N}")
fi
if [[ -n "${EVAL_CATEGORIES}" ]]; then
  # shellcheck disable=SC2206
  MAL_ARGS+=(--categories ${EVAL_CATEGORIES})
fi
if [[ -n "${COMPARE_JSON}" ]]; then
  test -s "${COMPARE_JSON}"
  MAL_ARGS+=(--compare-json "${COMPARE_JSON}" --compare-out "${RUN_DIR}/compare_sglang.json")
fi
if [[ "${ENABLE_THINKING}" == "0" || "${ENABLE_THINKING}" == "false" ]]; then
  MAL_ARGS+=(--disable-thinking)
fi

echo "=== ${EVAL_DATASET} SGLang DFLASH MAL ==="
python3 "${EVAL_PY}" sglang-mal "${MAL_ARGS[@]}" 2>&1 | tee "${RUN_DIR}/sglang_mal.log"
stop_server
if [[ "${REPLAY_OFFLINE}" == "1" || "${REPLAY_OFFLINE}" == "true" ]]; then
  echo "=== ${EVAL_DATASET} offline replay of SGLang trajectories ==="
  python3 "${EVAL_PY}" mal \
    --target "${TARGET_MODEL}" \
    --draft "${DRAFT_HF}" \
    --replay-json "${RUN_DIR}/sglang_mal.json" \
    --out "${RUN_DIR}/replay_mal.json" \
    --summary "${RUN_DIR}/replay_summary.md" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --mt-bench-turns "${MT_BENCH_TURNS}" \
    2>&1 | tee "${RUN_DIR}/replay_mal.log"
  python3 "${EVAL_PY}" compare-mal \
    --offline "${RUN_DIR}/replay_mal.json" \
    --sglang "${RUN_DIR}/sglang_mal.json" \
    --out "${RUN_DIR}/compare_replay.json" \
    --summary "${RUN_DIR}/compare_replay.md" \
    2>&1 | tee "${RUN_DIR}/compare_replay.log"
fi
echo "PASS: ${EVAL_DATASET} SGLang DFLASH MAL"
echo "run_dir=${RUN_DIR}"
