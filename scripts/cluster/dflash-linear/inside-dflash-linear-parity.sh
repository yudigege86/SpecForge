#!/bin/bash
# Live linear DFLASH MAL vs SGLang-aux teacher-force on frozen ids.
set -euo pipefail

cd /workspace/SpecForge
pip install -e . --no-deps
pip install datasets pandas tiktoken requests
bash /workspace/SpecForge/scripts/apply_sglang_spec_capture_patch.sh --target v0.5.18 || \
  echo "WARN: spec-capture patch not applied"

EVAL_PY="${EVAL_PY:-/workspace/SpecForge/scripts/eval/dflash_linear_eval.py}"
TARGET_MODEL="${TARGET_MODEL:?TARGET_MODEL is required}"
DRAFT_HF="${DRAFT_HF:?DRAFT_HF is required}"
EVAL_DATASET="${EVAL_DATASET:-humaneval}"
EVAL_N="${EVAL_N:-16}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4096}"
ENABLE_THINKING="${ENABLE_THINKING:-1}"
MT_BENCH_TURNS="${MT_BENCH_TURNS:-first}"
BLOCK_SIZE="${BLOCK_SIZE:-16}"
MEM_FRACTION="${MEM_FRACTION:-0.92}"
MAL_GATE="${MAL_GATE:-0.03}"
if [[ -z "${SERVER_PORT:-}" ]]; then
  SERVER_PORT=$((31000 + ${SLURM_JOB_ID:-0} % 1000))
fi
BASE="http://127.0.0.1:${SERVER_PORT}"
JSONL_DIR="/shared_nfs/naqin/Linear-Context-DFlash/speedbench"
case "${EVAL_DATASET}" in
  humaneval) DEFAULT_JSONL="${JSONL_DIR}/humaneval.jsonl" ;;
  mt-bench|mtbench) EVAL_DATASET="mt-bench"; DEFAULT_JSONL="${JSONL_DIR}/mtbench.jsonl" ;;
  *) echo "parity launcher expects humaneval or mt-bench, got ${EVAL_DATASET}"; exit 1 ;;
esac
EVAL_JSONL="${EVAL_JSONL:-${DEFAULT_JSONL}}"
mkdir -p "${RUN_DIR}" "$(dirname "${EVAL_JSONL}")"
if [[ "${FORCE_PREPARE:-0}" == "1" || ! -s "${EVAL_JSONL}" ]]; then
  python3 "${EVAL_PY}" prepare --dataset "${EVAL_DATASET}" --out "${EVAL_JSONL}" \
    2>&1 | tee "${RUN_DIR}/prepare.log"
fi
test -s "${EVAL_JSONL}"

python3 - <<'PY'
import os, sglang
print("sglang", getattr(sglang, "__version__", "?"))
print("SGLANG_GIT_SHA", os.environ.get("SGLANG_GIT_SHA", ""))
print("IMAGE_DIGEST", os.environ.get("IMAGE_DIGEST", ""))
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
  --max-running-requests "${MAX_RUNNING_REQUESTS:-1}"
)
if grep -q -- "--speculative-num-draft-tokens" <<<"${HELP}"; then
  LAUNCH+=(--speculative-num-draft-tokens "${BLOCK_SIZE}")
fi
if grep -q -- "--disable-radix-cache" <<<"${HELP}"; then
  LAUNCH+=(--disable-radix-cache)
fi
if grep -q -- "--mamba-scheduler-strategy" <<<"${HELP}"; then
  LAUNCH+=(--mamba-scheduler-strategy extra_buffer)
fi
if [[ "${OVERLAP_SCHEDULE:-1}" == "0" || "${OVERLAP_SCHEDULE:-1}" == "false" ]]; then
  if grep -q -- "--disable-overlap-schedule" <<<"${HELP}"; then
    LAUNCH+=(--disable-overlap-schedule)
  fi
fi

SERVER_PID=""
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
  tail -n 120 "${RUN_DIR}/sglang-server.log" || true
  exit 1
fi
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
  --summary "${RUN_DIR}/sglang_summary.md"
  --max-new-tokens "${MAX_NEW_TOKENS}"
  --mt-bench-turns "${MT_BENCH_TURNS}"
  --n "${EVAL_N}"
)
if [[ "${ENABLE_THINKING}" == "0" || "${ENABLE_THINKING}" == "false" ]]; then
  MAL_ARGS+=(--disable-thinking)
fi
echo "=== live sglang-mal ==="
python3 "${EVAL_PY}" sglang-mal "${MAL_ARGS[@]}" 2>&1 | tee "${RUN_DIR}/sglang_mal.log"
stop_server

echo "=== aux replay of frozen ids ==="
python3 "${EVAL_PY}" mal \
  --target "${TARGET_MODEL}" \
  --draft "${DRAFT_HF}" \
  --replay-json "${RUN_DIR}/sglang_mal.json" \
  --feature-source sglang \
  --out "${RUN_DIR}/replay_mal.json" \
  --summary "${RUN_DIR}/replay_summary.md" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --mt-bench-turns "${MT_BENCH_TURNS}" \
  2>&1 | tee "${RUN_DIR}/replay_mal.log"

python3 "${EVAL_PY}" compare-mal \
  --offline "${RUN_DIR}/replay_mal.json" \
  --sglang "${RUN_DIR}/sglang_mal.json" \
  --out "${RUN_DIR}/compare_parity.json" \
  --summary "${RUN_DIR}/compare_parity.md" \
  2>&1 | tee "${RUN_DIR}/compare_parity.log"

python3 - <<PY
import json, collections, sys
from pathlib import Path
run = Path("${RUN_DIR}")
live = json.loads((run / "sglang_mal.json").read_text())
cmp = json.loads((run / "compare_parity.json").read_text())
gate = float("${MAL_GATE}")
live_mean = live.get("spec_accept_length_mean")
delta = cmp.get("delta_mean")
divergent = cmp.get("n_token_divergent")
accepts = []
for row in live.get("raw") or []:
    for item in row.get("accepts") or []:
        accepts.append(int(item))
hist = dict(collections.Counter(accepts))
# Live sglang-mal records mean spec_accept_length, not per-block commit_lens.
# MAL in (1, B) implies a mix of accept and reject; MAL~1 is bonus-only and
# MAL==B is never-reject.
if hist:
    mixed = not (len(hist) == 1 and (16 in hist or 0 in hist))
    mixed_source = "accepts"
else:
    mixed = live_mean is not None and 1.05 < float(live_mean) < 15.5
    mixed_source = "mal_mean"
verdict = {
    "live_mean": live_mean,
    "replay_mean": cmp.get("offline_mean"),
    "delta_mean": delta,
    "abs_delta_mean": cmp.get("abs_delta_mean"),
    "n_token_identical": cmp.get("n_token_identical"),
    "n_token_divergent": divergent,
    "accept_hist": {str(k): hist[k] for k in sorted(hist)},
    "mixed_accept_reject": bool(mixed),
    "mixed_source": mixed_source,
    "gate": gate,
}
rel = None
if live_mean is not None and delta is not None:
    rel = abs(float(delta)) / max(abs(float(live_mean)), 1e-6)
verdict["rel_abs_delta"] = rel
verdict["passed"] = bool(
    rel is not None and rel <= gate and verdict["mixed_accept_reject"]
)
(run / "parity_verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")
print(json.dumps(verdict, indent=2), flush=True)
if Path("${STOCK_MAL_JSON:-}").is_file():
    stock = json.loads(Path("${STOCK_MAL_JSON}").read_text())
    stock_by = {str(row.get("question_id")): row for row in stock.get("raw") or []}
    n_same = n_diff = n_missing = 0
    for row in live.get("raw") or []:
        other = stock_by.get(str(row.get("question_id")))
        if other is None:
            n_missing += 1
            continue
        left = row.get("completion_ids")
        right = other.get("completion_ids")
        if left is None or right is None:
            n_missing += 1
            continue
        if list(map(int, left)) == list(map(int, right)):
            n_same += 1
        else:
            n_diff += 1
    verdict["stock_token_identical"] = n_same
    verdict["stock_token_divergent"] = n_diff
    verdict["stock_missing"] = n_missing
    (run / "parity_verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")
    print(json.dumps({"stock_token_identical": n_same, "stock_token_divergent": n_diff, "stock_missing": n_missing}, indent=2), flush=True)
if not verdict["passed"]:
    sys.exit("parity gate failed")
PY
echo "PASS: linear live vs aux-replay parity"
echo "run_dir=${RUN_DIR}"
