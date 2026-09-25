#!/bin/bash
# SPEED-Bench Qualitative MAL inside the Primus ROCm image.
set -euo pipefail

cd /workspace/SpecForge

echo "=== pip install -e . --no-deps ==="
pip install -e . --no-deps
pip install datasets pandas tiktoken

echo "=== CPU MAL helper / stock acceptance tests ==="
python3 -m unittest \
  tests.test_scripts.test_dflash_linear_eval \
  tests.test_modeling.test_dflash_acceptance \
  -v

TARGET_MODEL="${TARGET_MODEL:?TARGET_MODEL is required}"
DRAFT_HF="${DRAFT_HF:?DRAFT_HF is required}"
EVAL_DATASET="${EVAL_DATASET:-qualitative}"
EVAL_PY="${EVAL_PY:-/workspace/SpecForge/scripts/eval/dflash_linear_eval.py}"
EVAL_N="${EVAL_N:-}"
EVAL_CATEGORIES="${EVAL_CATEGORIES:-}"
MT_BENCH_TURNS="${MT_BENCH_TURNS:-first}"
FORCE_PREPARE="${FORCE_PREPARE:-0}"
ENABLE_THINKING="${ENABLE_THINKING:-}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-}"
REPLAY_JSON="${REPLAY_JSON:-}"
FEATURE_OFFSET="${FEATURE_OFFSET:-auto}"
FEATURE_SOURCE="${FEATURE_SOURCE:-hf}"
COMPARE_JSON="${COMPARE_JSON:-}"
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

python3 - <<PY
import json
from pathlib import Path
draft = Path("${DRAFT_HF}")
cfg_path = draft / "config.json" if draft.is_dir() else None
if cfg_path and cfg_path.is_file():
    cfg = json.loads(cfg_path.read_text())
    arch = cfg.get("architectures") or []
    print("architectures", arch)
    Path("${RUN_DIR}/draft_arch.json").write_text(json.dumps(cfg.get("dflash_config") or {}, indent=2))
else:
    print("draft_hf_id", "${DRAFT_HF}")
PY

ARCH="$(python3 - <<'PY'
import json
from pathlib import Path
import os
draft = Path(os.environ["DRAFT_HF"])
cfg = draft / "config.json"
if cfg.is_file():
    print(((json.loads(cfg.read_text()).get("architectures") or ["unknown"])[0]))
else:
    print("unknown")
PY
)"

if [[ "${ARCH}" == "DFlashLinearDraftModel" ]]; then
  echo "=== install flash-linear-attention[rocm] without replacing image torch ==="
  python3 - <<'PY'
import torch
print("torch_before_fla", torch.__version__, "hip", getattr(torch.version, "hip", None))
PY
  pip install einops
  pip install 'flash-linear-attention[rocm]'
  python3 - <<'PY'
import json
import os
from pathlib import Path
import torch
from specforge.modeling.draft.linear_context import fla_available, resolve_scan_backend
print("cuda_available", torch.cuda.is_available())
assert torch.cuda.is_available(), "MAL eval needs a GPU"
cfg_path = Path(os.environ["DRAFT_HF"]) / "config.json"
variant = "gdn"
if cfg_path.is_file():
    method = json.loads(cfg_path.read_text()).get("dflash_config") or {}
    variant = str((method.get("linear_context") or {}).get("variant") or "gdn")
print("linear_context.variant", variant)
assert fla_available(variant), f"MAL requires FLA {variant} on GPU"
resolved = resolve_scan_backend("auto", on_cuda=True, num_anchors=1)
print("auto_backend", resolved)
assert resolved == "fla", f"expected FLA scan backend, got {resolved!r}"
PY
  echo "=== linear teacher-force tests ==="
  python3 -m unittest \
    tests.test_modeling.test_dflash_linear_layer.DFlashLinearLayerTest.test_acceptance_along_sequence_does_not_use_kv_cache \
    tests.test_modeling.test_dflash_linear_layer.DFlashLinearLayerTest.test_acceptance_along_sequence_masks_future_block_tokens \
    -v
else
  echo "=== stock DFlash: skip FLA ==="
  python3 - <<'PY'
import torch
print("torch", torch.__version__, "hip", getattr(torch.version, "hip", None))
print("cuda_available", torch.cuda.is_available())
assert torch.cuda.is_available(), "MAL eval needs a GPU"
PY
fi

if [[ -n "${REPLAY_JSON}" ]]; then
  test -s "${REPLAY_JSON}"
else
  if [[ "${FORCE_PREPARE}" == "1" || ! -s "${EVAL_JSONL}" ]]; then
    echo "=== prepare ${EVAL_DATASET} ==="
    python3 "${EVAL_PY}" prepare \
      --dataset "${EVAL_DATASET}" \
      --out "${EVAL_JSONL}" \
      2>&1 | tee "${RUN_DIR}/prepare.log"
  fi
  test -s "${EVAL_JSONL}"
fi

MAL_ARGS=(
  --target "${TARGET_MODEL}"
  --draft "${DRAFT_HF}"
  --out "${RUN_DIR}/mal.json"
  --summary "${RUN_DIR}/summary.md"
  --max-new-tokens "${MAX_NEW_TOKENS}"
  --mt-bench-turns "${MT_BENCH_TURNS}"
  --feature-offset "${FEATURE_OFFSET}"
  --feature-source "${FEATURE_SOURCE}"
)
if [[ -n "${REPLAY_JSON}" ]]; then
  MAL_ARGS+=(--replay-json "${REPLAY_JSON}")
else
  MAL_ARGS+=(--eval-jsonl "${EVAL_JSONL}")
fi
if [[ "${ENABLE_THINKING}" == "0" || "${ENABLE_THINKING}" == "false" ]]; then
  MAL_ARGS+=(--disable-thinking)
fi
if [[ -n "${EVAL_N}" ]]; then
  MAL_ARGS+=(--n "${EVAL_N}")
fi
if [[ -n "${EVAL_CATEGORIES}" ]]; then
  # shellcheck disable=SC2206
  MAL_ARGS+=(--categories ${EVAL_CATEGORIES})
fi

echo "=== ${EVAL_DATASET} MAL feature_offset=${FEATURE_OFFSET} feature_source=${FEATURE_SOURCE} replay=${REPLAY_JSON:-none} ==="
python3 "${EVAL_PY}" mal "${MAL_ARGS[@]}" 2>&1 | tee "${RUN_DIR}/mal.log"
if [[ -n "${COMPARE_JSON}" ]]; then
  test -s "${COMPARE_JSON}"
  echo "=== compare replay vs ${COMPARE_JSON} ==="
  python3 "${EVAL_PY}" compare-mal \
    --offline "${RUN_DIR}/mal.json" \
    --sglang "${COMPARE_JSON}" \
    --out "${RUN_DIR}/compare_replay.json" \
    --summary "${RUN_DIR}/compare_replay.md" \
    2>&1 | tee "${RUN_DIR}/compare_replay.log"
fi
echo "PASS: ${EVAL_DATASET} MAL"
echo "run_dir=${RUN_DIR}"
