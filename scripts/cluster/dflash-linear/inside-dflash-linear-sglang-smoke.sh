#!/bin/bash
# M0: load DFlashLinearDraftModel under algorithm DFLASH and run one generate.
set -euo pipefail

cd /workspace/SpecForge
pip install -e . --no-deps
pip install datasets pandas tiktoken requests

TARGET_MODEL="${TARGET_MODEL:?TARGET_MODEL is required}"
DRAFT_HF="${DRAFT_HF:?DRAFT_HF is required}"
BLOCK_SIZE="${BLOCK_SIZE:-16}"
MEM_FRACTION="${MEM_FRACTION:-0.92}"
if [[ -z "${SERVER_PORT:-}" ]]; then
  SERVER_PORT=$((31000 + ${SLURM_JOB_ID:-0} % 1000))
fi
BASE="http://127.0.0.1:${SERVER_PORT}"
mkdir -p "${RUN_DIR}"

python3 - <<PY
import json
from pathlib import Path
from types import SimpleNamespace
from sglang.srt.models.registry import ModelRegistry
from sglang.srt.speculative.dflash_linear_worker_v2 import DFlashLinearWorkerV2
from sglang.srt.speculative.dflash_utils import is_dflash_linear_config, is_dflash_linear_draft
from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

cfg = json.loads((Path("${DRAFT_HF}") / "config.json").read_text())
arch = (cfg.get("architectures") or [None])[0]
print("architectures", cfg.get("architectures"))
if arch != "DFlashLinearDraftModel":
    raise SystemExit(f"expected DFlashLinearDraftModel, got {arch}")

supported = set(ModelRegistry.get_supported_archs())
print("registry_has_DFlash2DraftModel", "DFlash2DraftModel" in supported)
print("registry_has_DFlashLinearDraftModel", "DFlashLinearDraftModel" in supported)
if "DFlash2DraftModel" not in supported:
    raise SystemExit("DFlash2DraftModel missing from SGLang model registry")
if "DFlashLinearDraftModel" not in supported:
    raise SystemExit("DFlashLinearDraftModel missing from SGLang model registry")
print("is_dflash_linear_config", is_dflash_linear_config(cfg))
args = SimpleNamespace(
    speculative_draft_model_path="${DRAFT_HF}",
    json_model_override_args=None,
    trust_remote_code=True,
    speculative_draft_model_revision=None,
)
print("is_dflash_linear_draft", is_dflash_linear_draft(args))
if not is_dflash_linear_draft(args):
    raise SystemExit("create_worker would not route linear -> DFlashLinearWorkerV2")
if not issubclass(DFlashLinearWorkerV2, DFlashWorkerV2):
    raise SystemExit("DFlashLinearWorkerV2 must subclass DFlashWorkerV2")
try:
    worker_cls = SpeculativeAlgorithm.DFLASH.create_worker(args)
    print("create_worker", getattr(worker_cls, "__name__", worker_cls))
    if worker_cls is not DFlashLinearWorkerV2:
        raise SystemExit(f"expected DFlashLinearWorkerV2, got {worker_cls}")
except Exception as exc:
    print("create_worker_probe", type(exc).__name__, exc)
print("PASS: DFlash2 + DFlashLinear registry")
try:
    import torch
    print("cuda_is_available", torch.cuda.is_available())
    print("device_count", torch.cuda.device_count() if torch.cuda.is_available() else 0)
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info(0)
        print("gpu0_free_gb", round(free / 1024**3, 2), "total_gb", round(total / 1024**3, 2))
        print("gpu0_name", torch.cuda.get_device_name(0))
except Exception as exc:
    print("gpu_probe_failed", exc)
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
}
trap stop_server EXIT
echo "=== ${LAUNCH[*]} ==="
"${LAUNCH[@]}" >"${RUN_DIR}/sglang-server.log" 2>&1 &
SERVER_PID=$!
sleep 5
if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
  echo "FAIL: server exited immediately"
  tail -n 160 "${RUN_DIR}/sglang-server.log" || true
  exit 1
fi
ready=0
for _ in $(seq 1 180); do
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "FAIL: server died"
    tail -n 160 "${RUN_DIR}/sglang-server.log" || true
    exit 1
  fi
  if grep -q "The server is fired up and ready to roll" "${RUN_DIR}/sglang-server.log"; then
    ready=1
    break
  fi
  sleep 5
done
if [[ "${ready}" != "1" ]]; then
  echo "FAIL: not ready"
  tail -n 160 "${RUN_DIR}/sglang-server.log" || true
  exit 1
fi

python3 - <<PY
import json, urllib.request
base = "${BASE}"
info = json.loads(urllib.request.urlopen(base + "/get_server_info", timeout=30).read().decode())
(open("${RUN_DIR}/server_info.json", "w")).write(json.dumps(info, indent=2, default=str))
algo = str(info.get("speculative_algorithm") or info.get("spec_algorithm") or "")
print("speculative_algorithm", algo)
if "DFLASH" not in algo.upper():
    raise SystemExit(f"expected DFLASH, got {algo!r}")
payload = json.dumps({
    "text": "Write a Python function that returns 1.\n",
    "sampling_params": {"temperature": 0.0, "max_new_tokens": 32, "ignore_eos": False},
}).encode()
req = urllib.request.Request(base + "/generate", data=payload, headers={"Content-Type": "application/json"})
out = json.loads(urllib.request.urlopen(req, timeout=180).read().decode())
(open("${RUN_DIR}/generate.json", "w")).write(json.dumps(out, indent=2, default=str))
meta = out.get("meta_info") or {}
print("spec_verify_ct", meta.get("spec_verify_ct"))
print("spec_accept_length", meta.get("spec_accept_length"))
print("completion_tokens", meta.get("completion_tokens"))
if meta.get("spec_verify_ct") in (None, 0):
    raise SystemExit("generate returned no spec_verify_ct")
print("PASS: DFlashLinear smoke")
PY
echo "run_dir=${RUN_DIR}"
