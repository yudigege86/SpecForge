#!/bin/bash
# 1-GPU SPEED-Bench Qualitative MAL via stock SGLang DFLASH.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SPECFORGE_SRC="${SPECFORGE_SRC:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"

RUNTIME_IMAGE="${RUNTIME_IMAGE:-naqin/primus-specforge:v0.5.14-rocm700-mi35x}"
IMAGE_ARCHIVE="${IMAGE_ARCHIVE:-/shared_nfs/naqin/docker-images/primus-specforge-v0.5.14-rocm700-mi35x.tar.zst}"
HF_HOME="${HF_HOME:-/shared_nfs/naqin/hf-cache}"
HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
TARGET_MODEL="${TARGET_MODEL:-Qwen/Qwen3.5-4B}"
DRAFT_HF="${DRAFT_HF:-z-lab/Qwen3.5-4B-DFlash}"
EVAL_DATASET="${EVAL_DATASET:-qualitative}"
SPEEDBENCH_JSONL="${SPEEDBENCH_JSONL:-}"
EVAL_JSONL="${EVAL_JSONL:-}"
EVAL_N="${EVAL_N:-}"
EVAL_CATEGORIES="${EVAL_CATEGORIES:-}"
COMPARE_JSON="${COMPARE_JSON:-}"
BLOCK_SIZE="${BLOCK_SIZE:-16}"
MT_BENCH_TURNS="${MT_BENCH_TURNS:-first}"
FORCE_PREPARE="${FORCE_PREPARE:-0}"
REPLAY_OFFLINE="${REPLAY_OFFLINE:-0}"
MEM_FRACTION="${MEM_FRACTION:-0.92}"
if [[ -z "${MAX_NEW_TOKENS:-}" ]]; then
  if [[ "${EVAL_DATASET}" == "qualitative" || "${EVAL_DATASET}" == "speedbench" || "${EVAL_DATASET}" == "speedbench-qualitative" ]]; then
    MAX_NEW_TOKENS=512
  else
    MAX_NEW_TOKENS=4096
  fi
fi
if [[ -z "${ENABLE_THINKING:-}" ]]; then
  if [[ "${EVAL_DATASET}" == "qualitative" || "${EVAL_DATASET}" == "speedbench" || "${EVAL_DATASET}" == "speedbench-qualitative" ]]; then
    ENABLE_THINKING=0
  else
    ENABLE_THINKING=1
  fi
fi
if [[ -z "${SERVER_PORT:-}" ]]; then
  SERVER_PORT=$((31000 + ${SLURM_JOB_ID:-0} % 1000))
fi
if [[ -z "${RESULTS_DIR:-}" ]]; then
  if [[ "${EVAL_DATASET}" == "qualitative" ]]; then
    RESULTS_DIR="/shared_nfs/naqin/Linear-Context-DFlash/speedbench-sglang"
  else
    RESULTS_DIR="/shared_nfs/naqin/Linear-Context-DFlash/mal-eval/${EVAL_DATASET}-sglang"
  fi
fi
if [[ -z "${HIP_VISIBLE_DEVICES:-}" && -z "${CUDA_VISIBLE_DEVICES:-}" && -z "${ROCR_VISIBLE_DEVICES:-}" ]]; then
  HIP_VISIBLE_DEVICES=0
  CUDA_VISIBLE_DEVICES=0
fi
HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-${ROCR_VISIBLE_DEVICES:-0}}}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${HIP_VISIBLE_DEVICES}}"

if [[ -z "${HF_TOKEN:-}" ]]; then
  if [[ -f "${HOME}/.cache/huggingface/token" ]]; then
    HF_TOKEN="$(tr -d '\r\n' < "${HOME}/.cache/huggingface/token")"
  elif [[ -f "${HF_HOME}/token" ]]; then
    HF_TOKEN="$(tr -d '\r\n' < "${HF_HOME}/token")"
  fi
fi

mkdir -p "${RESULTS_DIR}" "${HF_HOME}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_DIR="${RESULTS_DIR}/${STAMP}"
mkdir -p "${RUN_DIR}"
exec > >(tee "${RUN_DIR}/run.log") 2>&1

echo "host=$(hostname) stamp=${STAMP} image=${RUNTIME_IMAGE}"
echo "job=${SLURM_JOB_ID:-none} node=${SLURMD_NODENAME:-$(hostname)}"
echo "HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "TARGET_MODEL=${TARGET_MODEL}"
echo "DRAFT_HF=${DRAFT_HF}"
echo "EVAL_DATASET=${EVAL_DATASET} EVAL_JSONL=${EVAL_JSONL:-auto} EVAL_N=${EVAL_N:-all} MAX_NEW_TOKENS=${MAX_NEW_TOKENS}"
echo "BLOCK_SIZE=${BLOCK_SIZE} ENABLE_THINKING=${ENABLE_THINKING} MT_BENCH_TURNS=${MT_BENCH_TURNS} REPLAY_OFFLINE=${REPLAY_OFFLINE} COMPARE_JSON=${COMPARE_JSON:-none}"
echo "MEM_FRACTION=${MEM_FRACTION} SERVER_PORT=${SERVER_PORT}"
echo "SPECFORGE_SRC=${SPECFORGE_SRC}"

test -f "${SPECFORGE_SRC}/scripts/eval/dflash_linear_eval.py"

run_docker() {
  if [ "${USE_SG_DOCKER:-0}" = 1 ]; then
    sg docker -c "$(printf '%q ' docker "$@")"
  else
    docker "$@"
  fi
}

if docker info >/dev/null 2>&1; then
  USE_SG_DOCKER=0
else
  echo "docker info failed on $(hostname); retrying under sg docker"
  USE_SG_DOCKER=1
  if ! run_docker info >/dev/null 2>&1; then
    echo "FAIL: docker unavailable on $(hostname)"
    exit 1
  fi
fi

if ! run_docker image inspect "${RUNTIME_IMAGE}" >/dev/null 2>&1; then
  echo "loading ${RUNTIME_IMAGE} from ${IMAGE_ARCHIVE}"
  test -s "${IMAGE_ARCHIVE}"
  zstd -dc "${IMAGE_ARCHIVE}" | run_docker load
fi
run_docker image inspect "${RUNTIME_IMAGE}" --format '{{.Id}}'

NAME="dflash-speedbench-sglang-${STAMP}"
run_docker rm -f "${NAME}" >/dev/null 2>&1 || true

DOCKER_ENV=()
if [[ -n "${HIP_VISIBLE_DEVICES:-}" ]]; then
  DOCKER_ENV+=(-e "HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES}")
fi
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  DOCKER_ENV+=(-e "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}")
fi

set +e
run_docker run --rm --name "${NAME}" \
  --network host --ipc=host --shm-size=32g \
  --ulimit memlock=-1:-1 \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --security-opt seccomp=unconfined \
  "${DOCKER_ENV[@]}" \
  -e HF_HOME="${HF_HOME}" \
  -e HF_HUB_CACHE="${HF_HUB_CACHE}" \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  -e RUN_DIR="${RUN_DIR}" \
  -e TARGET_MODEL="${TARGET_MODEL}" \
  -e DRAFT_HF="${DRAFT_HF}" \
  -e EVAL_DATASET="${EVAL_DATASET}" \
  -e EVAL_JSONL="${EVAL_JSONL}" \
  -e SPEEDBENCH_JSONL="${SPEEDBENCH_JSONL}" \
  -e EVAL_N="${EVAL_N}" \
  -e MAX_NEW_TOKENS="${MAX_NEW_TOKENS}" \
  -e BLOCK_SIZE="${BLOCK_SIZE}" \
  -e ENABLE_THINKING="${ENABLE_THINKING}" \
  -e MT_BENCH_TURNS="${MT_BENCH_TURNS}" \
  -e FORCE_PREPARE="${FORCE_PREPARE}" \
  -e REPLAY_OFFLINE="${REPLAY_OFFLINE}" \
  -e EVAL_CATEGORIES="${EVAL_CATEGORIES}" \
  -e COMPARE_JSON="${COMPARE_JSON}" \
  -e MEM_FRACTION="${MEM_FRACTION}" \
  -e SERVER_PORT="${SERVER_PORT}" \
  -e SLURM_JOB_ID="${SLURM_JOB_ID:-}" \
  -v /shared_nfs:/shared_nfs \
  -v "${SPECFORGE_SRC}:/workspace/SpecForge" \
  "${RUNTIME_IMAGE}" \
  bash /workspace/SpecForge/scripts/cluster/dflash-linear/inside-dflash-linear-speedbench-sglang.sh
rc=$?
set -e

echo "docker_exit=${rc}"
if [[ ${rc} -ne 0 ]]; then
  echo "FAIL: SPEED-Bench SGLang MAL exited ${rc}"
  exit "${rc}"
fi

echo "PASS: SPEED-Bench SGLang DFLASH MAL"
echo "run_dir=${RUN_DIR}"
