#!/bin/bash
# Shared docker runner for linear DFLASH cluster jobs.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SPECFORGE_SRC="${SPECFORGE_SRC:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
INSIDE_SCRIPT="${INSIDE_SCRIPT:?INSIDE_SCRIPT is required}"
CONTAINER_NAME="${CONTAINER_NAME:-dflash-linear-job}"

RUNTIME_IMAGE="${RUNTIME_IMAGE:-naqin/primus-specforge:v0.5.18-dflash-linear-rocm700-mi35x}"
IMAGE_ARCHIVE="${IMAGE_ARCHIVE:-/shared_nfs/naqin/docker-images/primus-specforge-v0.5.18-dflash-linear-rocm700-mi35x.tar.zst}"
HF_HOME="${HF_HOME:-/shared_nfs/naqin/hf-cache}"
HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
TARGET_MODEL="${TARGET_MODEL:-Qwen/Qwen3.5-4B}"
DRAFT_HF="${DRAFT_HF:-/shared_nfs/naqin/Linear-Context-DFlash/eval-1epoch/20260918T213552Z/draft_hf}"
if [[ -z "${SERVER_PORT:-}" ]]; then
  SERVER_PORT=$((31000 + ${SLURM_JOB_ID:-0} % 1000))
fi
if [[ -z "${HIP_VISIBLE_DEVICES:-}" && -z "${CUDA_VISIBLE_DEVICES:-}" && -z "${ROCR_VISIBLE_DEVICES:-}" ]]; then
  if [[ -n "${SLURM_JOB_GPUS:-}" ]]; then
    HIP_VISIBLE_DEVICES="${SLURM_JOB_GPUS}"
    CUDA_VISIBLE_DEVICES="${SLURM_JOB_GPUS}"
  else
    HIP_VISIBLE_DEVICES=0
    CUDA_VISIBLE_DEVICES=0
  fi
fi
HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-${ROCR_VISIBLE_DEVICES:-0}}}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${HIP_VISIBLE_DEVICES}}"
echo "resolved HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-}"

if [[ -z "${HF_TOKEN:-}" ]]; then
  if [[ -f "${HOME}/.cache/huggingface/token" ]]; then
    HF_TOKEN="$(tr -d '\r\n' < "${HOME}/.cache/huggingface/token")"
  elif [[ -f "${HF_HOME}/token" ]]; then
    HF_TOKEN="$(tr -d '\r\n' < "${HF_HOME}/token")"
  fi
fi

RESULTS_DIR="${RESULTS_DIR:?RESULTS_DIR is required}"
mkdir -p "${RESULTS_DIR}" "${HF_HOME}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_DIR="${RESULTS_DIR}/${STAMP}"
mkdir -p "${RUN_DIR}"
exec > >(tee "${RUN_DIR}/run.log") 2>&1

echo "host=$(hostname) stamp=${STAMP} image=${RUNTIME_IMAGE} inside=${INSIDE_SCRIPT}"

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
  USE_SG_DOCKER=1
  run_docker info >/dev/null 2>&1
fi
if ! run_docker image inspect "${RUNTIME_IMAGE}" >/dev/null 2>&1; then
  echo "loading ${RUNTIME_IMAGE} from ${IMAGE_ARCHIVE}"
  test -s "${IMAGE_ARCHIVE}"
  zstd -dc "${IMAGE_ARCHIVE}" | run_docker load
fi
IMAGE_DIGEST="$(run_docker image inspect "${RUNTIME_IMAGE}" --format '{{.Id}}')"
echo "IMAGE_DIGEST=${IMAGE_DIGEST}"
echo "${RUN_DIR}" > "${RESULTS_DIR}/latest_run_dir"

# shellcheck source=sglang-overlay-mounts.sh
source "${SCRIPT_DIR}/sglang-overlay-mounts.sh"

NAME="${CONTAINER_NAME}-${STAMP}"
run_docker rm -f "${NAME}" >/dev/null 2>&1 || true
set +e
run_docker run --rm --name "${NAME}" \
  --network host --ipc=host --shm-size=32g \
  --ulimit memlock=-1:-1 \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --security-opt seccomp=unconfined \
  -e HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES}" \
  -e CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  -e HF_HOME="${HF_HOME}" \
  -e HF_HUB_CACHE="${HF_HUB_CACHE}" \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  -e RUN_DIR="${RUN_DIR}" \
  -e TARGET_MODEL="${TARGET_MODEL}" \
  -e DRAFT_HF="${DRAFT_HF}" \
  -e EVAL_DATASET="${EVAL_DATASET:-humaneval}" \
  -e EVAL_JSONL="${EVAL_JSONL:-}" \
  -e EVAL_N="${EVAL_N:-}" \
  -e MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4096}" \
  -e BLOCK_SIZE="${BLOCK_SIZE:-16}" \
  -e ENABLE_THINKING="${ENABLE_THINKING:-1}" \
  -e MT_BENCH_TURNS="${MT_BENCH_TURNS:-first}" \
  -e MEM_FRACTION="${MEM_FRACTION:-0.92}" \
  -e SERVER_PORT="${SERVER_PORT}" \
  -e MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-1}" \
  -e SGLANG_DFLASH_LINEAR_SHADOW_CHECK="${SGLANG_DFLASH_LINEAR_SHADOW_CHECK:-0}" \
  -e SGLANG_GIT_SHA="${SGLANG_GIT_SHA:-}" \
  -e IMAGE_DIGEST="${IMAGE_DIGEST}" \
  -e RUNTIME_IMAGE="${RUNTIME_IMAGE}" \
  -e MAL_GATE="${MAL_GATE:-0.03}" \
  -e OVERLAP_SCHEDULE="${OVERLAP_SCHEDULE:-1}" \
  -e STOCK_MAL_JSON="${STOCK_MAL_JSON:-}" \
  -e HIDDEN_STATES_PATH="${HIDDEN_STATES_PATH:-}" \
  -e FEATURE_CONTRACT_GATE="${FEATURE_CONTRACT_GATE:-0.99}" \
  -e COMPARE_JSON="${COMPARE_JSON:-}" \
  -e REPLAY_OFFLINE="${REPLAY_OFFLINE:-0}" \
  -e FORCE_PREPARE="${FORCE_PREPARE:-0}" \
  -e SLURM_JOB_ID="${SLURM_JOB_ID:-}" \
  -v /shared_nfs:/shared_nfs \
  -v "${SPECFORGE_SRC}:/workspace/SpecForge" \
  "${OVERLAY_MOUNTS[@]}" \
  "${RUNTIME_IMAGE}" \
  bash "${INSIDE_SCRIPT}"
rc=$?
set -e
echo "docker_exit=${rc} run_dir=${RUN_DIR}"
echo "${RUN_DIR}" > "${RESULTS_DIR}/latest_run_dir"
exit "${rc}"
