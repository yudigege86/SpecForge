#!/bin/bash
# 1-GPU A/B of SGLang capture features vs HuggingFace hidden_states.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SPECFORGE_SRC="${SPECFORGE_SRC:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"

RUNTIME_IMAGE="${RUNTIME_IMAGE:-naqin/primus-specforge:v0.5.14-rocm700-mi35x}"
IMAGE_ARCHIVE="${IMAGE_ARCHIVE:-/shared_nfs/naqin/docker-images/primus-specforge-v0.5.14-rocm700-mi35x.tar.zst}"
RESULTS_DIR="${RESULTS_DIR:-/shared_nfs/naqin/Linear-Context-DFlash/feature-ab}"
HF_HOME="${HF_HOME:-/shared_nfs/naqin/hf-cache}"
HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
HIDDEN_STATES_PATH="${HIDDEN_STATES_PATH:-/shared_nfs/naqin/primus-specforge-smoke/qwen-capture-40k/valid-links}"
AB_N="${AB_N:-8}"
if [[ -z "${HIP_VISIBLE_DEVICES:-}" && -z "${CUDA_VISIBLE_DEVICES:-}" && -z "${ROCR_VISIBLE_DEVICES:-}" ]]; then
  HIP_VISIBLE_DEVICES=0
  CUDA_VISIBLE_DEVICES=0
fi
HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-${ROCR_VISIBLE_DEVICES:-0}}}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${HIP_VISIBLE_DEVICES}}"

mkdir -p "${RESULTS_DIR}" "${HF_HOME}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_DIR="${RESULTS_DIR}/${STAMP}"
mkdir -p "${RUN_DIR}"
exec > >(tee "${RUN_DIR}/run.log") 2>&1

echo "host=$(hostname) stamp=${STAMP} image=${RUNTIME_IMAGE}"
echo "job=${SLURM_JOB_ID:-none} node=${SLURMD_NODENAME:-$(hostname)}"
echo "HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "HIDDEN_STATES_PATH=${HIDDEN_STATES_PATH} AB_N=${AB_N}"
echo "SPECFORGE_SRC=${SPECFORGE_SRC}"

test -f "${SPECFORGE_SRC}/scripts/eval/dflash_linear_feature_ab.py"
test -d "${HIDDEN_STATES_PATH}"

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

NAME="dflash-linear-ab-${STAMP}"
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
  -e TRANSFORMERS_OFFLINE=1 \
  -e HF_HUB_OFFLINE=1 \
  -e RUN_DIR="${RUN_DIR}" \
  -e HIDDEN_STATES_PATH="${HIDDEN_STATES_PATH}" \
  -e AB_N="${AB_N}" \
  -v /shared_nfs:/shared_nfs \
  -v "${SPECFORGE_SRC}:/workspace/SpecForge" \
  "${RUNTIME_IMAGE}" \
  bash /workspace/SpecForge/scripts/cluster/dflash-linear/inside-dflash-linear-feature-ab.sh
rc=$?
set -e

echo "docker_exit=${rc}"
if [[ ${rc} -ne 0 ]]; then
  echo "FAIL: dflash_linear feature A/B exited ${rc}"
  exit "${rc}"
fi

echo "PASS: dflash_linear feature A/B"
echo "run_dir=${RUN_DIR}"
