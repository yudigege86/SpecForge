#!/bin/bash
# Pull/save the ROCm SGLang image, cache Qwen3.5-4B, prepare Open-PerfectBlend prompts, split shards.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=paths.sh
source "${SCRIPT_DIR}/paths.sh"

mkdir -p "${ROOT}" "${PROMPTS_DIR}" "${REGEN_DIR}" "${LOG_ROOT}" "${HF_HOME}" \
  "$(dirname "${IMAGE_ARCHIVE}")"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_DIR="${LOG_ROOT}/prepare-${STAMP}"
mkdir -p "${RUN_DIR}"
exec > >(tee "${RUN_DIR}/run.log") 2>&1

echo "host=$(hostname) stamp=${STAMP} image=${RUNTIME_IMAGE}"
echo "job=${SLURM_JOB_ID:-none} node=${SLURMD_NODENAME:-$(hostname)}"
echo "SPECFORGE_SRC=${SPECFORGE_SRC}"
echo "PROMPTS_DIR=${PROMPTS_DIR}"
echo "NUM_SHARDS=${NUM_SHARDS} SMOKE_ROWS=${SMOKE_ROWS}"

test -f "${SPECFORGE_SRC}/scripts/prepare_data.py"
test -f "${SCRIPT_DIR}/inside-prepare.sh"
test -f "${SCRIPT_DIR}/split_shards.py"

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
  if [[ -s "${IMAGE_ARCHIVE}" ]]; then
    echo "loading ${RUNTIME_IMAGE} from ${IMAGE_ARCHIVE}"
    zstd -dc "${IMAGE_ARCHIVE}" | run_docker load
  else
    echo "pulling ${RUNTIME_IMAGE}"
    run_docker pull "${RUNTIME_IMAGE}"
  fi
fi
run_docker image inspect "${RUNTIME_IMAGE}" --format '{{.Id}}'

if [[ ! -s "${IMAGE_ARCHIVE}" ]]; then
  echo "saving ${RUNTIME_IMAGE} to ${IMAGE_ARCHIVE}"
  run_docker save "${RUNTIME_IMAGE}" | zstd -T0 -o "${IMAGE_ARCHIVE}"
  ls -lh "${IMAGE_ARCHIVE}"
fi

NAME="pb-prepare-${STAMP}"
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
  --network host --ipc=host --shm-size=16g \
  --ulimit memlock=-1:-1 \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --security-opt seccomp=unconfined \
  "${DOCKER_ENV[@]}" \
  -e HF_HOME="${HF_HOME}" \
  -e HF_HUB_CACHE="${HF_HUB_CACHE}" \
  -e HF_DATASETS_CACHE="${HF_DATASETS_CACHE}" \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  -e PROMPTS_DIR="${PROMPTS_DIR}" \
  -e MODEL_PATH="${MODEL_PATH}" \
  -e NUM_SHARDS="${NUM_SHARDS}" \
  -e SMOKE_ROWS="${SMOKE_ROWS}" \
  -v /shared_nfs:/shared_nfs \
  -v "${SPECFORGE_SRC}:/workspace/SpecForge" \
  -v "${SCRIPT_DIR}:/opt/perfectblend-regen:ro" \
  "${RUNTIME_IMAGE}" \
  bash /opt/perfectblend-regen/inside-prepare.sh
rc=$?
set -e

echo "docker_exit=${rc}"
if [[ ${rc} -ne 0 ]]; then
  echo "FAIL: perfectblend prepare exited ${rc}"
  exit "${rc}"
fi

echo "PASS: perfectblend prepare"
echo "prompts_dir=${PROMPTS_DIR}"
