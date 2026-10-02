#!/bin/bash
# Build naqin/primus-specforge:v0.5.18-train-rocm700-mi35x on a compute node.
set -euo pipefail
SPECFORGE_SRC="${SPECFORGE_SRC:-/shared_nfs/naqin/Linear-Context-DFlash/SpecForge}"
TAG="${RUNTIME_IMAGE:-naqin/primus-specforge:v0.5.18-train-rocm700-mi35x}"
ARCHIVE="${IMAGE_ARCHIVE:-/shared_nfs/naqin/docker-images/primus-specforge-v0.5.18-train-rocm700-mi35x.tar.zst}"
DOCKERFILE="${SPECFORGE_SRC}/scripts/cluster/dflash-linear/Dockerfile.sglang-0.5.18-train"
STAGING="${STAGING:-/shared_nfs/naqin/Linear-Context-DFlash/docker-build/sglang-0.5.18-train}"

test -f "${DOCKERFILE}"
echo "building ${TAG} from ${DOCKERFILE}"

rm -rf "${STAGING}"
mkdir -p "${STAGING}"
cp "${DOCKERFILE}" "${STAGING}/Dockerfile"
sed -i 's/\r$//' "${STAGING}/Dockerfile"

run_docker() {
  if docker info >/dev/null 2>&1; then
    docker "$@"
  else
    sg docker -c "$(printf '%q ' docker "$@")"
  fi
}

run_docker pull lmsysorg/sglang:v0.5.18-rocm700-mi35x || true
run_docker build -t "${TAG}" -f "${STAGING}/Dockerfile" "${STAGING}"

DIGEST="$(run_docker image inspect "${TAG}" --format '{{.Id}}')"
echo "IMAGE_DIGEST=${DIGEST}"
mkdir -p "$(dirname "${ARCHIVE}")"
run_docker save "${TAG}" | zstd -T0 -o "${ARCHIVE}"
echo "wrote ${ARCHIVE}"
echo "${DIGEST}" > "${ARCHIVE}.digest"
echo "STAGING=${STAGING}"
