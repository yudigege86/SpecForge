#!/bin/bash
# Build naqin/primus-specforge:v0.5.19-dflash-linear-rocm700-mi35x on a compute node.
set -euo pipefail
SGLANG_SRC="${SGLANG_SRC:-/shared_nfs/naqin/Linear-Context-DFlash/sglang}"
SPECFORGE_SRC="${SPECFORGE_SRC:-/shared_nfs/naqin/Linear-Context-DFlash/SpecForge}"
TAG="${RUNTIME_IMAGE:-naqin/primus-specforge:v0.5.19-dflash-linear-rocm700-mi35x}"
ARCHIVE="${IMAGE_ARCHIVE:-/shared_nfs/naqin/docker-images/primus-specforge-v0.5.19-dflash-linear-rocm700-mi35x.tar.zst}"
DOCKERFILE="${SPECFORGE_SRC}/scripts/cluster/dflash-linear/Dockerfile.sglang-0.5.19"
STAGING="${STAGING:-/shared_nfs/naqin/Linear-Context-DFlash/docker-build/sglang-0.5.19-overlay}"

test -f "${SGLANG_SRC}/python/sglang/srt/models/dflash_linear.py"
test -f "${SGLANG_SRC}/python/sglang/srt/models/dflash.py"
if ! grep -q "DFlash2DraftModel" "${SGLANG_SRC}/python/sglang/srt/models/dflash.py"; then
  echo "error: ${SGLANG_SRC} is not a v0.5.19 tree (DFlash2DraftModel missing). Checkout dflash-linear-v0.5.19 before building." >&2
  exit 1
fi
test -f "${SPECFORGE_SRC}/patches/sglang/v0.5.19/spec-capture.patch"
test -f "${DOCKERFILE}"

SHA="$(git -C "${SGLANG_SRC}" rev-parse HEAD 2>/dev/null || echo unknown)"
echo "building ${TAG} overlay from ${SGLANG_SRC} sha=${SHA}"

rm -rf "${STAGING}"
mkdir -p "${STAGING}/overlay"
cp "${DOCKERFILE}" "${STAGING}/Dockerfile"
cp "${SPECFORGE_SRC}/patches/sglang/v0.5.19/spec-capture.patch" "${STAGING}/spec-capture.patch"
cp "${SPECFORGE_SRC}/scripts/apply_sglang_spec_capture_patch.sh" "${STAGING}/apply_sglang_spec_capture_patch.sh"
# Docker RUN scripts must be LF. Staging may inherit CRLF from a Windows scp.
sed -i 's/\r$//' "${STAGING}/Dockerfile" "${STAGING}/apply_sglang_spec_capture_patch.sh"

OVERLAY_FILES=(
  python/sglang/srt/models/dflash_linear.py
  python/sglang/srt/speculative/dflash_linear_state.py
  python/sglang/srt/speculative/dflash_linear_worker_v2.py
  python/sglang/srt/speculative/dflash_worker_v2.py
  python/sglang/srt/speculative/spec_info.py
  python/sglang/srt/speculative/dflash_utils.py
  python/sglang/srt/arg_groups/speculative_hook.py
)
for rel in "${OVERLAY_FILES[@]}"; do
  src="${SGLANG_SRC}/${rel}"
  test -f "${src}"
  # Installed package lives at .../sglang/; drop the python/sglang prefix.
  case "${rel}" in
    python/sglang/*)
      dest="${STAGING}/overlay/${rel#python/sglang/}"
      ;;
    *)
      dest="${STAGING}/overlay/${rel}"
      ;;
  esac
  mkdir -p "$(dirname "${dest}")"
  cp "${src}" "${dest}"
done

run_docker() {
  if docker info >/dev/null 2>&1; then
    docker "$@"
  else
    sg docker -c "$(printf '%q ' docker "$@")"
  fi
}

run_docker pull lmsysorg/sglang:v0.5.19-rocm700-mi35x || true
run_docker build \
  --build-arg "SGLANG_GIT_SHA=${SHA}" \
  -t "${TAG}" \
  -f "${STAGING}/Dockerfile" \
  "${STAGING}"

DIGEST="$(run_docker image inspect "${TAG}" --format '{{.Id}}')"
echo "IMAGE_DIGEST=${DIGEST}"
mkdir -p "$(dirname "${ARCHIVE}")"
run_docker save "${TAG}" | zstd -T0 -o "${ARCHIVE}"
echo "wrote ${ARCHIVE}"
echo "${DIGEST}" > "${ARCHIVE}.digest"
echo "${SHA}" > "${ARCHIVE}.sglang_sha"
echo "STAGING=${STAGING}"
