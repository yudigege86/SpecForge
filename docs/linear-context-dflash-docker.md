# Which Docker image: training vs testing

Two images. Do not mix them. SpecForge is **bind-mounted** at runtime in
both; a checkout move does not need a rebuild.

Eval protocol: [linear-context-dflash-sglang-eval.md](./linear-context-dflash-sglang-eval.md).
Status: [linear-context-dflash-sglang-status.md](./linear-context-dflash-sglang-status.md).

## Training (SpecForge)

Use this for 1-epoch / DFlash2-linear train, naive-steps, layer-test, FLA
scan, and capture-replay of a trained draft.

| | |
|---|---|
| Dockerfile | [scripts/cluster/dflash-linear/Dockerfile.sglang-0.5.18-train](../scripts/cluster/dflash-linear/Dockerfile.sglang-0.5.18-train) |
| Base | `lmsysorg/sglang:v0.5.18-rocm700-mi35x` |
| Tag | `naqin/primus-specforge:v0.5.18-train-rocm700-mi35x` |
| Archive (Crusoe) | `/shared_nfs/naqin/docker-images/primus-specforge-v0.5.18-train-rocm700-mi35x.tar.zst` |
| Baked in | `flash-linear-attention[rocm]`, `tensorboard`, `einops` |
| Not in the image | SpecForge (mount `/workspace/SpecForge`), SGLang `DFlash2DraftModel` |

SGLang 0.5.18 has **no** `DFlash2DraftModel`. DFlash2 / DFlash2-linear
**training** is SpecForge (`DFlash2LinearDraftModel`, strategy
`dflash_linear`). That is why train stays on 0.5.18.

```bash
# rebuild on any ROCm MI355X node
bash scripts/cluster/dflash-linear/build-sglang-0.5.18-train-image.sh
# or
docker build -t naqin/primus-specforge:v0.5.18-train-rocm700-mi35x \
  -f scripts/cluster/dflash-linear/Dockerfile.sglang-0.5.18-train \
  scripts/cluster/dflash-linear
```

Launchers: `cluster-dflash-linear-1epoch.sbatch`,
`cluster-dflash2-linear-1epoch.sbatch`. Override `RUNTIME_IMAGE`,
`SPECFORGE_SRC`, `HIDDEN_STATES_PATH`, and `RESULTS_DIR` on a new cluster.

## Testing / serving (live SGLang DFLASH)

Use this for registry smoke, feature-contract, live MAL, stock rebaseline,
and serving `DFlash2DraftModel` or `DFlashLinearDraftModel`.

| | |
|---|---|
| Dockerfile | [scripts/cluster/dflash-linear/Dockerfile.sglang-0.5.19](../scripts/cluster/dflash-linear/Dockerfile.sglang-0.5.19) |
| Base | `lmsysorg/sglang:v0.5.19-rocm700-mi35x` |
| Tag | `naqin/primus-specforge:v0.5.19-dflash-linear-rocm700-mi35x` |
| Digest | `sha256:673690725b3ebcaa12e1b6f27ea90d0609cca84aad548497e9f777a20a1c32a4` |
| Archive (Crusoe) | `/shared_nfs/naqin/docker-images/primus-specforge-v0.5.19-dflash-linear-rocm700-mi35x.tar.zst` |
| Overlay | linear DFLASH files from `yudigege86/sglang` `dflash-linear-v0.5.19` (`2e3695e59`) + `patches/sglang/v0.5.19/spec-capture.patch` |
| Not overlaid | `models/dflash.py` (`DFlash2DraftModel` comes from the 0.5.19 base) |

`SGLANG_SRC` must be branch `dflash-linear-v0.5.19`. A 0.5.18 checkout
would clobber DFlash2 worker hooks on this image.

```bash
bash scripts/cluster/dflash-linear/build-sglang-0.5.19-image.sh
```

Launchers: `cluster-dflash-linear-sglang-smoke.sbatch`,
`cluster-dflash-linear-feature-contract.sbatch`,
`cluster-dflash-linear-parity.sbatch`.

## Do not use for new work

| Dockerfile / tag | Why it still exists |
|---|---|
| [Dockerfile.sglang-0.5.18](../scripts/cluster/dflash-linear/Dockerfile.sglang-0.5.18) / `naqin/primus-specforge:v0.5.18-dflash-linear-rocm700-mi35x` | M1 eval evidence only |
| `naqin/primus-specforge:v0.5.14-rocm700-mi35x` | Retired. Launchers no longer default here |
| Train image for live `/generate` or `DFlash2DraftModel` | No DFlash2 serving, no linear overlay |
| Eval image for SpecForge 1-epoch train | No FLA bake |

HuggingFace `spec_generate` serve/eval is in
[scripts/cluster/dflash-linear/archive/](../scripts/cluster/dflash-linear/archive/).
