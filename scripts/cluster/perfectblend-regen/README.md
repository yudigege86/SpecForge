# PerfectBlend Qwen3.5-4B regeneration on SPUR

Target-model regeneration of [`mlabonne/open-perfectblend`](https://huggingface.co/datasets/mlabonne/open-perfectblend)
prompts with `Qwen/Qwen3.5-4B` in thinking mode. Original assistant text is
dropped. Prompt plus completion is capped at 4096 tokens.

Paths live under `/shared_nfs/naqin/Linear-Context-DFlash/perfectblend-qwen35-4b/`.
Jobs use account `amd-brain-models`, QOS `amd-burst-qos`, `--requeue`, and
resume by sample id after preemption.

## One-time prepare

From a login node (after `source /etc/profile.d/spur.sh`):

```bash
sbatch scripts/cluster/perfectblend-regen/prepare.sbatch
```

This pulls `lmsysorg/sglang:v0.5.18-rocm720-mi35x`, saves it to
`/shared_nfs/naqin/docker-images/sglang-v0.5.18-rocm720-mi35x.tar.zst`, caches
`Qwen/Qwen3.5-4B`, writes `perfectblend_train.jsonl`, and splits 64 shards plus
`shard-smoke.jsonl`.

## Smoke shard

```bash
bash scripts/cluster/perfectblend-regen/submit-shards.sh smoke
```

Kill and resubmit the same command to confirm `--resume-by-id` appends new ids.

## Full fan-out

```bash
bash scripts/cluster/perfectblend-regen/submit-shards.sh all
```

That submits one 8-GPU job per unfinished shard. Burst QOS currently allows
4 submitted jobs per user; `watch-and-merge.sh` refills as jobs finish,
releases `JobHoldMaxRequeue` holds, and concatenates
`regen/perfectblend_qwen35_4b_regen.jsonl` when every shard has a `.done`
marker.

Sampling used by `inside-regen.sh`: temperature 1.0, top_p 0.95, top_k 20,
min_p 0.0, presence_penalty 1.5, repetition_penalty 1.0, `--reasoning save`.
Each node runs 8 tp=1 SGLang servers under AITER with `--disable-radix-cache`
and `--disable-cuda-graph` (Qwen3.5-4B hybrid decode graph capture aborts on
gfx950). `SGLANG_USE_AITER_UNIFIED_ATTN` is left unset for the same reason.
