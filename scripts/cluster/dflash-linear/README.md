# Crusoe dflash_linear cluster tests

Launchers for linear-context DFlash on the SPUR MI355X cluster. Data and
results still live under `/shared_nfs/naqin/Linear-Context-DFlash/`.

**Docs**

- [How to run SGLang eval](../../../docs/linear-context-dflash-sglang-eval.md)
- [Implementation status and future work](../../../docs/linear-context-dflash-sglang-status.md)
- [Dockerfile.sglang-0.5.18](./Dockerfile.sglang-0.5.18) — eval overlay on `lmsysorg/sglang:v0.5.18-rocm700-mi35x`
- [Dockerfile.sglang-0.5.18-train](./Dockerfile.sglang-0.5.18-train) — same base plus FLA/tensorboard for offline 1-epoch train (`naqin/primus-specforge:v0.5.18-train-rocm700-mi35x`)

Live card MAL is SGLang `--speculative-algorithm DFLASH` plus
`scripts/eval/dflash_linear_eval.py` (`sglang-mal`, then `mal --replay-json`
with `--feature-source sglang`). Do not use HuggingFace `spec_generate`; those launchers are in
`scripts/cluster/dflash-linear/archive/`.

From a SpecForge checkout on the cluster (32 CPUs / 128G RAM for live
DFLASH; burst QOS submit cap is 4):

```bash
export TARGET_MODEL=Qwen/Qwen3.5-4B
export DRAFT_HF=/shared_nfs/naqin/Linear-Context-DFlash/eval-1epoch/20260918T213552Z/draft_hf

sbatch --account=amd-brain-models --qos=amd-burst-qos --partition=amd-spur \
  --nodes=1 --gres=gpu:1 --cpus-per-task=32 --mem=128G --time=02:00:00 \
  scripts/cluster/dflash-linear/cluster-dflash-linear-sglang-smoke.sbatch
```

Offline teacher-forced MAL on frozen live ids (same CLI for stock or linear):

```bash
python scripts/eval/dflash_linear_eval.py mal \
  --target Qwen/Qwen3.5-4B \
  --draft /path/to/draft_hf \
  --replay-json /path/to/sglang_mal.json \
  --feature-source sglang \
  --out replay_mal.json
```

Capture replay (training forward on SGLang `.ckpt` features, no HF hidden
states):

```bash
sbatch scripts/cluster/dflash-linear/cluster-dflash-linear-replay.sbatch
```

```bash
python scripts/eval/dflash_linear_capture_replay.py \
  --draft /path/to/draft_hf \
  --hidden-states-path /shared_nfs/naqin/primus-specforge-smoke/qwen-capture-40k/valid-links \
  --out replay.json --n 32
```

Feature A/B (same `input_ids`, capture vs HuggingFace `hidden_states`):

```bash
sbatch scripts/cluster/dflash-linear/cluster-dflash-linear-feature-ab.sbatch
```

SPEED-Bench Qualitative MAL (any target + stock DFlash or dflash_linear; stop at EOS; per-category MAL). Qualitative uses protocol `concat_user` and is **not** a z-lab card number.

```bash
python scripts/eval/dflash_linear_eval.py prepare-speedbench \
  --out /shared_nfs/naqin/Linear-Context-DFlash/speedbench/qualitative.jsonl

python scripts/eval/dflash_linear_eval.py prepare \
  --dataset humaneval \
  --out /shared_nfs/naqin/Linear-Context-DFlash/speedbench/humaneval.jsonl

python scripts/eval/dflash_linear_eval.py prepare \
  --dataset mt-bench \
  --out /shared_nfs/naqin/Linear-Context-DFlash/speedbench/mtbench.jsonl

EVAL_DATASET=humaneval \
TARGET_MODEL=Qwen/Qwen3.5-4B \
DRAFT_HF=z-lab/Qwen3.5-4B-DFlash \
sbatch scripts/cluster/dflash-linear/cluster-dflash-linear-speedbench-mal.sbatch
```

Same split through live SGLang DFLASH, then compare:

```bash
python scripts/eval/dflash_linear_eval.py sglang-mal \
  --target Qwen/Qwen3.5-4B \
  --draft z-lab/Qwen3.5-4B-DFlash \
  --eval-jsonl /shared_nfs/naqin/Linear-Context-DFlash/speedbench/humaneval.jsonl \
  --base http://127.0.0.1:30000 \
  --out sglang_mal.json

COMPARE_JSON=/path/to/speedbench_mal.json \
sbatch scripts/cluster/dflash-linear/cluster-dflash-linear-speedbench-sglang.sbatch
```

0.5.18 image, M0 smoke, feature-contract check, stock rebaseline, linear parity:

```bash
sbatch scripts/cluster/dflash-linear/cluster-dflash-linear-image.sbatch
sbatch scripts/cluster/dflash-linear/cluster-dflash-linear-sglang-smoke.sbatch
sbatch scripts/cluster/dflash-linear/cluster-dflash-linear-feature-contract.sbatch
sbatch scripts/cluster/dflash-linear/cluster-dflash-linear-rebaseline.sbatch
sbatch scripts/cluster/dflash-linear/cluster-dflash-linear-parity.sbatch
sbatch scripts/cluster/dflash-linear/cluster-dflash-linear-m1-gates.sbatch
```

0.5.18 train image + 1-epoch redo on the existing 40k capture (no recapture).
Image build is 1 GPU burst; 8-GPU 1-epoch train also uses `amd-burst-qos`:

```bash
IMG=$(sbatch --parsable scripts/cluster/dflash-linear/cluster-dflash-linear-train-image.sbatch)
sbatch --dependency=afterok:${IMG} scripts/cluster/dflash-linear/cluster-dflash-linear-1epoch.sbatch
```

Training-era launchers (`cluster-dflash-linear-1epoch.sbatch`, naive-steps,
layer-test, FLA scan, yaml configs) stay in this directory. They are not
the live MAL path.
