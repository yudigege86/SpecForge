# Crusoe dflash_linear cluster tests

Launchers for linear-context DFlash on the SPUR MI355X cluster. Data and
results still live under `/shared_nfs/naqin/Linear-Context-DFlash/`; this
directory is the in-repo copy of the scripts.

From a SpecForge checkout on the cluster:

```bash
sbatch scripts/cluster/dflash-linear/cluster-dflash-linear-mal.sbatch
```

Offline MAL (no SGLang, teacher-forced accept length):

```bash
python scripts/eval/dflash_linear_serve.py mal \
  --target Qwen/Qwen3.5-4B \
  --draft /path/to/draft_hf \
  --eval-jsonl /shared_nfs/naqin/primus-specforge-smoke/sharegpt_eval.holdout.jsonl \
  --out mal.json --n 256 --max-new-tokens 64 --no-ignore-eos
```

HTTP serve/eval (vanilla vs spec_generate) is the same module: `serve`, `eval`,
`compare`. Stock SGLang `--speculative-algorithm DFLASH` cannot load
`DFlashLinearDraftModel`.

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
