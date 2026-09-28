# Archived HuggingFace `spec_generate` eval (do not use)

These scripts predate live SGLang `--speculative-algorithm DFLASH` for
`DFlashLinearDraftModel`. They wrap `dflash_linear_serve.py`, a local HTTP
server that drafts with HuggingFace `spec_generate`. That path failed on
Qwen3.5 hybrid `DynamicCache` (`has_previous_state`) and is not the card
protocol.

Use instead:

- [How to run SGLang eval](../../../../docs/linear-context-dflash-sglang-eval.md)
- `scripts/eval/dflash_linear_eval.py` (`prepare`, `mal`, `sglang-mal`, `compare-mal`)
- `cluster-dflash-linear-{sglang-smoke,parity,m1-gates,feature-contract,rebaseline}.sbatch`

Kept here so old sbatch logs and NFS paths still make sense. The mal
launcher also pinned the **0.5.14** image; live eval is 0.5.18.

| File | Was |
|---|---|
| `dflash_linear_serve.py` | HF HTTP serve / ShareGPT `mal` / `eval` / `compare` |
| `inside-dflash-linear-eval.sh` / `run-dflash-linear-eval.sh` / `cluster-dflash-linear-eval.sbatch` / `deploy-eval.sh` | HF serve eval |
| `inside-dflash-linear-mal.sh` / `run-dflash-linear-mal.sh` / `cluster-dflash-linear-mal.sbatch` / `deploy-mal.sh` | HF serve MAL on ShareGPT holdout |
