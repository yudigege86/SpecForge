# Running SGLang eval for a linear-context DFlash drafter

Live card MAL for `DFlashLinearDraftModel` uses the same SGLang algorithm
name as stock DFlash (`DFLASH`). The linear draft is selected from
`config.json` `architectures[0]`. Metric:

\[
\mathrm{MAL} = \mathrm{completion\_tokens} / \mathrm{spec\_verify\_ct}
\]

averaged per prompt (HumanEval) or per turn (MT-Bench turn 1). Do not use
mean block accept length, HuggingFace `hidden_states` teacher-force, or
SPEED-Bench Qualitative `concat_user` as a z-lab card number.

Companion note: [linear-context-dflash-sglang-status.md](./linear-context-dflash-sglang-status.md).
Cluster launchers: `scripts/cluster/dflash-linear/`.

## What you need

| Piece | Pin |
|---|---|
| SGLang | Fork `yudigege86/sglang` branch `dflash-linear` off `v0.5.18` (validated at `2a73ad467`) |
| Image | `naqin/primus-specforge:v0.5.18-dflash-linear-rocm700-mi35x` (digest `sha256:ce8ff833c63aba22ec9606817926e2f7109c95173f1692fbff7cb4d7429a8a30`) |
| Archive | `/shared_nfs/naqin/docker-images/primus-specforge-v0.5.18-dflash-linear-rocm700-mi35x.tar.zst` |
| SpecForge | This branch (`dflash-linear`), eval CLI `scripts/eval/dflash_linear_eval.py` |
| Target | `Qwen/Qwen3.5-4B` (other targets need a feature-contract check first) |
| Linear draft | HF export whose `architectures` is `["DFlashLinearDraftModel"]` |
| 1-epoch export used for M1 | `/shared_nfs/naqin/Linear-Context-DFlash/eval-1epoch/20260918T213552Z/draft_hf` |
| Training capture (contract) | `/shared_nfs/naqin/primus-specforge-smoke/qwen-capture-40k/valid-links` |

The docker runner bind-mounts seven fork files over the image via
`scripts/cluster/dflash-linear/sglang-overlay-mounts.sh`. Keep
`SGLANG_SRC` pointed at a checkout of the fork (default
`/shared_nfs/naqin/Linear-Context-DFlash/sglang`).

Hard serving limits for this implementation (the server refuses otherwise):

- `--tp-size 1`
- `--disable-cuda-graph` (eager M1)
- `--disable-radix-cache`
- no `--speculative-dflash-draft-window-size`
- GDN only (`variant: kda` is not implemented)
- `model_type` in the draft config must stay `qwen3`

## Card protocol

Match [z-lab/Qwen3.5-4B-DFlash](https://huggingface.co/z-lab/Qwen3.5-4B-DFlash):

- thinking **on**
- `max_new_tokens=4096`, stop at EOS
- `--speculative-num-draft-tokens 16` (block size 16)
- HumanEval from the pinned `openai/human-eval` prepare commit
- MT-Bench **turn 1 only** (`--mt-bench-turns first`)

`dflash_linear_eval.py prepare` pins the upstream dataset SHAs. Do not
re-tokenize a chat-template string; `sglang-mal` dumps `prompt_ids` /
`completion_ids` for replay.

## Cluster (Crusoe SPUR / MI355X)

Account `amd-brain-models`, partition `amd-spur`, QOS `amd-burst-qos`
(submit cap 4). Host OOM killed earlier jobs that used the sbatch defaults
of 16 CPUs and no `--mem`; use **32 CPUs and 128G** for live DFLASH.

From a SpecForge checkout on NFS:

```bash
cd /shared_nfs/naqin/Linear-Context-DFlash/SpecForge
export TARGET_MODEL=Qwen/Qwen3.5-4B
export DRAFT_HF=/shared_nfs/naqin/Linear-Context-DFlash/eval-1epoch/20260918T213552Z/draft_hf
export SGLANG_GIT_SHA=$(git -C /shared_nfs/naqin/Linear-Context-DFlash/sglang rev-parse --short HEAD)
```

### Smoke (one generate)

```bash
sbatch --account=amd-brain-models --qos=amd-burst-qos --partition=amd-spur \
  --nodes=1 --gres=gpu:1 --cpus-per-task=32 --mem=128G --time=02:00:00 \
  --job-name=dflin-smoke \
  scripts/cluster/dflash-linear/cluster-dflash-linear-sglang-smoke.sbatch
```

Pass when `/get_server_info` reports `DFLASH`, the loaded draft is
`DFlashLinearDraftModel`, and one `/generate` returns `spec_verify_ct`.

### Live MAL + aux replay + compare (one scheduler setting)

```bash
export EVAL_DATASET=humaneval
export EVAL_N=16
export MAX_RUNNING_REQUESTS=1
export OVERLAP_SCHEDULE=1
export SGLANG_DFLASH_LINEAR_SHADOW_CHECK=1
export RESULTS_DIR=/shared_nfs/naqin/Linear-Context-DFlash/mal-eval/humaneval-linear-parity
export MAL_GATE=0.03
# optional: compare greedy ids to a stock live dump
export STOCK_MAL_JSON=/shared_nfs/naqin/Linear-Context-DFlash/mal-eval/rebaseline-0.5.18/he16/20260926T022416Z/sglang_mal.json

sbatch --account=amd-brain-models --qos=amd-burst-qos --partition=amd-spur \
  --nodes=1 --gres=gpu:1 --cpus-per-task=32 --mem=128G --time=08:00:00 \
  --job-name=dflin-parity \
  scripts/cluster/dflash-linear/cluster-dflash-linear-parity.sbatch
```

`inside-dflash-linear-parity.sh` does:

1. Launch SGLang (`DFLASH` + linear draft, eager, radix off, TP=1).
2. `sglang-mal` → `sglang_mal.json` (live MAL, dumped ids).
3. Stop the server.
4. `mal --replay-json sglang_mal.json --feature-source sglang` → teacher-force
   on the same ids with live-engine aux.
5. `compare-mal` and a 3% relative-MAL gate plus mixed accept/reject.

Live dumps record mean `spec_accept_length`, not per-block `commit_lens`.
The gate treats MAL in `(1.05, 15.5)` as mixed accept/reject on block 16.

### Full M1 matrix (batch 1/4 × overlap on/off)

```bash
export STOCK_MAL_JSON=/shared_nfs/naqin/Linear-Context-DFlash/mal-eval/rebaseline-0.5.18/he16/20260926T022416Z/sglang_mal.json
export RESULTS_DIR=/shared_nfs/naqin/Linear-Context-DFlash/mal-eval/m1-gates
sbatch --account=amd-brain-models --qos=amd-burst-qos --partition=amd-spur \
  --nodes=1 --gres=gpu:1 --cpus-per-task=32 --mem=128G --time=16:00:00 \
  --job-name=dflin-m1 \
  --wrap='bash /shared_nfs/naqin/Linear-Context-DFlash/SpecForge/scripts/cluster/dflash-linear/run-dflash-linear-m1-gates.sh'
```

Shadow check (`SGLANG_DFLASH_LINEAR_SHADOW_CHECK=1`) replays FLA prefill +
masked naive verify after every commit. It is O(commits²) and makes HE n=16
take on the order of an hour per scheduler setting. Turn it off for
throughput runs; leave it on when checking state ownership.

### Feature contract (training capture vs 0.5.18 live aux)

```bash
export HIDDEN_STATES_PATH=/shared_nfs/naqin/primus-specforge-smoke/qwen-capture-40k/valid-links
export FEATURE_CONTRACT_GATE=0.99
sbatch scripts/cluster/dflash-linear/cluster-dflash-linear-feature-contract.sbatch
```

Pass: mean per-token cosine ≥ 0.99 on n=16 samples. The 40k Qwen3.5-4B
capture on this stack measured mean cosine 0.99983.

### Stock rebaseline (concat-KV DFlash, same image)

```bash
# HumanEval n=16 / 164 with z-lab/Qwen3.5-4B-DFlash
sbatch scripts/cluster/dflash-linear/cluster-dflash-linear-rebaseline.sbatch
```

Use this dump as `STOCK_MAL_JSON` when you want stock-vs-linear
`completion_ids` divergence counts.

## Manual: server + CLI

Works inside the runtime image or any env with the fork installed
(`pip install -e python --no-deps` from `yudigege86/sglang`) and
`patches/sglang/v0.5.18/spec-capture.patch` applied for aux replay
(`scripts/apply_sglang_spec_capture_patch.sh --target v0.5.18`).

```bash
python3 -m sglang.launch_server \
  --model-path Qwen/Qwen3.5-4B \
  --speculative-algorithm DFLASH \
  --speculative-draft-model-path /path/to/draft_hf \
  --speculative-num-draft-tokens 16 \
  --tp-size 1 \
  --dtype bfloat16 \
  --mem-fraction-static 0.92 \
  --trust-remote-code \
  --host 127.0.0.1 \
  --port 30000 \
  --disable-cuda-graph \
  --disable-radix-cache \
  --mamba-scheduler-strategy extra_buffer \
  --max-running-requests 1
```

Optional: `--disable-overlap-schedule` for the overlap-off setting.
Optional: `SGLANG_DFLASH_LINEAR_SHADOW_CHECK=1`.

Prepare JSONL once:

```bash
python scripts/eval/dflash_linear_eval.py prepare \
  --dataset humaneval \
  --out /path/to/humaneval.jsonl

python scripts/eval/dflash_linear_eval.py prepare \
  --dataset mt-bench \
  --out /path/to/mtbench.jsonl
```

Live MAL (server already up):

```bash
python scripts/eval/dflash_linear_eval.py sglang-mal \
  --target Qwen/Qwen3.5-4B \
  --draft /path/to/draft_hf \
  --eval-jsonl /path/to/humaneval.jsonl \
  --base http://127.0.0.1:30000 \
  --out sglang_mal.json \
  --summary sglang_summary.md \
  --n 16 \
  --max-new-tokens 4096 \
  --mt-bench-turns first
```

Teacher-force on the frozen live ids (SGLang aux, not HF hidden states):

```bash
python scripts/eval/dflash_linear_eval.py mal \
  --target Qwen/Qwen3.5-4B \
  --draft /path/to/draft_hf \
  --replay-json sglang_mal.json \
  --feature-source sglang \
  --out replay_mal.json \
  --summary replay_summary.md \
  --max-new-tokens 4096 \
  --mt-bench-turns first
```

Compare:

```bash
python scripts/eval/dflash_linear_eval.py compare-mal \
  --offline replay_mal.json \
  --sglang sglang_mal.json \
  --out compare_parity.json \
  --summary compare_parity.md
```

`compare-mal` fails on incomplete `question_id` overlap unless you pass
`--allow-partial`. Deltas are computed on token-identical rows when both
sides have `completion_ids`; live-vs-replay of the same dump often has no
second copy of ids (`token_identical` null) and then uses all matched rows.

## Environment variables the docker runner forwards

| Variable | Default | Role |
|---|---|---|
| `TARGET_MODEL` | `Qwen/Qwen3.5-4B` | Target weights |
| `DRAFT_HF` | 1-epoch export path above | Linear (or stock) draft |
| `EVAL_DATASET` | `humaneval` | `humaneval` or `mt-bench` |
| `EVAL_N` | unset / 16 in M1 | Prompt cap |
| `MAX_NEW_TOKENS` | 4096 | Card length |
| `BLOCK_SIZE` | 16 | Draft tokens |
| `ENABLE_THINKING` | 1 | Card thinking |
| `MT_BENCH_TURNS` | `first` | Card MT-Bench |
| `MAX_RUNNING_REQUESTS` | 1 | Batch |
| `OVERLAP_SCHEDULE` | 1 | `0` adds `--disable-overlap-schedule` |
| `MEM_FRACTION` | 0.92 | SGLang static pool |
| `MAL_GATE` | 0.03 | Live vs aux relative MAL |
| `SGLANG_DFLASH_LINEAR_SHADOW_CHECK` | 0 (1 in parity) | State replay assert |
| `STOCK_MAL_JSON` | empty | Stock `completion_ids` compare |
| `HIDDEN_STATES_PATH` | 40k capture | Feature-contract-check |
| `SGLANG_GIT_SHA` | empty | Written into `run_record` |
| `SGLANG_SRC` | NFS sglang checkout | Overlay source |

Every `mal` / `sglang-mal` JSON embeds a `run_record` (eval JSONL sha256,
HF revisions, `sglang.__version__`, fork SHA, `IMAGE_DIGEST`,
`/get_server_info` snapshot, `feature_contract.json`).

## Reading results

| File | What to look at |
|---|---|
| `sglang_mal.json` | `spec_accept_length_mean`, `finished_on_eos`, `raw[].completion_ids` |
| `replay_mal.json` | Teacher-force MAL on the same ids |
| `compare_parity.json` | `delta_mean`, `abs_delta_mean` |
| `parity_verdict.json` | `rel_abs_delta` ≤ 0.03, `mixed_accept_reject` |
| `sglang-server.log` | `DFlashLinearDraftModel`, overlay, shadow failures (root-owned; copy with `dd`/`cp` before `scp`) |

M1 bar: live MAL within ~3% of SGLang-aux replay, mixed accept/reject,
shadow check clean, stock-vs-linear id divergence **reported** (not a hard
fail). On the 1-epoch draft, HE n=16 was live **2.459** vs replay **2.448**
(0.42%) at batch 1/4 and overlap on/off.

A 1-epoch linear MAL near 2.5 is the drafter, not a broken server. Stock
concat-KV on the same 0.5.18 image is about 7.9 on HE n=16.

## Common failures

- **Shadow `max_abs` against a full-prefix FLA rescan.** Serving mixes FLA
  prefill with a naive masked verify commit. The check must replay that mix
  (fork ≥ `2a73ad467`). Do not assert bit-identity of FP32 incremental state
  vs one FLA over the concatenated prefix.
- **`ModelRunner.decode_cuda_graph_runner` missing.** Linear still calls
  `_draft_worker.init_cuda_graphs(capture_decode_cuda_graph=False)`.
- **`expected BFloat16 found Float` on context-read.** State is FP32; the
  draft forward casts to the query dtype before the einsum.
- **Host SIGKILL / GPU OOM.** 32 CPUs, 128G RAM, `MEM_FRACTION=0.92`, do
  not pin `HIP_VISIBLE_DEVICES=0` when Slurm already set `SLURM_JOB_GPUS`.
- **`QOSMaxSubmitJobPerUserLimit`.** Burst cap is 4 including running jobs.
- **Login node `Remote commands are not allowed`.** Pipe a script into an
  interactive SSH session; do not `ssh host bash script`.
- **NFS hang on `sglang-server.log`.** The file is often root-owned inside
  docker. Copy it to a new name, then read.
