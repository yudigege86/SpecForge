# Linear-context DFlash in SGLang: status and future work

How to run eval: [linear-context-dflash-sglang-eval.md](./linear-context-dflash-sglang-eval.md).
Configs (default, no-ctx-residual, qkv): [linear-context-dflash-variants.md](./linear-context-dflash-variants.md).
Runtime image: [Dockerfile.sglang-0.5.19](../scripts/cluster/dflash-linear/Dockerfile.sglang-0.5.19).
Train image: [Dockerfile.sglang-0.5.18-train](../scripts/cluster/dflash-linear/Dockerfile.sglang-0.5.18-train).

Serving lives in `yudigege86/sglang` branch `dflash-linear-v0.5.19` (off v0.5.19).
The validated 0.5.18 history stays on `dflash-linear` (`2a73ad467`). Eval lives
in this SpecForge branch. Same algorithm name `DFLASH`; the linear draft is
dispatched when `architectures[0] == "DFlashLinearDraftModel"`. Stock DFlash2
(`DFlash2DraftModel`) stays on `DFlashWorkerV2`.

## Current status (M1 landed)

GDN-only eager serving is in. On the 1-epoch Qwen3.5-4B linear export,
HumanEval n=16 live MAL matches SGLang-aux teacher-force within 0.42% at
batch 1 and 4, overlap on and off. Absolute linear MAL is comparable to the
training capture (feature-contract cosine 0.9998). It is **not** a card
number: the 1-epoch drafter sits near MAL 2.5 vs stock concat-KV ~7.9.

| Milestone | State | Evidence |
|---|---|---|
| M-1 benchmark contract | Landed | `run_record`, pinned prepare SHAs, strict `compare-mal`, `feature_contract.json`, `feature-contract-check` |
| M0 load | Landed | Server starts, `/get_server_info` = `DFLASH`, one `/generate` returns `spec_verify_ct` |
| M1 live vs aux parity | Landed | Job 175655, `mal-eval/m1-gates/20260926T034914Z`, fork `2a73ad467` |
| M2 HE 164 + MT-Bench t1 stock vs linear | Not started | Same live protocol, first publishable relative live MAL |
| M3 graphs / fused commit / TP / radix | Not started | See below |
| M4 FLA kernel tolerance on gfx950 | Outlined | Serving FP32 vs training BF16 FLA windows |
| M5 other targets | Outlined | Per-target feature contract first |

### M1 numbers (1-epoch export, HE n=16, thinking on, 4096, block 16)

All four scheduler settings agreed:

- live MAL **2.459**
- SGLang-aux replay **2.448** (rel **0.42%**, gate 3%)
- 16/16 EOS, mixed accept/reject (MAL ~2.5 on block 16)
- shadow check held after mixed-path replay
- vs stock `z-lab/Qwen3.5-4B-DFlash` on the same prompts: **6/16**
  `completion_ids` identical, **10/16** diverge; first difference never in
  the first 8 tokens (range 90–459). Late numeric drift, not a first-token
  verify-path failure.

Stock rebaseline on the same 0.5.18 image: HE n=16 live **7.879** / aux
**7.527**; HE n=164 live **8.019** / aux **7.767**. Feature contract vs the
40k training capture: mean cosine **0.99983** (min 0.99951, gate 0.99).

### What is implemented

```
target extend (capture aux)
  -> commit_prefill: varlen GDN scan into ctx_state[req_pool_idx]
  -> draft forward: gather state, block SDPA, absolute RoPE
  -> target verify B tokens (unchanged stock path)
  -> accept: commit_lens = accept+1 (unchanged)
  -> commit_verify: identity-masked B-step recurrence
```

Invariant: state before a draft round holds features of `[0, prefix_len)`.
Verify commits bonus + accepted tokens. The correction token is the next
bonus and is **not** folded in yet.

SGLang v0.5.19 files (fork `dflash-linear-v0.5.19`):

| File | Role |
|---|---|
| `python/sglang/srt/models/dflash_linear.py` | `DFlashLinearDraftModel`, weight names 1:1 with SpecForge |
| `python/sglang/srt/speculative/dflash_linear_state.py` | GDN step, masked block commit, FLA varlen wrapper |
| `python/sglang/srt/speculative/dflash_linear_worker_v2.py` | State pool, ownership, prefill/verify commits, shadow check |
| `python/sglang/srt/speculative/dflash_worker_v2.py` | 0.5.19 stock worker (DFlash2 selector / quantized lm_head / TP-sync); linear commit hooks extracted onto this file, never copied from 0.5.18 |
| `python/sglang/srt/models/dflash.py` | Unmodified 0.5.19 `EntryClass` includes `DFlash2DraftModel`; do not overlay |
| `python/sglang/srt/speculative/spec_info.py` | `create_worker` dispatch: linear → `DFlashLinearWorkerV2`, else stock `DFlashWorkerV2` |
| `python/sglang/srt/speculative/dflash_utils.py` | Keep 0.5.19 helpers (`is_dense_head_weight`); re-add `is_dflash_linear_*` |
| `python/sglang/srt/arg_groups/speculative_hook.py` | Linear refuses draft-window, TP>1, radix |

SpecForge:

| File | Role |
|---|---|
| `scripts/eval/dflash_linear_eval.py` | `prepare`, `mal`, `sglang-mal`, `compare-mal`, `feature-contract-check` |
| [Dockerfile.sglang-0.5.19](../scripts/cluster/dflash-linear/Dockerfile.sglang-0.5.19) | Overlay fork files + v0.5.19 spec-capture patch onto `lmsysorg/sglang:v0.5.19-rocm700-mi35x` |
| [Dockerfile.sglang-0.5.18](../scripts/cluster/dflash-linear/Dockerfile.sglang-0.5.18) | Previous eval overlay (M1 evidence); leave in place |
| [Dockerfile.sglang-0.5.18-train](../scripts/cluster/dflash-linear/Dockerfile.sglang-0.5.18-train) | 0.5.18 base + FLA/tensorboard; tag `naqin/primus-specforge:v0.5.18-train-rocm700-mi35x` |
| [build-sglang-0.5.19-image.sh](../scripts/cluster/dflash-linear/build-sglang-0.5.19-image.sh) | Stages the Dockerfile and tags `naqin/primus-specforge:v0.5.19-dflash-linear-rocm700-mi35x` |
| `scripts/cluster/dflash-linear/` | Docker/sbatch launchers, overlay mounts |
| `scripts/cluster/dflash-linear/archive/` | Pre-SGLang HuggingFace `spec_generate` serve/eval (do not use) |
| `specforge/modeling/draft/dflash_linear.py` | Training / teacher-force draft |
| `specforge/modeling/draft/linear_context.py` | `commit_block_masked` reference |
| `tests/test_modeling/test_linear_context_serving_state.py` | Incremental commit == full scan; == `_teacher_force_block` |

Design choices that must not be “fixed”:

- Absolute block RoPE `prefix_len + [0..B-1]`, not rebased `0..B-1`.
- Serving state is FP32; training FLA windows stay BF16. Parity is a
  tolerance, not bit-identity.
- Draft `model_type` stays `qwen3` so hybrid GDN config does not treat it
  as Mamba. `linear_context.backend` is ignored at serve time.
- Capture is DFLASH aux (`target_layer_ids` `[1,8,15,22,29]` on Qwen3.5-4B,
  hybrid marks layer \(k\) with no +1). HF `hidden_states` are a different
  ABI.

State size for this config (D=5, H=8, K=V=64): 163,840 elements / request
(640 KiB FP32). Stock draft KV is 20,480 B/token BF16, so memory breaks even
near 16 BF16 or 32 FP32 tokens of context.

## Known limitations

- Eager only. CUDA-graph capture of the draft forward is skipped on purpose.
- TP=1, radix cache off, no compact draft-window KV.
- Verify commit is an eager torch loop over B steps with
  `beta=0, log_decay=0` as identity for uncommitted positions.
- Shadow check replays the **serving** mix (FLA prefill + naive verify). A
  full-prefix FLA rescan diverges (~0.10 on HE prompts) and is not a bug
  signal.
- Live HTTP dumps have mean MAL, not a `commit_lens` histogram.
- Draft KV pool is still allocated even though linear does not use concat
  KV (`_resolve_dflash_draft_cell_size` still stock-sized).
- Training `max_length: 2048` makes long-\(L\) eval out of distribution.
- 10/16 stock-vs-linear completions diverge late on thinking traces. Report
  it; do not treat it as a failed verify path unless the first tokens differ.
- M1 was one 1-epoch checkpoint. Retrain or recapture the 40k set only if
  feature-contract-check fails for a new target or the 0.5.19 image.

## Future work

### M2 — publishable relative live MAL

Same card protocol, stock vs linear, with M-1 `run_record`s:

- HumanEval 164
- MT-Bench turn 1 (80)

Keep shadow check optional (too slow for 164). Compare live linear MAL to
live stock **and** to SGLang-aux replay of the linear dump. Do not mix HF
hidden-state MAL into that table.

### M3 — serving performance (after M2)

- CUDA-graph the draft forward (state gather is graph-safe).
- Fused verify-commit kernel (replace the B-step Python loop).
- Shrink the unused draft KV pool; fix `_resolve_dflash_draft_cell_size`.
- TP-shard state heads.
- KDA variant.
- Radix-cache via Mamba-style copy-on-write of GDN state.
- Latency vs context length \(L\). Needs long-context training data before
  the sweep is in-distribution.

### M4 — kernel numerics

Match training FLA math on gfx950 within an explicit BF16 tolerance, plus
a short HE smoke for acceptance-level parity. Incremental FP32 serving vs
chunked BF16 training will not be bit-identical.

### M5 — new targets

For each target: n=16 live stock vs SGLang-aux vs live linear, under that
target’s recorded feature contract (layer ids, pre/post-norm, dtype,
fusion). Do not reuse the Qwen3.5-4B `target_layer_ids` blindly.

### Training (parallel, not blocked on M3)

Keep using frozen live ids + SGLang-aux teacher-force as the checkpoint
metric, labeled as teacher-forced, until a new export re-runs M1. Do not
hold training for CUDA graphs or the \(L\)-sweep.

## What not to claim yet

- Card accept length for linear DFlash (need M2 + a trained-enough draft).
- Linear vs stock latency (need M3 graphs / fused commit).
- Bit-identical greedy vs stock (M1 showed late drift on 10/16 HE prompts).
- Interchangeability of HF `hidden_states` and SGLang DFLASH aux.
- Qualitative `concat_user` MAL as a z-lab number.
