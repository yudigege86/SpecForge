# Linear-Context DFlash: evaluation status and SGLang plan

This note records what the Qwen3.5-4B evaluation campaign actually measured,
what is still wrong, and how to put `DFlashLinearDraftModel` into SGLang so
future trained drafters can be scored the same way as stock DFlash.

Treat **SGLang DFLASH aux as a versioned model interface**, not an
implementation detail. A drafter trained or scored on one engine's hidden
states is silently tied to that engine's capture ABI.

Setup: SpecForge fork `yudigege86/SpecForge` branch `dflash-linear`, cluster
image `naqin/primus-specforge:v0.5.14-rocm700-mi35x` (SGLang 0.5.14, MI355X /
gfx950), target `Qwen/Qwen3.5-4B`, stock draft `z-lab/Qwen3.5-4B-DFlash`,
one-epoch linear export under
`/shared_nfs/naqin/Linear-Context-DFlash/eval-1epoch/.../draft_hf`.
**Target runtime is SGLang 0.5.18** (what SpecForge pins). Rebuild the cluster
image rather than keep 0.5.14 compatibility patches, unless a 0.5.18 ROCm
MI355X image fails validation.

## 1. Goal

MAL (mean accept length) is the main metric. The intended use is:

- train stock DFlash and DFlash-linear drafters for several target models;
- compare them fairly;
- eventually report card-style accept length **and** latency vs context
  length \(L\) (the reason linear context exists).

Card MAL is live SGLang greedy decoding:
\(\mathrm{MAL} = \mathrm{completion\_tokens} / \mathrm{spec\_verify\_ct}\),
averaged per turn. z-lab quotes HumanEval **7.719** and MT-Bench **5.933**
at `block_size=16`.

## 2. Harness that exists today

Reusable entry point: `SpecForge/scripts/eval/dflash_linear_eval.py`.

| Command | Role |
|---|---|
| `prepare` / `prepare-speedbench` | Materialize Qualitative / HumanEval / MT-Bench JSONL; refuse leftover SPEED-Bench placeholders |
| `mal` | Offline teacher-forced MAL on any `DFlashDraftModel` or `DFlashLinearDraftModel` |
| `sglang-mal` | Live stock SGLang `--speculative-algorithm DFLASH` |
| `compare-mal` | Per-prompt / per-category offline vs live |

Cluster launchers: `scripts/cluster/dflash-linear/cluster-dflash-linear-speedbench-{mal,sglang}.sbatch` (burst QOS, 1 GPU).

Offline `mal` supports:

- `--replay-json` — freeze prompt/completion ids from a live dump;
- `--feature-offset auto\|0\|1` — HF `hidden_states` index (`auto` is 1);
- `--feature-source hf\|sglang` — HF hidden states, or SGLang DFLASH aux captured on the frozen ids then injected into `acceptance_along_sequence`.

Direct offline scoring (no replay) must use `render_prompt_ids()`
(`apply_chat_template(..., tokenize=True)`). Do not render a chat-template
string and re-tokenize it: the tokenizer's default `add_special_tokens=True`
can change the prompt. Replay of dumped ids is still the safer comparison
path.

Stock and linear share `acceptance_along_sequence`. Linear only overrides
`_teacher_force_block` (block-only RoPE + `anchor_positions=start`). Mask
protocol: only `block_ids[:, 0]` is the current token; the rest of the block
is the mask token (fix `a124734`; pre-fix ShareGPT MAL was 1.008).

## 3. Locked card protocol vs custom protocols

After matching [z-lab/Qwen3.5-4B-DFlash](https://huggingface.co/z-lab/Qwen3.5-4B-DFlash#accept-length):

- thinking **on** (Qwen chat default);
- `max_new_tokens=4096`, stop at EOS;
- `block_size=16`;
- HumanEval wrap from z-lab `dflash/benchmark.py`;
- MT-Bench **turn 1 only**;
- MAL = completion / verify count, not mean block length.

SPEED-Bench Qualitative uses protocol `concat_user`: every user turn is
concatenated into one prompt with **no** generated assistant replies between
turns. z-lab's loop generates, then appends the assistant response, then
continues. Qualitative numbers in this note are a **custom protocol**, not a
card match, unless that loop is implemented. Keep Qualitative for category
mix; use HumanEval + MT-Bench turn 1 for accept-length claims.

MT-Bench `mt_bench_turns=all` inserts prior assistant replies and is a
different protocol from the card.

## 4. Results (Qwen3.5-4B, card protocol unless noted)

### 4.1 Stock DFlash (`z-lab/Qwen3.5-4B-DFlash`)

| Scorer | HumanEval n=164 | MT-Bench n=80 t1 |
|---|---:|---:|
| HF hidden states, offset 1 | 3.36 | 2.90 |
| HF hidden states, offset 0 | 3.18 | GPU fault at 48/80 |
| Frozen SGLang ids + HF features | 3.32 | 2.91 |
| Frozen SGLang ids + **SGLang aux** | **7.275** | **5.525** |
| Live SGLang DFLASH | **7.508** | **5.668** |
| z-lab card | 7.72 | 5.93 |

n=16 SGLang-aux smoke was 7.32 vs live 7.51 before the full 164.

Qualitative n=880 (thinking off, 512, **concat_user**): offline HF 2.38,
live SGLang 3.99. Custom protocol, not card.

Thinking-off 512 HE/MT was even more misleading (HE 4.98 offline / 11.61
live) because it was not the card setting.

### 4.2 Linear GDN (1-epoch export)

No live SGLang path. Only HF-feature offline MAL: Qualitative 880 **1.51**.
That number uses the scorer that undercounted stock by ~2×. It is **not** a
real accept length and must not be compared to the card or to live stock.

ShareGPT holdout after the mask-leak fix: teacher-forced MAL **2.26**
(training EAL ~2.10 at step 620). Same caveat: HF features.

### 4.3 What closed the stock gap

1. Trajectory: dump live `prompt_ids` / `completion_ids`, replay through
   teacher-force. Completions were similar length; verify counts were not.
   Gap survived, so it was not HF `generate` vs SGLang generate.
2. Layer index: dense Qwen3 SGLang capture marks layer \(k+1\); Qwen3.5
   hybrid marks layer \(k\). Offset 0 (hybrid-shaped) **did not** close MAL
   (3.18 vs 7.51). Default offset stays **1**.
3. Features: HF `hidden_states` vs SGLang aux cosine was only ~0.73–0.86,
   and that was enough to halve MAL. Injecting SGLang aux into the HF draft
   recovered live MAL to within ~0.23 (HE) / ~0.14 (MT).

So: **same ids + same SGLang aux + HF draft ≈ live stock MAL** on this
target. The leftover 0.23 / 0.14 is a **remaining implementation
difference, possibly fused versus concat KV**. That is a hypothesis, not
an established cause. Other candidates: dtype, projection ordering,
normalization, verifier accounting.

## 5. Reliability verdict

| Question | Reliable? |
|---|---|
| Absolute MAL for **stock** DFlash on Qwen3.5-4B | Yes: live `sglang-mal`. SGLang-aux replay is a ~3% offline cross-check on this target. |
| Ranking two drafters on the same frozen ids and SGLang aux | Ranks **teacher-forced draft quality** on that trajectory. Does **not** guarantee live ranking until linear serving validates state lifecycle, kernel numerics, and block execution. |
| Default `--feature-source hf` as a published MAL | No. ~2× undercount on this target. Cosine vs SGLang aux does not certify the scorer. |
| Absolute MAL for **linear** | No live number. SGLang-aux teacher-force is the best proxy; it is not served accept length. |
| New target models | Not proven. Capture layer map, hybrid vs dense +1, ROCm Mamba radix / `extra_buffer`, and mem-fraction all broke on Qwen3.5 until patched. |
| Latency, draft-state bytes, tokens/s vs \(L\) | Not measured. Teacher-force cannot produce them. |

## 6. Current issues

**Linear cannot be served.** Stock SGLang `DFLASH` loads `DFlashDraftModel`
and concat-KV. `DFlashLinearDraftModel` is rejected. SpecForge
`dflash_linear_serve.py` (`spec_generate`) failed on Qwen3.5 hybrid
`DynamicCache` (`has_previous_state`). The serving engine that already runs
this target is SGLang, not HuggingFace.

**Offline capture is version-fragile.** SpecForge’s `offline_capture` stack
is written for SGLang 0.5.18. The cluster image is 0.5.14. Getting
`--feature-source sglang` running required 0.5.14 fallbacks,
`disable_radix_cache=True` (Mamba `extra_buffer` asserts CUDA/FLA on ROCm),
and `mem_fraction_static=0.85`. Prefer rebuilding on 0.5.18 and dropping
the shims.

**HF and SGLang aux are not interchangeable** on Qwen3.5 even at offset 1.
`target_layer_ids` is not a complete feature contract (see §8.1).

**One-epoch linear has no calibrated MAL.** Qualitative 1.51 and ShareGPT
2.26 are HF-feature scores. Re-score the export with SGLang-aux replay
before comparing to stock.

**Paper metrics beyond MAL are blocked** until linear runs in a real decode
loop: draft-cache bytes, draft/verify latency, end-to-end throughput vs
\(L\).

**Benchmark artifacts are under-specified.** Prepared JSONL, model
revisions, and image digest are not pinned in the run record. Direct
offline used to re-tokenize chat text; replay dumps are the comparison
standard.

## 7. Interim eval protocol (until linear is in SGLang)

Do not wait on the SGLang port to train. After **M-1** artifacts exist,
for each target that has been calibrated like Qwen3.5:

1. If a stock DFlash exists, run live `sglang-mal` on HumanEval and MT-Bench
   turn 1. Dump `prompt_ids` / `completion_ids`.
2. Score every trained draft (stock and linear) with
   `mal --replay-json … --feature-source sglang`.
3. Report those numbers as **teacher-forced MAL on frozen live trajectories**,
   not as live ranking and not as a card.
4. Interpret per-question MAL deltas only when greedy token ids are
   identical and every compared prompt is present (complete overlap).
5. On every **new** target, n=16 HumanEval smoke: live stock (if any) vs
   SGLang-aux vs HF-aux. Adopt the target only if SGLang-aux is within a few
   percent of live. If only HF-aux is close, do not trust that target yet.

Linear already consumes injected `target_hidden`, so step 2 does not need
new draft code.

## 8. High-level plan: linear-context DFlash in SGLang

Follow the DFlash2 pattern: **same serving algorithm `DFLASH`**, different
draft architecture. Reuse target capture, verify, and
`set_dflash_layers_to_capture`. Branch only the draft worker.

### 8.1 Feature contract (versioned aux ABI)

Persist this with every draft checkpoint and every MAL report. Layer ids
alone are not enough:

- capture implementation and version (SGLang git / SpecForge capture patch);
- pre- vs post-layer and pre- vs post-norm location;
- layer-index convention (embeddings at 0? dense `k+1` vs hybrid `k`);
- dtype and quantization;
- fusion ordering (concat then fc+RMSNorm, scale, etc.).

Training, teacher-force, and serving must name the same contract. A silent
engine change is how HF vs SGLang halved MAL.

### 8.2 Design and state lifecycle

Stock DFLASH today:

1. Target prefill/decode captures aux at `target_layer_ids`.
2. `project_target_hidden` (fc + RMSNorm) then `k/v_proj` + RoPE into a
   growing draft KV (fused KV).
3. Dense (or sliding) attention over context KV plus the B-token block.
4. Verify accepted prefix; append those tokens’ features to draft KV.

Linear should keep (1) and the **accepted-token** update rule, and replace
(2)–(3) with GDN/KDA prefix state plus dense B-token attention.
Block-only RoPE / `position_ids` of length \(B\), not
`arange(start+block_size)`.

**Invariant:** persistent state *before* a draft round contains target
features strictly **before** the current anchor. Verification appends the
old anchor plus accepted draft tokens. The newly sampled correction token
remains the next explicit anchor and is **not** folded into persistent
state until the following verify.

The port must handle, explicitly:

- state allocation per request and per draft layer;
- prompt prefill scan into GDN/KDA;
- tentative block state (must not commit on reject);
- commit at the accepted-prefix index;
- rejection rollback;
- batch compaction / reordering;
- preemption and request migration;
- prefix-cache cloning, or an explicit “incompatible with radix prefix
  cache” flag;
- CUDA-graph shape constraints (fixed block size \(B\), fixed state
  tensors).

Teacher-force never exercises rollback, compaction, preemption, or cache
clone. Live ranking can invert vs frozen-id ranking until those paths are
correct.

Do not invent a second speculative algorithm name unless SGLang’s loader
cannot dispatch on `architectures`.

### 8.3 Version pin

**Prefer SGLang 0.5.18.** SpecForge is written for it. Rebuild
`naqin/primus-specforge` on `lmsysorg/sglang:v0.5.18-rocm700-mi35x` (or the
validated gfx950 equivalent), drop the 0.5.14 shims, and pin the image
digest in M-1. Keep 0.5.14 only until that image is proven on this
cluster. Do not implement the draft worker against a third version.

### 8.4 Milestones

**M-1 — benchmark contract (before any claimed number, including M0).**

- Pin dataset revisions and the remote SPEED-Bench / HumanEval / MT-Bench
  prepare-script commits (do not float `prepare.py` from `HEAD`).
- Save SHA256 of every prepared JSONL used in a report.
- Save target, draft, and tokenizer revisions plus serving/capture image
  digest.
- Store prompt and completion token ids with every live or replay run.
- `compare-mal` requires complete prompt / `question_id` overlap.
- Require token-identical greedy trajectories before interpreting
  per-question MAL deltas.
- Record the §8.1 feature contract next to the checkpoint.

**M0 — load and refuse-nothing.** SGLang draft loader accepts
`DFlashLinearDraftModel` / `linear_context` in `dflash_config` without
rewriting the class to `DFlashDraftModel`. Smoke: server starts, `/get_server_info`
still reports `DFLASH`.

**M1 — greedy MAL parity (correctness), including state lifecycle.** Serve
the 1-epoch linear (or a tiny overfit) greedy. Run `sglang-mal` HE n=16.
Compare to the same weights scored with `--feature-source sglang`
teacher-force **and** exercise reject/commit (not only all-accept
smokes). If live and teacher-force disagree, serving kernels or state
lifecycle are wrong — fix before any latency work.

**M2 — stock vs linear on the same live protocol.** HumanEval + MT-Bench
turn 1, thinking on, 4096, block 16. Stock stays the current DFLASH worker;
linear uses the new branch. Same target, same prompts, M-1 artifacts.
This is the first publishable **relative live** MAL. Frozen-id ranking is
not a substitute.

**M3 — long-context efficiency.** Sweep context length \(L\): accept
length, draft-state bytes, draft time, verify time, end-to-end tokens/s.
Confirm state size is \(O(1)\) in \(L\) and that rejected tokens do not
commit GDN/KDA updates.

**M4 — kernels on gfx950.** Training uses FLA `[rocm]` inside SpecForge.
Serving must match that math within an explicit BF16 tolerance, and
**acceptance-level parity** on a short HE smoke (naive vs FLA). Do not
require bit-identical tensors.

**M5 — new targets.** For each target: n=16 live stock vs SGLang-aux vs
live linear, under that target’s recorded feature contract. Do not assume
Qwen3.5’s layer map.

### 8.5 What not to do

- Do not extend HuggingFace `spec_generate` as the long-term eval server
  for hybrid Mamba+attention targets.
- Do not change stock DFlash except where the draft worker must dispatch
  (shared capture / verify).
- Do not treat fused-KV vs GDN as a training-feature change; capture stays
  DFLASH aux under the same contract.
- Do not block the next training runs on M0–M2. Keep using frozen ids +
  SGLang-aux as the checkpoint metric, labeled as teacher-forced.
- Do not publish Qualitative `concat_user` MAL as a z-lab card number.

### 8.6 Success criteria

Linear-in-SGLang is done when:

1. `sglang-mal` runs for `DFlashLinearDraftModel` with the same CLI as stock;
2. greedy HE n=16 live MAL is within a few percent of SGLang-aux teacher-force
   for that checkpoint, including mixed accept/reject;
3. a latency-vs-\(L\) sweep exists for stock concat-KV vs linear GDN on one
   target.

Until (1)–(2), published MAL for linear remains teacher-forced and labeled
as such. Sequencing stays: **calibrated replay now, live MAL parity before
latency work, then the \(L\)-sweep.**

## 9. Pointers

| Item | Location |
|---|---|
| Eval CLI | `SpecForge/scripts/eval/dflash_linear_eval.py` |
| Cluster MAL / SGLang | `SpecForge/scripts/cluster/dflash-linear/` |
| Stock teacher-force / aux inject | `specforge/modeling/draft/dflash.py` (`acceptance_along_sequence`) |
| Linear block hook | `specforge/modeling/draft/dflash_linear.py` |
| HE live dump | `/shared_nfs/naqin/Linear-Context-DFlash/mal-eval/humaneval-sglang/20260924T175823Z/sglang_mal.json` |
| HE SGLang-aux result | `.../mal-eval/humaneval-sglang-aux/20260924T213555Z/` |
| MT SGLang-aux result | `.../mal-eval/mt-bench-sglang-aux/20260924T213556Z/` |
| Architecture note | `main.tex` |
