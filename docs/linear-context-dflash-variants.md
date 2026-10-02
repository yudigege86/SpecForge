# Linear-context DFlash: configs and algorithms

Every checked-in linear recipe uses **training strategy** `dflash_linear`.
The default draft class is `DFlashLinearDraftModel`. DFlash2-linear is the
same strategy with local convolution and a candidate selector
(`DFlash2LinearDraftModel`). Stock concat-KV DFlash (`training.strategy:
dflash`, `DFlashDraftModel`) is a different algorithm and is not listed here.

Eval and serving: [linear-context-dflash-sglang-eval.md](./linear-context-dflash-sglang-eval.md),
[linear-context-dflash-sglang-status.md](./linear-context-dflash-sglang-status.md).
Docker images: [linear-context-dflash-docker.md](./linear-context-dflash-docker.md).
Layer code: `specforge/modeling/draft/dflash_linear.py`,
`specforge/modeling/draft/dflash2_linear.py`.

## Shared backbone

Stock DFlash concatenates target-context K/V with the B-token block, so
attention and draft cache are \(O(L)\). Linear-context DFlash instead:

1. fuses captured target features (`fc` then RMSNorm);
2. scans them with GDN (default) or KDA into a fixed-size state \(S\);
3. gathers \(S_{p-1}\) for each sampled anchor \(p\);
4. reads that state into a retrieved vector per block offset;
5. mixes **only** inside the B-token block (dense bidirectional SDPA).

```
target features  --GDN/KDA scan-->  S_{p-1}
block [anchor | MASK…]  --LN-->  query  --read S-->  retrieved
retrieved  --injection-->  block Q/K/V  --B×B SDPA-->  attn
hidden = block + attn  [+ context residual]  + MLP
```

All Qwen3.5-4B linear JSONs share:

| Field | Value |
|---|---|
| `architectures` | `DFlashLinearDraftModel` or `DFlash2LinearDraftModel` |
| `block_size` | 16 |
| `target_layer_ids` | `[1, 8, 15, 22, 29]` (5 draft layers) |
| `mask_token_id` | 248070 |
| `embedding_key` (YAML) | `model.language_model.embed_tokens.weight` |
| Capture method | stock DFlash (`hidden_states`); caches are reusable |
| `linear_context.num_heads` / `key_dim` / `value_dim` | 8 / 64 / 64 |
| `normalize_qk` | true |
| `backend` | `auto` (FLA if installed, else naive). Serving ignores this and uses eager GDN. |

Training YAML always sets `training.strategy: dflash_linear` and points
`model.draft_model_config` at one of the JSONs below. A 1-epoch cluster
recipe (`scripts/cluster/dflash-linear/qwen3.5-4b-dflash-linear-1epoch.yaml`)
uses the **default** JSON. DFlash2-linear 1-epoch is
`scripts/cluster/dflash-linear/qwen3.5-4b-dflash2-linear-1epoch.yaml`.

Checkpoints are **not** interchangeable across variants: missing or extra
modules (`inject_gate`, `q_r_proj`, `context_residual`, KDA vs GDN gate
shapes) will fail to load.

## Knobs

`dflash_config.linear_context` has two independent axes.

**Scan (`variant`)** — how prefix state is written.

| `variant` | Decay | Extra params vs GDN |
|---|---|---|
| `gdn` (default) | scalar per head, \(\alpha_t I\) | — |
| `kda` | channel-wise, \(\mathrm{diag}(\alpha_t)\) | `g_proj` is \(H \times K\) instead of \(H\) |

**How retrieved context enters the block** — `injection` plus
`context_residual`.

Let \(h\) be RMSNorm of the B-token block and \(r\) the retrieved
prefix read (horizon embeddings are added only on this retrieve path).
Then:

| `injection` | Input to B×B attention | Extra modules |
|---|---|---|
| `gated_residual` | \(h + \sigma(W_g[h; r]) \odot W_v r\) | `inject_gate`, `inject_value` |
| `qkv_conditioning` | \(h\) unchanged; \(Q \mathrel{+}= W_{qr} r\) (same for K, V) | `q_r_proj`, `k_r_proj`, `v_r_proj` |
| `independent` | \(h\) only; attention never sees \(r\) | none |

After attention, if `context_residual` is true (default):

\[
\mathrm{hidden} = \mathrm{block} + \mathrm{attn} + W_r r
\]

If false, drop \(+ W_r r\). MLP is unchanged.

`independent` still reads \(S_{p-1}\) and (with the default residual) adds
\(W_r r\) after attention. The ablation is “no context in Q/K/V”, not “no
context at all”. `no-ctx-residual` is the opposite cut: gated injection
into Q/K/V, but no post-attention skip from \(r\).

## Checked-in recipes

Short names match the JSON / YAML suffixes. Injection ablations live under
`configs/qwen3.5-4b-dflash-linear*.json`. DFlash2-linear is
`configs/qwen3.5-4b-dflash2-linear.json`.

| Short name | JSON | `variant` | `injection` | `context_residual` | What it tests |
|---|---|---|---|---|---|
| **dflash-linear** (default) | `configs/qwen3.5-4b-dflash-linear.json` | gdn | gated_residual | true | Context-first GDN: gate retrieved features into the block, then residual-add them after attention. This is the main architecture and the 1-epoch train. |
| **dflash2-linear** | `configs/qwen3.5-4b-dflash2-linear.json` | gdn | gated_residual | true | Same GDN retrieve as default, plus DFlash2 grouped conv around **local** B×B attention and MLP, and a top-k candidate selector. Conv does not wrap the GDN scan or prefix read. |
| **no-ctx-residual** | `configs/qwen3.5-4b-dflash-linear-no-ctx-residual.json` | gdn | gated_residual | false | Same gated injection, but retrieved context cannot skip around attention. Isolates whether the post-attn residual is load-bearing. |
| **qkv** | `configs/qwen3.5-4b-dflash-linear-qkv.json` | gdn | qkv_conditioning | true | Direct Q/K/V conditioning instead of a gated residual on \(h\). Paper hypothesis: this preserves more acceptance than independent branches. |
| independent | `configs/qwen3.5-4b-dflash-linear-independent.json` | gdn | independent | true | Parallel GDN-context and dense-local branches. Context only via \(W_r r\). |
| kda | `configs/qwen3.5-4b-dflash-linear-kda.json` | kda | gated_residual | true | Same injection as default, channel-wise decay. |

Train one of them:

```bash
specforge train -c examples/configs/offline/colocated/qwen3.5-4b-dflash-linear-offline.yaml
specforge train -c examples/configs/offline/colocated/qwen3.5-4b-dflash2-linear-offline.yaml
specforge train -c examples/configs/offline/colocated/qwen3.5-4b-dflash-linear-no-ctx-residual-offline.yaml
specforge train -c examples/configs/offline/colocated/qwen3.5-4b-dflash-linear-qkv-offline.yaml
```

Offline feature caches from stock DFlash capture are valid for all of these.
Do not mix a checkpoint trained on one JSON with another JSON at export
or serve time. DFlash2-linear checkpoints are not loadable as
`DFlashLinearDraftModel` (extra conv and selector modules).

## The three named ablations

### dflash-linear (default gated residual)

```json
"linear_context": {
  "variant": "gdn",
  "injection": "gated_residual",
  "context_residual": true
}
```

Retrieved \(r\) is fused into the block **before** local attention
(learned gate) and again **after** attention (linear skip). Local SDPA
still has no context KV; it only sees the conditioned B tokens.

This is paper item “context-first GDN with dense block attention”.

### no-ctx-residual

```json
"linear_context": {
  "variant": "gdn",
  "injection": "gated_residual",
  "context_residual": false
}
```

Same `inject_gate` / `inject_value` as default. `context_residual` is
omitted, so the only path from prefix state into the residual stream is
the pre-attention gate. If MAL collapses relative to default, the skip
was carrying context that attention did not keep.

### qkv

```json
"linear_context": {
  "variant": "gdn",
  "injection": "qkv_conditioning",
  "context_residual": true
}
```

No `inject_gate`. Block tokens stay \(h\). Extra projections add \(r\)
onto Q, K, and V **after** the usual `q_proj` / `k_proj` / `v_proj` and
**before** Q/K RMSNorm and RoPE. The post-attention \(W_r r\) skip is
still present.

This is paper item “direct Q/K/V conditioning instead of residual
injection”. Weight count grows by three retrieved-bias linears
(`q_r_proj`, `k_r_proj`, `v_r_proj`) and drops the two gated-residual
linears.

### dflash2-linear

Same GDN gated-residual backbone as the default. `DFlashGroupedConv`
wraps **only** dense B×B attention and the MLP. The GDN/KDA scan and
prefix read stay unconvolved: local conv is a local-coherence correction,
not a retrieval filter. `CandidateSelector` re-ranks the frozen target
head’s top-k tokens the same way stock DFlash2 does.

JSON extras: `conv_kernel_size: 2`, `conv_group_size: 16` (must divide
hidden 2560), `selector_rank: 256`, `selector_top_k: 16`. Block size
stays 16. YAML keeps `training.strategy: dflash_linear` and adds the
DFlash2 selector-loss schedule. Checkpoints are not loadable as
`DFlashLinearDraftModel`.

## Serving

SGLang still uses `--speculative-algorithm DFLASH`. The worker is chosen
from `architectures[0] == "DFlashLinearDraftModel"`. Injection and
`context_residual` are read from the exported `config.json`; there is no
extra serve flag.

`DFlash2LinearDraftModel` is a SpecForge training and Hugging Face export
class. Offline MAL / capture-replay load it through `AutoDraftModel`.
This repo’s SGLang overlay does not dispatch that architecture yet.

The 1-epoch export used for M1 is the **default** gated-residual GDN
drafter. A qkv or no-ctx-residual export is a different model: rebuild
the HF directory from that run, keep `feature_contract.json` next to it,
and do not compare MAL across variants as if they were the same checkpoint.

## Defaults if JSON omits a field

`resolve_linear_context_settings` fills gaps:

| Field | Default |
|---|---|
| `variant` | `gdn` |
| `injection` | `gated_residual` |
| `context_residual` | `true` |
| `num_heads` | `min(8, num_key_value_heads)` |
| `key_dim` / `value_dim` | `min(64, head_dim)` |
| `normalize_qk` | `true` |
| `backend` | `auto` |

Sliding-window DFlash configs are rejected: the scan is full-prefix.
