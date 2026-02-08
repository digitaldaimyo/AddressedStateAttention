# AddressedStateAttention — User Guide

This is the companion guide for the unified `AddressedStateAttention` research harness.
It merges two prior variants (slot-mask controls and refine-delta intervention) into a
single module with one `forward()` signature and a sane configuration surface.

---

## Quick start

```python
from addressed_state_attention import AddressedStateAttention

asa = AddressedStateAttention(embed_dim=256, num_heads=8, num_slots=8)

x = torch.randn(2, 128, 256)
out, info = asa(x, return_info=True)
# out:  [2, 128, 256]
# info: dict of diagnostics (or None when return_info=False)
```

---

## Checkpoint compatibility

All parameter and buffer names are preserved from earlier harness versions.
You can `load_state_dict()` from any checkpoint trained with either prior variant
without renaming keys.  **Do not rename** any of: `slot_keys`, `Wk_write`, `Wv_write`,
`Wq_read`, `out_proj`, `_alibi_slopes`, `_alibi_strength_param`,
`_content_read_gamma_raw`, `slot_in`, `slot_q`, `slot_k`, `slot_v`, `slot_out`,
`_slotspace_gate_raw`, `rope`, `rope_slotspace`.

---

## Constructor arguments (unchanged)

| Argument | Default | Description |
|---|---|---|
| `embed_dim` | — | Model dimension (must be divisible by `num_heads`) |
| `num_heads` | 8 | Number of attention heads |
| `num_slots` | 8 | Number of addressable slots per head |
| `dropout` | 0.1 | Output dropout |
| `read_temperature` | 1.0 | Softmax temperature for read routing |
| `write_temperature` | 1.0 | Temperature for write logits |
| `state_fp32` | True | Force fp32 for streaming prefix-softmax state |
| `slot_dropout` | 0.0 | Per-slot dropout during training |
| `normalize_k` | False | L2-normalize write keys |
| `use_rope_keys` | True | RoPE on write keys |
| `rope_base` | 10000 | RoPE frequency base |
| `use_alibi_write` | True | ALiBi bias on write logits |
| `alibi_strength_init` | 0.1 | Initial ALiBi strength |
| `learn_alibi_strength` | True | Make ALiBi strength learnable |
| `min_strength` | 0.0 | Floor for ALiBi strength |
| `use_content_read` | True | Content-conditioned read term |
| `content_read_init` | -4.0 | Initial gamma (pre-softplus) |
| `content_read_max_gamma` | 3.0 | Gamma clamp ceiling |
| `use_slotspace_refine` | True | Enable slot-space refinement path |
| `slotspace_dim` | 32 | Dimension of slot-space coordinate space |
| `slotspace_gate_init` | -4.0 | Initial gate (pre-softplus, starts near 0) |
| `slotspace_dropout` | 0.05 | Dropout in slotspace linear attention |
| `slotspace_signed_weights` | True | tanh (signed) vs softmax (positive) slot weights |
| `use_rope_slotspace` | True | RoPE on slotspace Q/K |
| `rope_base_slotspace` | 100000 | RoPE base for slotspace |
| `write_chunk_size` | 128 | Chunk size for streaming write scan |
| `slotspace_chunk_size` | 128 | Chunk size for slotspace linear attention |

---

## `forward()` arguments

```python
out, info = asa(
    x,                          # [B, T, C]  input embeddings
    attention_mask=None,        # [B, T]  padding mask (1=valid, 0=pad)
    return_info=False,          # collect diagnostics

    # --- routing ---
    routing_mode="softmax",     # "softmax" | "top1" | "topk" | "external"
    routing_topk=2,             # k for topk mode
    read_weights_override=None, # [B,H,T,K] for external routing
    routing_noise=None,         # None | "gumbel" | "gaussian"
    routing_noise_scale=1.0,

    # --- slot mask (causal intervention) ---
    slot_mask=None,             # [K] float/bool, 1=keep, 0=mask
    slot_mask_where="read",     # where the mask is applied
    slot_mask_scope="all",      # which positions get masked

    # --- info controls ---
    info_level="full",          # "basic" | "logits" | "full"
    info_cfg=None,              # dict (see below)
)
```

### Routing modes

| Mode | Behavior |
|---|---|
| `"softmax"` | Standard softmax over slot logits |
| `"top1"` | Hard one-hot to highest-scoring slot |
| `"topk"` | Softmax restricted to top-k slots |
| `"external"` | Use `read_weights_override` directly |

You can also set `asa.routing_override` to a tensor `[B,H,T,K]` or a callable
`fn(t0, t1, read_logits, read_logits_key, read_logits_content, ctx) → [B,H,L,K]`
for full programmatic control.

### Routing noise

Set `routing_noise="gumbel"` or `"gaussian"` with `routing_noise_scale` to inject
noise into read logits before the softmax.  Useful for exploration or Gumbel-softmax
straight-through estimators.

---

## Slot mask (causal intervention)

Selectively suppress individual slots during a forward pass without retraining.

```python
# mask out slot 3 (0-indexed, K=8)
mask = torch.ones(8)
mask[3] = 0.0

out, info = asa(x, slot_mask=mask, slot_mask_where="read")
```

### `slot_mask_where` options

| Value | Effect |
|---|---|
| `"read"` | Zero out masked slots in both key and content read logits, then renormalize read weights. Strongest effect. |
| `"content_read_only"` | Zero out the content contribution for masked slots; key routing is untouched. Useful for probing what the content term contributes for a slot. |
| `"slotspace_only"` | Zero out masked slots in the slotspace refinement weights only; base read path is untouched. Tests refine-path reliance on specific slots. |

### `slot_mask_scope` options

| Value | Effect |
|---|---|
| `"all"` | Mask applies to every position in the sequence. |
| `"last_pos_only"` | Mask applies only at the final position (useful for generation-time probing). |

You can also set `asa.slot_mask = mask_tensor` as an instance attribute; it will be
used as a fallback when the `slot_mask` kwarg is `None`.

---

## Info controls

### `info_level`

Controls the breadth of diagnostics stored:

| Level | What's stored |
|---|---|
| `"basic"` | Scalars only (gamma, gate, strengths, norms) |
| `"logits"` | + read logits (combined, key, content) |
| `"full"` | + write logits, ALiBi bias, slot state norms |

### `info_cfg`

Fine-grained control over which tensors are stored and how.
Get a default dict with `AddressedStateAttention.default_info_cfg()`, then override:

```python
cfg = AddressedStateAttention.default_info_cfg()
cfg["store_read_weights"] = True   # read weights [B,H,T,K]
cfg["store_read_logits"]  = True   # read logits (combined, key, content)
cfg["store_write_logits"] = True   # write logits (raw and processed)
cfg["store_slot_state_norm"] = True  # slot state norms
cfg["store_out1"]  = False         # base output (pre-refine)
cfg["store_delta"] = False         # refine delta (pre-intervention)
cfg["store_slot_w"] = False        # slotspace weights (pre-tanh)
cfg["detach_to_cpu"] = False       # offload diagnostics to CPU
cfg["time_stride"]  = 1            # subsample T dimension
cfg["batch_stride"] = 1            # subsample B dimension

out, info = asa(x, return_info=True, info_cfg=cfg)
```

Setting `store_out1=True`, `store_delta=True`, `store_slot_w=True` enables the
tensors needed for refine-delta intervention analysis.  They are off by default
because they double the memory footprint.

`time_stride` and `batch_stride` let you subsample large tensors to reduce memory
during long-context analysis runs.

---

## Refine-delta intervention

Decompose the slot-space refinement delta into components parallel and orthogonal
to the base output, then selectively gate them.  This is a **post-hoc analysis tool**
— it modifies the model output at inference time without changing any learned parameters.

All intervention settings are instance attributes (not forward args), so you set them
once and they take effect on all subsequent forward calls until changed.

### Enabling intervention

```python
# Decompose: keep only the parallel component
asa._intv_mode = "delta_par"

# Or only orthogonal
asa._intv_mode = "delta_orth"

# Or use orth_gate: keep parallel, selectively gate orthogonal
asa._intv_mode = "orth_gate"

# Turn off (default)
asa._intv_mode = "off"
```

### `_intv_mode` options

| Mode | Output modification |
|---|---|
| `"off"` | None (default). Delta is applied unchanged. |
| `"delta_par"` | Replace delta with its parallel component only. |
| `"delta_orth"` | Replace delta with its orthogonal component only. |
| `"delta_par_plus_orth"` | No-op sanity check (par + orth = original). |
| `"orth_gate"` | Keep parallel; gate orthogonal by a score-based mask. |

### `orth_gate` configuration

When `_intv_mode = "orth_gate"`, the orthogonal component is scaled by a per-token
mask derived from a score, a threshold (tau), and a temperature.

```python
asa._intv_mode = "orth_gate"

# Scoring function
asa._intv_score_kind = "orth_frac"  # "orth_frac" | "orth_ratio" | "alpha_abs" | "slot_peaked"

# Threshold
asa._intv_tau_kind = "pctl"         # "pctl" (percentile) | "abs" (absolute)
asa._intv_tau_pctl = 75.0           # used when tau_kind="pctl"
asa._intv_tau = 0.15                # used when tau_kind="abs"

# Mask sharpness
asa._intv_mask_mode = "soft"        # "soft" (sigmoid) | "hard" (binary)
asa._intv_soft_temp = 0.05          # sigmoid temperature (lower = sharper)

# Scaling
asa._intv_beta = 1.0               # scale on gated orthogonal component
asa._intv_par_beta = 1.0            # scale on parallel component

# Score clipping (prevents saturation)
asa._intv_score_clip_pctl = 99.0    # clip score at this percentile
```

#### Score kinds

| Kind | Formula | Intuition |
|---|---|---|
| `"orth_frac"` | ‖δ_orth‖ / ‖δ‖ | Fraction of delta that is orthogonal |
| `"orth_ratio"` | ‖δ_orth‖ / ‖out1‖ | Orthogonal norm relative to base output |
| `"alpha_abs"` | |α| | Absolute projection coefficient |
| `"slot_peaked"` | 1 − H(softmax(slot_w))/log(K) | Normalized entropy of slot weights (requires `store_slot_w=True`) |

### Head targeting

Restrict intervention to specific heads:

```python
# Only intervene on heads 0, 2, 5 (out of 8)
head_mask = torch.zeros(8, dtype=torch.bool)
head_mask[[0, 2, 5]] = True
asa._intv_head_mask = head_mask

# Clear (intervene on all heads)
asa._intv_head_mask = None
```

---

## Refine-geometry logging

Enable per-head geometry vectors in the info dict **without changing any outputs**.
Useful for understanding the geometric relationship between the base output and the
refinement delta across heads.

```python
asa._log_refine_geom = True
out, info = asa(x, return_info=True)

# Per-head vectors (shape [H]):
info["geom_alpha_mean"]   # mean projection coefficient α
info["geom_alpha_abs"]    # mean |α|
info["geom_sign_pos"]     # fraction of α > 0
info["geom_orth_frac"]    # mean ‖δ_orth‖ / ‖δ‖
info["geom_d_ratio"]      # mean ‖δ‖ / ‖out1‖
info["geom_dpar_ratio"]   # mean ‖δ_par‖ / ‖δ‖
```

These are averaged over batch and time for each write-chunk, then averaged across
chunks.  No computational graph is modified — this is pure logging.

---

## Info dict reference

When `return_info=True`, the second return value is a dict.  Which keys are populated
depends on `info_level` and `info_cfg`.  Keys that are not populated are set to `None`.

### Always present (when return_info=True)

| Key | Shape / type | Description |
|---|---|---|
| `content_read_gamma` | scalar | Current gamma value |
| `routing_mode` | str | Routing mode used |
| `slot_mask_where` | str | Slot mask application point |
| `slot_mask_scope` | str | Slot mask scope |
| `intv_mode` | str | Current intervention mode |

### Conditional on info_level and info_cfg

| Key | Shape | Condition |
|---|---|---|
| `read_weights` | [B,H,T,K] | `store_read_weights=True` |
| `read_logits` | [B,H,T,K] | `store_read_logits=True` + level ≥ logits |
| `read_logits_key` | [B,H,T,K] | same |
| `read_logits_content` | [B,H,T,K] | same + `use_content_read` |
| `write_logits_raw` | [B,H,K,T] | `store_write_logits=True` + level = full |
| `write_logits` | [B,H,K,T] | same |
| `slot_state_norm` | [B,H,K,T] | `store_slot_state_norm=True` + level = full |
| `alibi_bias_applied` | [1,H,1,T] | level = full + `use_alibi_write` |
| `alibi_strength` | scalar | `learn_alibi_strength=True` |
| `slotspace_gate` | scalar | `use_slotspace_refine=True` |
| `slotspace_delta_norm` | scalar | `use_slotspace_refine=True` |
| `use_rope_slotspace` | bool tensor | `use_slotspace_refine=True` |
| `out1` | [B,H,T,d] | `store_out1=True` |
| `delta` | [B,H,T,d] | `store_delta=True` |
| `slot_w` | [B,H,T,K] | `store_slot_w=True` |

### Intervention / geometry logs (when active)

| Key | Shape | Source |
|---|---|---|
| `geom_alpha_mean` | [H] | `_log_refine_geom=True` |
| `geom_alpha_abs` | [H] | same |
| `geom_sign_pos` | [H] | same |
| `geom_orth_frac` | [H] | same |
| `geom_d_ratio` | [H] | same |
| `geom_dpar_ratio` | [H] | same |
| `score` | scalar or [H] | `_intv_mode="orth_gate"` |
| `tau` | scalar | same |
| `mask` | scalar or [H] | same |
| `alpha` | scalar or [H] | any intervention mode |
| `out1_norm` | scalar or [H] | `_intv_mode="orth_gate"` |
| `dpar_norm` | scalar or [H] | same |
| `dorth_norm` | scalar or [H] | same |

---

## Typical workflows

### 1. Training (no diagnostics needed)

```python
out, _ = asa(x)  # return_info=False by default
loss = criterion(out, target)
loss.backward()
```

### 2. Quick validation check

```python
out, info = asa(x, return_info=True, info_level="basic")
print(f"gamma={info['content_read_gamma']:.3f}  gate={info['slotspace_gate']:.3f}")
```

### 3. Full analysis with memory budget

```python
cfg = AddressedStateAttention.default_info_cfg()
cfg["time_stride"] = 4       # subsample time by 4x
cfg["detach_to_cpu"] = True  # offload to CPU
cfg["store_out1"] = True
cfg["store_delta"] = True

out, info = asa(x, return_info=True, info_level="full", info_cfg=cfg)
```

### 4. Causal slot knockout

```python
mask = torch.ones(8); mask[3] = 0
out_ko, info = asa(x, return_info=True, slot_mask=mask, slot_mask_where="read")
# compare out_ko vs baseline to measure slot 3's causal contribution
```

### 5. Orth-gate sweep

```python
asa._intv_mode = "orth_gate"
asa._log_refine_geom = True

for pctl in [50, 75, 90, 95]:
    asa._intv_tau_pctl = pctl
    out, info = asa(x, return_info=True)
    print(f"pctl={pctl}  orth_frac={info['geom_orth_frac'].mean():.3f}")

asa._intv_mode = "off"  # always reset when done
```
