
"""
Addressed State Attention (ASA) - Analysis Harness

Research implementation with mechanistic intervention capabilities.
For efficient training without interventions, use asm_training.py instead.

Features:
    - Slot-mask causal interventions (slot_mask, slot_mask_where, slot_mask_scope)
    - Refinement decomposition (orthogonal/parallel gating)
    - Per-head geometry logging
    - Configurable information storage (info_level, info_cfg)

Checkpoint Compatibility:
    All parameter/buffer names match asm_training.py for weight sharing.
    Do NOT rename: slot_keys, Wk_write, Wv_write, Wq_read, out_proj,
    _alibi_slopes, _alibi_strength_param, _content_read_gamma_raw,
    slot_in/slot_q/slot_k/slot_v/slot_out, _slotspace_gate_raw,
    rope/rope_slotspace buffers.

Repository: https://github.com/DigitalDaimyo/AddressedStateAttention
Paper: https://github.com/DigitalDaimyo/AddressedStateAttention/tree/main/paper_drafts
"""

import math
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = [
    'AddressedStateAttention',
    'ASMBlock',
    'ASMLanguageModel',
    'ASMTrainConfig',
    'build_model_from_cfg',
]


# ------------------------------------------------------------------ helpers ---

def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        assert dim % 2 == 0, "RoPE requires even dim"
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._cos_cached = None
        self._sin_cached = None
        self._t_cached = None
        self._device_cached = None

    def get_cos_sin(self, T: int, device, dtype):
        if (
            self._t_cached == T
            and self._cos_cached is not None
            and self._device_cached == device
        ):
            return self._cos_cached.to(dtype=dtype), self._sin_cached.to(dtype=dtype)
        t = torch.arange(T, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("t,f->tf", t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        cos = emb.cos()[None, None, :, :]
        sin = emb.sin()[None, None, :, :]
        self._t_cached = T
        self._device_cached = device
        self._cos_cached = cos
        self._sin_cached = sin
        return cos.to(dtype=dtype), sin.to(dtype=dtype)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return (x * cos) + (_rotate_half(x) * sin)


def alibi_slopes(num_heads: int, device=None, dtype=torch.float32) -> torch.Tensor:
    def get_slopes(n):
        def power_of_2_slopes(n):
            start = 2.0 ** (-(2.0 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * (ratio ** i) for i in range(n)]
        if math.log2(n).is_integer():
            return power_of_2_slopes(n)
        closest = 2 ** math.floor(math.log2(n))
        return power_of_2_slopes(closest) + get_slopes(2 * closest)[0::2][: n - closest]
    return torch.tensor(get_slopes(num_heads), device=device, dtype=dtype)


def _inv_softplus(y: torch.Tensor) -> torch.Tensor:
    return torch.log(torch.expm1(y))


def phi(x: torch.Tensor) -> torch.Tensor:
    """Performer-style feature map (elu + 1)."""
    return F.elu(x) + 1.0


# --------------------------------------------------------- main module ---

class AddressedStateAttention(nn.Module):
    """
    Addressed State Attention (ASA) — unified research harness.

    Core mechanism
    --------------
    * Prefix-softmax WRITE into K learned slots  (streaming, O(T))
    * READ routing from tokens → slots  (softmax / top-k / external)
    * Content-conditioned READ term  (gamma-weighted)
    * RoPE on write keys  (geometry)
    * ALiBi bias on write logits  (prefix-friendly)

    Slot-space refinement
    ---------------------
    * Causal linear attention in a low-dim slot-address coordinate space
    * Produces per-token signed weights over slots
    * Decoded through the same streaming slot-state basis
    * Gated by learnable ``slotspace_gate`` (softplus)

    Causal intervention (slot mask)
    -------------------------------
    * ``slot_mask``  [K] float/bool, 1=keep 0=mask
    * ``slot_mask_where``  "read" | "content_read_only" | "slotspace_only"
    * ``slot_mask_scope``  "all" | "last_pos_only"

    Refine-delta intervention  (instance attrs, NO-OP by default)
    ----------------------------------------------------------------
    * ``_intv_mode``  "off" | "delta_par" | "delta_orth" | "orth_gate" | …
    * Decomposes refine delta into parallel / orthogonal vs base output
    * See User Guide for configuration details.

    Refine-geometry logging  (NO output change)
    ------------------------------------------------
    * ``_log_refine_geom = True`` enables per-head geometry vectors in info dict.

    Info storage
    ------------
    * ``info_level``  "basic" | "logits" | "full"
    * ``info_cfg``  dict controlling which tensors to store, downsampling, CPU offload.
    """

    # ---------------------------------------------------------------- init ---

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        num_slots: int = 8,
        dropout: float = 0.1,
        # temperatures / numerics
        read_temperature: float = 1.0,
        write_temperature: float = 1.0,
        state_fp32: bool = True,
        slot_dropout: float = 0.0,
        normalize_k: bool = False,
        # positions (write geometry)
        use_rope_keys: bool = True,
        rope_base: float = 10000.0,
        # write bias (ALiBi)
        use_alibi_write: bool = True,
        alibi_strength_init: float = 0.1,
        learn_alibi_strength: bool = True,
        min_strength: float = 0.0,
        # content-conditioned read term
        use_content_read: bool = True,
        content_read_init: float = -4.0,
        content_read_max_gamma: float = 3.0,
        # slot-space refinement
        use_slotspace_refine: bool = True,
        slotspace_dim: int = 32,
        slotspace_gate_init: float = -4.0,
        slotspace_dropout: float = 0.05,
        slotspace_signed_weights: bool = True,
        # RoPE in slot-space matcher
        use_rope_slotspace: bool = True,
        rope_base_slotspace: float = 100000.0,
        # perf knobs
        write_chunk_size: int = 128,
        slotspace_chunk_size: int = 128,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_slots = num_slots
        self.head_dim = embed_dim // num_heads

        self.dropout = nn.Dropout(dropout)

        self.read_temperature = float(read_temperature)
        self.write_temperature = float(write_temperature)
        self.state_fp32 = bool(state_fp32)
        self.slot_dropout = float(slot_dropout)
        self.normalize_k = bool(normalize_k)
        self.routing_override = None

        self.use_rope_keys = bool(use_rope_keys)
        self.use_alibi_write = bool(use_alibi_write)
        self.learn_alibi_strength = bool(learn_alibi_strength)
        self.min_strength = float(min_strength)

        self.use_content_read = bool(use_content_read)
        self.content_read_max_gamma = float(content_read_max_gamma)

        self.use_slotspace_refine = bool(use_slotspace_refine)
        self.slotspace_dim = int(slotspace_dim)
        self.slotspace_dropout = nn.Dropout(float(slotspace_dropout))
        self.slotspace_signed_weights = bool(slotspace_signed_weights)

        self.write_chunk_size = int(write_chunk_size)
        self.slotspace_chunk_size = int(slotspace_chunk_size)

        # Learned slot keys: [H, K, d]
        self.slot_keys = nn.Parameter(
            torch.randn(num_heads, num_slots, self.head_dim) / math.sqrt(self.head_dim)
        )

        # Projections
        self.Wk_write = nn.Linear(embed_dim, embed_dim, bias=False)
        self.Wv_write = nn.Linear(embed_dim, embed_dim, bias=False)
        self.Wq_read  = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj  = nn.Linear(embed_dim, embed_dim, bias=False)

        # RoPE (write geometry)
        self.rope = RotaryEmbedding(self.head_dim, base=rope_base) if self.use_rope_keys else None

        # ALiBi
        if self.use_alibi_write:
            self.register_buffer("_alibi_slopes", alibi_slopes(num_heads), persistent=False)
        else:
            self.register_buffer("_alibi_slopes", torch.zeros(num_heads), persistent=False)

        if self.use_alibi_write and self.learn_alibi_strength:
            init = torch.tensor(float(alibi_strength_init) - self.min_strength).clamp_min(1e-8)
            self._alibi_strength_param = nn.Parameter(_inv_softplus(init))
        else:
            self._alibi_strength_param = None
            self.alibi_strength = float(alibi_strength_init)

        # Content read gamma
        if self.use_content_read:
            self._content_read_gamma_raw = nn.Parameter(torch.tensor(float(content_read_init)))
        else:
            self._content_read_gamma_raw = None

        # Slot-space refinement
        self.use_rope_slotspace = bool(use_rope_slotspace) and bool(self.use_slotspace_refine)
        if self.use_slotspace_refine:
            self.slot_in  = nn.Linear(num_slots, self.slotspace_dim, bias=False)
            self.slot_q   = nn.Linear(self.slotspace_dim, self.slotspace_dim, bias=False)
            self.slot_k   = nn.Linear(self.slotspace_dim, self.slotspace_dim, bias=False)
            self.slot_v   = nn.Linear(self.slotspace_dim, self.slotspace_dim, bias=False)
            self.slot_out = nn.Linear(self.slotspace_dim, num_slots, bias=False)
            self._slotspace_gate_raw = nn.Parameter(torch.tensor(float(slotspace_gate_init)))
            if self.use_rope_slotspace:
                assert (self.slotspace_dim % 2) == 0, "use_rope_slotspace requires even slotspace_dim"
                self.rope_slotspace = RotaryEmbedding(self.slotspace_dim, base=float(rope_base_slotspace))
            else:
                self.rope_slotspace = None
        else:
            self.slot_in = None
            self.slot_q = self.slot_k = self.slot_v = None
            self.slot_out = None
            self._slotspace_gate_raw = None
            self.rope_slotspace = None

        # ----- intervention defaults (NO-OP) -----
        self._intv_mode: str = "off"
        self._intv_beta: float = 1.0
        self._intv_score_kind: str = "orth_frac"
        self._intv_tau_kind: str = "pctl"
        self._intv_tau: float = 0.15
        self._intv_tau_pctl: float = 75.0
        self._intv_mask_mode: str = "soft"
        self._intv_soft_temp: float = 0.05
        self._intv_par_beta: float = 1.0
        self._intv_head_mask: Optional[torch.Tensor] = None
        self._intv_score_clip_pctl: float = 99.0

        # ----- refine-geometry logging (no compute change) -----
        self._log_refine_geom: bool = False

    # -------------------------------------------------------- scalar params ---

    def _alibi_strength(self, dtype, device) -> torch.Tensor:
        if not (self.use_alibi_write and self.learn_alibi_strength):
            return torch.tensor(self.alibi_strength, dtype=dtype, device=device)
        return (F.softplus(self._alibi_strength_param) + self.min_strength).to(dtype=dtype, device=device)

    def _content_read_gamma(self, dtype, device) -> torch.Tensor:
        if not self.use_content_read:
            return torch.tensor(0.0, dtype=dtype, device=device)
        g = F.softplus(self._content_read_gamma_raw)
        if self.content_read_max_gamma is not None and self.content_read_max_gamma > 0:
            g = g.clamp(max=self.content_read_max_gamma)
        return g.to(dtype=dtype, device=device)

    def _slotspace_gate(self, dtype, device) -> torch.Tensor:
        if not self.use_slotspace_refine:
            return torch.tensor(0.0, dtype=dtype, device=device)
        return F.softplus(self._slotspace_gate_raw).to(dtype=dtype, device=device)

    # --------------------------------------------------------- numerics ---

    @staticmethod
    def _safe_exp_sub_max(s: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        diff = s - m
        diff = diff.masked_fill(~torch.isfinite(m), float("-inf"))
        return torch.exp(diff)

    # ------------------------------------------------------ slot mask ---

    def _resolve_slot_mask(
        self,
        slot_mask: Optional[torch.Tensor],
        *,
        B: int, H: int, L: int, K: int,
        device, dtype, scope: str,
    ) -> Optional[torch.Tensor]:
        """Expand [K] mask → [B,H,L,K]. Falls back to self.slot_mask attr."""
        if slot_mask is None:
            slot_mask = getattr(self, "slot_mask", None)
        if slot_mask is None:
            return None
        sm = slot_mask.to(device=device, dtype=dtype)
        if sm.ndim != 1 or sm.numel() != K:
            raise ValueError(f"slot_mask must be shape [K]={K}, got {tuple(sm.shape)}")
        sm = sm.view(1, 1, 1, K)
        if scope == "all":
            return sm.expand(B, H, L, K)
        if scope == "last_pos_only":
            out = torch.ones((B, H, L, K), device=device, dtype=dtype)
            out[:, :, -1:, :] = sm.expand(B, H, 1, K)
            return out
        raise ValueError(f"Unknown slot_mask_scope={scope!r}")

    @staticmethod
    def _apply_hard_mask_and_renorm(w: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
        w = w * keep.to(w.dtype)
        return w / w.sum(dim=-1, keepdim=True).clamp_min(1e-8)

    # --------------------------------------------------- info helpers ---

    @staticmethod
    def default_info_cfg() -> Dict:
        """Return default info_cfg dict.  Copy and modify before passing to forward()."""
        return dict(
            store_read_weights=True,
            store_read_logits=True,
            store_write_logits=True,
            store_slot_state_norm=True,
            store_out1=False,
            store_delta=False,
            store_slot_w=False,
            detach_to_cpu=False,
            time_stride=1,
            batch_stride=1,
        )

    @staticmethod
    def _store_tensor(
        t: Optional[torch.Tensor], *, cfg: Dict, kind: str,
    ) -> Optional[torch.Tensor]:
        """Downsample + detach (+ optional CPU offload)."""
        if t is None:
            return None
        bstride = int(cfg.get("batch_stride", 1))
        tstride = int(cfg.get("time_stride", 1))
        to_cpu  = bool(cfg.get("detach_to_cpu", False))
        x = t
        if x.dim() >= 1 and bstride > 1:
            x = x[::bstride]
        if x.dim() == 4 and tstride > 1:
            if kind == "bhtk":
                x = x[:, :, ::tstride, :]
            elif kind == "bhkt":
                x = x[:, :, :, ::tstride]
        x = x.detach()
        if to_cpu:
            x = x.to("cpu", non_blocking=True)
        return x

    # ------------------------------------------------ read-weight routing ---

    def _compute_read_weights(
        self,
        *,
        read_logits: torch.Tensor,
        read_logits_key: torch.Tensor,
        read_logits_content: Optional[torch.Tensor],
        routing_mode: str,
        routing_topk: int,
        read_weights_override: Optional[torch.Tensor],
        routing_noise: Optional[str],
        routing_noise_scale: float,
        rtemp: float,
        sm: Optional[torch.Tensor],
        slot_mask_where: str,
        B: int, H: int, L: int, K: int,
        T_total: int,
        t0: int, t1: int,
        q_read_c: torch.Tensor,
        slot_keys: torch.Tensor,
        slot_state_t: torch.Tensor,
        valid: Optional[torch.Tensor],
        state_dtype,
    ) -> torch.Tensor:
        """Compute read weights for one write-chunk.  Handles noise, overrides, masks."""
        # routing noise
        if routing_noise is not None:
            if routing_noise == "gumbel":
                u = torch.rand_like(read_logits)
                g = -torch.log(-torch.log(u.clamp_min(1e-8)).clamp_min(1e-8))
                read_logits = read_logits + routing_noise_scale * g
            elif routing_noise == "gaussian":
                read_logits = read_logits + routing_noise_scale * torch.randn_like(read_logits)
            else:
                raise ValueError(f"Unknown routing_noise={routing_noise}")

        # routing override (external callable or tensor)
        if self.routing_override is not None:
            if callable(self.routing_override):
                ctx = dict(
                    t0=t0, t1=t1, B=B, H=H, T=T_total, K=K, d=self.head_dim,
                    rtemp=rtemp, state_dtype=state_dtype,
                    q_read_c=q_read_c, slot_keys=slot_keys,
                    slot_state_t=slot_state_t, valid=valid,
                )
                read_w = self.routing_override(
                    t0, t1, read_logits, read_logits_key, read_logits_content, ctx,
                )
            else:
                read_w = self.routing_override[:, :, t0:t1, :].to(read_logits.dtype)
            read_w = torch.nan_to_num(read_w, nan=0.0, posinf=0.0, neginf=0.0)
            read_w = read_w.clamp_min(0.0)
            read_w = read_w / read_w.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        else:
            if routing_mode == "softmax":
                read_w = torch.softmax(read_logits / rtemp, dim=-1)
            elif routing_mode == "top1":
                top = read_logits.argmax(dim=-1)
                read_w = F.one_hot(top, num_classes=K).to(read_logits.dtype)
            elif routing_mode == "topk":
                kk = max(1, min(K, int(routing_topk)))
                vals, idx = torch.topk(read_logits, k=kk, dim=-1)
                masked = torch.full_like(read_logits, float("-inf"))
                masked.scatter_(-1, idx, vals)
                read_w = torch.softmax(masked / rtemp, dim=-1)
            elif routing_mode == "external":
                if read_weights_override is None:
                    raise ValueError("routing_mode='external' requires read_weights_override")
                if read_weights_override.shape[-2] == T_total:
                    read_w = read_weights_override[:, :, t0:t1, :]
                else:
                    read_w = read_weights_override
                read_w = read_w / read_w.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            else:
                raise ValueError(f"Unknown routing_mode={routing_mode}")

        # slot mask at read stage
        if slot_mask_where == "read" and sm is not None:
            read_w = self._apply_hard_mask_and_renorm(read_w, (sm > 0.0))

        return read_w

    # ------------------------------------------- refine-delta intervention ---

    def _apply_refine_intervention(
        self,
        out1: torch.Tensor,
        delta: torch.Tensor,
        slot_w: Optional[torch.Tensor],
    ):
        """Decompose refine delta into par/orth vs base output, optionally gate."""
        eps = 1e-8
        B, H, L, d = out1.shape

        # head mask
        hm = getattr(self, "_intv_head_mask", None)
        if hm is not None:
            hm = hm.to(device=out1.device).view(1, H, 1, 1).to(dtype=out1.dtype)

        out1_norm2 = (out1 * out1).sum(dim=-1, keepdim=True).clamp_min(eps)
        alpha = (delta * out1).sum(dim=-1, keepdim=True) / out1_norm2
        delta_par  = alpha * out1
        delta_orth = delta - delta_par

        logs = None

        # geometry logging (no output change)
        if getattr(self, "_log_refine_geom", False):
            out1n  = out1.norm(dim=-1).clamp_min(eps)
            dn     = delta.norm(dim=-1).clamp_min(eps)
            dparn  = delta_par.norm(dim=-1)
            dorthn = delta_orth.norm(dim=-1)
            a = alpha.squeeze(-1)
            logs = dict(
                geom_alpha_mean=a.mean(dim=(0, 2)),
                geom_alpha_abs=a.abs().mean(dim=(0, 2)),
                geom_sign_pos=(a > 0).float().mean(dim=(0, 2)),
                geom_orth_frac=(dorthn / dn).mean(dim=(0, 2)),
                geom_d_ratio=(dn / out1n).mean(dim=(0, 2)),
                geom_dpar_ratio=(dparn / dn).mean(dim=(0, 2)),
            )

        mode = getattr(self, "_intv_mode", "off")
        if mode is None or mode == "off":
            return delta, logs

        # --- intervention modes ---
        if mode == "delta_par":
            delta_mod = delta_par
            logs = logs or {}
            logs["alpha"] = alpha.squeeze(-1)

        elif mode == "delta_orth":
            delta_mod = delta_orth
            logs = logs or {}
            logs["alpha"] = alpha.squeeze(-1)

        elif mode == "delta_par_plus_orth":
            delta_mod = delta_par + delta_orth
            logs = logs or {}
            logs["alpha"] = alpha.squeeze(-1)

        elif mode == "orth_gate":
            beta = float(getattr(self, "_intv_beta", 1.0))
            sk = getattr(self, "_intv_score_kind", "orth_frac")
            out1n = out1.norm(dim=-1).clamp_min(eps)
            dorthn = delta_orth.norm(dim=-1)
            dn = delta.norm(dim=-1).clamp_min(eps)

            if sk == "orth_ratio":
                score = dorthn / out1n
            elif sk == "orth_frac":
                score = dorthn / dn
            elif sk == "alpha_abs":
                score = alpha.abs().squeeze(-1)
            elif sk == "slot_peaked":
                if slot_w is None:
                    raise ValueError("score_kind='slot_peaked' requires slot_w")
                p = torch.softmax(slot_w.float(), dim=-1).clamp_min(1e-8)
                Hrw = -(p * p.log()).sum(dim=-1)
                K = p.shape[-1]
                score = (1.0 - Hrw / max(1e-8, math.log(K))).to(dtype=out1.dtype)
            else:
                raise ValueError(f"Unknown _intv_score_kind={sk}")

            # score clipping
            clip_p = getattr(self, "_intv_score_clip_pctl", None)
            if clip_p is not None:
                clip_p = float(clip_p)
                if 0.0 < clip_p < 100.0:
                    smax = torch.quantile(score.detach().flatten(), clip_p / 100.0).to(score.dtype)
                    score = torch.clamp(score, max=smax)

            # tau
            tk = getattr(self, "_intv_tau_kind", "pctl")
            if tk == "abs":
                tau = torch.tensor(float(getattr(self, "_intv_tau", 0.15)),
                                   device=score.device, dtype=score.dtype)
            elif tk == "pctl":
                tau = torch.quantile(
                    score.detach().flatten(),
                    float(getattr(self, "_intv_tau_pctl", 75.0)) / 100.0,
                ).to(score.dtype)
            else:
                raise ValueError(f"Unknown _intv_tau_kind={tk}")

            # mask
            mm = getattr(self, "_intv_mask_mode", "soft")
            if mm == "hard":
                mask = (score > tau).to(out1.dtype)
            elif mm == "soft":
                temp = max(1e-6, float(getattr(self, "_intv_soft_temp", 0.05)))
                mask = torch.sigmoid((score - tau) / temp).to(out1.dtype)
            else:
                raise ValueError(f"Unknown _intv_mask_mode={mm}")

            par_beta = float(getattr(self, "_intv_par_beta", 1.0))
            delta_mod = par_beta * delta_par + beta * mask.unsqueeze(-1) * delta_orth

            logs = logs or {}
            logs.update(dict(
                score=score, tau=tau, mask=mask,
                alpha=alpha.squeeze(-1),
                out1_norm=out1n,
                dpar_norm=delta_par.norm(dim=-1),
                dorth_norm=dorthn,
            ))
        else:
            raise ValueError(f"Unknown _intv_mode={mode}")

        # head targeting
        if hm is not None:
            delta_mod = hm * delta_mod + (1.0 - hm) * delta
            logs = logs or {}
            logs["head_mask"] = hm.squeeze(0).squeeze(-1).squeeze(-1).detach()

        return delta_mod, logs

    # ============================================================ forward ===

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_info: bool = False,

        # routing
        routing_mode: str = "softmax",
        routing_topk: int = 2,
        read_weights_override: Optional[torch.Tensor] = None,
        routing_noise: Optional[str] = None,
        routing_noise_scale: float = 1.0,

        # slot mask (causal intervention)
        slot_mask: Optional[torch.Tensor] = None,
        slot_mask_where: str = "read",
        slot_mask_scope: str = "all",

        # info controls
        info_level: str = "full",
        info_cfg: Optional[Dict] = None,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        Parameters
        ----------
        x : [B, T, C]
        attention_mask : [B, T] optional padding mask (1=valid, 0=pad)
        return_info : if True, return diagnostics dict as second element
        routing_mode : "softmax" | "top1" | "topk" | "external"
        routing_topk : k for topk mode
        read_weights_override : [B,H,T,K] or [B,H,L,K] for external routing
        routing_noise : None | "gumbel" | "gaussian"
        routing_noise_scale : scale for routing noise
        slot_mask : [K] where 1=keep, 0=mask
        slot_mask_where : "read" | "content_read_only" | "slotspace_only"
        slot_mask_scope : "all" | "last_pos_only"
        info_level : "basic" | "logits" | "full"
        info_cfg : dict (see default_info_cfg())

        Returns
        -------
        (output, info) where info is None if return_info=False.
        """

        B, T, C = x.shape
        H, K, d = self.num_heads, self.num_slots, self.head_dim

        # ---- resolve info config ----
        if info_cfg is None:
            info_cfg = self.default_info_cfg()
        store_read_weights = bool(info_cfg.get("store_read_weights", True))
        store_read_logits  = bool(info_cfg.get("store_read_logits", True)) and info_level in ("logits", "full")
        store_write_logits = bool(info_cfg.get("store_write_logits", True)) and info_level == "full"
        store_slot_norm    = bool(info_cfg.get("store_slot_state_norm", True)) and info_level == "full"
        store_out1   = bool(info_cfg.get("store_out1", False)) and return_info
        store_delta  = bool(info_cfg.get("store_delta", False)) and return_info
        store_slot_w = bool(info_cfg.get("store_slot_w", False)) and return_info

        # ---- projections ----
        k_write = self.Wk_write(x).view(B, T, H, d).transpose(1, 2)
        v_write = self.Wv_write(x).view(B, T, H, d).transpose(1, 2)
        q_read  = self.Wq_read(x).view(B, T, H, d).transpose(1, 2)

        if self.normalize_k:
            k_write = F.normalize(k_write, dim=-1, eps=1e-8)

        if self.use_rope_keys:
            cos, sin = self.rope.get_cos_sin(T, device=x.device, dtype=k_write.dtype)
            k_write = apply_rope(k_write, cos, sin)

        # slot dropout
        slot_keys = self.slot_keys
        if self.training and self.slot_dropout > 0.0:
            drop = (torch.rand((H, K), device=x.device) < self.slot_dropout)
            slot_keys = slot_keys * (~drop).to(slot_keys.dtype).unsqueeze(-1)

        # ---- WRITE logits ----
        write_logits_raw = torch.einsum("hkd,bhtd->bhkt", slot_keys, k_write) / math.sqrt(d)
        state_dtype = torch.float32 if (self.state_fp32 and x.dtype != torch.float32) else x.dtype
        write_logits = write_logits_raw.to(state_dtype) / max(1e-6, self.write_temperature)

        # ALiBi
        alibi_bias_applied = None
        if self.use_alibi_write:
            strength = self._alibi_strength(dtype=state_dtype, device=x.device)
            slopes = self._alibi_slopes.to(device=x.device, dtype=state_dtype) * strength
            pos_i = torch.arange(T, device=x.device, dtype=state_dtype)
            alibi_bias = slopes.view(1, H, 1, 1) * pos_i.view(1, 1, 1, T)
            write_logits = write_logits + alibi_bias
            alibi_bias_applied = alibi_bias

        # padding mask
        if attention_mask is not None:
            valid = attention_mask.to(dtype=torch.bool)
            write_logits = write_logits.masked_fill(~valid.view(B, 1, 1, T), float("-inf"))
        else:
            valid = None

        # ================================================================
        # STREAMING WRITE + READ
        # ================================================================
        content_read_gamma = self._content_read_gamma(dtype=q_read.dtype, device=x.device)
        rtemp = max(1e-6, self.read_temperature)

        out_h = torch.empty((B, H, T, d), device=x.device, dtype=state_dtype)

        out1_full   = torch.empty((B, H, T, d), device=x.device, dtype=state_dtype) if store_out1 else None
        delta_full  = torch.empty((B, H, T, d), device=x.device, dtype=state_dtype) if store_delta else None
        slot_w_full = torch.empty((B, H, T, K), device=x.device, dtype=state_dtype) if store_slot_w else None

        need_rw = bool(self.use_slotspace_refine) or (return_info and store_read_weights)
        read_weights = torch.empty((B, H, T, K), device=x.device, dtype=q_read.dtype) if need_rw else None

        slot_state_norm_t = (
            torch.empty((B, H, T, K), device=x.device, dtype=torch.float32)
            if (return_info and store_slot_norm) else None
        )

        if return_info and store_read_logits:
            read_logits_full = torch.empty((B, H, T, K), device=x.device, dtype=state_dtype)
            read_logits_key_full = torch.empty((B, H, T, K), device=x.device, dtype=state_dtype)
            read_logits_content_full = (
                torch.empty((B, H, T, K), device=x.device, dtype=state_dtype) if self.use_content_read else None
            )
        else:
            read_logits_full = read_logits_key_full = read_logits_content_full = None

        # streaming state
        denom_state = torch.zeros((B, H, K), device=x.device, dtype=state_dtype)
        numer_state = torch.zeros((B, H, K, d), device=x.device, dtype=state_dtype)
        m_state = torch.full((B, H, K), float("-inf"), device=x.device, dtype=state_dtype)

        WRITE_CHUNK = self.write_chunk_size

        for t0 in range(0, T, WRITE_CHUNK):
            t1 = min(T, t0 + WRITE_CHUNK)
            L = t1 - t0

            wlog_c = write_logits[:, :, :, t0:t1]
            m_c, _ = torch.cummax(wlog_c, dim=-1)
            m_new = torch.maximum(m_state.unsqueeze(-1), m_c)

            scale = torch.exp(m_state.unsqueeze(-1) - m_new)
            denom_c = denom_state.unsqueeze(-1) * scale
            numer_c = numer_state.unsqueeze(-2) * scale.unsqueeze(-1)

            w_new = self._safe_exp_sub_max(wlog_c, m_new)
            denom_c = denom_c + torch.cumsum(w_new, dim=-1)

            v_c = v_write[:, :, t0:t1, :].to(state_dtype)
            add = torch.cumsum(w_new.unsqueeze(-1) * v_c.unsqueeze(2), dim=-2)
            numer_c = numer_c + add

            slot_state_c = numer_c / denom_c.clamp_min(1e-8).unsqueeze(-1)
            slot_state_t = slot_state_c.permute(0, 1, 3, 2, 4).contiguous()

            # READ logits
            q_read_c = q_read[:, :, t0:t1, :]
            read_logits_key = torch.einsum("bhld,hkd->bhlk", q_read_c, slot_keys) / math.sqrt(d)

            read_logits_content = None
            if self.use_content_read:
                read_logits_content = torch.einsum(
                    "bhld,bhlkd->bhlk", q_read_c, slot_state_t.to(q_read_c.dtype),
                ) / math.sqrt(d)

            # slot mask for this chunk
            sm = self._resolve_slot_mask(
                slot_mask, B=B, H=H, L=L, K=K,
                device=x.device, dtype=read_logits_key.dtype, scope=slot_mask_scope,
            )

            # apply mask to logits according to slot_mask_where
            if slot_mask_where == "read":
                if sm is not None:
                    read_logits_key = read_logits_key.masked_fill(sm <= 0.0, float("-inf"))
                    if self.use_content_read and read_logits_content is not None:
                        read_logits_content = read_logits_content.masked_fill(sm <= 0.0, float("-inf"))
            elif slot_mask_where == "content_read_only":
                if sm is not None and self.use_content_read and read_logits_content is not None:
                    read_logits_content = read_logits_content.masked_fill(sm <= 0.0, 0.0)
            elif slot_mask_where == "slotspace_only":
                pass  # applied later on slot_w
            else:
                raise ValueError(f"Unknown slot_mask_where={slot_mask_where!r}")

            # combine
            rl = read_logits_key
            if self.use_content_read and read_logits_content is not None:
                rl = rl + content_read_gamma.to(rl.dtype) * read_logits_content

            if return_info and store_read_logits:
                read_logits_full[:, :, t0:t1, :] = rl.to(state_dtype)
                read_logits_key_full[:, :, t0:t1, :] = read_logits_key.to(state_dtype)
                if self.use_content_read and read_logits_content_full is not None:
                    read_logits_content_full[:, :, t0:t1, :] = read_logits_content.to(state_dtype)

            # read weights
            read_w_c = self._compute_read_weights(
                read_logits=rl, read_logits_key=read_logits_key,
                read_logits_content=read_logits_content,
                routing_mode=routing_mode, routing_topk=routing_topk,
                read_weights_override=read_weights_override,
                routing_noise=routing_noise, routing_noise_scale=routing_noise_scale,
                rtemp=rtemp, sm=sm, slot_mask_where=slot_mask_where,
                B=B, H=H, L=L, K=K, T_total=T, t0=t0, t1=t1,
                q_read_c=q_read_c, slot_keys=slot_keys,
                slot_state_t=slot_state_t, valid=valid,
                state_dtype=state_dtype,
            )

            if read_weights is not None:
                read_weights[:, :, t0:t1, :] = read_w_c

            # base output
            out_h[:, :, t0:t1, :] = torch.einsum(
                "bhlk,bhlkd->bhld", read_w_c.to(state_dtype), slot_state_t.to(state_dtype),
            )

            if out1_full is not None:
                out1_full[:, :, t0:t1, :] = out_h[:, :, t0:t1, :]

            if slot_state_norm_t is not None:
                slot_state_norm_t[:, :, t0:t1, :] = slot_state_t.to(torch.float32).norm(dim=-1)

            m_state = m_new[:, :, :, -1]
            denom_state = denom_c[:, :, :, -1]
            numer_state = numer_c[:, :, :, -1, :]

        # ================================================================
        # SLOT-SPACE REFINEMENT
        # ================================================================
        slotspace_delta_norm_mean = None
        intv_logs_acc: Optional[Dict] = None
        intv_logs_count = 0

        if self.use_slotspace_refine:
            slotspace_dtype = state_dtype
            M = self.slotspace_dim
            assert read_weights is not None

            u = self.slot_in(read_weights.to(slotspace_dtype))
            q_s = self.slot_q(u)
            k_s = self.slot_k(u)
            v_s = self.slot_v(u)

            if self.use_rope_slotspace:
                cos_s, sin_s = self.rope_slotspace.get_cos_sin(T, device=x.device, dtype=q_s.dtype)
                q_s = apply_rope(q_s, cos_s, sin_s)
                k_s = apply_rope(k_s, cos_s, sin_s)

            qf = phi(q_s)
            kf = phi(k_s)

            if valid is not None:
                vmask = valid.view(B, 1, T, 1).to(slotspace_dtype)
                qf = qf * vmask
                kf = kf * vmask
                v_s = v_s * vmask

            u2 = torch.empty((B, H, T, M), device=x.device, dtype=slotspace_dtype)
            S_state = torch.zeros((B, H, M, M), device=x.device, dtype=slotspace_dtype)
            Z_state = torch.zeros((B, H, M), device=x.device, dtype=slotspace_dtype)

            SS_CHUNK = self.slotspace_chunk_size
            for t0 in range(0, T, SS_CHUNK):
                t1 = min(T, t0 + SS_CHUNK)
                qf_c = qf[:, :, t0:t1, :]
                kf_c = kf[:, :, t0:t1, :]
                v_c  = v_s[:, :, t0:t1, :]

                kv = torch.einsum("bhlm,bhln->bhlmn", kf_c, v_c)
                S_c = torch.cumsum(kv, dim=2) + S_state.unsqueeze(2)
                Z_c = (torch.cumsum(kf_c, dim=2) + Z_state.unsqueeze(2)).clamp_min(1e-8)

                num = torch.einsum("bhlm,bhlmn->bhln", qf_c, S_c)
                den = torch.einsum("bhlm,bhlm->bhl", qf_c, Z_c).unsqueeze(-1).clamp_min(1e-8)
                u2[:, :, t0:t1, :] = num / den

                S_state = S_c[:, :, -1, :, :]
                Z_state = Z_c[:, :, -1, :]

            u2 = self.slotspace_dropout(u2)
            slot_w = self.slot_out(u2)

            if slot_w_full is not None:
                slot_w_full[:] = slot_w.to(state_dtype)

            if self.slotspace_signed_weights:
                slot_w_eff = torch.tanh(slot_w)
            else:
                slot_w_eff = torch.softmax(slot_w, dim=-1)

            # slotspace-only mask
            if slot_mask_where == "slotspace_only":
                sm_full = self._resolve_slot_mask(
                    slot_mask, B=B, H=H, L=T, K=K,
                    device=x.device, dtype=slot_w_eff.dtype, scope=slot_mask_scope,
                )
                if sm_full is not None:
                    slot_w_eff = slot_w_eff * (sm_full > 0.0).to(slot_w_eff.dtype)
                    if not self.slotspace_signed_weights:
                        slot_w_eff = slot_w_eff / slot_w_eff.sum(dim=-1, keepdim=True).clamp_min(1e-8)

            gate = self._slotspace_gate(dtype=state_dtype, device=x.device).to(state_dtype)

            # second streaming pass: decode delta through slot states
            denom_state2 = torch.zeros((B, H, K), device=x.device, dtype=state_dtype)
            numer_state2 = torch.zeros((B, H, K, d), device=x.device, dtype=state_dtype)
            m_state2 = torch.full((B, H, K), float("-inf"), device=x.device, dtype=state_dtype)

            delta_norm_sum = torch.zeros((), device=x.device, dtype=torch.float32)
            delta_norm_count = 0

            for t0 in range(0, T, WRITE_CHUNK):
                t1 = min(T, t0 + WRITE_CHUNK)
                Lc = t1 - t0

                wlog_c = write_logits[:, :, :, t0:t1]
                m_c, _ = torch.cummax(wlog_c, dim=-1)
                m_new = torch.maximum(m_state2.unsqueeze(-1), m_c)

                scale = torch.exp(m_state2.unsqueeze(-1) - m_new)
                denom_c = denom_state2.unsqueeze(-1) * scale
                numer_c = numer_state2.unsqueeze(-2) * scale.unsqueeze(-1)

                w_new = self._safe_exp_sub_max(wlog_c, m_new)
                denom_c = denom_c + torch.cumsum(w_new, dim=-1)

                v_c = v_write[:, :, t0:t1, :].to(state_dtype)
                add = torch.cumsum(w_new.unsqueeze(-1) * v_c.unsqueeze(2), dim=-2)
                numer_c = numer_c + add

                slot_state_c = numer_c / denom_c.clamp_min(1e-8).unsqueeze(-1)
                slot_state_t2 = slot_state_c.permute(0, 1, 3, 2, 4).contiguous()

                slot_w_c = slot_w_eff[:, :, t0:t1, :].to(state_dtype)
                delta_c = torch.einsum("bhlk,bhlkd->bhld", slot_w_c, slot_state_t2.to(state_dtype))

                delta = gate * delta_c

                if delta_full is not None:
                    delta_full[:, :, t0:t1, :] = delta

                # intervention
                slot_w_for_score = slot_w[:, :, t0:t1, :] if store_slot_w else None
                delta_mod, logs = self._apply_refine_intervention(
                    out1=out_h[:, :, t0:t1, :], delta=delta, slot_w=slot_w_for_score,
                )

                out_h[:, :, t0:t1, :] = out_h[:, :, t0:t1, :] + delta_mod

                # accumulate logs
                if logs is not None and return_info:
                    if intv_logs_acc is None:
                        intv_logs_acc = {}
                        for klog, v in logs.items():
                            if torch.is_tensor(v):
                                vv = v.detach().to(torch.float32)
                                intv_logs_acc[klog] = vv if vv.ndim == 1 else vv.mean()
                        intv_logs_count = 1
                    else:
                        for klog, v in logs.items():
                            if torch.is_tensor(v) and klog in intv_logs_acc:
                                vv = v.detach().to(torch.float32)
                                intv_logs_acc[klog] = intv_logs_acc[klog] + (vv if vv.ndim == 1 else vv.mean())
                        intv_logs_count += 1

                delta_norm_sum = delta_norm_sum + delta.detach().to(torch.float32).norm(dim=-1).sum()
                delta_norm_count += B * H * Lc

                m_state2 = m_new[:, :, :, -1]
                denom_state2 = denom_c[:, :, :, -1]
                numer_state2 = numer_c[:, :, :, -1, :]

            slotspace_delta_norm_mean = (delta_norm_sum / max(1, delta_norm_count)).detach().cpu()

        # ================================================================
        # OUTPUT
        # ================================================================
        out = out_h.transpose(1, 2).contiguous().view(B, T, C)
        out = self.out_proj(out)
        out = self.dropout(out)

        # ---- info dict ----
        info = None
        if return_info:
            info = {
                "content_read_gamma": content_read_gamma.detach().to(torch.float32).cpu(),
                "routing_mode": routing_mode,
                "slot_mask_where": slot_mask_where,
                "slot_mask_scope": slot_mask_scope,
                "intv_mode": getattr(self, "_intv_mode", "off"),
            }

            if alibi_bias_applied is not None and info_level == "full":
                info["alibi_bias_applied"] = self._store_tensor(alibi_bias_applied.to(torch.float32), cfg=info_cfg, kind="other")

            if self.use_alibi_write and self.learn_alibi_strength:
                info["alibi_strength"] = self._alibi_strength(dtype=torch.float32, device=x.device).detach().cpu()

            if self.use_slotspace_refine:
                info["slotspace_gate"] = self._slotspace_gate(dtype=torch.float32, device=x.device).detach().cpu()
                info["use_rope_slotspace"] = torch.tensor(bool(self.use_rope_slotspace))
                if slotspace_delta_norm_mean is not None:
                    info["slotspace_delta_norm"] = slotspace_delta_norm_mean

            # read weights
            if store_read_weights and read_weights is not None:
                info["read_weights"] = self._store_tensor(read_weights, cfg=info_cfg, kind="bhtk")
            else:
                info["read_weights"] = None

            # slot state norm
            if store_slot_norm and slot_state_norm_t is not None:
                s = slot_state_norm_t.permute(0, 1, 3, 2).contiguous()
                info["slot_state_norm"] = self._store_tensor(s, cfg=info_cfg, kind="bhkt")
            else:
                info["slot_state_norm"] = None

            # read logits
            if store_read_logits and read_logits_full is not None:
                info["read_logits"] = self._store_tensor(read_logits_full.to(torch.float32), cfg=info_cfg, kind="bhtk")
                info["read_logits_key"] = self._store_tensor(read_logits_key_full.to(torch.float32), cfg=info_cfg, kind="bhtk")
                info["read_logits_content"] = (
                    self._store_tensor(read_logits_content_full.to(torch.float32), cfg=info_cfg, kind="bhtk")
                    if read_logits_content_full is not None else None
                )
            else:
                info["read_logits"] = info["read_logits_key"] = info["read_logits_content"] = None

            # write logits
            if store_write_logits and info_level == "full":
                info["write_logits_raw"] = self._store_tensor(write_logits_raw, cfg=info_cfg, kind="bhkt")
                info["write_logits"] = self._store_tensor(write_logits.to(torch.float32), cfg=info_cfg, kind="bhkt")
            else:
                info["write_logits_raw"] = info["write_logits"] = None

            # out1 / delta / slot_w
            info["out1"]   = self._store_tensor(out1_full.to(torch.float32), cfg=info_cfg, kind="other") if out1_full is not None else None
            info["delta"]  = self._store_tensor(delta_full.to(torch.float32), cfg=info_cfg, kind="other") if delta_full is not None else None
            info["slot_w"] = self._store_tensor(slot_w_full.to(torch.float32), cfg=info_cfg, kind="bhtk") if slot_w_full is not None else None

            # averaged intervention / geometry logs
            if intv_logs_acc is not None and intv_logs_count > 0:
                for klog, v in intv_logs_acc.items():
                    info[klog] = (v / float(intv_logs_count)).detach().cpu()

                # backward-compatible scalar aliases
                for alias_from, alias_to in [
                    ("score", "intv_score_mean"), ("mask", "intv_mask_mean"),
                    ("tau", "intv_tau"), ("alpha", "intv_alpha_mean"),
                    ("out1_norm", "intv_out1_norm_mean"),
                    ("dpar_norm", "intv_dpar_norm_mean"),
                    ("dorth_norm", "intv_dorth_norm_mean"),
                ]:
                    if alias_from in intv_logs_acc:
                        val = info.get(alias_from)
                        if torch.is_tensor(val) and val.ndim != 1:
                            info[alias_to] = val

        return out, info


# Addressed State Models (ASM): Config + Block + LM
#
# Unified companion for the consolidated AddressedStateAttention harness.
# Block.forward() and LM.forward() pass through the full ASA forward() surface:
#   routing controls, slot mask, info_level, info_cfg.
#
# ============================================================================
# Config
# ============================================================================
@dataclass
class ASMTrainConfig:
    # Data
    dataset_name: str = "wikitext"
    dataset_config: str = "wikitext-103-raw-v1"
    tokenizer_name: str = "gpt2"

    max_seq_len: int = 256
    stride_frac_val: float = 0.50
    seed: int = 1337

    micro_batch_size: int = 2
    grad_accum_steps: int = 8
    train_samples_target: int = 100_000_000
    val_samples_target: int = 25_000

    # Training
    batch_size: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    betas: Tuple[float, float] = (0.9, 0.95)
    grad_clip: float = 1.0
    warmup_steps: int = 1_000
    total_steps: int = 75_000
    eval_interval: int = 1_000
    log_interval: int = 100

    # Model
    vocab_size: int = 50257
    embed_dim: int = 384
    num_layers: int = 23
    num_heads: int = 8
    num_slots: int = 32
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    tie_weights: bool = True

    # ASA / numerics
    read_temperature: float = 1.0
    write_temperature: float = 1.0
    slot_dropout: float = 0.05
    state_fp32: bool = True
    normalize_k: bool = False

    # Positions
    use_abs_pos: bool = False
    use_rope_keys: bool = True
    rope_base: float = 10000.0
    use_alibi_write: bool = True
    alibi_strength_init: float = 0.1
    learn_alibi_strength: bool = True
    min_strength: float = 0.0

    # Content-conditioned read (gamma)
    use_content_read: bool = True
    content_read_init: float = -4.0
    content_read_max_gamma: float = 3.0

    # Slot-space refinement
    use_slotspace_refine: bool = True
    slotspace_dim: int = 64
    slotspace_gate_init: float = -4.0
    slotspace_dropout: float = 0.05
    slotspace_signed_weights: bool = True

    # RoPE inside slot-space matcher
    use_rope_slotspace: bool = True
    rope_base_slotspace: float = 100000.0

    # Perf knobs
    write_chunk_size: int = 128
    slotspace_chunk_size: int = 128
    enable_compiled: bool = False

    # Analytics
    eval_max_batches: int = 150
    analytics_last_k: int = 32

    # IO / caches
    output_dir: str = "./drive/MyDrive/asm_outputs"
    tag: str = "asm_wikitext"
    cache_dir: str = "./drive/MyDrive/asm_caches"
    val_windows_cache: str = "./drive/MyDrive/asm_val_cache_windows_1024.pkl"


# ============================================================================
# Block
# ============================================================================
class ASMBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_slots: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        # temperatures / numerics
        read_temperature: float = 1.0,
        write_temperature: float = 1.0,
        state_fp32: bool = True,
        slot_dropout: float = 0.0,
        normalize_k: bool = False,
        # positions
        use_rope_keys: bool = True,
        rope_base: float = 10000.0,
        use_alibi_write: bool = True,
        # ALiBi
        alibi_strength_init: float = 0.1,
        learn_alibi_strength: bool = True,
        min_strength: float = 0.0,
        # content-conditioned read (gamma)
        use_content_read: bool = True,
        content_read_init: float = -4.0,
        content_read_max_gamma: float = 3.0,
        # slot-space refinement
        use_slotspace_refine: bool = True,
        slotspace_dim: int = 32,
        slotspace_gate_init: float = -10.0,
        slotspace_dropout: float = 0.0,
        slotspace_signed_weights: bool = True,
        # RoPE inside slot-space matcher
        use_rope_slotspace: bool = True,
        rope_base_slotspace: float = 100000.0,
        # chunk sizes
        write_chunk_size: int = 128,
        slotspace_chunk_size: int = 128,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)

        self.asa = AddressedStateAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_slots=num_slots,
            dropout=dropout,
            read_temperature=read_temperature,
            write_temperature=write_temperature,
            state_fp32=state_fp32,
            slot_dropout=slot_dropout,
            normalize_k=normalize_k,
            use_rope_keys=use_rope_keys,
            rope_base=rope_base,
            use_alibi_write=use_alibi_write,
            alibi_strength_init=alibi_strength_init,
            learn_alibi_strength=learn_alibi_strength,
            min_strength=min_strength,
            use_content_read=use_content_read,
            content_read_init=content_read_init,
            content_read_max_gamma=content_read_max_gamma,
            use_slotspace_refine=use_slotspace_refine,
            slotspace_dim=slotspace_dim,
            slotspace_gate_init=slotspace_gate_init,
            slotspace_dropout=slotspace_dropout,
            slotspace_signed_weights=slotspace_signed_weights,
            use_rope_slotspace=use_rope_slotspace,
            rope_base_slotspace=rope_base_slotspace,
            write_chunk_size=write_chunk_size,
            slotspace_chunk_size=slotspace_chunk_size,
        )

        self.norm2 = nn.LayerNorm(embed_dim)
        hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, embed_dim, bias=False),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_info: bool = False,
        # routing
        routing_mode: str = "softmax",
        routing_topk: int = 2,
        read_weights_override: Optional[torch.Tensor] = None,
        routing_noise: Optional[str] = None,
        routing_noise_scale: float = 1.0,
        # slot mask
        slot_mask: Optional[torch.Tensor] = None,
        slot_mask_where: str = "read",
        slot_mask_scope: str = "all",
        # info controls
        info_level: str = "full",
        info_cfg: Optional[Dict] = None,
    ):
        a, info = self.asa(
            self.norm1(x),
            attention_mask=attention_mask,
            return_info=return_info,
            routing_mode=routing_mode,
            routing_topk=routing_topk,
            read_weights_override=read_weights_override,
            routing_noise=routing_noise,
            routing_noise_scale=routing_noise_scale,
            slot_mask=slot_mask,
            slot_mask_where=slot_mask_where,
            slot_mask_scope=slot_mask_scope,
            info_level=info_level,
            info_cfg=info_cfg,
        )
        x = x + a
        x = x + self.mlp(self.norm2(x))
        return x, info


# ============================================================================
# LM
# ============================================================================
class ASMLanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 384,
        num_layers: int = 6,
        num_heads: int = 8,
        num_slots: int = 8,
        max_seq_len: int = 1024,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        # temperatures / numerics
        read_temperature: float = 1.0,
        write_temperature: float = 1.0,
        state_fp32: bool = True,
        slot_dropout: float = 0.05,
        normalize_k: bool = False,
        tie_weights: bool = True,
        # LM-level abs pos
        use_abs_pos: bool = False,
        # positions
        use_rope_keys: bool = True,
        rope_base: float = 10000.0,
        use_alibi_write: bool = True,
        # ALiBi
        alibi_strength_init: float = 0.1,
        learn_alibi_strength: bool = True,
        min_strength: float = 0.0,
        # content-conditioned read (gamma)
        use_content_read: bool = True,
        content_read_init: float = -4.0,
        content_read_max_gamma: float = 3.0,
        # slot-space refinement
        use_slotspace_refine: bool = True,
        slotspace_dim: int = 32,
        slotspace_gate_init: float = -10.0,
        slotspace_dropout: float = 0.0,
        slotspace_signed_weights: bool = True,
        # RoPE inside slot-space matcher
        use_rope_slotspace: bool = True,
        rope_base_slotspace: float = 100000.0,
        # chunk sizes
        write_chunk_size: int = 128,
        slotspace_chunk_size: int = 128,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.use_abs_pos = bool(use_abs_pos)

        self.tok = nn.Embedding(vocab_size, embed_dim)
        self.pos = nn.Embedding(max_seq_len, embed_dim) if self.use_abs_pos else None
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            ASMBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_slots=num_slots,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                read_temperature=read_temperature,
                write_temperature=write_temperature,
                state_fp32=state_fp32,
                slot_dropout=slot_dropout,
                normalize_k=normalize_k,
                use_rope_keys=use_rope_keys,
                rope_base=rope_base,
                use_alibi_write=use_alibi_write,
                alibi_strength_init=alibi_strength_init,
                learn_alibi_strength=learn_alibi_strength,
                min_strength=min_strength,
                use_content_read=use_content_read,
                content_read_init=content_read_init,
                content_read_max_gamma=content_read_max_gamma,
                use_slotspace_refine=use_slotspace_refine,
                slotspace_dim=slotspace_dim,
                slotspace_gate_init=slotspace_gate_init,
                slotspace_dropout=slotspace_dropout,
                slotspace_signed_weights=slotspace_signed_weights,
                use_rope_slotspace=use_rope_slotspace,
                rope_base_slotspace=rope_base_slotspace,
                write_chunk_size=write_chunk_size,
                slotspace_chunk_size=slotspace_chunk_size,
            )
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)
        if tie_weights:
            self.lm_head.weight = self.tok.weight

        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_info: bool = False,
        # routing
        routing_mode: str = "softmax",
        routing_topk: int = 2,
        read_weights_override: Optional[torch.Tensor] = None,
        routing_noise: Optional[str] = None,
        routing_noise_scale: float = 1.0,
        # slot mask
        slot_mask: Optional[torch.Tensor] = None,
        slot_mask_where: str = "read",
        slot_mask_scope: str = "all",
        # info controls
        info_level: str = "full",
        info_cfg: Optional[Dict] = None,
    ):
        B, T = input_ids.shape

        x = self.tok(input_ids)
        if self.use_abs_pos:
            pos = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, -1)
            x = x + self.pos(pos)
        x = self.drop(x)

        infos: List[Optional[Dict[str, torch.Tensor]]] = []
        for blk in self.blocks:
            x, info = blk(
                x,
                attention_mask=attention_mask,
                return_info=return_info,
                routing_mode=routing_mode,
                routing_topk=routing_topk,
                read_weights_override=read_weights_override,
                routing_noise=routing_noise,
                routing_noise_scale=routing_noise_scale,
                slot_mask=slot_mask,
                slot_mask_where=slot_mask_where,
                slot_mask_scope=slot_mask_scope,
                info_level=info_level,
                info_cfg=info_cfg,
            )
            if return_info:
                infos.append(info)

        x = self.norm(x)
        logits = self.lm_head(x)
        return (logits, infos) if return_info else logits


# ============================================================================
# Convenience: build model from config
# ============================================================================
def build_model_from_cfg(cfg: ASMTrainConfig) -> ASMLanguageModel:
    return ASMLanguageModel(
        vocab_size=cfg.vocab_size,
        embed_dim=cfg.embed_dim,
        num_layers=cfg.num_layers,
        num_heads=cfg.num_heads,
        num_slots=cfg.num_slots,
        max_seq_len=cfg.max_seq_len,
        mlp_ratio=cfg.mlp_ratio,
        dropout=cfg.dropout,
        read_temperature=cfg.read_temperature,
        write_temperature=cfg.write_temperature,
        state_fp32=cfg.state_fp32,
        slot_dropout=cfg.slot_dropout,
        normalize_k=cfg.normalize_k,
        tie_weights=cfg.tie_weights,
        use_abs_pos=cfg.use_abs_pos,
        use_rope_keys=cfg.use_rope_keys,
        rope_base=cfg.rope_base,
        use_alibi_write=cfg.use_alibi_write,
        alibi_strength_init=cfg.alibi_strength_init,
        learn_alibi_strength=cfg.learn_alibi_strength,
        min_strength=cfg.min_strength,
        use_content_read=cfg.use_content_read,
        content_read_init=cfg.content_read_init,
        content_read_max_gamma=cfg.content_read_max_gamma,
        use_slotspace_refine=cfg.use_slotspace_refine,
        slotspace_dim=cfg.slotspace_dim,
        slotspace_gate_init=cfg.slotspace_gate_init,
        slotspace_dropout=cfg.slotspace_dropout,
        slotspace_signed_weights=cfg.slotspace_signed_weights,
        use_rope_slotspace=cfg.use_rope_slotspace,
        rope_base_slotspace=cfg.rope_base_slotspace,
        write_chunk_size=cfg.write_chunk_size,
        slotspace_chunk_size=cfg.slotspace_chunk_size,
    )
