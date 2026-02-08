
"""
Addressed State Attention (ASA) - Training Harness

Efficient implementation optimized for language model training.
For mechanistic analysis and interventions, use asm_analysis.py instead.

Repository: https://github.com/DigitalDaimyo/AddressedStateAttention
Paper:  https://github.com/DigitalDaimyo/AddressedStateAttention/paper_drafts
"""

import math
from dataclasses import dataclass
from typing import Optional, Dict, Tuple

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

# -------------------------
# RoPE helper (rotate-half)
# -------------------------
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
        if self._t_cached == T and self._cos_cached is not None and self._device_cached == device:
            return self._cos_cached.to(dtype=dtype), self._sin_cached.to(dtype=dtype)

        t = torch.arange(T, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("t,f->tf", t, self.inv_freq)  # [T, d/2]
        emb = torch.cat([freqs, freqs], dim=-1)            # [T, d]
        cos = emb.cos()[None, None, :, :]                  # [1,1,T,d]
        sin = emb.sin()[None, None, :, :]                  # [1,1,T,d]

        self._t_cached = T
        self._device_cached = device
        self._cos_cached = cos
        self._sin_cached = sin
        return cos.to(dtype=dtype), sin.to(dtype=dtype)

def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return (x * cos) + (_rotate_half(x) * sin)

# -------------------------
# ALiBi slopes helper
# -------------------------
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

class AddressedStateAttention(nn.Module):
    """
    ASA with integral slotspace refine fused into the compiled chunk kernel.
    Fixes included:
      (1) pad slotspace RoPE cos/sin to CH (identity on padded positions)
      (2) build valid_mask_c even when attention_mask is None (padding-only)
      (3) pad write logits with -inf (so padded positions contribute zero to scan)
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 12,
        num_slots: int = 16,
        dropout: float = 0.1,

        # temps / numerics
        read_temperature: float = 1.0,
        write_temperature: float = 1.0,
        state_fp32: bool = True,
        slot_dropout: float = 0.0,
        normalize_k: bool = False,

        # write geometry
        use_rope_keys: bool = True,
        rope_base: float = 10000.0,

        # write bias
        use_alibi_write: bool = True,
        alibi_strength_init: float = 0.1,
        learn_alibi_strength: bool = True,
        min_strength: float = 0.0,

        # content read gamma
        use_content_read: bool = True,
        content_read_init: float = -4.0,
        content_read_max_gamma: float = 3.0,

        # slotspace refine (INTEGRAL)
        use_slotspace_refine: bool = True,  # compat only
        slotspace_dim: int = 8,
        slotspace_gate_init: float = -4.0,
        slotspace_dropout: float = 0.05,
        slotspace_signed_weights: bool = True,

        # slotspace RoPE (Q/K only)
        use_rope_slotspace: bool = True,
        rope_base_slotspace: float = 100000.0,

        # perf
        write_chunk_size: int = 1024,
        enable_compiled: bool = True,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0
        assert (slotspace_dim % 2) == 0, "slotspace_dim must be even if RoPE enabled"

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

        self.use_rope_keys = bool(use_rope_keys)
        self.use_alibi_write = bool(use_alibi_write)
        self.learn_alibi_strength = bool(learn_alibi_strength)
        self.min_strength = float(min_strength)

        self.use_content_read = bool(use_content_read)
        self.content_read_max_gamma = float(content_read_max_gamma)

        self.slotspace_dim = int(slotspace_dim)
        self.slotspace_dropout = nn.Dropout(float(slotspace_dropout))
        self.slotspace_signed_weights = bool(slotspace_signed_weights)

        self.use_rope_slotspace = bool(use_rope_slotspace)
        self.write_chunk_size = int(write_chunk_size)

        H, K, d = self.num_heads, self.num_slots, self.head_dim
        M = self.slotspace_dim

        self.slot_keys = nn.Parameter(torch.randn(H, K, d) / math.sqrt(d))

        self.Wk_write = nn.Linear(embed_dim, embed_dim, bias=False)
        self.Wv_write = nn.Linear(embed_dim, embed_dim, bias=False)
        self.Wq_read  = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self.rope = RotaryEmbedding(d, base=rope_base) if self.use_rope_keys else None

        if self.use_alibi_write:
            self.register_buffer("_alibi_slopes", alibi_slopes(H), persistent=False)
        else:
            self.register_buffer("_alibi_slopes", torch.zeros(H), persistent=False)

        if self.use_alibi_write and self.learn_alibi_strength:
            init = torch.tensor(float(alibi_strength_init) - self.min_strength).clamp_min(1e-8)
            self._alibi_strength_param = nn.Parameter(_inv_softplus(init))
        else:
            self._alibi_strength_param = None
            self.alibi_strength = float(alibi_strength_init)

        if self.use_content_read:
            self._content_read_gamma_raw = nn.Parameter(torch.tensor(float(content_read_init)))
        else:
            self._content_read_gamma_raw = None

        self.slot_in  = nn.Linear(K, M, bias=False)
        self.slot_q   = nn.Linear(M, M, bias=False)
        self.slot_k   = nn.Linear(M, M, bias=False)
        self.slot_v   = nn.Linear(M, M, bias=False)
        self.slot_out = nn.Linear(M, K, bias=False)

        self._slotspace_gate_raw = nn.Parameter(torch.tensor(float(slotspace_gate_init)))

        self.rope_slotspace = RotaryEmbedding(M, base=float(rope_base_slotspace)) if self.use_rope_slotspace else None

        self._compiled = None
        if enable_compiled:
            self.enable_compiled_kernel()

    def enable_compiled_kernel(self):
        if self._compiled is None:
            self._compiled = torch.compile(self._asa_chunk_fused, dynamic=False, fullgraph=False)

    def _alibi_strength(self, dtype, device) -> torch.Tensor:
        if not (self.use_alibi_write and self.learn_alibi_strength):
            return torch.tensor(getattr(self, "alibi_strength", 0.0), dtype=dtype, device=device)
        return (F.softplus(self._alibi_strength_param) + self.min_strength).to(dtype=dtype, device=device)

    def _content_read_gamma(self, dtype, device) -> torch.Tensor:
        if not self.use_content_read:
            return torch.tensor(0.0, dtype=dtype, device=device)
        g = F.softplus(self._content_read_gamma_raw)
        if self.content_read_max_gamma is not None and self.content_read_max_gamma > 0:
            g = g.clamp(max=self.content_read_max_gamma)
        return g.to(dtype=dtype, device=device)

    def _slotspace_gate(self, dtype, device) -> torch.Tensor:
        return F.softplus(self._slotspace_gate_raw).to(dtype=dtype, device=device)

    @staticmethod
    def _safe_exp_sub_max(s: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        diff = s - m
        diff = diff.masked_fill(~torch.isfinite(m), float("-inf"))
        return torch.exp(diff)

    @staticmethod
    def _phi(x: torch.Tensor) -> torch.Tensor:
        return F.elu(x) + 1.0

    @staticmethod
    def _pad_time_slice(x: torch.Tensor, t0: int, L: int, CH: int, dim: int):
        sl = x.narrow(dim, t0, L)
        if L == CH:
            return sl, None
        pad_shape = list(sl.shape)
        pad_shape[dim] = CH - L
        pad = torch.zeros(pad_shape, device=sl.device, dtype=sl.dtype)
        xpad = torch.cat([sl, pad], dim=dim)
        mask = torch.zeros((CH,), device=sl.device, dtype=torch.bool)
        mask[:L] = True
        return xpad, mask

    def _asa_chunk_fused(
        self,
        wlog_c: torch.Tensor,         # [B,H,K,CH]
        v_c: torch.Tensor,            # [B,H,CH,d]
        q_c: torch.Tensor,            # [B,H,CH,d]
        slot_keys_dk: torch.Tensor,   # [1,H,d,K]
        pos_cos_s: Optional[torch.Tensor],  # [1,1,CH,M] or None
        pos_sin_s: Optional[torch.Tensor],  # [1,1,CH,M] or None
        content_gamma: torch.Tensor,
        rtemp_t: torch.Tensor,
        gate_t: torch.Tensor,
        m_state: torch.Tensor,        # [B,H,K]
        denom_state: torch.Tensor,    # [B,H,K]
        numer_state: torch.Tensor,    # [B,H,K,d]
        S_state: torch.Tensor,        # [B,H,M,M]
        Z_state: torch.Tensor,        # [B,H,M]
        valid_mask_c: Optional[torch.Tensor],  # [B,1,CH,1] or None
        do_dropout: bool,
        dropout_p: float,
        signed_slot_w: bool,
    ):
        B, H, K, CH = wlog_c.shape
        d = numer_state.shape[-1]
        M = S_state.shape[-1]
        inv_sqrt_d = 1.0 / math.sqrt(d)

        # ----- WRITE prefix-softmax scan -----
        m_c, _ = torch.cummax(wlog_c, dim=-1)              # [B,H,K,CH]
        m_new = torch.maximum(m_state.unsqueeze(-1), m_c)  # [B,H,K,CH]
        scale = torch.exp(m_state.unsqueeze(-1) - m_new)   # [B,H,K,CH]

        denom_c = denom_state.unsqueeze(-1) * scale                          # [B,H,K,CH]
        numer_c = numer_state.unsqueeze(-2) * scale.unsqueeze(-1)            # [B,H,K,CH,d]

        w_new = self._safe_exp_sub_max(wlog_c, m_new)                        # [B,H,K,CH]
        denom_c = denom_c + torch.cumsum(w_new, dim=-1)                      # [B,H,K,CH]
        numer_c = numer_c + torch.cumsum(w_new.unsqueeze(-1) * v_c.unsqueeze(2), dim=-2)  # [B,H,K,CH,d]

        # ----- Routing logits -----
        read_logits_key = torch.matmul(q_c, slot_keys_dk) * inv_sqrt_d        # [B,H,CH,K]

        if self.use_content_read:
            numer_for_dot = numer_c.to(q_c.dtype).permute(0, 1, 3, 2, 4)      # [B,H,CH,K,d]
            denom_for_div = denom_c.to(q_c.dtype).permute(0, 1, 3, 2)         # [B,H,CH,K]
            read_logits_content = (q_c.unsqueeze(-2) * numer_for_dot).sum(dim=-1) * inv_sqrt_d
            read_logits_content = read_logits_content / denom_for_div.clamp_min(1e-8)
            read_logits = read_logits_key + content_gamma.to(read_logits_key.dtype) * read_logits_content
        else:
            read_logits = read_logits_key

        read_w = torch.softmax(read_logits / rtemp_t, dim=-1)                # [B,H,CH,K]

        # ----- EXACT base output -----
        inv_denom = (1.0 / denom_c.clamp_min(1e-8)).to(numer_c.dtype)         # [B,H,K,CH]
        w_scaled = read_w.to(numer_c.dtype).permute(0, 1, 3, 2) * inv_denom   # [B,H,K,CH]
        out_base = (w_scaled.unsqueeze(-1) * numer_c).sum(dim=2)              # [B,H,CH,d]

        # ----- Slotspace refine -----
        u = self.slot_in(read_w.to(out_base.dtype))                           # [B,H,CH,M]
        q_s = self.slot_q(u)
        k_s = self.slot_k(u)
        v_s = self.slot_v(u)

        if self.use_rope_slotspace and (pos_cos_s is not None) and (pos_sin_s is not None):
            q_s = apply_rope(q_s, pos_cos_s, pos_sin_s)
            k_s = apply_rope(k_s, pos_cos_s, pos_sin_s)

        if valid_mask_c is not None:
            q_s = q_s * valid_mask_c
            k_s = k_s * valid_mask_c
            v_s = v_s * valid_mask_c

        qf = self._phi(q_s)
        kf = self._phi(k_s)

        kv = kf.unsqueeze(-1) * v_s.unsqueeze(-2)                             # [B,H,CH,M,M]
        S_c = torch.cumsum(kv, dim=2) + S_state.unsqueeze(2)                  # [B,H,CH,M,M]
        Z_c = torch.cumsum(kf, dim=2) + Z_state.unsqueeze(2)                  # [B,H,CH,M]
        Z_c = Z_c.clamp_min(1e-8)

        num = torch.matmul(qf.unsqueeze(-2), S_c).squeeze(-2)                 # [B,H,CH,M]
        den = (qf * Z_c).sum(dim=-1, keepdim=True).clamp_min(1e-8)            # [B,H,CH,1]
        u2 = num / den                                                        # [B,H,CH,M]

        S_state_new = S_c[:, :, -1, :, :]
        Z_state_new = Z_c[:, :, -1, :]

        if do_dropout and dropout_p > 0.0:
            keep = (torch.rand_like(u2) > dropout_p).to(u2.dtype) / (1.0 - dropout_p)
            u2 = u2 * keep

        slot_w = self.slot_out(u2)                                            # [B,H,CH,K]
        if signed_slot_w:
            slot_w = torch.tanh(slot_w)
        else:
            slot_w = torch.softmax(slot_w, dim=-1)

        slot_w_scaled = slot_w.to(numer_c.dtype).permute(0, 1, 3, 2) * inv_denom
        delta = (slot_w_scaled.unsqueeze(-1) * numer_c).sum(dim=2)             # [B,H,CH,d]

        out = out_base + gate_t.to(out_base.dtype) * delta

        m_state_new = m_new[:, :, :, -1]
        denom_state_new = denom_c[:, :, :, -1]
        numer_state_new = numer_c[:, :, :, -1, :]

        return out, read_w, m_state_new, denom_state_new, numer_state_new, S_state_new, Z_state_new

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_info: bool = False,
        return_light_stats: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:

        B, T, C = x.shape
        H, K, d = self.num_heads, self.num_slots, self.head_dim
        M = self.slotspace_dim

        k_write = self.Wk_write(x).reshape(B, T, H, d).transpose(1, 2)  # [B,H,T,d]
        v_write = self.Wv_write(x).reshape(B, T, H, d).transpose(1, 2)  # [B,H,T,d]
        q_read  = self.Wq_read(x).reshape(B, T, H, d).transpose(1, 2)   # [B,H,T,d]

        if self.normalize_k:
            k_write = F.normalize(k_write, dim=-1, eps=1e-8)

        if self.use_rope_keys:
            cos, sin = self.rope.get_cos_sin(T, device=x.device, dtype=k_write.dtype)
            k_write = apply_rope(k_write, cos, sin)

        slot_keys = self.slot_keys
        if self.training and self.slot_dropout > 0.0:
            drop = (torch.rand((H, K), device=x.device) < self.slot_dropout)
            slot_keys = slot_keys * (~drop).to(slot_keys.dtype).unsqueeze(-1)

        slot_keys_dk = slot_keys.transpose(-1, -2).unsqueeze(0).to(q_read.dtype)  # [1,H,d,K]

        write_logits_raw = torch.matmul(k_write.to(q_read.dtype), slot_keys_dk).permute(0, 1, 3, 2) / math.sqrt(d)

        state_dtype = torch.float32 if (self.state_fp32 and x.dtype != torch.float32) else x.dtype
        write_logits = write_logits_raw.to(state_dtype)

        wtemp = max(1e-6, self.write_temperature)
        write_logits = write_logits / wtemp

        if self.use_alibi_write:
            strength = self._alibi_strength(dtype=state_dtype, device=x.device)
            slopes = self._alibi_slopes.to(device=x.device, dtype=state_dtype) * strength
            pos = torch.arange(T, device=x.device, dtype=state_dtype)
            write_logits = write_logits + slopes.view(1, H, 1, 1) * pos.view(1, 1, 1, T)

        valid = None
        if attention_mask is not None:
            valid = attention_mask.to(dtype=torch.bool)
            write_logits = write_logits.masked_fill(~valid.view(B, 1, 1, T), float("-inf"))

        content_gamma = self._content_read_gamma(dtype=q_read.dtype, device=x.device)
        rtemp_t = torch.tensor(max(1e-6, self.read_temperature), device=x.device, dtype=q_read.dtype)
        gate_t = self._slotspace_gate(dtype=state_dtype, device=x.device)

        denom_state = torch.zeros((B, H, K), device=x.device, dtype=state_dtype)
        numer_state = torch.zeros((B, H, K, d), device=x.device, dtype=state_dtype)
        m_state = torch.full((B, H, K), float("-inf"), device=x.device, dtype=state_dtype)

        S_state = torch.zeros((B, H, M, M), device=x.device, dtype=state_dtype)
        Z_state = torch.zeros((B, H, M), device=x.device, dtype=state_dtype)

        out_h = torch.empty((B, H, T, d), device=x.device, dtype=state_dtype)

        if self.use_rope_slotspace:
            cos_s_full, sin_s_full = self.rope_slotspace.get_cos_sin(T, device=x.device, dtype=state_dtype)
        else:
            cos_s_full = sin_s_full = None

        CH = self.write_chunk_size
        kernel = self._compiled if self._compiled is not None else self._asa_chunk_fused

        do_dropout = bool(self.training and self.slotspace_dropout.p > 0.0)
        dropout_p = float(self.slotspace_dropout.p)
        signed_slot_w = bool(self.slotspace_signed_weights)

        for t0 in range(0, T, CH):
            t1 = min(T, t0 + CH)
            L = t1 - t0

            wlog_c, mask = self._pad_time_slice(write_logits, t0, L, CH, dim=3)              # [B,H,K,CH]
            v_c,   _     = self._pad_time_slice(v_write.to(state_dtype), t0, L, CH, dim=2)  # [B,H,CH,d]
            q_c,   _     = self._pad_time_slice(q_read, t0, L, CH, dim=2)                   # [B,H,CH,d]

            # (3) ensure padded write logits contribute zero mass
            if mask is not None:
                wlog_c = wlog_c.clone()
                wlog_c[:, :, :, L:] = float("-inf")

            # (2) build valid_mask_c even when attention_mask is None (padding-only)
            valid_mask_c = None
            if (valid is not None) or (mask is not None):
                if valid is None:
                    vm_pad = mask.view(1, CH).expand(B, CH)  # [B,CH]
                else:
                    if mask is None:
                        vm_pad = valid[:, t0:t1]
                    else:
                        vm = valid[:, t0:t1]
                        vm_pad = torch.zeros((B, CH), device=x.device, dtype=torch.bool)
                        vm_pad[:, :L] = vm
                valid_mask_c = vm_pad.view(B, 1, CH, 1).to(state_dtype)

            # (1) slotspace RoPE slice PADDED TO CH (identity on padded positions)
            if self.use_rope_slotspace:
                cos_slice = cos_s_full[:, :, t0:t1, :]  # [1,1,L,M]
                sin_slice = sin_s_full[:, :, t0:t1, :]  # [1,1,L,M]
                if L == CH:
                    cos_s, sin_s = cos_slice, sin_slice
                else:
                    cos_s = torch.ones((1, 1, CH, M), device=x.device, dtype=state_dtype)
                    sin_s = torch.zeros((1, 1, CH, M), device=x.device, dtype=state_dtype)
                    cos_s[:, :, :L, :] = cos_slice
                    sin_s[:, :, :L, :] = sin_slice
            else:
                cos_s = sin_s = None

            out_c, read_w_c, m_state, denom_state, numer_state, S_state, Z_state = kernel(
                wlog_c, v_c, q_c, slot_keys_dk,
                cos_s, sin_s,
                content_gamma, rtemp_t, gate_t,
                m_state, denom_state, numer_state,
                S_state, Z_state,
                valid_mask_c,
                do_dropout, dropout_p,
                signed_slot_w,
            )

            if mask is not None:
                out_c = out_c * mask.view(1, 1, CH, 1).to(out_c.dtype)

            out_h[:, :, t0:t1, :] = out_c[:, :, :L, :]

        out = out_h.transpose(1, 2).reshape(B, T, C)
        out = self.out_proj(out)
        out = self.dropout(out)

        info = None
        if return_info or return_light_stats:
            info = {
                "content_read_gamma": content_gamma.detach().to(torch.float32).cpu(),
                "slotspace_gate": gate_t.detach().to(torch.float32).cpu(),
            }
        return out, info


# ============================================================================
# Addressed State Models (ASM): Config + Block + LM
# - Naming aligned with paper: slots, read/write, slot-space refinement
# - No compatibility layer (fresh public tooling)
# ============================================================================


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
    # Sample budgets
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

    # Addressed State Attention (ASA) / numerics
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

    # Content-conditioned read term (gamma)
    use_content_read: bool = True
    content_read_init: float = -4.0
    content_read_max_gamma: float = 3.0

    # Optional slot-space refinement (formerly "k-space")
    use_slotspace_refine: bool = True
    slotspace_dim: int = 64
    slotspace_gate_init: float = -4.0
    slotspace_dropout: float = 0.05
    slotspace_signed_weights: bool = True

    # RoPE inside slot-space matcher (Q/K only)
    use_rope_slotspace: bool = True
    rope_base_slotspace: float = 100000.0

    # Perf knobs (behavior-identical)
    write_chunk_size: int = 128
    enable_compiled: bool = True

    # Analytics
    eval_max_batches: int = 150
    analytics_last_k: int = 4

    # IO / caches
    output_dir: str = "./drive/MyDrive/asm_outputs"
    tag: str = "asm_wikitext"
    cache_dir: str = "./drive/MyDrive/asm_caches/fineweb/1B"
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

        # ALiBi params
        alibi_strength_init: float = 0.1,
        learn_alibi_strength: bool = True,
        min_strength: float = 0.0,

        # content-conditioned read (gamma)
        use_content_read: bool = True,
        content_read_init: float = -4.0,
        content_read_max_gamma: float = 3.0,

        # optional slot-space refinement
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
        enable_compiled: bool = False,
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
            enable_compiled=enable_compiled,

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

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, return_info: bool = False, return_light_stats: Optional[bool] = None):
        a, info = self.asa(self.norm1(x), attention_mask=attention_mask, return_info=return_info, return_light_stats=return_light_stats)
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

        # optional slot-space refinement
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
        enable_compiled: bool = False,
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
                slotspace_dim=slotspace_dim,            slotspace_gate_init=slotspace_gate_init,
                slotspace_dropout=slotspace_dropout,
                slotspace_signed_weights=slotspace_signed_weights,
                use_rope_slotspace=use_rope_slotspace,
                rope_base_slotspace=rope_base_slotspace,

                write_chunk_size=write_chunk_size,
                enable_compiled=enable_compiled,
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
        return_light_stats: Optional[bool] = None,
    ):
        B, T = input_ids.shape
        assert T <= self.max_seq_len, f"T={T} exceeds max_seq_len={self.max_seq_len}"

        x = self.tok(input_ids)
        if self.use_abs_pos:
            pos = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, -1)
            x = x + self.pos(pos)

        x = self.drop(x)

        infos = []
        for blk in self.blocks:
            x, info = blk(x, attention_mask=attention_mask, return_info=return_info, return_light_stats=return_light_stats)
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
        enable_compiled=cfg.enable_compiled,
    )
