"""
Text Generation Utilities for ASA Models

Simple, dependency-free text generation with common decoding strategies.

Repository: https://github.com/DigitalDaimyo/AddressedStateAttention
"""

import torch
import torch.nn.functional as F
from typing import Optional, Set, Tuple, List


__all__ = ['generate']


def _forward_logits(model, input_ids, attention_mask=None):
    """Extract logits from various model output formats."""
    out = model(input_ids, attention_mask=attention_mask) if attention_mask is not None else model(input_ids)
    
    if isinstance(out, torch.Tensor):
        return out
    if isinstance(out, (tuple, list)):
        return out[0]
    if isinstance(out, dict):
        for key in ["logits", "out", "y", "pred"]:
            if key in out:
                return out[key]
    raise TypeError(f"Unrecognized model output type: {type(out)}")


def _apply_repetition_penalty(logits: torch.Tensor, input_ids: torch.Tensor, penalty: float):
    """Apply repetition penalty to logits (GPT-2 style)."""
    if penalty is None or penalty == 1.0:
        return logits
    
    B = logits.size(0)
    for b in range(B):
        prev_tokens = torch.unique(input_ids[b])
        l = logits[b, prev_tokens]
        logits[b, prev_tokens] = torch.where(l < 0, l * penalty, l / penalty)
    return logits


def _top_k_top_p_filtering(
    logits: torch.Tensor, 
    top_k: int = 0, 
    top_p: float = 1.0, 
    min_tokens_to_keep: int = 1
):
    """Filter logits using top-k and nucleus (top-p) filtering."""
    B, V = logits.shape
    top_k = int(top_k) if top_k is not None else 0
    top_p = float(top_p) if top_p is not None else 1.0

    if top_k > 0 and top_k < V:
        kth = torch.topk(logits, top_k, dim=-1).values[:, -1].unsqueeze(-1)
        logits = logits.masked_fill(logits < kth, float("-inf"))

    if top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        probs = F.softmax(sorted_logits, dim=-1)
        cum = probs.cumsum(dim=-1)

        remove = cum > top_p
        if min_tokens_to_keep > 1:
            remove[:, :min_tokens_to_keep] = False
        remove = torch.cat([
            torch.zeros((B, 1), device=logits.device, dtype=torch.bool), 
            remove[:, :-1]
        ], dim=-1)

        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        logits = torch.full_like(logits, float("-inf"))
        logits.scatter_(dim=-1, index=sorted_idx, src=sorted_logits)

    return logits


def _update_seen_ngrams(seen: Set, tokens: List[int], n: int):
    """Add n-gram to seen set."""
    if n > 0 and len(tokens) >= n:
        seen.add(tuple(tokens[-n:]))


def _seed_seen_ngrams(input_ids: torch.Tensor, n: int) -> Set:
    """Initialize seen n-grams from input."""
    seen = set()
    if n <= 0:
        return seen
    tokens = input_ids[0].tolist()
    if len(tokens) >= n:
        for i in range(len(tokens) - n + 1):
            seen.add(tuple(tokens[i:i+n]))
    return seen


def _banned_from_seen(seen: Set, input_ids: torch.Tensor, n: int) -> Set:
    """Get tokens banned by n-gram constraint."""
    if n <= 0 or input_ids.shape[1] < n - 1:
        return set()
    
    prefix = tuple(input_ids[0, -(n - 1):].tolist())
    banned = set()
    for ng in seen:
        if ng[:-1] == prefix:
            banned.add(ng[-1])
    return banned


@torch.no_grad()
def generate(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 120,
    max_seq_len: int = 1024,
    strategy: str = "sample",
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 0.9,
    repetition_penalty: float = 1.0,
    no_repeat_ngram_size: int = 0,
    eos_token_id: Optional[int] = None,
    device: str = "cuda",
) -> str:
    """
    Generate text from a prompt using various decoding strategies.
    
    Args:
        model: ASA language model
        tokenizer: HuggingFace tokenizer
        prompt: Input text prompt
        max_new_tokens: Maximum tokens to generate
        max_seq_len: Maximum sequence length (truncates context if exceeded)
        strategy: "greedy" or "sample"
        temperature: Sampling temperature (higher = more random)
        top_k: Keep only top k tokens (0 = disabled)
        top_p: Nucleus sampling threshold (1.0 = disabled)
        repetition_penalty: Penalty for repeating tokens (1.0 = disabled)
        no_repeat_ngram_size: Block repeating n-grams (0 = disabled)
        eos_token_id: Stop generation at this token
        device: Device to run on
        
    Returns:
        Generated text (including prompt)
        
    Example:
        >>> text = generate(
        ...     model, tokenizer,
        ...     prompt="The capital of France is",
        ...     max_new_tokens=20,
        ...     strategy="greedy"
        ... )
    """
    model.eval()
    
    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc.input_ids.to(device)
    
    if eos_token_id is None:
        eos_token_id = tokenizer.eos_token_id
    
    seen = _seed_seen_ngrams(input_ids, no_repeat_ngram_size)
    
    for _ in range(max_new_tokens):
        # Truncate if exceeding context length
        if input_ids.shape[1] > max_seq_len:
            input_ids = input_ids[:, -max_seq_len:]
            seen = _seed_seen_ngrams(input_ids, no_repeat_ngram_size)
        
        logits = _forward_logits(model, input_ids)
        next_logits = logits[:, -1, :].to(torch.float32).clone()
        
        # Apply repetition penalty
        next_logits = _apply_repetition_penalty(next_logits, input_ids, repetition_penalty)
        
        # Block repeated n-grams
        banned = _banned_from_seen(seen, input_ids, no_repeat_ngram_size)
        if banned:
            next_logits[0, list(banned)] = float("-inf")
        
        # Decode strategy
        if strategy == "greedy":
            next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
        elif strategy == "sample":
            temp = max(1e-6, float(temperature))
            next_logits = next_logits / temp
            next_logits = _top_k_top_p_filtering(next_logits, top_k=top_k, top_p=top_p)
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            raise ValueError(f"Unknown strategy '{strategy}'. Use 'greedy' or 'sample'.")
        
        input_ids = torch.cat([input_ids, next_token], dim=1)
        
        # Update n-gram tracking
        tokens = input_ids[0].tolist()
        _update_seen_ngrams(seen, tokens, no_repeat_ngram_size)
        
        # Check for EOS
        if eos_token_id is not None and next_token.item() == eos_token_id:
            break
    
    return tokenizer.decode(input_ids[0], skip_special_tokens=False)
