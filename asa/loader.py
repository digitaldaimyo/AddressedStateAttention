

"""
Universal Checkpoint Loader for ASA Models

Loads checkpoints into either training or analysis harness.

Repository: https://github.com/DigitalDaimyo/AddressedStateAttention
"""

import torch
from typing import Literal, Tuple, Dict, Any


__all__ = ['load_asm_checkpoint']


def load_asm_checkpoint(
    checkpoint_path: str,
    mode: Literal["train", "analysis"] = "train",
    device: str = None
) -> Tuple[Any, Any, Dict]:
    """
    Universal ASM checkpoint loader.
    
    Args:
        checkpoint_path: Path to .pt checkpoint file
        mode: "train" (efficient) or "analysis" (intervention harness)  
        device: Device to load on (defaults to cuda if available)
        
    Returns:
        model: Loaded ASMLanguageModel
        cfg: ASMTrainConfig object
        ckpt: Full checkpoint dict (for step, loss metadata)
        
    Example:
        >>> model, cfg, ckpt = load_asm_checkpoint(
        ...     "best.pt", mode="analysis", device="cuda"
        ... )
        >>> print(f"Step {ckpt['step']}, Loss {ckpt['val_loss']:.3f}")
    """    

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    
    cfg_dict = ckpt.get("cfg")
    if cfg_dict is None:
        raise KeyError(f"Missing 'cfg' key. Available: {list(ckpt.keys())}")
    
    # Import appropriate harness
    if mode == "train":
        from .training import ASMTrainConfig, build_model_from_cfg
    else:  # analysis
        from .analysis import ASMTrainConfig, build_model_from_cfg
    
    # Build model using helper
    cfg = ASMTrainConfig(**cfg_dict)
    model = build_model_from_cfg(cfg)
    
    # Load weights
    state_dict = ckpt.get("model")
    if state_dict is None:
        raise KeyError(f"Missing 'model' key. Available: {list(ckpt.keys())}")
    
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    
    if missing:
        print(f"⚠ Missing keys: {len(missing)}")
    if unexpected:
        print(f"⚠ Unexpected keys: {len(unexpected)}")
    
    model = model.to(device).eval()
    
    return model, cfg, ckpt
