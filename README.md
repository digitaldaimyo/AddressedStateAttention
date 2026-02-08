# Addressed State Attention (ASA)

**Interpretable slot-based attention achieving competitive language modeling performance**

📊 **187M params:** 3.73 val loss / 41.6 PPL (FineWeb, 75k steps)  
🔬 **Mechanistic:** Refinement = directional suppression (α = -0.81)  
⚡ **Efficient:** O(T·K) vs O(T²) standard attention (K=16 slots)

[

![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)

](link)
[Paper](paper_drafts/ASA_Mechanistic.pdf) | [Weights (HF)](https://huggingface.co/DigitalDaimyo/AddressedStateAttention) | [Demo](#quick-start)

---

## Quick Start

```python
# In Colab: download from GitHub
!wget https://raw.githubusercontent.com/DigitalShogun/AddressedStateAttention/main/asa/analysis.py
!wget https://raw.githubusercontent.com/DigitalShogun/AddressedStateAttention/main/asa/universal_loader.py

#or:
!pip install git+https://github.com/DigitalDaimyo/AddressedStateAttention.git

from asa import load_asm_checkpoint, generate

# Load model
from universal_loader import load_asm_checkpoint

model, cfg, ckpt = load_asm_checkpoint(
    "path/to/checkpoint.pt",  # download from HF
    mode="analysis"
)

# Generate
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
# ... inference code
