# Addressed State Attention (ASA)

**Interpretable slot-based attention achieving competitive language modeling performance**

📊 **187M params:** 3.73 val loss / 41.6 PPL (FineWeb, 75k steps)  
🔬 **Mechanistic:** Refinement = directional suppression (α = -0.81)  
⚡ **Efficient:** O(T·K) vs O(T²) standard attention (K=16 slots)

[

![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)
](link)

## Notebooks

[![Open Analysis Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/digitaldaimyo/AddressedStateAttention/blob/main/notebooks/asa_analysis.ipynb)

[![Open Training Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/digitaldaimyo/AddressedStateAttention/blob/main/notebooks/asa_analysis.ipynb)

Browse all notebooks: https://github.com/digitaldaimyo/AddressedStateAttention/tree/main/notebooks




[Paper](paper_drafts/ASA_Mechanistic.pdf) | [Weights (HF)](https://huggingface.co/DigitalDaimyo/AddressedStateAttention) | [Demo](#quick-start)

---

## Quick Start

```python
# Install directly from GitHub
!pip install git+https://github.com/DigitalDaimyo/AddressedStateAttention.git

from asa import load_asm_checkpoint, generate
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download

# Download checkpoint from Hugging Face
ckpt_path = hf_hub_download(
    repo_id="DigitalDaimyo/AddressedStateAttention",
    filename="checkpoints/fineweb_187M_75k.pt"
)

# Load checkpoint
model, cfg, ckpt = load_asm_checkpoint(
    ckpt_path,
    mode="analysis"
)

tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Generate text
print(generate(model, tokenizer, "Once upon a time"))

```
---
## Results

| Model | Params | Dataset | Steps | Loss ↓ | PPL ↓ |
|:------|------:|:--------|-----:|------:|------:|
| ASA   | 187M  | FineWeb | 75k  | 3.73  | 41.6  |
