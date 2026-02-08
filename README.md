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
# Install directly from GitHub
!pip install git+https://github.com/DigitalDaimyo/AddressedStateAttention.git

from asa import load_asm_checkpoint, generate
from transformers import AutoTokenizer

# Load checkpoint
model, cfg, ckpt = load_asm_checkpoint(
    "path/to/checkpoint.pt",
    mode="analysis"
)

tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Generate text
print(generate(model, tokenizer, "Once upon a time"))


---

## 7) Add a short “Results” section
Turn your header stats into a proper table.

```markdown
## Results

| Model | Params | Dataset | Steps | Val Loss | PPL |
|------|--------|---------|------|----------|-----|
| ASA | 187M | FineWeb | 75k | 3.73 | 41.6 |
