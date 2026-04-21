# Low-Rank Demo

A small standalone demo for compressing local Hugging Face causal language models with low-rank decomposition.

## What It Does

- Replaces selected `nn.Linear` layers with a factored `LowRankLinear`
- Builds the factors from SVD
- Skips layers when the requested rank would not actually reduce parameter count
- Saves a `low_rank_manifest.json` so the compressed structure can be rebuilt on load

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Compress A Local Model

```bash
python3 scripts/low_rank_compress.py \
  --model-path /path/to/local-model \
  --output-dir /path/to/compressed-model \
  --rank-ratio 0.25 \
  --modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj
```

Use a fixed rank instead of a ratio if you want tighter control:

```bash
python3 scripts/low_rank_compress.py \
  --model-path /path/to/local-model \
  --output-dir /path/to/compressed-model \
  --rank 128 \
  --min-rank 1
```

## Load The Compressed Model

```python
import sys
import torch

sys.path.insert(0, "scripts")
from low_rank_compress import load_low_rank_model

model = load_low_rank_model(
    "/path/to/compressed-model",
    device = "cuda:0",
    torch_dtype = torch.float16,
)
```

## Test

```bash
pytest tests/test_low_rank_compress.py
```
