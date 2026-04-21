# Low-Rank Demo

A small standalone demo for compressing local Hugging Face causal language models with low-rank decomposition.

## What It Does

- Replaces selected `nn.Linear` layers with a factored `LowRankLinear`
- Builds the factors from SVD
- Supports fixed-rank compression and perplexity-guided rank search
- Skips layers when the requested rank would not actually reduce parameter count
- Saves a `low_rank_manifest.json` so the compressed structure can be rebuilt on load
- Writes a `search_report.json` when perplexity search is enabled

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Fixed-Rank Compression

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

## Perplexity-Guided Rank Search

Prepare a small local validation set as `txt`, `json`, or `jsonl`, then run:

```bash
python3 scripts/low_rank_compress.py \
  --model-path /path/to/local-model \
  --output-dir /path/to/compressed-model \
  --search-by-perplexity \
  --eval-file /path/to/eval.txt \
  --search-rank-ratios 0.125 0.25 0.375 0.5 \
  --max-perplexity-ratio 1.05 \
  --min-rank 1
```

The script will:

- measure baseline perplexity on the original model
- try every candidate rank or rank ratio
- choose the strongest compression that stays within `baseline_perplexity * max_perplexity_ratio`
- save a `search_report.json` alongside the compressed model

## One-Click Project Launcher

If you want config filling and startup in one place, run:

```bash
python3 scripts/project_launcher.py
```

Behavior:

- if `project_config.json` does not exist, it opens an interactive config wizard
- after the config is saved, it immediately starts compression
- if the config already exists, it launches directly

Useful options:

```bash
python3 scripts/project_launcher.py --init-only
python3 scripts/project_launcher.py --reconfigure
python3 scripts/project_launcher.py --config /path/to/project_config.json
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
