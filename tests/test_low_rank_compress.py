import importlib.util
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from transformers import LlamaConfig, LlamaForCausalLM


_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SCRIPT_PATH = os.path.join(_REPO_ROOT, "scripts", "low_rank_compress.py")


def _load_low_rank_script():
    spec = importlib.util.spec_from_file_location("low_rank_compress", _SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["low_rank_compress"] = module
    spec.loader.exec_module(module)
    return module


low_rank = _load_low_rank_script()


def test_low_rank_linear_is_exact_for_rank_limited_matrix():
    linear = nn.Linear(8, 6, bias = True)
    left = torch.randn(6, 2)
    right = torch.randn(2, 8)
    with torch.no_grad():
        linear.weight.copy_(left @ right)
        linear.bias.zero_()

    replacement = low_rank.LowRankLinear.from_linear(linear, rank = 2, svd_method = "exact")
    inputs = torch.randn(4, 8)
    expected = linear(inputs)
    actual = replacement(inputs)
    assert torch.allclose(actual, expected, atol = 1e-5, rtol = 1e-5)


def test_compress_and_reload_tiny_llama(tmp_path: Path):
    config = LlamaConfig(
        vocab_size = 128,
        hidden_size = 32,
        intermediate_size = 64,
        num_hidden_layers = 1,
        num_attention_heads = 4,
        num_key_value_heads = 4,
        max_position_embeddings = 64,
    )
    model = LlamaForCausalLM(config)
    input_ids = torch.randint(0, config.vocab_size, (1, 8))

    manifest, summary = low_rank.compress_model(
        model,
        modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        exclude_modules = [],
        rank = 4,
        rank_ratio = None,
        min_rank = 1,
        max_rank = None,
        svd_method = "exact",
        svd_niter = 2,
        verbose = False,
    )
    assert summary.replaced_modules > 0

    logits_before = model(input_ids).logits.detach()
    low_rank.save_compressed_model(
        model,
        tmp_path,
        manifest,
        tokenizer = None,
        safe_serialization = True,
        max_shard_size = "100GB",
    )

    reloaded = low_rank.load_low_rank_model(tmp_path)
    logits_after = reloaded(input_ids).logits.detach()
    assert torch.allclose(logits_before, logits_after, atol = 1e-5, rtol = 1e-5)
    assert (tmp_path / low_rank.MANIFEST_FILENAME).exists()
