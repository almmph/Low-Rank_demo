import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

pytest.importorskip("tokenizers")
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace


_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SCRIPT_PATH = os.path.join(_REPO_ROOT, "scripts", "low_rank_compress.py")
_LAUNCHER_PATH = os.path.join(_REPO_ROOT, "scripts", "project_launcher.py")


def _load_low_rank_script():
    spec = importlib.util.spec_from_file_location("low_rank_compress", _SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["low_rank_compress"] = module
    spec.loader.exec_module(module)
    return module


def _write_eval_file(path: Path) -> None:
    path.write_text("hello world\nsmall local model\nhello model world\n", encoding = "utf-8")


def _save_tiny_model_bundle(model_dir: Path) -> None:
    config = LlamaConfig(
        vocab_size = 16,
        hidden_size = 32,
        intermediate_size = 64,
        num_hidden_layers = 1,
        num_attention_heads = 4,
        num_key_value_heads = 4,
        max_position_embeddings = 64,
    )
    model = LlamaForCausalLM(config)
    model.save_pretrained(model_dir, safe_serialization = True)

    vocab = {
        "<pad>": 0,
        "<bos>": 1,
        "<eos>": 2,
        "<unk>": 3,
        "hello": 4,
        "world": 5,
        "small": 6,
        "local": 7,
        "model": 8,
    }
    tokenizer_object = Tokenizer(WordLevel(vocab, unk_token = "<unk>"))
    tokenizer_object.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object = tokenizer_object,
        bos_token = "<bos>",
        eos_token = "<eos>",
        unk_token = "<unk>",
        pad_token = "<pad>",
    )
    tokenizer.save_pretrained(model_dir)


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


def test_select_best_search_result_prefers_stronger_compression_under_threshold():
    baseline = 10.0
    result_a = low_rank.SearchResult(
        candidate = low_rank.SearchCandidate(rank = 8),
        perplexity = 10.4,
        summary = low_rank.CompressionSummary(
            total_params_before = 1000,
            total_params_after = 400,
            target_params_before = 600,
            target_params_after = 160,
            replaced_modules = 7,
            skipped_modules = 0,
        ),
    )
    result_b = low_rank.SearchResult(
        candidate = low_rank.SearchCandidate(rank = 16),
        perplexity = 10.2,
        summary = low_rank.CompressionSummary(
            total_params_before = 1000,
            total_params_after = 500,
            target_params_before = 600,
            target_params_after = 220,
            replaced_modules = 7,
            skipped_modules = 0,
        ),
    )

    best = low_rank.select_best_search_result(
        [result_b, result_a],
        baseline_perplexity = baseline,
        max_perplexity_ratio = 1.05,
    )
    assert best.candidate.rank == 8


def test_search_mode_smoke(tmp_path: Path):
    model_dir = tmp_path / "model"
    output_dir = tmp_path / "compressed"
    eval_file = tmp_path / "eval.txt"
    model_dir.mkdir()
    _save_tiny_model_bundle(model_dir)
    _write_eval_file(eval_file)

    subprocess.run(
        [
            sys.executable,
            _SCRIPT_PATH,
            "--model-path",
            str(model_dir),
            "--output-dir",
            str(output_dir),
            "--search-by-perplexity",
            "--search-ranks",
            "2",
            "4",
            "--eval-file",
            str(eval_file),
            "--min-rank",
            "1",
            "--dtype",
            "float32",
            "--quiet",
        ],
        check = True,
    )

    assert (output_dir / low_rank.MANIFEST_FILENAME).exists()
    search_report = output_dir / low_rank.SEARCH_REPORT_FILENAME
    assert search_report.exists()
    payload = json.loads(search_report.read_text(encoding = "utf-8"))
    assert payload["selected_result"]["candidate"]["rank"] in {2, 4}


def test_project_launcher_runs_from_config(tmp_path: Path):
    model_dir = tmp_path / "model"
    output_dir = tmp_path / "launcher-output"
    config_path = tmp_path / "project_config.json"
    model_dir.mkdir()
    _save_tiny_model_bundle(model_dir)

    config = low_rank.build_default_project_config()
    config.update(
        {
            "model_path": str(model_dir),
            "output_dir": str(output_dir),
            "rank": 4,
            "rank_ratio": None,
            "min_rank": 1,
            "dtype": "float32",
            "quiet": True,
        }
    )
    config_path.write_text(json.dumps(config, indent = 2), encoding = "utf-8")

    subprocess.run(
        [
            sys.executable,
            _LAUNCHER_PATH,
            "--config",
            str(config_path),
            "--non-interactive",
        ],
        check = True,
    )

    assert (output_dir / low_rank.MANIFEST_FILENAME).exists()
