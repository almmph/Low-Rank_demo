#!/usr/bin/env python3
"""Compress a local Hugging Face causal LM with fixed-rank or perplexity-guided low-rank search."""

from __future__ import annotations

import argparse
import gc
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

try:
    from safetensors.torch import load_file as load_safetensors
except ImportError:  # pragma: no cover - exercised only when safetensors is absent
    load_safetensors = None


DEFAULT_TARGET_SUFFIXES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)
DEFAULT_SEARCH_RANK_RATIOS = (0.125, 0.25, 0.375, 0.5)
MANIFEST_FILENAME = "low_rank_manifest.json"
SEARCH_REPORT_FILENAME = "search_report.json"
DTYPE_MAP = {
    "auto": None,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


@dataclass
class LowRankModuleSpec:
    name: str
    in_features: int
    out_features: int
    rank: int
    has_bias: bool


@dataclass
class CompressionSummary:
    total_params_before: int
    total_params_after: int
    target_params_before: int
    target_params_after: int
    replaced_modules: int
    skipped_modules: int

    @property
    def total_compression_ratio(self) -> float:
        if self.total_params_after == 0:
            return 0.0
        return self.total_params_before / self.total_params_after

    @property
    def target_compression_ratio(self) -> float:
        if self.target_params_after == 0:
            return 0.0
        return self.target_params_before / self.target_params_after

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["total_compression_ratio"] = self.total_compression_ratio
        data["target_compression_ratio"] = self.target_compression_ratio
        return data


@dataclass(frozen = True)
class SearchCandidate:
    rank: int | None = None
    rank_ratio: float | None = None

    def __post_init__(self) -> None:
        has_rank = self.rank is not None
        has_ratio = self.rank_ratio is not None
        if has_rank == has_ratio:
            raise ValueError("Exactly one of rank or rank_ratio must be set for SearchCandidate.")

    @property
    def label(self) -> str:
        if self.rank is not None:
            return f"rank={self.rank}"
        return f"rank_ratio={self.rank_ratio:.4f}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "rank_ratio": self.rank_ratio,
            "label": self.label,
        }


@dataclass
class SearchResult:
    candidate: SearchCandidate
    perplexity: float
    summary: CompressionSummary

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate": self.candidate.to_dict(),
            "perplexity": self.perplexity,
            "summary": self.summary.to_dict(),
        }


class LowRankLinear(nn.Module):
    """A Linear layer factored into two smaller Linear layers."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        *,
        bias: bool = True,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError(f"rank must be positive, got {rank}")

        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.input_factor = nn.Linear(
            in_features,
            rank,
            bias = False,
            device = device,
            dtype = dtype,
        )
        self.output_factor = nn.Linear(
            rank,
            out_features,
            bias = bias,
            device = device,
            dtype = dtype,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.input_factor(inputs)
        return self.output_factor(hidden)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"rank={self.rank}, bias={self.output_factor.bias is not None}"
        )

    @classmethod
    @torch.no_grad()
    def from_linear(
        cls,
        linear: nn.Linear,
        rank: int,
        *,
        svd_method: str = "auto",
        niter: int = 2,
    ) -> "LowRankLinear":
        min_dim = min(linear.in_features, linear.out_features)
        if rank >= min_dim:
            raise ValueError(
                f"rank={rank} must be smaller than min(in_features, out_features)={min_dim}"
            )

        module = cls(
            linear.in_features,
            linear.out_features,
            rank,
            bias = linear.bias is not None,
            device = linear.weight.device,
            dtype = linear.weight.dtype,
        )

        matrix = linear.weight.detach()
        compute_dtype = (
            torch.float32
            if matrix.dtype in {torch.float16, torch.bfloat16}
            else matrix.dtype
        )
        matrix = matrix.to(dtype = compute_dtype)

        use_randomized = _should_use_randomized_svd(
            matrix = matrix,
            rank = rank,
            svd_method = svd_method,
        )

        if use_randomized:
            q = min(min_dim, max(rank + 8, rank + 1))
            u, singular_values, v = torch.svd_lowrank(matrix, q = q, niter = niter)
            u = u[:, :rank]
            singular_values = singular_values[:rank]
            v = v[:, :rank]
            sqrt_s = singular_values.sqrt()
            left_weight = u * sqrt_s.unsqueeze(0)
            right_weight = sqrt_s.unsqueeze(1) * v.transpose(0, 1)
        else:
            u, singular_values, vh = torch.linalg.svd(matrix, full_matrices = False)
            u = u[:, :rank]
            singular_values = singular_values[:rank]
            vh = vh[:rank, :]
            sqrt_s = singular_values.sqrt()
            left_weight = u * sqrt_s.unsqueeze(0)
            right_weight = sqrt_s.unsqueeze(1) * vh

        module.input_factor.weight.copy_(
            right_weight.to(device = linear.weight.device, dtype = linear.weight.dtype)
        )
        module.output_factor.weight.copy_(
            left_weight.to(device = linear.weight.device, dtype = linear.weight.dtype)
        )
        if linear.bias is not None:
            module.output_factor.bias.copy_(linear.bias.detach())
        return module


def build_default_project_config() -> dict[str, Any]:
    return {
        "model_path": "",
        "output_dir": "",
        "modules": list(DEFAULT_TARGET_SUFFIXES),
        "exclude_modules": [],
        "rank": None,
        "rank_ratio": 0.25,
        "min_rank": 64,
        "max_rank": None,
        "dtype": "auto",
        "device": None,
        "svd_method": "auto",
        "svd_niter": 2,
        "max_shard_size": "100GB",
        "trust_remote_code": False,
        "safe_serialization": True,
        "quiet": False,
        "search_by_perplexity": False,
        "search_ranks": [],
        "search_rank_ratios": list(DEFAULT_SEARCH_RANK_RATIOS),
        "eval_file": "",
        "eval_text_key": "text",
        "eval_max_samples": 128,
        "eval_max_length": 512,
        "eval_batch_size": 4,
        "max_perplexity_ratio": 1.05,
        "search_report_name": SEARCH_REPORT_FILENAME,
    }


def normalize_job_config(config: dict[str, Any]) -> dict[str, Any]:
    normalized = build_default_project_config()
    normalized.update(config)
    normalized["modules"] = list(normalized.get("modules") or DEFAULT_TARGET_SUFFIXES)
    normalized["exclude_modules"] = list(normalized.get("exclude_modules") or [])
    normalized["search_ranks"] = list(normalized.get("search_ranks") or [])
    normalized["search_rank_ratios"] = list(
        normalized.get("search_rank_ratios") or DEFAULT_SEARCH_RANK_RATIOS
    )
    normalized["quiet"] = bool(normalized.get("quiet", False))
    normalized["search_by_perplexity"] = bool(
        normalized.get("search_by_perplexity", False)
    )
    normalized["trust_remote_code"] = bool(normalized.get("trust_remote_code", False))
    normalized["safe_serialization"] = bool(normalized.get("safe_serialization", True))
    normalized["dtype"] = normalized.get("dtype") or "auto"
    normalized["svd_method"] = normalized.get("svd_method") or "auto"
    normalized["eval_text_key"] = normalized.get("eval_text_key") or "text"
    normalized["search_report_name"] = (
        normalized.get("search_report_name") or SEARCH_REPORT_FILENAME
    )
    return normalized


def validate_job_config(config: dict[str, Any]) -> None:
    if not config.get("model_path"):
        raise ValueError("model_path is required.")
    if not config.get("output_dir"):
        raise ValueError("output_dir is required.")
    if config["dtype"] not in DTYPE_MAP:
        raise ValueError(f"Unsupported dtype={config['dtype']}.")
    if config["svd_method"] not in {"auto", "exact", "randomized"}:
        raise ValueError(f"Unsupported svd_method={config['svd_method']}.")
    if config["min_rank"] is not None and int(config["min_rank"]) <= 0:
        raise ValueError("min_rank must be positive.")
    if config["max_rank"] is not None and int(config["max_rank"]) <= 0:
        raise ValueError("max_rank must be positive when set.")
    if config["max_perplexity_ratio"] <= 0:
        raise ValueError("max_perplexity_ratio must be positive.")

    if config["search_by_perplexity"]:
        if not config.get("eval_file"):
            raise ValueError("eval_file is required when search_by_perplexity is enabled.")
        if config["eval_batch_size"] <= 0:
            raise ValueError("eval_batch_size must be positive.")
        if config["eval_max_length"] <= 1:
            raise ValueError("eval_max_length must be greater than 1.")
    elif config["rank"] is None and config["rank_ratio"] is None:
        raise ValueError(
            "Either rank or rank_ratio must be provided when perplexity search is disabled."
        )


def _should_use_randomized_svd(
    *,
    matrix: torch.Tensor,
    rank: int,
    svd_method: str,
) -> bool:
    if svd_method not in {"auto", "exact", "randomized"}:
        raise ValueError(f"Unsupported svd_method={svd_method}")
    if svd_method == "exact":
        return False
    if svd_method == "randomized":
        return True

    min_dim = min(matrix.shape)
    return min_dim >= 2048 and rank * 2 < min_dim


def _compile_patterns(patterns: Iterable[str]) -> list[re.Pattern[str]]:
    return [re.compile(pattern) for pattern in patterns]


def _suffixes_to_patterns(suffixes: Iterable[str]) -> list[str]:
    return [rf"{re.escape(suffix)}$" for suffix in suffixes]


def _module_matches(name: str, patterns: Sequence[re.Pattern[str]]) -> bool:
    return any(pattern.search(name) for pattern in patterns)


def _iter_named_linears(
    model: nn.Module,
) -> Iterable[tuple[str, nn.Module, str, nn.Linear]]:
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if "." in name:
            parent_name, child_name = name.rsplit(".", 1)
            parent = model.get_submodule(parent_name)
        else:
            parent = model
            child_name = name
        yield name, parent, child_name, module


def _linear_parameter_count(module: nn.Linear) -> int:
    return module.weight.numel() + (module.bias.numel() if module.bias is not None else 0)


def _low_rank_parameter_count(spec: LowRankModuleSpec) -> int:
    bias_params = spec.out_features if spec.has_bias else 0
    return spec.rank * (spec.in_features + spec.out_features) + bias_params


def resolve_rank(
    module: nn.Linear,
    *,
    rank: int | None,
    rank_ratio: float | None,
    min_rank: int,
    max_rank: int | None,
) -> int:
    if rank is None and rank_ratio is None:
        raise ValueError("Either rank or rank_ratio must be provided.")

    limit = min(module.in_features, module.out_features)
    resolved = rank if rank is not None else math.ceil(limit * rank_ratio)
    resolved = max(1, resolved)
    resolved = max(min_rank, resolved)
    if max_rank is not None:
        resolved = min(resolved, max_rank)
    return min(limit, resolved)


@torch.no_grad()
def compress_model(
    model: nn.Module,
    *,
    modules: Sequence[str],
    exclude_modules: Sequence[str],
    rank: int | None,
    rank_ratio: float | None,
    min_rank: int,
    max_rank: int | None,
    svd_method: str,
    svd_niter: int,
    verbose: bool = True,
) -> tuple[dict[str, object], CompressionSummary]:
    include_patterns = _compile_patterns(_suffixes_to_patterns(modules))
    exclude_patterns = _compile_patterns(_suffixes_to_patterns(exclude_modules))
    total_params_before = sum(param.numel() for param in model.parameters())

    specs: list[LowRankModuleSpec] = []
    target_params_before = 0
    target_params_after = 0
    skipped_modules = 0

    for name, parent, child_name, module in _iter_named_linears(model):
        if not _module_matches(name, include_patterns):
            continue
        if exclude_patterns and _module_matches(name, exclude_patterns):
            continue

        target_params_before += _linear_parameter_count(module)
        resolved_rank = resolve_rank(
            module,
            rank = rank,
            rank_ratio = rank_ratio,
            min_rank = min_rank,
            max_rank = max_rank,
        )
        spec = LowRankModuleSpec(
            name = name,
            in_features = module.in_features,
            out_features = module.out_features,
            rank = resolved_rank,
            has_bias = module.bias is not None,
        )

        compressed_params = _low_rank_parameter_count(spec)
        original_params = _linear_parameter_count(module)
        if compressed_params >= original_params or resolved_rank >= min(
            module.in_features, module.out_features
        ):
            skipped_modules += 1
            if verbose:
                print(
                    f"[skip] {name}: rank={resolved_rank} would not reduce parameters "
                    f"({original_params} -> {compressed_params})"
                )
            continue

        replacement = LowRankLinear.from_linear(
            module,
            rank = resolved_rank,
            svd_method = svd_method,
            niter = svd_niter,
        )
        setattr(parent, child_name, replacement)
        specs.append(spec)
        target_params_after += compressed_params
        if verbose:
            print(
                f"[compress] {name}: rank={resolved_rank} "
                f"({original_params} -> {compressed_params})"
            )

    total_params_after = sum(param.numel() for param in model.parameters())
    manifest: dict[str, object] = {
        "format": "low_rank_linear",
        "version": 1,
        "modules": [asdict(spec) for spec in specs],
    }
    summary = CompressionSummary(
        total_params_before = total_params_before,
        total_params_after = total_params_after,
        target_params_before = target_params_before,
        target_params_after = target_params_after,
        replaced_modules = len(specs),
        skipped_modules = skipped_modules,
    )
    return manifest, summary


def apply_low_rank_manifest(model: nn.Module, manifest: dict[str, object]) -> nn.Module:
    for module_info in manifest.get("modules", []):
        spec = LowRankModuleSpec(**module_info)
        if "." in spec.name:
            parent_name, child_name = spec.name.rsplit(".", 1)
            parent = model.get_submodule(parent_name)
        else:
            parent = model
            child_name = spec.name

        existing = getattr(parent, child_name)
        if isinstance(existing, LowRankLinear):
            continue
        if not isinstance(existing, nn.Linear):
            raise TypeError(
                f"Manifest expected `{spec.name}` to be nn.Linear, found {type(existing)!r}"
            )

        setattr(
            parent,
            child_name,
            LowRankLinear(
                spec.in_features,
                spec.out_features,
                spec.rank,
                bias = spec.has_bias,
                device = existing.weight.device,
                dtype = existing.weight.dtype,
            ),
        )
    return model


def save_compressed_model(
    model: nn.Module,
    output_dir: str | Path,
    manifest: dict[str, object],
    *,
    tokenizer = None,
    safe_serialization: bool = True,
    max_shard_size: str = "100GB",
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents = True, exist_ok = True)
    model.save_pretrained(
        output_dir,
        safe_serialization = safe_serialization,
        max_shard_size = max_shard_size,
    )
    if tokenizer is not None:
        tokenizer.save_pretrained(output_dir)

    with (output_dir / MANIFEST_FILENAME).open("w", encoding = "utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii = False, indent = 2)


def load_low_rank_model(
    model_dir: str | Path,
    *,
    torch_dtype: torch.dtype | None = None,
    device: str | torch.device | None = None,
    trust_remote_code: bool = False,
) -> nn.Module:
    model_dir = Path(model_dir)
    manifest_path = model_dir / MANIFEST_FILENAME
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")

    with manifest_path.open("r", encoding = "utf-8") as handle:
        manifest = json.load(handle)

    config = AutoConfig.from_pretrained(
        model_dir,
        trust_remote_code = trust_remote_code,
        local_files_only = True,
    )
    model = AutoModelForCausalLM.from_config(
        config,
        trust_remote_code = trust_remote_code,
    )
    apply_low_rank_manifest(model, manifest)
    state_dict = _load_saved_state_dict(model_dir)
    model.load_state_dict(state_dict, strict = True)

    if torch_dtype is not None:
        model = model.to(dtype = torch_dtype)
    if device is not None:
        model = model.to(device)
    model.eval()
    return model


def _load_saved_state_dict(model_dir: Path) -> dict[str, torch.Tensor]:
    safetensor_path = model_dir / "model.safetensors"
    if safetensor_path.exists():
        if load_safetensors is None:
            raise ImportError(
                "safetensors is required to load model.safetensors. Install safetensors first."
            )
        return load_safetensors(safetensor_path)

    pytorch_path = model_dir / "pytorch_model.bin"
    if pytorch_path.exists():
        return torch.load(pytorch_path, map_location = "cpu")

    sharded_safetensors = model_dir / "model.safetensors.index.json"
    sharded_pytorch = model_dir / "pytorch_model.bin.index.json"
    if sharded_safetensors.exists() or sharded_pytorch.exists():
        raise FileNotFoundError(
            "Found a sharded checkpoint. Save with a larger --max-shard-size, "
            "or extend the loader to read shard indexes."
        )
    raise FileNotFoundError(f"No model weights found under {model_dir}")


def load_base_model(
    model_path: str | Path,
    *,
    dtype_name: str,
    device: str | torch.device | None,
    trust_remote_code: bool,
) -> nn.Module:
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype = DTYPE_MAP[dtype_name],
        low_cpu_mem_usage = True,
        trust_remote_code = trust_remote_code,
        local_files_only = True,
    )
    if device is not None:
        model = model.to(device)
    model.eval()
    return model


def load_optional_tokenizer(
    model_path: str | Path,
    *,
    trust_remote_code: bool,
    quiet: bool,
    required: bool,
):
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code = trust_remote_code,
            local_files_only = True,
        )
    except Exception as exc:
        if required:
            raise RuntimeError(
                "A local tokenizer is required for perplexity evaluation but could not be loaded."
            ) from exc
        if not quiet:
            print(f"[warn] tokenizer was not saved because it could not be loaded: {exc}")
        return None

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
    return tokenizer


def _append_eval_text(texts: list[str], value: Any, *, text_key: str) -> None:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped:
            texts.append(stripped)
        return

    if isinstance(value, dict):
        if text_key in value:
            _append_eval_text(texts, value[text_key], text_key = text_key)
            return
        for fallback_key in ("text", "content", "prompt"):
            if fallback_key in value:
                _append_eval_text(texts, value[fallback_key], text_key = text_key)
                return
        return

    if isinstance(value, list):
        for item in value:
            _append_eval_text(texts, item, text_key = text_key)


def load_eval_texts(
    eval_file: str | Path,
    *,
    text_key: str,
    max_samples: int | None,
) -> list[str]:
    path = Path(eval_file)
    if not path.exists():
        raise FileNotFoundError(f"Evaluation file does not exist: {path}")

    texts: list[str] = []
    suffix = path.suffix.lower()

    if suffix == ".jsonl":
        with path.open("r", encoding = "utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                _append_eval_text(texts, json.loads(stripped), text_key = text_key)
                if max_samples is not None and len(texts) >= max_samples:
                    break
    elif suffix == ".json":
        with path.open("r", encoding = "utf-8") as handle:
            payload = json.load(handle)

        if isinstance(payload, dict):
            if "texts" in payload:
                _append_eval_text(texts, payload["texts"], text_key = text_key)
            elif text_key in payload:
                _append_eval_text(texts, payload[text_key], text_key = text_key)
            else:
                _append_eval_text(texts, payload, text_key = text_key)
        else:
            _append_eval_text(texts, payload, text_key = text_key)
    else:
        with path.open("r", encoding = "utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if stripped:
                    texts.append(stripped)
                if max_samples is not None and len(texts) >= max_samples:
                    break

    if max_samples is not None:
        texts = texts[:max_samples]
    if not texts:
        raise ValueError(f"No evaluation texts were loaded from {path}")
    return texts


def _batch_items(items: Sequence[str], batch_size: int) -> Iterable[list[str]]:
    for start in range(0, len(items), batch_size):
        yield list(items[start : start + batch_size])


@torch.inference_mode()
def compute_perplexity(
    model: nn.Module,
    tokenizer,
    texts: Sequence[str],
    *,
    max_length: int,
    batch_size: int,
) -> float:
    if not texts:
        raise ValueError("texts must be non-empty for perplexity evaluation.")
    if tokenizer.pad_token_id is None:
        raise ValueError(
            "Tokenizer needs a pad token for perplexity evaluation. "
            "Set one before calling compute_perplexity."
        )

    device = next(model.parameters()).device
    total_nll = 0.0
    total_tokens = 0
    previous_use_cache = getattr(model.config, "use_cache", None)
    if previous_use_cache is not None:
        model.config.use_cache = False

    try:
        for batch in _batch_items(list(texts), batch_size):
            encoded = tokenizer(
                batch,
                return_tensors = "pt",
                padding = True,
                truncation = True,
                max_length = max_length,
            )
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded.get("attention_mask")
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids)
            attention_mask = attention_mask.to(device)
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100

            outputs = model(
                input_ids = input_ids,
                attention_mask = attention_mask,
                labels = labels,
            )
            token_count = int((labels[:, 1:] != -100).sum().item())
            if token_count == 0:
                continue

            total_nll += float(outputs.loss.detach().cpu()) * token_count
            total_tokens += token_count
    finally:
        if previous_use_cache is not None:
            model.config.use_cache = previous_use_cache

    if total_tokens == 0:
        raise ValueError("Perplexity evaluation saw zero valid tokens.")
    return math.exp(total_nll / total_tokens)


def build_search_candidates(
    search_ranks: Sequence[int],
    search_rank_ratios: Sequence[float],
) -> list[SearchCandidate]:
    candidates: list[SearchCandidate] = []
    seen: set[tuple[str, float]] = set()

    for rank in search_ranks:
        if rank <= 0:
            raise ValueError(f"search rank must be positive, got {rank}")
        key = ("rank", float(rank))
        if key not in seen:
            seen.add(key)
            candidates.append(SearchCandidate(rank = int(rank)))

    for rank_ratio in search_rank_ratios:
        if rank_ratio <= 0:
            raise ValueError(f"search rank ratio must be positive, got {rank_ratio}")
        key = ("rank_ratio", float(rank_ratio))
        if key not in seen:
            seen.add(key)
            candidates.append(SearchCandidate(rank_ratio = float(rank_ratio)))

    if not candidates:
        for default_ratio in DEFAULT_SEARCH_RANK_RATIOS:
            candidates.append(SearchCandidate(rank_ratio = default_ratio))
    return candidates


def select_best_search_result(
    results: Sequence[SearchResult],
    *,
    baseline_perplexity: float,
    max_perplexity_ratio: float,
) -> SearchResult:
    if not results:
        raise ValueError("No search results were provided.")

    threshold = baseline_perplexity * max_perplexity_ratio
    eligible = [result for result in results if result.perplexity <= threshold]
    if eligible:
        return min(
            eligible,
            key = lambda result: (
                result.summary.total_params_after,
                result.perplexity,
                result.summary.target_params_after,
            ),
        )

    return min(
        results,
        key = lambda result: (
            result.perplexity,
            result.summary.total_params_after,
            result.summary.target_params_after,
        ),
    )


def write_search_report(
    output_dir: str | Path,
    *,
    baseline_perplexity: float,
    max_perplexity_ratio: float,
    selected_result: SearchResult,
    results: Sequence[SearchResult],
    report_name: str,
) -> Path:
    output_dir = Path(output_dir)
    report = {
        "baseline_perplexity": baseline_perplexity,
        "max_perplexity_ratio": max_perplexity_ratio,
        "selected_result": selected_result.to_dict(),
        "results": [result.to_dict() for result in results],
    }
    report_path = output_dir / report_name
    with report_path.open("w", encoding = "utf-8") as handle:
        json.dump(report, handle, ensure_ascii = False, indent = 2)
    return report_path


def format_count(value: int) -> str:
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if value >= 1_000:
        return f"{value / 1_000:.2f}K"
    return str(value)


def release_model(model: nn.Module | None) -> None:
    if model is None:
        return
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_perplexity_search(config: dict[str, Any]) -> tuple[float, SearchResult, list[SearchResult]]:
    tokenizer = load_optional_tokenizer(
        config["model_path"],
        trust_remote_code = config["trust_remote_code"],
        quiet = config["quiet"],
        required = True,
    )
    eval_texts = load_eval_texts(
        config["eval_file"],
        text_key = config["eval_text_key"],
        max_samples = config["eval_max_samples"],
    )

    if not config["quiet"]:
        print(
            f"[search] loaded {len(eval_texts)} evaluation samples from {config['eval_file']}"
        )

    baseline_model = load_base_model(
        config["model_path"],
        dtype_name = config["dtype"],
        device = config["device"],
        trust_remote_code = config["trust_remote_code"],
    )
    baseline_perplexity = compute_perplexity(
        baseline_model,
        tokenizer,
        eval_texts,
        max_length = config["eval_max_length"],
        batch_size = config["eval_batch_size"],
    )
    release_model(baseline_model)

    if not config["quiet"]:
        print(f"[search] baseline perplexity: {baseline_perplexity:.4f}")

    candidates = build_search_candidates(
        config["search_ranks"],
        config["search_rank_ratios"],
    )
    results: list[SearchResult] = []
    for candidate in candidates:
        model = load_base_model(
            config["model_path"],
            dtype_name = config["dtype"],
            device = config["device"],
            trust_remote_code = config["trust_remote_code"],
        )
        _, summary = compress_model(
            model,
            modules = config["modules"],
            exclude_modules = config["exclude_modules"],
            rank = candidate.rank,
            rank_ratio = candidate.rank_ratio,
            min_rank = config["min_rank"],
            max_rank = config["max_rank"],
            svd_method = config["svd_method"],
            svd_niter = config["svd_niter"],
            verbose = False,
        )
        if summary.replaced_modules == 0:
            release_model(model)
            if not config["quiet"]:
                print(f"[search] skipped {candidate.label}: no layers were compressible")
            continue

        perplexity = compute_perplexity(
            model,
            tokenizer,
            eval_texts,
            max_length = config["eval_max_length"],
            batch_size = config["eval_batch_size"],
        )
        release_model(model)
        result = SearchResult(
            candidate = candidate,
            perplexity = perplexity,
            summary = summary,
        )
        results.append(result)
        if not config["quiet"]:
            print(
                f"[search] {candidate.label}: perplexity={perplexity:.4f}, "
                f"total_compression={summary.total_compression_ratio:.2f}x"
            )

    if not results:
        raise RuntimeError("No valid search candidates produced a compressed model.")

    selected_result = select_best_search_result(
        results,
        baseline_perplexity = baseline_perplexity,
        max_perplexity_ratio = config["max_perplexity_ratio"],
    )
    if not config["quiet"]:
        threshold = baseline_perplexity * config["max_perplexity_ratio"]
        print(
            f"[search] selected {selected_result.candidate.label} "
            f"(ppl={selected_result.perplexity:.4f}, threshold={threshold:.4f})"
        )
    return baseline_perplexity, selected_result, results


def run_job(config: dict[str, Any]) -> dict[str, Any]:
    config = normalize_job_config(config)
    validate_job_config(config)

    baseline_perplexity = None
    selected_result = None
    search_results: list[SearchResult] = []
    chosen_rank = config["rank"]
    chosen_rank_ratio = config["rank_ratio"]

    if config["search_by_perplexity"]:
        baseline_perplexity, selected_result, search_results = run_perplexity_search(config)
        chosen_rank = selected_result.candidate.rank
        chosen_rank_ratio = selected_result.candidate.rank_ratio

    model = load_base_model(
        config["model_path"],
        dtype_name = config["dtype"],
        device = config["device"],
        trust_remote_code = config["trust_remote_code"],
    )
    tokenizer = load_optional_tokenizer(
        config["model_path"],
        trust_remote_code = config["trust_remote_code"],
        quiet = config["quiet"],
        required = False,
    )

    manifest, summary = compress_model(
        model,
        modules = config["modules"],
        exclude_modules = config["exclude_modules"],
        rank = chosen_rank,
        rank_ratio = chosen_rank_ratio,
        min_rank = config["min_rank"],
        max_rank = config["max_rank"],
        svd_method = config["svd_method"],
        svd_niter = config["svd_niter"],
        verbose = not config["quiet"],
    )
    if summary.replaced_modules == 0:
        raise RuntimeError("No layers were compressed. Relax rank settings or update modules.")

    if selected_result is not None:
        manifest["search"] = {
            "baseline_perplexity": baseline_perplexity,
            "selected_result": selected_result.to_dict(),
            "max_perplexity_ratio": config["max_perplexity_ratio"],
        }

    save_compressed_model(
        model,
        config["output_dir"],
        manifest,
        tokenizer = tokenizer,
        safe_serialization = config["safe_serialization"],
        max_shard_size = config["max_shard_size"],
    )
    release_model(model)

    search_report_path = None
    if selected_result is not None:
        search_report_path = write_search_report(
            config["output_dir"],
            baseline_perplexity = baseline_perplexity,
            max_perplexity_ratio = config["max_perplexity_ratio"],
            selected_result = selected_result,
            results = search_results,
            report_name = config["search_report_name"],
        )

    print("")
    print(f"Saved compressed model to: {config['output_dir']}")
    print(f"Replaced modules: {summary.replaced_modules}")
    print(f"Skipped modules: {summary.skipped_modules}")
    if selected_result is not None:
        print(f"Selected candidate: {selected_result.candidate.label}")
        print(f"Baseline perplexity: {baseline_perplexity:.4f}")
        print(f"Compressed perplexity: {selected_result.perplexity:.4f}")
    print(
        f"Target params: {format_count(summary.target_params_before)} -> "
        f"{format_count(summary.target_params_after)} "
        f"({summary.target_compression_ratio:.2f}x)"
    )
    print(
        f"Total params: {format_count(summary.total_params_before)} -> "
        f"{format_count(summary.total_params_after)} "
        f"({summary.total_compression_ratio:.2f}x)"
    )
    if search_report_path is not None:
        print(f"Search report: {search_report_path}")
    print("Reload later in Python with: from low_rank_compress import load_low_rank_model")

    return {
        "output_dir": str(config["output_dir"]),
        "summary": summary.to_dict(),
        "manifest_path": str(Path(config["output_dir"]) / MANIFEST_FILENAME),
        "search_report_path": str(search_report_path) if search_report_path else None,
        "selected_candidate": selected_result.to_dict() if selected_result else None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument("--model-path", required = True, help = "Local Hugging Face model path.")
    parser.add_argument("--output-dir", required = True, help = "Where to save the compressed model.")
    parser.add_argument(
        "--modules",
        nargs = "+",
        default = list(DEFAULT_TARGET_SUFFIXES),
        help = "Linear layer suffixes to compress.",
    )
    parser.add_argument(
        "--exclude-modules",
        nargs = "*",
        default = [],
        help = "Layer suffixes to skip even if they match --modules.",
    )
    parser.add_argument("--rank", type = int, default = None, help = "Fixed low-rank width.")
    parser.add_argument(
        "--rank-ratio",
        type = float,
        default = 0.25,
        help = "Rank as a fraction of min(in_features, out_features). Ignored when --rank is set.",
    )
    parser.add_argument("--min-rank", type = int, default = 64, help = "Minimum allowed rank.")
    parser.add_argument("--max-rank", type = int, default = None, help = "Optional maximum rank.")
    parser.add_argument(
        "--dtype",
        choices = sorted(DTYPE_MAP),
        default = "auto",
        help = "Weight dtype used when loading the base model.",
    )
    parser.add_argument(
        "--device",
        default = None,
        help = "Optional torch device to move the model onto before compression, e.g. cuda:0.",
    )
    parser.add_argument(
        "--svd-method",
        choices = ["auto", "exact", "randomized"],
        default = "auto",
        help = "SVD backend. randomized is faster for very large matrices.",
    )
    parser.add_argument(
        "--svd-niter",
        type = int,
        default = 2,
        help = "Power iterations used by torch.svd_lowrank when randomized SVD is selected.",
    )
    parser.add_argument(
        "--max-shard-size",
        default = "100GB",
        help = "Passed through to save_pretrained; defaults high to avoid sharded output.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action = "store_true",
        help = "Allow custom model code from the local model folder.",
    )
    parser.add_argument(
        "--safe-serialization",
        action = argparse.BooleanOptionalAction,
        default = True,
        help = "Save weights as safetensors when enabled.",
    )
    parser.add_argument(
        "--search-by-perplexity",
        action = "store_true",
        help = "Search candidate ranks and select the best one by validation perplexity.",
    )
    parser.add_argument(
        "--search-ranks",
        nargs = "*",
        type = int,
        default = None,
        help = "Candidate fixed ranks to try during perplexity search.",
    )
    parser.add_argument(
        "--search-rank-ratios",
        nargs = "*",
        type = float,
        default = None,
        help = "Candidate rank ratios to try during perplexity search.",
    )
    parser.add_argument(
        "--eval-file",
        default = "",
        help = "Validation text file used for perplexity search. Supports txt, json, jsonl.",
    )
    parser.add_argument(
        "--eval-text-key",
        default = "text",
        help = "Text key to read when --eval-file is json/jsonl.",
    )
    parser.add_argument(
        "--eval-max-samples",
        type = int,
        default = 128,
        help = "Maximum number of evaluation samples to load.",
    )
    parser.add_argument(
        "--eval-max-length",
        type = int,
        default = 512,
        help = "Maximum sequence length used during perplexity evaluation.",
    )
    parser.add_argument(
        "--eval-batch-size",
        type = int,
        default = 4,
        help = "Batch size used during perplexity evaluation.",
    )
    parser.add_argument(
        "--max-perplexity-ratio",
        type = float,
        default = 1.05,
        help = "Pick the strongest compression whose perplexity stays under baseline * ratio.",
    )
    parser.add_argument(
        "--search-report-name",
        default = SEARCH_REPORT_FILENAME,
        help = "Filename for the saved search report inside output-dir.",
    )
    parser.add_argument(
        "--quiet",
        action = "store_true",
        help = "Suppress per-layer progress logs.",
    )
    return parser.parse_args()


def main() -> int:
    run_job(vars(parse_args()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
