#!/usr/bin/env python3
"""Compress a local Hugging Face causal LM by replacing selected Linear layers with low-rank factors.

Example:
    python3 scripts/low_rank_compress.py \
        --model-path /models/Qwen2.5-7B-Instruct \
        --output-dir /models/Qwen2.5-7B-Instruct-lowrank \
        --rank-ratio 0.25 \
        --modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

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
MANIFEST_FILENAME = "low_rank_manifest.json"
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


def format_count(value: int) -> str:
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if value >= 1_000:
        return f"{value / 1_000:.2f}K"
    return str(value)


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
        "--quiet",
        action = "store_true",
        help = "Suppress per-layer progress logs.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.rank is None and args.rank_ratio is None:
        raise SystemExit("Either --rank or --rank-ratio must be provided.")

    torch_dtype = DTYPE_MAP[args.dtype]
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype = torch_dtype,
        low_cpu_mem_usage = True,
        trust_remote_code = args.trust_remote_code,
        local_files_only = True,
    )
    if args.device is not None:
        model = model.to(args.device)
    tokenizer = None
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path,
            trust_remote_code = args.trust_remote_code,
            local_files_only = True,
        )
    except Exception as exc:
        if not args.quiet:
            print(f"[warn] tokenizer was not saved because it could not be loaded: {exc}")

    manifest, summary = compress_model(
        model,
        modules = args.modules,
        exclude_modules = args.exclude_modules,
        rank = args.rank,
        rank_ratio = args.rank_ratio,
        min_rank = args.min_rank,
        max_rank = args.max_rank,
        svd_method = args.svd_method,
        svd_niter = args.svd_niter,
        verbose = not args.quiet,
    )
    if summary.replaced_modules == 0:
        raise SystemExit("No layers were compressed. Relax rank settings or update --modules.")

    save_compressed_model(
        model,
        args.output_dir,
        manifest,
        tokenizer = tokenizer,
        safe_serialization = args.safe_serialization,
        max_shard_size = args.max_shard_size,
    )

    print("")
    print(f"Saved compressed model to: {args.output_dir}")
    print(f"Replaced modules: {summary.replaced_modules}")
    print(f"Skipped modules: {summary.skipped_modules}")
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
    print(
        "Reload later in Python with: "
        "from low_rank_compress import load_low_rank_model"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
