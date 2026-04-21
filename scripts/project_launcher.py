#!/usr/bin/env python3
"""Interactive config writer and one-click launcher for the low-rank compression demo."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from low_rank_compress import build_default_project_config, normalize_job_config, run_job


def load_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        return build_default_project_config()
    with config_path.open("r", encoding = "utf-8") as handle:
        return normalize_job_config(json.load(handle))


def save_config(config_path: Path, config: dict[str, Any]) -> None:
    config_path.parent.mkdir(parents = True, exist_ok = True)
    with config_path.open("w", encoding = "utf-8") as handle:
        json.dump(config, handle, ensure_ascii = False, indent = 2)


def prompt_text(prompt: str, default: str = "", *, required: bool = False) -> str:
    suffix = f" [{default}]" if default else ""
    while True:
        value = input(f"{prompt}{suffix}: ").strip()
        if value:
            return value
        if default:
            return default
        if not required:
            return ""
        print("This field is required.")


def prompt_bool(prompt: str, default: bool) -> bool:
    default_hint = "Y/n" if default else "y/N"
    while True:
        value = input(f"{prompt} ({default_hint}): ").strip().lower()
        if not value:
            return default
        if value in {"y", "yes"}:
            return True
        if value in {"n", "no"}:
            return False
        print("Please answer with y or n.")


def prompt_int(prompt: str, default: int | None = None, *, required: bool = False) -> int | None:
    default_text = "" if default is None else str(default)
    while True:
        value = prompt_text(prompt, default_text, required = required)
        if not value:
            return None
        try:
            return int(value)
        except ValueError:
            print("Please enter an integer.")


def prompt_float(
    prompt: str,
    default: float | None = None,
    *,
    required: bool = False,
) -> float | None:
    default_text = "" if default is None else str(default)
    while True:
        value = prompt_text(prompt, default_text, required = required)
        if not value:
            return None
        try:
            return float(value)
        except ValueError:
            print("Please enter a number.")


def prompt_string_list(prompt: str, default: list[str]) -> list[str]:
    raw = prompt_text(prompt, ", ".join(default))
    return [item.strip() for item in raw.split(",") if item.strip()]


def prompt_int_list(prompt: str, default: list[int]) -> list[int]:
    raw = prompt_text(prompt, ", ".join(str(item) for item in default))
    if not raw.strip():
        return []
    try:
        return [int(item.strip()) for item in raw.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError("Expected a comma-separated list of integers.") from exc


def prompt_float_list(prompt: str, default: list[float]) -> list[float]:
    raw = prompt_text(prompt, ", ".join(str(item) for item in default))
    if not raw.strip():
        return []
    try:
        return [float(item.strip()) for item in raw.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError("Expected a comma-separated list of numbers.") from exc


def configure_interactively(existing: dict[str, Any]) -> dict[str, Any]:
    config = normalize_job_config(existing)

    print("Fill the project config. Press Enter to keep the current/default value.")
    config["model_path"] = prompt_text("Local model path", config["model_path"], required = True)
    config["output_dir"] = prompt_text(
        "Output directory",
        config["output_dir"] or "outputs/low-rank-model",
        required = True,
    )
    config["modules"] = prompt_string_list("Modules to compress", config["modules"])
    config["exclude_modules"] = prompt_string_list(
        "Modules to exclude",
        config["exclude_modules"],
    )
    config["dtype"] = prompt_text("Dtype", config["dtype"] or "auto", required = True)
    device_value = prompt_text("Device (blank for CPU/default)", config["device"] or "")
    config["device"] = device_value or None
    config["svd_method"] = prompt_text(
        "SVD method",
        config["svd_method"] or "auto",
        required = True,
    )
    config["svd_niter"] = prompt_int("SVD lowrank iterations", config["svd_niter"], required = True)
    config["min_rank"] = prompt_int("Minimum rank", config["min_rank"], required = True)

    max_rank_value = prompt_text(
        "Maximum rank (blank means no cap)",
        "" if config["max_rank"] is None else str(config["max_rank"]),
    )
    config["max_rank"] = int(max_rank_value) if max_rank_value else None

    config["search_by_perplexity"] = prompt_bool(
        "Enable perplexity-guided rank search",
        config["search_by_perplexity"],
    )
    if config["search_by_perplexity"]:
        config["eval_file"] = prompt_text(
            "Evaluation text file",
            config["eval_file"],
            required = True,
        )
        config["search_ranks"] = prompt_int_list(
            "Candidate fixed ranks (comma-separated, blank to skip)",
            config["search_ranks"],
        )
        config["search_rank_ratios"] = prompt_float_list(
            "Candidate rank ratios (comma-separated)",
            config["search_rank_ratios"],
        )
        config["eval_max_samples"] = prompt_int(
            "Evaluation max samples",
            config["eval_max_samples"],
            required = True,
        )
        config["eval_max_length"] = prompt_int(
            "Evaluation max length",
            config["eval_max_length"],
            required = True,
        )
        config["eval_batch_size"] = prompt_int(
            "Evaluation batch size",
            config["eval_batch_size"],
            required = True,
        )
        config["max_perplexity_ratio"] = prompt_float(
            "Max perplexity ratio against baseline",
            config["max_perplexity_ratio"],
            required = True,
        )
        config["rank"] = None
    else:
        use_fixed_rank = prompt_bool(
            "Use a fixed rank instead of a rank ratio",
            config["rank"] is not None,
        )
        if use_fixed_rank:
            config["rank"] = prompt_int(
                "Fixed rank",
                config["rank"] or 128,
                required = True,
            )
            config["rank_ratio"] = None
        else:
            config["rank_ratio"] = prompt_float(
                "Rank ratio",
                config["rank_ratio"] or 0.25,
                required = True,
            )
            config["rank"] = None

    config["quiet"] = prompt_bool("Quiet mode", config["quiet"])
    return normalize_job_config(config)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument(
        "--config",
        default = "project_config.json",
        help = "Path to the JSON config file.",
    )
    parser.add_argument(
        "--init-only",
        action = "store_true",
        help = "Write or update the config and exit without starting compression.",
    )
    parser.add_argument(
        "--reconfigure",
        action = "store_true",
        help = "Force the interactive config wizard even if the config file already exists.",
    )
    parser.add_argument(
        "--non-interactive",
        action = "store_true",
        help = "Fail instead of opening interactive prompts when config is missing or invalid.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config_exists = config_path.exists()

    if not config_exists and args.non_interactive:
        raise SystemExit(f"Config file does not exist: {config_path}")

    config = load_config(config_path)
    should_prompt = args.reconfigure or not config_exists

    if should_prompt:
        if args.non_interactive:
            raise SystemExit("Interactive configuration is required but --non-interactive was set.")
        config = configure_interactively(config)
        save_config(config_path, config)
        print(f"Saved config to: {config_path}")

    if args.init_only:
        if not should_prompt and not config_exists:
            save_config(config_path, config)
            print(f"Saved config to: {config_path}")
        return 0

    try:
        run_job(config)
    except ValueError as exc:
        if args.non_interactive:
            raise
        print(f"Config is incomplete or invalid: {exc}")
        config = configure_interactively(config)
        save_config(config_path, config)
        print(f"Saved config to: {config_path}")
        run_job(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
