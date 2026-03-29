"""
Entry point for LLMs-Based QDA in Requirements Engineering experiments.

Examples
--------
# Run all models, all datasets, all conditions (full paper replication):
    python main.py

# Quick test with GPT-4 only, small sample, 2 runs:
    python main.py --models gpt4 --sample-size 50 --n-runs 2

# Run only the LMS dataset with few-shot, long prompts:
    python main.py --datasets lms --shot-types few_shot --lengths long

# Run with 4-bit quantisation for local models (saves VRAM):
    python main.py --models mistral llama2 --load-4bit

# Skip already-completed experiments (resume a partial run):
    python main.py --skip-existing
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

import yaml
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Load environment variables (.env file)
# ---------------------------------------------------------------------------
load_dotenv()


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging(log_level: str, log_file: Optional[str] = None) -> None:
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(level=getattr(logging, log_level.upper()), format=fmt, handlers=handlers)


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def apply_cli_overrides(config: dict, args: argparse.Namespace) -> dict:
    """Patch the config dict with any CLI overrides."""
    exp = config.setdefault("experiment", {})

    if args.sample_size is not None:
        exp["sample_size"] = args.sample_size
    if args.n_runs is not None:
        exp["n_runs"] = args.n_runs
    if args.shot_types:
        exp["shot_types"] = args.shot_types
    if args.lengths:
        exp["prompt_lengths"] = args.lengths
    if args.contexts:
        exp["context_levels"] = args.contexts
    if args.skip_existing:
        exp["skip_existing"] = True
    if args.no_skip:
        exp["skip_existing"] = False

    # Enable 4-bit quantisation for local models
    if args.load_4bit:
        for key in ("mistral", "llama2"):
            if key in config.get("models", {}):
                config["models"][key]["load_in_4bit"] = True

    # Disable models not in the requested list
    if args.models:
        for mkey in list(config.get("models", {}).keys()):
            if mkey not in args.models:
                config["models"][mkey]["enabled"] = False

    # Disable datasets not in the requested list
    if args.datasets:
        for dkey in list(config.get("datasets", {}).keys()):
            if dkey not in args.datasets:
                config["datasets"].pop(dkey, None)

    return config


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run LLM-based QDA annotation experiments for RE.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    p.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to YAML config file (default: config/config.yaml).",
    )
    p.add_argument(
        "--models",
        nargs="+",
        choices=["gpt4", "mistral", "llama2"],
        metavar="MODEL",
        help="Models to run (default: all enabled in config).",
    )
    p.add_argument(
        "--datasets",
        nargs="+",
        choices=["lms", "smart"],
        metavar="DATASET",
        help="Datasets to evaluate (default: lms smart).",
    )
    p.add_argument(
        "--shot-types",
        nargs="+",
        choices=["zero_shot", "one_shot", "few_shot"],
        dest="shot_types",
        metavar="SHOT",
        help="Shot types to evaluate.",
    )
    p.add_argument(
        "--lengths",
        nargs="+",
        choices=["short", "medium", "long"],
        metavar="LENGTH",
        help="Prompt lengths to evaluate.",
    )
    p.add_argument(
        "--contexts",
        nargs="+",
        choices=["no_context", "some_context", "full_context"],
        metavar="CTX",
        help="Context levels to evaluate.",
    )
    p.add_argument(
        "--sample-size",
        type=int,
        default=None,
        dest="sample_size",
        help="Randomly sample N requirements per dataset (for quick testing).",
    )
    p.add_argument(
        "--n-runs",
        type=int,
        default=None,
        dest="n_runs",
        help="Number of repeated runs per condition (for consistency analysis).",
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        dest="skip_existing",
        default=False,
        help="Skip conditions that already have saved results (resume mode).",
    )
    p.add_argument(
        "--no-skip",
        action="store_true",
        dest="no_skip",
        default=False,
        help="Re-run all conditions even if results exist.",
    )
    p.add_argument(
        "--load-4bit",
        action="store_true",
        dest="load_4bit",
        default=False,
        help="Load local HuggingFace models in 4-bit quantisation (saves VRAM).",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        dest="log_level",
        help="Logging verbosity.",
    )

    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"ERROR: Config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    config = load_config(str(config_path))
    config = apply_cli_overrides(config, args)

    # Setup logging
    log_file = config.get("output", {}).get("log_file")
    setup_logging(args.log_level, log_file)

    logger = logging.getLogger(__name__)
    logger.info("Starting QDA annotation experiments.")
    logger.info("Config: %s", args.config)

    # Check API key for GPT-4
    enabled_models = [
        k for k, v in config.get("models", {}).items() if v.get("enabled", True)
    ]
    if "gpt4" in enabled_models and not os.getenv("OPENAI_API_KEY"):
        logger.warning(
            "OPENAI_API_KEY not set. GPT-4 experiments will fail. "
            "Copy .env.example → .env and add your key."
        )

    # Run pipeline
    from src.pipeline import ExperimentPipeline

    models_to_run = [
        k for k, v in config.get("models", {}).items() if v.get("enabled", True)
    ]
    datasets_to_run = list(config.get("datasets", {}).keys())

    pipeline = ExperimentPipeline(
        config=config,
        models_to_run=models_to_run,
        datasets_to_run=datasets_to_run,
    )

    try:
        pipeline.run()
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    except Exception as exc:
        logger.exception("Pipeline failed: %s", exc)
        sys.exit(1)

    logger.info("Done. Generate result tables with: python analysis/generate_tables.py")


if __name__ == "__main__":
    main()
