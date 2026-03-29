"""
Generate result tables matching those in the paper.

Reads all JSON metrics files from ``results/metrics/`` and produces:
  - Table 2  (tab:comp)             – Kappa by model × shot_type × dataset
  - Table 3  (tab:promptlength)     – Kappa by model × prompt_length × dataset
  - Table 4  (tab:consistency)      – SD and ICC by model × dataset
  - Table 5  (tab:context)          – Kappa by context_level × model × dataset
  - Table 6  (tab:performance)      – Full metrics by setting × model × dataset

Outputs:
  - CSV files to ``results/tables/``
  - LaTeX tables to ``results/tables/``
  - Console summary

Usage::

    python analysis/generate_tables.py
    python analysis/generate_tables.py --results-dir results/metrics --output-dir results/tables
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Human-readable labels for models/shot-types/etc.
# ---------------------------------------------------------------------------

MODEL_DISPLAY = {
    "llama2": "LLaMA-2",
    "mistral": "Mistral",
    "gpt4": "GPT-4",
}
SHOT_DISPLAY = {
    "zero_shot": "Zero-shot",
    "one_shot": "One-shot",
    "few_shot": "Few-shot",
}
LENGTH_DISPLAY = {
    "short": "Short",
    "medium": "Medium",
    "long": "Long",
}
CONTEXT_DISPLAY = {
    "no_context": "No Context",
    "some_context": "Some Context",
    "full_context": "Full Context",
}
DATASET_DISPLAY = {
    "lms": "Library Management",
    "smart": "Smart Home",
}

MODEL_ORDER = ["llama2", "mistral", "gpt4"]
SHOT_ORDER = ["zero_shot", "one_shot", "few_shot"]
LENGTH_ORDER = ["short", "medium", "long"]
CONTEXT_ORDER = ["no_context", "some_context", "full_context"]
DATASET_ORDER = ["lms", "smart"]


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def load_all_metrics(metrics_dir: Path) -> List[dict]:
    """Load all JSON metric files and attach parsed metadata."""
    records = []
    for path in sorted(metrics_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Could not read %s: %s", path, exc)
            continue

        exp_id: str = data.get("exp_id", path.stem)
        parts = exp_id.split("__")
        if len(parts) < 5:
            logger.warning("Unexpected exp_id format: %s", exp_id)
            continue

        dataset_key, model_key, shot_type, length, context_level = parts[:5]

        best = data.get("best_metrics", {})
        consistency = data.get("consistency", {})

        records.append(
            {
                "dataset": dataset_key,
                "model": model_key,
                "shot_type": shot_type,
                "length": length,
                "context_level": context_level,
                "kappa": best.get("kappa", float("nan")),
                "accuracy": best.get("accuracy", float("nan")),
                "precision": best.get("precision", float("nan")),
                "recall": best.get("recall", float("nan")),
                "f1": best.get("f1", float("nan")),
                "mean_kappa": consistency.get("mean_kappa", float("nan")),
                "std_kappa": consistency.get("std_kappa", float("nan")),
                "icc": consistency.get("icc", float("nan")),
            }
        )

    logger.info("Loaded %d experiment results.", len(records))
    return records


# ---------------------------------------------------------------------------
# Table builders
# ---------------------------------------------------------------------------


def table_kappa_by_shot(df: pd.DataFrame) -> pd.DataFrame:
    """
    Table 2 – Cohen's Kappa by model × shot_type, for each dataset.
    Aggregated using the best (long, full_context) prompt condition.
    """
    # Use best-performing condition per (dataset, model, shot_type)
    grp = (
        df.groupby(["dataset", "model", "shot_type"])["kappa"]
        .max()
        .reset_index()
    )
    pivot = grp.pivot_table(
        index=["dataset", "shot_type"],
        columns="model",
        values="kappa",
    )
    # Reorder
    pivot = pivot.reindex(
        index=pd.MultiIndex.from_product(
            [DATASET_ORDER, SHOT_ORDER], names=["dataset", "shot_type"]
        ),
        columns=MODEL_ORDER,
    )
    pivot = pivot.rename(
        columns=MODEL_DISPLAY,
        index={k: SHOT_DISPLAY.get(k, k) for k in SHOT_ORDER},
        level=1,
    )
    return pivot.round(3)


def table_kappa_by_length(df: pd.DataFrame) -> pd.DataFrame:
    """
    Table 3 – Kappa by model × prompt_length, best context per cell.
    """
    grp = (
        df.groupby(["dataset", "model", "length"])["kappa"]
        .max()
        .reset_index()
    )
    pivot = grp.pivot_table(
        index=["dataset", "model"],
        columns="length",
        values="kappa",
    )
    pivot = pivot.reindex(columns=LENGTH_ORDER)
    pivot = pivot.rename(columns=LENGTH_DISPLAY)
    return pivot.round(3)


def table_consistency(df: pd.DataFrame) -> pd.DataFrame:
    """
    Table 4 – Standard Deviation (SD) and ICC by model × dataset.
    Uses long-prompt / full-context / few-shot as the primary condition.
    """
    mask = (
        (df["length"] == "long")
        & (df["context_level"] == "full_context")
        & (df["shot_type"] == "few_shot")
    )
    subset = df[mask].copy()

    records = []
    for dataset in DATASET_ORDER:
        for model in MODEL_ORDER:
            row = subset[(subset["dataset"] == dataset) & (subset["model"] == model)]
            if row.empty:
                sd = float("nan")
                icc = float("nan")
            else:
                sd = float(row["std_kappa"].values[0])
                icc = float(row["icc"].values[0])
            records.append(
                {
                    "Dataset": DATASET_DISPLAY.get(dataset, dataset),
                    "Model": MODEL_DISPLAY.get(model, model),
                    "SD": round(sd, 3),
                    "ICC": round(icc, 3),
                }
            )

    return pd.DataFrame(records).set_index(["Dataset", "Model"])


def table_kappa_by_context(df: pd.DataFrame) -> pd.DataFrame:
    """
    Table 5 – Kappa by context_level × model, best shot/length per cell.
    """
    grp = (
        df.groupby(["dataset", "model", "context_level"])["kappa"]
        .max()
        .reset_index()
    )
    pivot = grp.pivot_table(
        index=["dataset", "context_level"],
        columns="model",
        values="kappa",
    )
    pivot = pivot.reindex(
        index=pd.MultiIndex.from_product(
            [DATASET_ORDER, CONTEXT_ORDER],
            names=["dataset", "context_level"],
        ),
        columns=MODEL_ORDER,
    )
    pivot = pivot.rename(
        columns=MODEL_DISPLAY,
        index={k: CONTEXT_DISPLAY.get(k, k) for k in CONTEXT_ORDER},
        level=1,
    )
    return pivot.round(3)


def table_full_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Table 6 – Accuracy, Precision, Recall, F1 by shot_type × model × dataset.
    Uses the best (length, context) per cell.
    """
    agg = (
        df.groupby(["dataset", "model", "shot_type"])[
            ["accuracy", "precision", "recall", "f1"]
        ]
        .max()
        .reset_index()
    )

    records = []
    for dataset in DATASET_ORDER:
        for shot in SHOT_ORDER:
            for model in MODEL_ORDER:
                row = agg[
                    (agg["dataset"] == dataset)
                    & (agg["shot_type"] == shot)
                    & (agg["model"] == model)
                ]
                if row.empty:
                    acc = prec = rec = f1 = float("nan")
                else:
                    acc = float(row["accuracy"].values[0])
                    prec = float(row["precision"].values[0])
                    rec = float(row["recall"].values[0])
                    f1 = float(row["f1"].values[0])

                records.append(
                    {
                        "Dataset": DATASET_DISPLAY.get(dataset, dataset),
                        "Setting": SHOT_DISPLAY.get(shot, shot),
                        "Model": MODEL_DISPLAY.get(model, model),
                        "Accuracy": round(acc, 3),
                        "Precision": round(prec, 3),
                        "Recall": round(rec, 3),
                        "F1": round(f1, 3),
                    }
                )

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# LaTeX formatters
# ---------------------------------------------------------------------------


def to_latex(df: pd.DataFrame, caption: str, label: str) -> str:
    """Render a DataFrame as a LaTeX table string."""
    latex = df.to_latex(
        caption=caption,
        label=label,
        na_rep="—",
        float_format="%.3f",
        bold_rows=False,
    )
    return latex


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate result tables from experiment metrics."
    )
    parser.add_argument(
        "--results-dir",
        default="results/metrics",
        dest="results_dir",
        help="Directory containing JSON metrics files.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/tables",
        dest="output_dir",
        help="Directory to write output tables.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    metrics_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not metrics_dir.exists():
        logger.error("Metrics directory not found: %s", metrics_dir)
        sys.exit(1)

    records = load_all_metrics(metrics_dir)
    if not records:
        logger.error(
            "No metrics found in %s. Run main.py first.", metrics_dir
        )
        sys.exit(1)

    df = pd.DataFrame(records)

    # ------------------------------------------------------------------
    # Generate each table
    # ------------------------------------------------------------------

    tables = [
        (
            "kappa_by_shot",
            table_kappa_by_shot(df),
            "Cohen's Kappa by Model and Shot Type",
            "tab:comp",
        ),
        (
            "kappa_by_length",
            table_kappa_by_length(df),
            "Cohen's Kappa by Model and Prompt Length",
            "tab:promptlength",
        ),
        (
            "consistency",
            table_consistency(df),
            "Consistency Analysis (SD and ICC) Across Multiple Runs",
            "tab:consistency",
        ),
        (
            "kappa_by_context",
            table_kappa_by_context(df),
            "Cohen's Kappa by Context Level and Model",
            "tab:context",
        ),
        (
            "full_metrics",
            table_full_metrics(df),
            "Detailed Performance Metrics by Setting and Model",
            "tab:performance",
        ),
    ]

    for name, tbl, caption, label in tables:
        # CSV
        csv_path = output_dir / f"{name}.csv"
        tbl.to_csv(csv_path)
        logger.info("Saved CSV: %s", csv_path)

        # LaTeX
        tex_path = output_dir / f"{name}.tex"
        tex_path.write_text(
            to_latex(tbl, caption, label), encoding="utf-8"
        )
        logger.info("Saved LaTeX: %s", tex_path)

        # Console
        print(f"\n{'='*60}")
        print(f"  {caption}")
        print(f"{'='*60}")
        print(tbl.to_string())

    # ------------------------------------------------------------------
    # Summary statistics
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("  Overall Summary")
    print(f"{'='*60}")
    for dataset in DATASET_ORDER:
        for model in MODEL_ORDER:
            subset = df[(df["dataset"] == dataset) & (df["model"] == model)]
            if subset.empty:
                continue
            best_row = subset.loc[subset["kappa"].idxmax()]
            print(
                f"  {DATASET_DISPLAY.get(dataset, dataset)} | "
                f"{MODEL_DISPLAY.get(model, model):10s}: "
                f"best kappa={best_row['kappa']:.3f}  "
                f"({best_row['shot_type']} / {best_row['length']} / {best_row['context_level']})"
            )

    logger.info("Table generation complete. Output: %s", output_dir)


if __name__ == "__main__":
    main()
