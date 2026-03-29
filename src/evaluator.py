"""
Evaluation metrics for QDA annotation experiments.

Implements:
  - Cohen's Kappa (inter-rater agreement with human analysts)
  - Accuracy, macro Precision, Recall, F1-score
  - Standard Deviation of Kappa across multiple runs (RQ3)
  - Intraclass Correlation Coefficient across multiple runs (RQ3)
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    f1_score,
    precision_score,
    recall_score,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Single-run metrics
# ---------------------------------------------------------------------------


def compute_kappa(
    y_true: List[str],
    y_pred: List[str],
) -> float:
    """
    Cohen's Kappa between ground-truth and predicted labels.

    Parameters
    ----------
    y_true : list of str
        Ground-truth labels from human analysts.
    y_pred : list of str
        Labels predicted by the LLM (``None`` values → "UNKNOWN").

    Returns
    -------
    float
        Cohen's Kappa score.
    """
    y_true_c, y_pred_c = _align_labels(y_true, y_pred)
    return float(cohen_kappa_score(y_true_c, y_pred_c))


def compute_classification_metrics(
    y_true: List[str],
    y_pred: List[str],
) -> Dict[str, float]:
    """
    Compute accuracy, macro precision, recall, and F1-score.

    Returns
    -------
    dict with keys: accuracy, precision, recall, f1
    """
    y_true_c, y_pred_c = _align_labels(y_true, y_pred)
    return {
        "accuracy": float(accuracy_score(y_true_c, y_pred_c)),
        "precision": float(
            precision_score(
                y_true_c, y_pred_c, average="macro", zero_division=0
            )
        ),
        "recall": float(
            recall_score(
                y_true_c, y_pred_c, average="macro", zero_division=0
            )
        ),
        "f1": float(
            f1_score(y_true_c, y_pred_c, average="macro", zero_division=0)
        ),
    }


def compute_all_metrics(
    y_true: List[str],
    y_pred: List[str],
) -> Dict[str, float]:
    """Compute kappa + classification metrics in one call."""
    metrics = compute_classification_metrics(y_true, y_pred)
    metrics["kappa"] = compute_kappa(y_true, y_pred)
    return metrics


# ---------------------------------------------------------------------------
# Multi-run consistency metrics (RQ3)
# ---------------------------------------------------------------------------


def compute_consistency(
    y_true: List[str],
    all_run_predictions: List[List[Optional[str]]],
) -> Dict[str, float]:
    """
    Compute cross-run consistency metrics for repeated experiments.

    Parameters
    ----------
    y_true : list of str
        Ground-truth labels.
    all_run_predictions : list of list
        One inner list of predictions per run.

    Returns
    -------
    dict with keys:
        mean_kappa, std_kappa, icc
    """
    kappas = []
    for run_preds in all_run_predictions:
        kappas.append(compute_kappa(y_true, run_preds))

    mean_kappa = float(np.mean(kappas))
    std_kappa = float(np.std(kappas, ddof=1) if len(kappas) > 1 else 0.0)
    icc = _compute_icc(y_true, all_run_predictions)

    return {
        "mean_kappa": mean_kappa,
        "std_kappa": std_kappa,
        "icc": icc,
        "kappas_per_run": kappas,
    }


def _compute_icc(
    y_true: List[str],
    all_run_predictions: List[List[Optional[str]]],
) -> float:
    """
    Intraclass Correlation Coefficient (ICC 2,1 – two-way random, absolute
    agreement) across multiple experimental runs.

    Each run is treated as a "rater" and each requirement as a "subject".
    Labels are encoded as integers for the computation.

    Returns ``float("nan")`` if the ICC cannot be computed (e.g. < 2 runs).
    """
    if len(all_run_predictions) < 2:
        logger.warning("ICC requires at least 2 runs; returning NaN.")
        return float("nan")

    try:
        import pingouin as pg
    except ImportError:
        logger.warning("pingouin not installed; ICC computation skipped.")
        return float("nan")

    # Build combined label vocabulary (ground truth + all predictions)
    all_labels: set = set(y_true)
    for run_preds in all_run_predictions:
        all_labels.update(p for p in run_preds if p is not None)
    all_labels.add("UNKNOWN")
    label_to_int = {lbl: idx for idx, lbl in enumerate(sorted(all_labels))}
    unknown_int = label_to_int["UNKNOWN"]

    records = []
    for run_idx, run_preds in enumerate(all_run_predictions):
        aligned_preds = [
            p if p is not None else "UNKNOWN" for p in run_preds
        ]
        for subj_idx, pred in enumerate(aligned_preds):
            records.append(
                {
                    "Subject": subj_idx,
                    "Rater": f"run_{run_idx}",
                    "Rating": label_to_int.get(pred, unknown_int),
                }
            )

    df = pd.DataFrame(records)
    try:
        icc_df = pg.intraclass_corr(
            data=df,
            targets="Subject",
            raters="Rater",
            ratings="Rating",
        )
        # ICC2 = two-way random, absolute agreement
        icc_row = icc_df[icc_df["Type"] == "ICC2"]
        if icc_row.empty:
            icc_row = icc_df.iloc[0:1]
        return float(icc_row["ICC"].values[0])
    except Exception as exc:
        logger.warning("ICC computation failed: %s", exc)
        return float("nan")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _align_labels(
    y_true: List[str],
    y_pred: List[Optional[str]],
) -> Tuple[List[str], List[str]]:
    """
    Replace ``None`` predictions with ``"UNKNOWN"`` and trim to same length.
    """
    if len(y_true) != len(y_pred):
        min_len = min(len(y_true), len(y_pred))
        logger.warning(
            "Length mismatch: y_true=%d, y_pred=%d. Truncating to %d.",
            len(y_true),
            len(y_pred),
            min_len,
        )
        y_true = y_true[:min_len]
        y_pred = y_pred[:min_len]

    y_pred_clean = [p if p is not None else "UNKNOWN" for p in y_pred]
    return y_true, y_pred_clean
