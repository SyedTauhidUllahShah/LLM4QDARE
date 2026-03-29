"""
Data loading and preprocessing for QDA annotation experiments.

Loads requirement datasets (LMS and Smart Home), handles label cleaning,
and creates stratified few-shot / test splits.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Dataset:
    """Holds a loaded requirement dataset."""
    name: str                       # Human-readable name (e.g. "Library Management System")
    system_type: str                # Short system identifier (e.g. "Library Management")
    requirements: List[str]         # Requirement statements
    labels: List[str]               # Ground-truth labels (consensus)
    label_set: List[str]            # Sorted list of unique labels

    def __len__(self) -> int:
        return len(self.requirements)

    def __repr__(self) -> str:
        return (
            f"Dataset(name={self.name!r}, n={len(self)}, "
            f"n_labels={len(self.label_set)})"
        )


@dataclass
class ExperimentSplit:
    """Train/test split with few-shot examples extracted."""
    few_shot_examples: List[Tuple[str, str]]   # (requirement, label) pairs
    test_requirements: List[str]
    test_labels: List[str]
    example_indices: List[int]                  # Original indices used as examples


# ---------------------------------------------------------------------------
# DataLoader
# ---------------------------------------------------------------------------

class DataLoader:
    """
    Loads and preprocesses requirement datasets for QDA experiments.

    Usage::

        loader = DataLoader(config)
        lms = loader.load_dataset("lms")
        split = loader.create_split(lms, n_examples_per_class=2)
    """

    def __init__(self, config: dict) -> None:
        self.config = config
        exp_cfg = config.get("experiment", {})
        self.random_seed: int = exp_cfg.get("random_seed", 42)
        self._rng = np.random.RandomState(self.random_seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_dataset(self, dataset_key: str) -> Dataset:
        """
        Load a dataset from its configured CSV file.

        Parameters
        ----------
        dataset_key : str
            Key from ``config['datasets']``, e.g. ``"lms"`` or ``"smart"``.

        Returns
        -------
        Dataset
        """
        ds_cfg = self.config["datasets"][dataset_key]
        path = Path(ds_cfg["path"])

        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path.resolve()}")

        df = pd.read_csv(path, encoding=ds_cfg.get("encoding", "utf-8-sig"))

        req_col = ds_cfg["requirement_col"]
        label_col = ds_cfg["label_col"]

        # Drop rows with missing or empty values
        df = df.dropna(subset=[req_col, label_col])
        df[req_col] = df[req_col].str.strip()
        df[label_col] = df[label_col].str.strip()
        df = df[df[req_col] != ""]
        df = df[df[label_col] != ""]

        requirements = df[req_col].tolist()
        labels = df[label_col].tolist()
        label_set = sorted(df[label_col].unique().tolist())

        logger.info(
            "Loaded dataset '%s': %d requirements, %d unique labels: %s",
            dataset_key,
            len(requirements),
            len(label_set),
            label_set,
        )

        return Dataset(
            name=ds_cfg["name"],
            system_type=ds_cfg["system_type"],
            requirements=requirements,
            labels=labels,
            label_set=label_set,
        )

    def create_split(
        self,
        dataset: Dataset,
        n_examples_per_class: int = 2,
    ) -> ExperimentSplit:
        """
        Stratified split: draw ``n_examples_per_class`` examples per label
        as the few-shot pool; remaining rows become the test set.

        The same split is deterministically reproducible via ``random_seed``.

        Parameters
        ----------
        dataset : Dataset
        n_examples_per_class : int
            Number of labeled examples to hold out per label class.

        Returns
        -------
        ExperimentSplit
        """
        # Group indices by label
        label_to_indices: Dict[str, List[int]] = {}
        for i, label in enumerate(dataset.labels):
            label_to_indices.setdefault(label, []).append(i)

        example_indices: List[int] = []
        for label, indices in sorted(label_to_indices.items()):
            n = min(n_examples_per_class, len(indices))
            chosen = self._rng.choice(indices, size=n, replace=False).tolist()
            example_indices.extend(chosen)

        example_index_set = set(example_indices)
        few_shot_examples = [
            (dataset.requirements[i], dataset.labels[i])
            for i in example_indices
        ]

        test_requirements: List[str] = []
        test_labels: List[str] = []
        for i, (req, label) in enumerate(
            zip(dataset.requirements, dataset.labels)
        ):
            if i not in example_index_set:
                test_requirements.append(req)
                test_labels.append(label)

        logger.info(
            "Split created: %d few-shot examples (%d classes), %d test requirements",
            len(few_shot_examples),
            len(label_to_indices),
            len(test_requirements),
        )

        return ExperimentSplit(
            few_shot_examples=few_shot_examples,
            test_requirements=test_requirements,
            test_labels=test_labels,
            example_indices=example_indices,
        )

    def sample(
        self,
        requirements: List[str],
        labels: List[str],
        sample_size: Optional[int],
    ) -> Tuple[List[str], List[str]]:
        """
        Optionally subsample the dataset for quick testing.

        Parameters
        ----------
        sample_size : int or None
            If ``None`` or >= dataset length, returns the full dataset.
        """
        if sample_size is None or sample_size >= len(requirements):
            return requirements, labels

        indices = sorted(
            self._rng.choice(len(requirements), size=sample_size, replace=False)
        )
        logger.info(
            "Sampled %d / %d requirements", sample_size, len(requirements)
        )
        return (
            [requirements[i] for i in indices],
            [labels[i] for i in indices],
        )
