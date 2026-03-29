"""
Main experiment pipeline orchestrating all QDA annotation runs.

The pipeline iterates over all configured combinations of:
  dataset × model × shot_type × prompt_length × context_level × run

Results are saved incrementally so interrupted runs can be resumed.

Usage::

    from src.pipeline import ExperimentPipeline
    pipeline = ExperimentPipeline(config)
    pipeline.run()
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

from .data_loader import DataLoader, Dataset, ExperimentSplit
from .evaluator import compute_all_metrics, compute_consistency
from .label_extractor import LabelExtractor
from .models import GPT4Model, LlamaModel, MistralModel
from .models.base_model import BaseModel
from .prompt_builder import PromptBuilder

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Experiment identifier
# ---------------------------------------------------------------------------


def make_run_id(
    dataset_key: str,
    model_key: str,
    shot_type: str,
    length: str,
    context_level: str,
    run_idx: int,
) -> str:
    """Canonical string key for one experiment run."""
    return f"{dataset_key}__{model_key}__{shot_type}__{length}__{context_level}__run{run_idx}"


def make_experiment_id(
    dataset_key: str,
    model_key: str,
    shot_type: str,
    length: str,
    context_level: str,
) -> str:
    """Key for all runs of a single experimental condition."""
    return f"{dataset_key}__{model_key}__{shot_type}__{length}__{context_level}"


# ---------------------------------------------------------------------------
# ExperimentPipeline
# ---------------------------------------------------------------------------


class ExperimentPipeline:
    """
    Runs all QDA annotation experiments described in the paper.

    Parameters
    ----------
    config : dict
        Parsed YAML config (see ``config/config.yaml``).
    models_to_run : list of str, optional
        Subset of model keys to run (default: all enabled models).
    datasets_to_run : list of str, optional
        Subset of dataset keys to run (default: all datasets).
    """

    def __init__(
        self,
        config: dict,
        models_to_run: Optional[List[str]] = None,
        datasets_to_run: Optional[List[str]] = None,
    ) -> None:
        self.config = config
        self.exp_cfg = config["experiment"]
        out_cfg = config["output"]

        self.raw_dir = Path(out_cfg["raw_dir"])
        self.metrics_dir = Path(out_cfg["metrics_dir"])
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

        self.skip_existing: bool = self.exp_cfg.get("skip_existing", True)
        self.n_runs: int = self.exp_cfg.get("n_runs", 5)
        self.sample_size: Optional[int] = self.exp_cfg.get("sample_size", None)

        # Which models / datasets to run
        self.models_to_run = models_to_run or list(config["models"].keys())
        self.datasets_to_run = datasets_to_run or list(config["datasets"].keys())

        self._data_loader = DataLoader(config)

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Execute all configured experiments."""
        shot_types = self.exp_cfg["shot_types"]
        lengths = self.exp_cfg["prompt_lengths"]
        contexts = self.exp_cfg["context_levels"]
        n_few = self.exp_cfg.get("n_few_shot_examples", 3)
        n_one = self.exp_cfg.get("n_one_shot_examples", 1)
        n_per_class = self.exp_cfg.get("n_examples_per_class", 2)

        for dataset_key in self.datasets_to_run:
            logger.info("=== Dataset: %s ===", dataset_key)

            dataset = self._data_loader.load_dataset(dataset_key)
            split = self._data_loader.create_split(dataset, n_examples_per_class=n_per_class)

            extractor = LabelExtractor(dataset.label_set)
            prompt_builder = PromptBuilder(dataset.system_type, dataset.label_set)

            test_reqs, test_labels = self._data_loader.sample(
                split.test_requirements,
                split.test_labels,
                self.sample_size,
            )

            # Build example pools for different shot counts
            few_shot_examples = self._select_examples(
                split.few_shot_examples, n_few
            )
            one_shot_examples = self._select_examples(
                split.few_shot_examples, n_one
            )

            for model_key in self.models_to_run:
                model_cfg = self.config["models"].get(model_key, {})
                if not model_cfg.get("enabled", True):
                    logger.info("Model %s is disabled. Skipping.", model_key)
                    continue

                logger.info("--- Model: %s ---", model_key)
                model = self._build_model(model_key, model_cfg)

                try:
                    for shot_type in shot_types:
                        examples = (
                            few_shot_examples
                            if shot_type == "few_shot"
                            else one_shot_examples
                            if shot_type == "one_shot"
                            else []
                        )

                        for length in lengths:
                            for context in contexts:
                                exp_id = make_experiment_id(
                                    dataset_key, model_key, shot_type, length, context
                                )
                                self._run_condition(
                                    exp_id=exp_id,
                                    model=model,
                                    prompt_builder=prompt_builder,
                                    extractor=extractor,
                                    test_requirements=test_reqs,
                                    test_labels=test_labels,
                                    shot_type=shot_type,
                                    length=length,
                                    context_level=context,
                                    examples=examples,
                                )
                finally:
                    model.close()
                    logger.info("Model %s closed.", model_key)

        logger.info("All experiments complete.")

    # ------------------------------------------------------------------
    # Single experimental condition (all runs)
    # ------------------------------------------------------------------

    def _run_condition(
        self,
        exp_id: str,
        model: BaseModel,
        prompt_builder: PromptBuilder,
        extractor: LabelExtractor,
        test_requirements: List[str],
        test_labels: List[str],
        shot_type: str,
        length: str,
        context_level: str,
        examples: List[Tuple[str, str]],
    ) -> None:
        """Run all N repeated runs for one (model, shot, length, context) condition."""
        logger.info(
            "  Condition: %s | %s | %s  [%d runs × %d reqs]",
            shot_type,
            length,
            context_level,
            self.n_runs,
            len(test_requirements),
        )

        all_run_preds: List[List[Optional[str]]] = []

        for run_idx in range(self.n_runs):
            run_id = f"{exp_id}__run{run_idx}"
            raw_path = self.raw_dir / f"{run_id}.json"

            if self.skip_existing and raw_path.exists():
                logger.debug("Skipping existing run: %s", run_id)
                saved = self._load_json(raw_path)
                all_run_preds.append(saved["predictions"])
                continue

            # Support resume: partial file stores progress so far
            partial_path = self.raw_dir / f"{run_id}.partial.json"
            partial_data = {}
            if partial_path.exists():
                partial_data = self._load_json(partial_path)
                n_done = len(partial_data.get("raw_outputs", []))
                logger.info("  Resuming run%d from %d / %d", run_idx, n_done, len(test_requirements))

            saved_raw = partial_data.get("raw_outputs", [])
            start_idx = len(saved_raw)

            # Build prompts only for remaining requirements
            all_prompts = [
                prompt_builder.build(
                    requirement=req,
                    shot_type=shot_type,
                    length=length,
                    context_level=context_level,
                    examples=examples,
                )
                for req in test_requirements
            ]
            remaining_prompts = all_prompts[start_idx:]

            t0 = time.time()
            raw_outputs = list(saved_raw)  # start from already-completed

            for prompt in tqdm(
                remaining_prompts,
                desc=f"[{model.model_name}]",
                initial=start_idx,
                total=len(all_prompts),
                leave=False,
            ):
                raw_outputs.append(model.predict(prompt))
                # Save partial progress after every request
                self._save_json(partial_path, {
                    "run_id": run_id,
                    "raw_outputs": raw_outputs,
                })

            predictions = extractor.extract_batch(raw_outputs)
            elapsed = time.time() - t0

            run_data = {
                "run_id": run_id,
                "test_labels": test_labels,
                "raw_outputs": raw_outputs,
                "predictions": predictions,
                "elapsed_seconds": round(elapsed, 2),
            }
            self._save_json(raw_path, run_data)
            partial_path.unlink(missing_ok=True)  # clean up partial file
            all_run_preds.append(predictions)

            single_metrics = compute_all_metrics(test_labels, predictions)
            logger.info(
                "    run%d → kappa=%.3f  acc=%.3f  f1=%.3f  (%.1fs)",
                run_idx,
                single_metrics["kappa"],
                single_metrics["accuracy"],
                single_metrics["f1"],
                elapsed,
            )

        # Aggregate across runs
        consistency = compute_consistency(test_labels, all_run_preds)

        # Best-run metrics (for reporting, use run with highest kappa)
        best_idx = int(
            max(range(len(all_run_preds)), key=lambda i: compute_all_metrics(test_labels, all_run_preds[i])["kappa"])
        )
        best_metrics = compute_all_metrics(test_labels, all_run_preds[best_idx])

        summary = {
            "exp_id": exp_id,
            "best_run_idx": best_idx,
            "best_metrics": best_metrics,
            "consistency": consistency,
        }
        metrics_path = self.metrics_dir / f"{exp_id}.json"
        self._save_json(metrics_path, summary)

        logger.info(
            "  → best kappa=%.3f  mean=%.3f  std=%.3f  icc=%.3f",
            best_metrics["kappa"],
            consistency["mean_kappa"],
            consistency["std_kappa"],
            consistency["icc"],
        )

    # ------------------------------------------------------------------
    # Inference with progress bar
    # ------------------------------------------------------------------

    def _run_inference(
        self,
        model: BaseModel,
        prompts: List[str],
    ) -> List[Optional[str]]:
        """
        Call model.predict_batch and wrap with a tqdm progress bar.
        Falls back to sequential per-call inference with individual progress.
        """
        try:
            # Try bulk prediction first
            results = []
            batch_size = getattr(model, "model_cfg", {}).get("batch_size", 1)

            if batch_size > 1:
                # Batched HuggingFace path
                for start in tqdm(
                    range(0, len(prompts), batch_size),
                    desc=f"[{model.model_name}] batches",
                    leave=False,
                ):
                    batch = prompts[start : start + batch_size]
                    results.extend(model.predict_batch(batch))
            else:
                # Sequential path (GPT-4 API)
                for prompt in tqdm(
                    prompts,
                    desc=f"[{model.model_name}]",
                    leave=False,
                ):
                    results.append(model.predict(prompt))

            return results

        except Exception as exc:
            logger.error("Inference error: %s", exc)
            return [None] * len(prompts)

    # ------------------------------------------------------------------
    # Model factory
    # ------------------------------------------------------------------

    def _build_model(self, model_key: str, model_cfg: dict) -> BaseModel:
        api_type = model_cfg.get("api_type", "openai")
        if api_type == "openai":
            return GPT4Model(model_cfg)
        elif model_key == "mistral":
            return MistralModel(model_cfg)
        elif model_key == "llama2":
            return LlamaModel(model_cfg)
        else:
            raise ValueError(f"Unknown model key: {model_key!r}")

    # ------------------------------------------------------------------
    # Example selection
    # ------------------------------------------------------------------

    @staticmethod
    def _select_examples(
        examples: List[Tuple[str, str]],
        n: int,
    ) -> List[Tuple[str, str]]:
        """
        Select up to ``n`` examples from the pool, trying to cover diverse
        labels. If the pool has < n items, return all of them.
        """
        if n >= len(examples):
            return examples

        # Round-robin across labels for diversity
        by_label: Dict[str, List[Tuple[str, str]]] = {}
        for req, lbl in examples:
            by_label.setdefault(lbl, []).append((req, lbl))

        selected: List[Tuple[str, str]] = []
        label_iters = {lbl: iter(items) for lbl, items in by_label.items()}
        labels_cycle = list(label_iters.keys())
        idx = 0
        while len(selected) < n:
            lbl = labels_cycle[idx % len(labels_cycle)]
            try:
                selected.append(next(label_iters[lbl]))
            except StopIteration:
                labels_cycle.remove(lbl)
                if not labels_cycle:
                    break
            idx += 1

        return selected[:n]

    # ------------------------------------------------------------------
    # I/O helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _save_json(path: Path, data: Any) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)

    @staticmethod
    def _load_json(path: Path) -> Any:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
