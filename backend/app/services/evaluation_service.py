from __future__ import annotations
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence
import numpy as np
from ..utils.io import read_json, to_jsonable, utc_now
from ..utils.metrics import (
    bootstrap_ci,
    classification_report,
    expected_calibration_error,
    hausdorff_distance_95,
    reliability_curve,
    roc_auc_binary,
    average_precision_binary,
    segmentation_metrics,
)
logger = logging.getLogger(__name__)
NOT_EVALUATED = "Not evaluated"
RESULT_STATE_KEYS = {
    "evaluated": "Metrics were computed against labelled reference data.",
    "not_evaluated": (
        "No labelled evaluation data is available in this deployment, so no metric was computed. "
        "The project documentation must not report an accuracy, Dice or AUC value until the "
        "reserved test set has been evaluated with this same protocol."
    ),
}
@dataclass
class EvaluationReport:
    state: str
    reason: str
    dataset: dict = field(default_factory=dict)
    segmentation: dict = field(default_factory=dict)
    classification: dict = field(default_factory=dict)
    runtime: dict = field(default_factory=dict)
    generated_at: str = utc_now()
    def to_dict(self) -> dict:
        return to_jsonable(
            {
                "state": self.state,
                "reason": self.reason,
                "dataset": self.dataset,
                "segmentation": self.segmentation,
                "classification": self.classification,
                "runtime": self.runtime,
                "generated_at": self.generated_at,
                "state_description": RESULT_STATE_KEYS.get(self.state, ""),
            }
        )
class EvaluationService:
    def __init__(self, registry, config, segmentation=None, ensemble=None):
        self.registry = registry
        self.config = config
        self.segmentation = segmentation
        self.ensemble = ensemble
        data_dir = Path(config.DATA_DIR)
        self.results_path = data_dir / "models" / "evaluation_results.json"
        self.model_selection_path = data_dir / "models" / "model_selection.json"
    def stored_results(self) -> dict:
        payload = read_json(self.results_path, default=None)
        if payload:
            payload.setdefault("state", "evaluated")
            payload.setdefault("source", str(self.results_path))
            return payload
        return {
            "state": "not_evaluated",
            "reason": RESULT_STATE_KEYS["not_evaluated"],
            "expected_artifact": str(self.results_path),
            "produced_by": "scripts/evaluate_models.py",
        }
    def model_selection(self) -> dict:
        payload = read_json(self.model_selection_path, default=None)
        if payload:
            return payload
        return {
            "state": "not_evaluated",
            "reason": (
                "No model-selection record exists. Model selection should be performed on the "
                "validation split and recorded (with the random seed, the configuration and the "
                "selection metric) before the reserved test set is touched."
            ),
            "expected_artifact": str(self.model_selection_path),
            "produced_by": "scripts/evaluate_models.py --select",
        }
    def segmentation_summary(self) -> dict:
        stored = self.stored_results()
        if stored.get("state") != "evaluated":
            return {
                "state": "not_evaluated",
                "reason": stored.get("reason"),
                "models": [
                    {
                        "model_id": descriptor.model_id,
                        "display_name": descriptor.display_name,
                        "metrics": None,
                        "status": "not_evaluated",
                    }
                    for descriptor in self.registry.descriptors.values()
                    if descriptor.task == "segmentation"
                ],
            }
        return {
            "state": "evaluated",
            "dataset": stored.get("dataset", {}),
            "models": stored.get("segmentation", {}).get("models", []),
            "ensemble": stored.get("segmentation", {}).get("ensemble"),
            "ablation": stored.get("segmentation", {}).get("ablation", []),
        }
    def classification_summary(self) -> dict:
        stored = self.stored_results()
        if stored.get("state") != "evaluated" or "classification" not in stored:
            return {
                "state": "not_evaluated",
                "reason": (
                    "No classification evaluation artefact was found. It is produced by "
                    "scripts/train_classifier.py, which requires a labelled radiomics table."
                ),
            }
        return stored["classification"]
    def runtime_summary(self) -> dict:
        stored = self.stored_results()
        if stored.get("runtime"):
            return stored["runtime"]
        if self.segmentation is None:
            return {"state": "not_measured"}
        entries = []
        from .model_health_service import brainai_reference_phantom_4ch, reference_phantom_1ch
        for model_id in self.registry.segmentation_model_ids():
            loaded = self.registry.load_model(model_id)
            if not loaded.is_ready:
                entries.append(
                    {
                        "model_id": model_id,
                        "state": "not_measured",
                        "reason": loaded.message,
                    }
                )
                continue
            spec = loaded.descriptor.input_spec
            if spec.channels == 4:
                tensor = brainai_reference_phantom_4ch(spec.height)[None]
            else:
                tensor = reference_phantom_1ch(spec.height)[None]
            import torch
            started = time.perf_counter()
            with torch.no_grad():
                loaded.module(torch.from_numpy(tensor).to(loaded.device))
            elapsed = time.perf_counter() - started
            entries.append(
                {
                    "model_id": model_id,
                    "state": "measured",
                    "device": str(loaded.device),
                    "input_shape": list(tensor.shape),
                    "parameter_count": loaded.parameter_count,
                    "seconds_per_image": round(elapsed, 4),
                    "images_per_second": round(1.0 / elapsed, 3) if elapsed > 0 else None,
                    "note": (
                        "Measured on this machine with a single reference input; it is an inference "
                        "cost, not a training cost."
                    ),
                }
            )
        return {"state": "measured", "generated_at": utc_now(), "models": entries}
    @staticmethod
    def evaluate_segmentation_predictions(
        predictions: dict[str, Sequence[np.ndarray]],
        references: Sequence[np.ndarray],
        *,
        threshold: float = 0.5,
        spacing: Sequence[float] | None = None,
        case_ids: Sequence[str] | None = None,
    ) -> dict:
        n_cases = len(references)
        if n_cases == 0:
            return {"state": "not_evaluated", "reason": "No labelled cases were supplied."}
        model_reports = {}
        for model_id, maps in predictions.items():
            if len(maps) != n_cases:
                raise ValueError(
                    f"{model_id}: {len(maps)} predictions for {n_cases} references"
                )
            per_case = []
            for index, (probability, reference) in enumerate(zip(maps, references)):
                metrics = segmentation_metrics(probability, reference, threshold)
                metrics["hausdorff_95"] = hausdorff_distance_95(
                    np.asarray(probability) > threshold, np.asarray(reference) > 0.5, spacing
                )
                metrics["case_id"] = (
                    str(case_ids[index]) if case_ids is not None else f"case-{index:04d}"
                )
                metrics["reference_empty"] = bool(not np.asarray(reference).astype(bool).any())
                per_case.append(metrics)
            model_reports[model_id] = {
                "model_id": model_id,
                "threshold": threshold,
                "n_cases": n_cases,
                "per_case": per_case,
                "aggregate": _aggregate_segmentation(per_case, n_cases),
            }
        return {
            "state": "evaluated",
            "threshold": threshold,
            "n_cases": n_cases,
            "models": model_reports,
            "comparison_table": _comparison_table(model_reports),
            "generated_at": utc_now(),
        }
    @staticmethod
    def evaluate_classification(
        y_true: Sequence[int],
        y_pred: Sequence[int],
        probabilities: Sequence[Sequence[float]],
        *,
        n_classes: int,
        label_mapping: dict | None = None,
    ) -> dict:
        if len(y_true) == 0:
            return {"state": "not_evaluated", "reason": "No labelled samples were supplied."}
        report = classification_report(y_true, y_pred, n_classes)
        report["label_mapping"] = label_mapping or {}
        report["state"] = "evaluated"
        if n_classes == 2:
            positive_scores = [p[1] if len(p) > 1 else p[0] for p in probabilities]
            report["roc_auc"] = roc_auc_binary(y_true, positive_scores)
            report["pr_auc"] = average_precision_binary(y_true, positive_scores)
            report["expected_calibration_error"] = expected_calibration_error(
                y_true, positive_scores
            )
            report["reliability_curve"] = reliability_curve(y_true, positive_scores)
        else:
            report["roc_auc"] = _macro_roc_auc(y_true, probabilities, n_classes)
            report["pr_auc"] = None
            report["expected_calibration_error"] = None
            report["reliability_curve"] = []
        return report
def _aggregate_segmentation(per_case: list[dict], n_cases: int) -> dict:
    keys = [
        "dice",
        "iou",
        "precision",
        "recall",
        "sensitivity",
        "specificity",
        "accuracy",
        "hausdorff_95",
    ]
    aggregate = {"n_cases": n_cases}
    for key in keys:
        values = [case.get(key) for case in per_case]
        aggregate[key] = bootstrap_ci(values)
    empty_references = sum(1 for case in per_case if case.get("reference_empty"))
    aggregate["cases_with_empty_reference"] = empty_references
    aggregate["note"] = (
        "Dice and IoU are defined as 1.0 when both the prediction and the reference are empty. "
        f"{empty_references} of {n_cases} reference masks are empty; report this explicitly when "
        "comparing against published results."
    )
    return aggregate
def _comparison_table(model_reports: dict[str, dict]) -> list[dict]:
    rows = []
    for model_id, report in model_reports.items():
        aggregate = report["aggregate"]
        rows.append(
            {
                "model_id": model_id,
                "n_cases": aggregate["n_cases"],
                "dice_mean": (aggregate.get("dice") or {}).get("mean"),
                "dice_ci": [
                    (aggregate.get("dice") or {}).get("ci_lower"),
                    (aggregate.get("dice") or {}).get("ci_upper"),
                ],
                "iou_mean": (aggregate.get("iou") or {}).get("mean"),
                "precision_mean": (aggregate.get("precision") or {}).get("mean"),
                "recall_mean": (aggregate.get("recall") or {}).get("mean"),
                "specificity_mean": (aggregate.get("specificity") or {}).get("mean"),
                "hausdorff_95_mean": (aggregate.get("hausdorff_95") or {}).get("mean"),
            }
        )
    rows.sort(key=lambda row: (row["dice_mean"] is None, -(row["dice_mean"] or 0)))
    for position, row in enumerate(rows, start=1):
        row["rank"] = position
    return rows
def _macro_roc_auc(y_true: Sequence[int], probabilities: Sequence[Sequence[float]], n_classes: int):
    aucs = []
    for class_index in range(n_classes):
        binary_truth = [1 if int(label) == class_index else 0 for label in y_true]
        scores = [row[class_index] for row in probabilities]
        value = roc_auc_binary(binary_truth, scores)
        if value is not None:
            aucs.append(value)
    return float(np.mean(aucs)) if aucs else None
