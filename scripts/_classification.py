from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any, Mapping, Sequence

import numpy as np

IDENTIFIER_COLUMNS = {
    "case_id",
    "case",
    "id",
    "patient",
    "patient_id",
    "subject",
    "subject_id",
    "scan_id",
    "study",
    "study_id",
    "name",
    "filename",
    "path",
    "label",
    "target",
    "grade",
    "class",
    "diagnosis",
    "split",
    "group",
    "mask_source",
    "radiomics_backend",
    "image_channel",
    "foreground_pixels",
    "slice_index",
    "roi_mask_dice",
    "segmentation_model",
}


def select_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    label_column: str,
    patient_column: str | None,
    id_column: str | None,
) -> tuple[list[dict], list[dict]]:
    usable: list[dict] = []
    rejected: list[dict] = []
    for index, row in enumerate(rows):
        label = str(row.get(label_column, "")).strip()
        case_id = str(row.get(id_column or "case_id", "")).strip() or f"row-{index:05d}"
        patient_id = ""
        if patient_column:
            patient_id = str(row.get(patient_column, "")).strip()
        if not patient_id:
            patient_id = str(row.get("patient_id", "")).strip()
        if not patient_id:
            rejected.append(
                {
                    "case_id": case_id,
                    "reason": "no patient identifier; a patient-level split is impossible",
                }
            )
            continue
        if not label:
            rejected.append({"case_id": case_id, "reason": "no label"})
            continue
        usable.append(
            {
                "case_id": case_id,
                "patient_id": patient_id,
                "label": label,
                "declared_split": str(row.get("split", "")).strip().lower(),
                "values": dict(row),
            }
        )
    return usable, rejected


def feature_names(rows: Sequence[Mapping[str, Any]], columns: Sequence[str], explicit: Sequence[str] | None) -> list[str]:
    if explicit:
        missing = [name for name in explicit if name not in columns]
        if missing:
            raise ValueError(f"feature columns not present in the table: {missing[:5]}")
        return list(explicit)
    names: list[str] = []
    for column in columns:
        if column in IDENTIFIER_COLUMNS:
            continue
        sample = next((row.get(column) for row in rows if row.get(column) not in (None, "")), None)
        if sample is None:
            continue
        try:
            float(sample)
        except (TypeError, ValueError):
            continue
        names.append(column)
    return names


def build_matrix(rows: Sequence[Mapping[str, Any]], names: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.full((len(rows), len(names)), np.nan, dtype=np.float64)
    for row_index, row in enumerate(rows):
        values = row["values"]
        for column_index, name in enumerate(names):
            raw = values.get(name)
            if raw in (None, ""):
                continue
            try:
                matrix[row_index, column_index] = float(raw)
            except (TypeError, ValueError):
                continue
    return matrix, np.isnan(matrix)


def declared_splits(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[int]] | None:
    names = {row["declared_split"] for row in rows}
    if not names or names <= {"", "unassigned", "none"}:
        return None
    mapping = {"valid": "validation", "val": "validation", "train": "train", "test": "test", "validation": "validation"}
    splits: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        key = mapping.get(row["declared_split"])
        if key is None:
            return None
        splits[key].append(index)
    if set(splits) != {"train", "validation", "test"}:
        return None
    return dict(splits)


def assign_patient_splits(
    rows: Sequence[Mapping[str, Any]],
    *,
    train: float,
    val: float,
    test: float,
    seed: int,
    logger: Any = None,
) -> dict[str, list[int]]:
    if abs((train + val + test) - 1.0) > 1e-6:
        raise ValueError("train/validation/test must sum to 1.0")
    patients: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        entry = patients.setdefault(row["patient_id"], {"indices": [], "labels": Counter()})
        entry["indices"].append(index)
        entry["labels"][row["label"]] += 1
    strata: dict[str, list[str]] = defaultdict(list)
    conflicts: list[str] = []
    for patient_id, entry in patients.items():
        if len(entry["labels"]) > 1:
            conflicts.append(patient_id)
        key = "+".join(sorted(entry["labels"]))
        strata[key].append(patient_id)
    if conflicts and logger is not None:
        logger.warning(
            "%d patient(s) carry more than one label (%s). The split keeps each patient whole; the "
            "label inconsistency must be resolved before the numbers are trusted.",
            len(conflicts),
            ", ".join(sorted(conflicts)[:5]),
        )
    rng = np.random.default_rng(seed)
    splits: dict[str, list[int]] = {"train": [], "validation": [], "test": []}
    for key, patient_ids in sorted(strata.items()):
        ordered = sorted(patient_ids)
        rng.shuffle(ordered)
        count = len(ordered)
        if count < 3:
            for patient_id in ordered:
                splits["train"].extend(patients[patient_id]["indices"])
            if logger is not None:
                logger.warning(
                    "Stratum '%s' has %d patient(s); kept in training because no metric can be "
                    "reported from a smaller split.",
                    key,
                    count,
                )
            continue
        n_val = max(1, int(round(count * val)))
        n_test = max(1, int(round(count * test)))
        n_train = max(1, count - n_val - n_test)
        if n_train + n_val + n_test > count:
            n_test = max(0, count - n_train - n_val)
        for patient_id in ordered[:n_train]:
            splits["train"].extend(patients[patient_id]["indices"])
        for patient_id in ordered[n_train : n_train + n_val]:
            splits["validation"].extend(patients[patient_id]["indices"])
        remainder = ordered[n_train + n_val :]
        for patient_id in remainder:
            splits["test"].extend(patients[patient_id]["indices"])
    return splits


def split_summary(rows: Sequence[Mapping[str, Any]], splits: Mapping[str, Sequence[int]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for name in ("train", "validation", "test"):
        indices = list(splits.get(name, []))
        patients = sorted({rows[index]["patient_id"] for index in indices})
        classes = Counter(rows[index]["label"] for index in indices)
        summary[name] = {
            "rows": len(indices),
            "patients": len(patients),
            "class_distribution": dict(sorted(classes.items())),
            "patient_ids": patients,
        }
    placements: dict[str, set[str]] = defaultdict(set)
    for name in ("train", "validation", "test"):
        for index in splits.get(name, []):
            placements[rows[index]["patient_id"]].add(name)
    overlap = sorted(patient for patient, names in placements.items() if len(names) > 1)
    summary["patient_overlap"] = overlap
    summary["leakage_free"] = not overlap
    return summary


def impute(matrix: np.ndarray, missing: np.ndarray, train_indices: Sequence[int]) -> tuple[np.ndarray, list[float]]:
    filled = matrix.copy()
    train_rows = np.asarray(list(train_indices), dtype=int)
    medians: list[float] = []
    for column in range(matrix.shape[1]):
        observed = matrix[train_rows, column]
        observed = observed[~np.isnan(observed)]
        median = float(np.median(observed)) if observed.size else 0.0
        medians.append(median)
        column_missing = np.isnan(filled[:, column])
        filled[column_missing, column] = median
    return filled, medians


def standardise(filled: np.ndarray, train_indices: Sequence[int]) -> tuple[np.ndarray, list[float], list[float]]:
    train_rows = np.asarray(list(train_indices), dtype=int)
    mean = filled[train_rows].mean(axis=0)
    scale = filled[train_rows].std(axis=0)
    scale = np.where(scale < 1e-8, 1.0, scale)
    return (filled - mean) / scale, [float(item) for item in mean], [float(item) for item in scale]


def evaluate(y_true: Sequence[int], scores: np.ndarray, n_classes: int) -> dict[str, Any]:
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_recall_fscore_support,
        roc_auc_score,
    )

    truth = np.asarray(list(y_true), dtype=int)
    probabilities = np.asarray(scores, dtype=float)
    if probabilities.ndim == 1:
        probabilities = np.stack([1.0 - probabilities, probabilities], axis=1)
    predicted = probabilities.argmax(axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, predicted, labels=list(range(n_classes)), zero_division=0
    )
    per_class = {
        str(index): {
            "precision": _round(precision[index]),
            "recall": _round(recall[index]),
            "f1": _round(f1[index]),
            "support": int(support[index]),
        }
        for index in range(n_classes)
    }
    payload: dict[str, Any] = {
        "accuracy": _round(accuracy_score(truth, predicted)),
        "macro_f1": _round(f1_score(truth, predicted, average="macro", zero_division=0)),
        "weighted_f1": _round(f1_score(truth, predicted, average="weighted", zero_division=0)),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(truth, predicted, labels=list(range(n_classes))).tolist(),
        "confusion_matrix_order": [str(index) for index in range(n_classes)],
        "samples": int(truth.size),
        "class_distribution": dict(sorted(Counter(truth.tolist()).items())),
    }
    try:
        if n_classes == 2:
            payload["roc_auc"] = _round(roc_auc_score(truth, probabilities[:, 1]))
            payload["roc_auc_average"] = "binary"
        else:
            payload["roc_auc"] = _round(
                roc_auc_score(truth, probabilities, multi_class="ovr", average="macro", labels=list(range(n_classes)))
            )
            payload["roc_auc_average"] = "macro one-vs-rest"
    except ValueError as exc:
        payload["roc_auc"] = None
        payload["roc_auc_reason"] = f"{type(exc).__name__}: {exc}"
    return payload


def _round(value: Any, digits: int = 6) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return round(number, digits)


def text_report(y_true: Sequence[int], scores: np.ndarray, label_names: Sequence[str]) -> str:
    from sklearn.metrics import classification_report

    truth = np.asarray(list(y_true), dtype=int)
    probabilities = np.asarray(scores, dtype=float)
    predicted = probabilities.argmax(axis=1)
    return classification_report(
        truth,
        predicted,
        labels=list(range(len(label_names))),
        target_names=list(label_names),
        zero_division=0,
        digits=4,
    )


def class_weights(labels: Sequence[int], n_classes: int) -> np.ndarray:
    counts = Counter(labels)
    total = sum(counts.values()) or 1
    weights = []
    for index in range(n_classes):
        count = counts.get(index, 0)
        weights.append(total / (n_classes * count) if count else 0.0)
    return np.asarray(weights, dtype=float)


def quantiles(matrix: np.ndarray, train_indices: Sequence[int], *, count: int = 16) -> list[list[float]]:
    train_rows = np.asarray(list(train_indices), dtype=int)
    payload: list[list[float]] = []
    for column in range(matrix.shape[1]):
        values = matrix[train_rows, column]
        values = values[~np.isnan(values)]
        if values.size == 0:
            payload.append([])
            continue
        payload.append([float(item) for item in np.quantile(values, np.linspace(0.0, 1.0, count))])
    return payload


def linear_coefficients(model: Any, n_classes: int) -> dict[str, list[float]] | None:
    coefficients = getattr(model, "coef_", None)
    if coefficients is None:
        return None
    array = np.atleast_2d(np.asarray(coefficients, dtype=float))
    if array.shape[0] == 1 and n_classes == 2:
        return {"0": (-array[0]).tolist(), "1": array[0].tolist()}
    return {str(index): array[index].tolist() for index in range(min(n_classes, array.shape[0]))}
