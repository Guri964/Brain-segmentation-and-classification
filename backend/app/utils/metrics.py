from __future__ import annotations
from typing import Iterable, Sequence
import numpy as np
EPS = 1e-7
def _as_bool(array: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    return np.asarray(array) > threshold
def confusion_counts(pred: np.ndarray, target: np.ndarray, threshold: float = 0.5) -> dict:
    p = _as_bool(pred, threshold).ravel()
    t = _as_bool(target, threshold).ravel()
    tp = float(np.logical_and(p, t).sum())
    fp = float(np.logical_and(p, ~t).sum())
    fn = float(np.logical_and(~p, t).sum())
    tn = float(np.logical_and(~p, ~t).sum())
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}
def dice(pred: np.ndarray, target: np.ndarray, threshold: float = 0.5) -> float:
    c = confusion_counts(pred, target, threshold)
    denom = 2 * c["tp"] + c["fp"] + c["fn"]
    if denom == 0:
        return 1.0
    return float((2 * c["tp"]) / denom)
def iou(pred: np.ndarray, target: np.ndarray, threshold: float = 0.5) -> float:
    c = confusion_counts(pred, target, threshold)
    denom = c["tp"] + c["fp"] + c["fn"]
    if denom == 0:
        return 1.0
    return float(c["tp"] / denom)
def precision(pred: np.ndarray, target: np.ndarray, threshold: float = 0.5) -> float:
    c = confusion_counts(pred, target, threshold)
    denom = c["tp"] + c["fp"]
    return float(c["tp"] / denom) if denom else 0.0
def recall(pred: np.ndarray, target: np.ndarray, threshold: float = 0.5) -> float:
    c = confusion_counts(pred, target, threshold)
    denom = c["tp"] + c["fn"]
    return float(c["tp"] / denom) if denom else 0.0
def specificity(pred: np.ndarray, target: np.ndarray, threshold: float = 0.5) -> float:
    c = confusion_counts(pred, target, threshold)
    denom = c["tn"] + c["fp"]
    return float(c["tn"] / denom) if denom else 0.0
def accuracy(pred: np.ndarray, target: np.ndarray, threshold: float = 0.5) -> float:
    c = confusion_counts(pred, target, threshold)
    total = sum(c.values())
    return float((c["tp"] + c["tn"]) / total) if total else 0.0
def segmentation_metrics(pred: np.ndarray, target: np.ndarray, threshold: float = 0.5) -> dict:
    counts = confusion_counts(pred, target, threshold)
    return {
        "dice": dice(pred, target, threshold),
        "iou": iou(pred, target, threshold),
        "precision": precision(pred, target, threshold),
        "recall": recall(pred, target, threshold),
        "sensitivity": recall(pred, target, threshold),
        "specificity": specificity(pred, target, threshold),
        "accuracy": accuracy(pred, target, threshold),
        "confusion": counts,
        "predicted_positive_pixels": int(counts["tp"] + counts["fp"]),
        "reference_positive_pixels": int(counts["tp"] + counts["fn"]),
    }
def hausdorff_distance_95(pred: np.ndarray, target: np.ndarray, spacing: Sequence[float] | None = None) -> float | None:
    p = _as_bool(pred)
    t = _as_bool(target)
    if not p.any() or not t.any():
        return None
    dt_pred = _edt(~p, spacing)
    dt_target = _edt(~t, spacing)
    surface_p = p & (_neighbour_min(dt_pred) > 0)
    surface_t = t & (_neighbour_min(dt_target) > 0)
    if not surface_p.any():
        surface_p = p
    if not surface_t.any():
        surface_t = t
    d1 = dt_target[surface_p]
    d2 = dt_pred[surface_t]
    return float(np.percentile(np.concatenate([d1, d2]), 95))
def _neighbour_min(array: np.ndarray) -> np.ndarray:
    out = np.full_like(array, np.inf, dtype=np.float64)
    for axis in range(array.ndim):
        for shift in (-1, 1):
            rolled = np.roll(array, shift, axis=axis)
            if shift == 1:
                sl = [slice(None)] * array.ndim
                sl[axis] = 0
                rolled[tuple(sl)] = np.inf
            else:
                sl = [slice(None)] * array.ndim
                sl[axis] = -1
                rolled[tuple(sl)] = np.inf
            out = np.minimum(out, rolled)
    return out
def _edt(mask: np.ndarray, spacing: Sequence[float] | None = None) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if spacing is None:
        spacing = [1.0] * mask.ndim
    dist = np.where(mask, np.inf, 0.0).astype(np.float64)
    for axis, sp in enumerate(spacing):
        dist = np.apply_along_axis(lambda line: _dt_1d(line, sp), axis, dist)
    return np.sqrt(np.maximum(dist, 0.0))
def _dt_1d(line: np.ndarray, spacing: float) -> np.ndarray:
    n = line.shape[0]
    f = line.astype(np.float64)
    d = np.empty(n, dtype=np.float64)
    v = np.zeros(n, dtype=int)
    z = np.empty(n + 1, dtype=np.float64)
    k = 0
    v[0] = 0
    z[0] = -np.inf
    z[1] = np.inf
    for q in range(1, n):
        if not np.isfinite(f[q]):
            continue
        s = ((f[q] + (q * spacing) ** 2) - (f[v[k]] + (v[k] * spacing) ** 2)) / (
            2 * spacing * (q - v[k])
        )
        while s <= z[k]:
            k -= 1
            if k < 0:
                break
            s = ((f[q] + (q * spacing) ** 2) - (f[v[k]] + (v[k] * spacing) ** 2)) / (
                2 * spacing * (q - v[k])
            )
        if k < 0:
            k = 0
            v[k] = q
            z[k] = -np.inf
            z[k + 1] = np.inf
            continue
        k += 1
        v[k] = q
        z[k] = s
        z[k + 1] = np.inf
    k = 0
    for q in range(n):
        while z[k + 1] < q * spacing:
            k += 1
        d[q] = ((q * spacing) - (v[k] * spacing)) ** 2 + f[v[k]]
    d[~np.isfinite(f)] = np.inf
    return d
def agreement_metrics(mask_a: np.ndarray, mask_b: np.ndarray) -> dict:
    return {
        "dice_agreement": dice(mask_a, mask_b, 0.5),
        "iou_agreement": iou(mask_a, mask_b, 0.5),
        "pixels_a": int(_as_bool(mask_a).sum()),
        "pixels_b": int(_as_bool(mask_b).sum()),
    }
def multilabel_confusion(y_true: Sequence[int], y_pred: Sequence[int], n_classes: int) -> np.ndarray:
    matrix = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        matrix[int(t), int(p)] += 1
    return matrix
def classification_report(y_true: Sequence[int], y_pred: Sequence[int], n_classes: int) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    matrix = multilabel_confusion(y_true, y_pred, n_classes)
    per_class = {}
    f1_scores = []
    for index in range(n_classes):
        tp = matrix[index, index]
        fp = matrix[:, index].sum() - tp
        fn = matrix[index, :].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        f1_scores.append(float(f1))
        per_class[str(index)] = {
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "support": int(matrix[index, :].sum()),
        }
    total = matrix.sum()
    return {
        "accuracy": float(np.trace(matrix) / total) if total else None,
        "macro_f1": float(np.mean(f1_scores)) if f1_scores else None,
        "weighted_f1": float(
            np.average(f1_scores, weights=[matrix[i, :].sum() for i in range(n_classes)])
        )
        if total
        else None,
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
        "n_samples": int(total),
    }
def roc_auc_binary(y_true: Sequence[int], y_score: Sequence[float]) -> float | None:
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    positives = y_true == 1
    negatives = ~positives
    if positives.sum() == 0 or negatives.sum() == 0:
        return None
    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(y_score) + 1)
    sorted_scores = y_score[order]
    i = 0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = np.mean(ranks[order[i : j + 1]])
        i = j + 1
    auc = (ranks[positives].sum() - positives.sum() * (positives.sum() + 1) / 2) / (
        positives.sum() * negatives.sum()
    )
    return float(auc)
def average_precision_binary(y_true: Sequence[int], y_score: Sequence[float]) -> float | None:
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    if y_true.sum() == 0:
        return None
    order = np.argsort(-y_score)
    y_sorted = y_true[order]
    tp = np.cumsum(y_sorted)
    precision = tp / (np.arange(len(y_sorted)) + 1)
    recall = tp / y_true.sum()
    return float(np.trapz(precision, recall))
def expected_calibration_error(
    y_true: Sequence[int], probabilities: Sequence[float], n_bins: int = 10
) -> float | None:
    y_true = np.asarray(y_true, dtype=int)
    probs = np.asarray(probabilities, dtype=float)
    if len(y_true) == 0:
        return None
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lower, upper in zip(bins[:-1], bins[1:]):
        in_bin = (probs > lower) & (probs <= upper)
        if not in_bin.any():
            continue
        confidence = probs[in_bin].mean()
        accuracy_in_bin = y_true[in_bin].mean()
        ece += (in_bin.sum() / len(probs)) * abs(confidence - accuracy_in_bin)
    return float(ece)
def reliability_curve(
    y_true: Sequence[int], probabilities: Sequence[float], n_bins: int = 10
) -> list[dict]:
    y_true = np.asarray(y_true, dtype=int)
    probs = np.asarray(probabilities, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    curve = []
    for lower, upper in zip(bins[:-1], bins[1:]):
        in_bin = (probs > lower) & (probs <= upper)
        curve.append(
            {
                "bin_lower": float(lower),
                "bin_upper": float(upper),
                "count": int(in_bin.sum()),
                "mean_confidence": float(probs[in_bin].mean()) if in_bin.any() else None,
                "empirical_accuracy": float(y_true[in_bin].mean()) if in_bin.any() else None,
            }
        )
    return curve
def bootstrap_ci(
    values: Iterable[float], n_resamples: int = 1000, alpha: float = 0.05, seed: int = 42
) -> dict:
    arr = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return {"mean": None, "ci_lower": None, "ci_upper": None, "n": 0}
    rng = np.random.default_rng(seed)
    samples = rng.choice(arr, size=(n_resamples, arr.size), replace=True).mean(axis=1)
    return {
        "mean": float(arr.mean()),
        "ci_lower": float(np.quantile(samples, alpha / 2)),
        "ci_upper": float(np.quantile(samples, 1 - alpha / 2)),
        "n": int(arr.size),
    }
