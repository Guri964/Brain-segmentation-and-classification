from __future__ import annotations

import random
from collections import defaultdict
from typing import Any, Mapping, Sequence

SPLIT_NAMES = ("train", "validation", "test")


def patient_groups(cases: Sequence[Mapping[str, Any]]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = defaultdict(list)
    for index, entry in enumerate(cases):
        case_id = str(entry.get("case_id") or entry.get("id") or "").strip()
        if not case_id:
            raise ValueError(f"case entry {index} has no case_id")
        patient_id = str(entry.get("patient_id") or entry.get("patient") or "").strip()
        if not patient_id:
            raise ValueError(
                f"case {case_id} has no patient_id; patient-level splitting is impossible"
            )
        groups[patient_id].append(case_id)
    return dict(groups)


def patient_labels(cases: Sequence[Mapping[str, Any]]) -> dict[str, list[Any]]:
    labels: dict[str, set[Any]] = defaultdict(set)
    for entry in cases:
        patient_id = str(entry.get("patient_id") or entry.get("patient") or "").strip()
        if not patient_id:
            continue
        label = entry.get("label")
        labels[patient_id].add(None if label is None else str(label))
    return {key: sorted(value, key=lambda item: str(item)) for key, value in labels.items()}


def stratum_for(labels: Sequence[Any]) -> str:
    known = sorted({str(item) for item in labels if item is not None})
    if not known:
        return "unlabelled"
    if len(known) > 1:
        return f"conflict:{'+'.join(known)}"
    return known[0]


def assign_splits(
    cases: Sequence[Mapping[str, Any]],
    *,
    train: float = 0.70,
    val: float = 0.15,
    test: float = 0.15,
    seed: int = 20240101,
    logger: Any = None,
) -> tuple[dict[str, list[str]], dict[str, Any]]:
    if abs((train + val + test) - 1.0) > 1e-6:
        raise ValueError(f"train/validation/test must sum to 1.0, got {train + val + test:.4f}")
    groups = patient_groups(cases)
    labels = patient_labels(cases)
    strata: dict[str, list[str]] = defaultdict(list)
    conflicts: list[str] = []
    for patient_id in groups:
        key = stratum_for(labels.get(patient_id, []))
        if key.startswith("conflict:"):
            conflicts.append(f"{patient_id} ({key.split(':', 1)[1]})")
        strata[key].append(patient_id)

    rng = random.Random(seed)
    splits: dict[str, list[str]] = {name: [] for name in SPLIT_NAMES}
    for name, patient_ids in sorted(strata.items()):
        ordered = sorted(patient_ids)
        rng.shuffle(ordered)
        count = len(ordered)
        if count < 3:
            if logger is not None:
                logger.warning(
                    "Stratum '%s' has %d patient(s); kept in training because a split of that size "
                    "cannot support a reported metric.",
                    name,
                    count,
                )
            for patient_id in ordered:
                splits["train"].extend(groups[patient_id])
            continue
        n_val = max(1, int(round(count * val)))
        n_test = max(1, int(round(count * test)))
        n_train = count - n_val - n_test
        if n_train < 1:
            n_train = 1
            n_test = max(0, count - n_train - n_val)
        for patient_id in ordered[:n_train]:
            splits["train"].extend(groups[patient_id])
        for patient_id in ordered[n_train : n_train + n_val]:
            splits["validation"].extend(groups[patient_id])
        for patient_id in ordered[n_train + n_val : n_train + n_val + n_test]:
            splits["test"].extend(groups[patient_id])

    metadata = describe_splits(cases, splits, method="patient_level_stratified_by_label", seed=seed)
    metadata["ratios"] = {"train": train, "validation": val, "test": test}
    if conflicts:
        metadata["label_conflicts"] = conflicts[:20]
        metadata["label_conflict_count"] = len(conflicts)
        if logger is not None:
            logger.warning(
                "%d patient(s) carry more than one label in the manifest; the split is stratified "
                "on the conflicting key and the inconsistency is recorded. Fix the label table "
                "before reporting metrics.",
                len(conflicts),
            )
    return splits, metadata


def describe_splits(
    cases: Sequence[Mapping[str, Any]],
    splits: Mapping[str, Sequence[str]],
    *,
    method: str,
    seed: int | None = None,
) -> dict[str, Any]:
    by_case = {
        str(entry.get("case_id") or entry.get("id")): entry for entry in cases
    }
    labels = patient_labels(cases)
    detail: dict[str, Any] = {}
    assigned_case_ids: set[str] = set()
    for name in SPLIT_NAMES:
        case_ids = [str(item) for item in splits.get(name, [])]
        patients = sorted(
            {
                str(by_case[case_id].get("patient_id") or by_case[case_id].get("patient"))
                for case_id in case_ids
                if case_id in by_case
            }
        )
        class_counts: dict[str, int] = defaultdict(int)
        for patient_id in patients:
            for label in labels.get(patient_id, [None]):
                class_counts[str(label)] += 1
        detail[name] = {
            "cases": len(case_ids),
            "patients": len(patients),
            "patient_ids": patients,
            "class_distribution": dict(sorted(class_counts.items())),
        }
        assigned_case_ids.update(case_ids)
    detail["method"] = method
    detail["seed"] = seed
    detail["cases_assigned"] = len(assigned_case_ids)
    detail["cases_total"] = len(cases)
    overlap = _patient_overlap(by_case, splits)
    detail["patient_overlap"] = overlap
    detail["leakage_free"] = not overlap
    return detail


def _patient_overlap(
    by_case: Mapping[str, Mapping[str, Any]], splits: Mapping[str, Sequence[str]]
) -> list[str]:
    placement: dict[str, set[str]] = defaultdict(set)
    for name in SPLIT_NAMES:
        for case_id in splits.get(name, []):
            entry = by_case.get(str(case_id))
            if entry is None:
                continue
            patient_id = str(entry.get("patient_id") or entry.get("patient") or case_id)
            placement[patient_id].add(name)
    return sorted(
        patient for patient, names in placement.items() if len(names) > 1
    )[:20]
