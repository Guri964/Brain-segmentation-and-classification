from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest


def _table(path: Path, *, patients: int = 12, per_patient: int = 1, classes=("HGG", "LGG")) -> Path:
    """A tiny synthetic feature table: two classes that are trivially separable."""
    import random

    rng = random.Random(7)
    rows = []
    for patient in range(patients):
        label = classes[patient % len(classes)]
        for slice_index in range(per_patient):
            offset = 5.0 if label == classes[0] else -5.0
            rows.append(
                {
                    "case_id": f"volume_{patient:03d}_{slice_index}",
                    "patient_id": f"volume_{patient:03d}",
                    "label": label,
                    "firstorder_mean": offset + rng.gauss(0, 0.4),
                    "glcm_contrast": offset * 0.5 + rng.gauss(0, 0.4),
                    "shape_volume": 100.0 + patient + rng.gauss(0, 0.5),
                    "spacing_x": "",
                }
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_feature_names_exclude_identifiers_and_labels():
    from _classification import feature_names, select_rows

    columns = ["case_id", "patient_id", "label", "split", "firstorder_mean", "glcm_contrast", "notes"]
    raw = [
        {"case_id": "a", "patient_id": "p1", "label": "HGG", "split": "train", "firstorder_mean": "1.5", "glcm_contrast": "2.5", "notes": "text"},
        {"case_id": "b", "patient_id": "p2", "label": "LGG", "split": "test", "firstorder_mean": "0.5", "glcm_contrast": "0.1", "notes": ""},
    ]
    rows, rejected = select_rows(
        [
            {"case_id": "a", "patient_id": "p1", "label": "HGG", "split": "train", "firstorder_mean": "1.5", "glcm_contrast": "2.5", "notes": "text"},
            {"case_id": "b", "patient_id": "p2", "label": "", "firstorder_mean": "1.0"},
            {"case_id": "c", "label": "LGG", "firstorder_mean": "1.0"},
        ],
        label_column="label",
        patient_column="patient_id",
        id_column="case_id",
    )
    assert len(rows) == 1
    assert {entry["case_id"] for entry in rejected} == {"b", "c"}
    names = feature_names(raw, columns, None)
    assert names == ["firstorder_mean", "glcm_contrast"]
    assert "label" not in names and "case_id" not in names and "notes" not in names
    kept = feature_names(raw, columns, ["firstorder_mean"])
    assert kept == ["firstorder_mean"]
    with pytest.raises(ValueError):
        feature_names(raw, columns, ["not_a_column"])


def test_imputation_and_standardisation_use_training_rows_only():
    from _classification import impute, standardise

    matrix = np.array([[1.0, np.nan], [3.0, 10.0], [np.nan, 20.0]])
    missing = np.isnan(matrix)
    filled, medians = impute(matrix, missing, train_indices=[0, 1])
    assert medians[0] == pytest.approx(2.0)
    assert filled[2, 0] == pytest.approx(2.0)
    assert filled[0, 1] == pytest.approx(10.0)

    scaled, mean, scale = standardise(filled, train_indices=[0, 1])
    train_scaled = scaled[[0, 1]]
    assert np.allclose(train_scaled.mean(axis=0), 0.0, atol=1e-9)
    # The held-out row is transformed with the training statistics, not its own.
    assert scaled[2, 0] == pytest.approx((2.0 - mean[0]) / scale[0])


def test_evaluate_reports_per_class_metrics_and_confusion_matrix():
    from _classification import evaluate

    scores = np.array([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4], [0.1, 0.9]])
    report = evaluate([0, 1, 0, 1], scores, 2)
    assert report["accuracy"] == pytest.approx(1.0)
    assert report["macro_f1"] == pytest.approx(1.0)
    assert report["roc_auc"] == pytest.approx(1.0)
    assert set(report["per_class"]) == {"0", "1"}
    assert report["confusion_matrix"] == [[2, 0], [0, 2]]
    assert report["samples"] == 4


def test_patient_splits_never_share_a_patient():
    from _classification import assign_patient_splits, split_summary

    rows = [
        {"case_id": f"c{index}", "patient_id": f"p{index}", "label": "HGG" if index % 2 else "LGG", "values": {}}
        for index in range(20)
    ]
    splits = assign_patient_splits(rows, train=0.6, val=0.2, test=0.2, seed=11)
    seen: dict[str, str] = {}
    for name, indices in splits.items():
        for index in indices:
            patient = rows[index]["patient_id"]
            assert seen.setdefault(patient, name) == name
    summary = split_summary(rows, splits)
    assert {"train", "validation", "test"} <= set(summary)
    assert summary["leakage_free"] is True
    assert summary["patient_overlap"] == []
    assert sum(summary[name]["rows"] for name in ("train", "validation", "test")) == len(rows)
    assert sum(
        summary[name]["rows"] is not None for name in ("train", "validation", "test")
    ) == 3


def test_declared_splits_are_respected_only_when_complete():
    from _classification import declared_splits

    rows = [
        {"declared_split": "train"},
        {"declared_split": "valid"},
        {"declared_split": "test"},
    ]
    assert declared_splits(rows) == {"train": [0], "validation": [1], "test": [2]}
    # An incomplete or unknown column is ignored rather than half-honoured.
    assert declared_splits(rows[:2]) is None
    assert declared_splits([{"declared_split": ""}]) is None
    assert declared_splits([*rows, {"declared_split": "unassigned"}]) is None


def test_trainer_writes_a_servable_artefact_and_honest_metrics(tmp_path):
    from train_classification import main

    table = _table(tmp_path / "features.csv", patients=16)
    data_dir = tmp_path / "data"
    exit_code = main(
        [
            "--features", str(table),
            "--data-dir", str(data_dir),
            "--model-id", "unit_grade_model",
            "--target-description", "unit-test label, synthetic rows",
            "--seed", "3",
        ]
    )
    assert exit_code == 0
    artefacts = sorted((data_dir / "models" / "classification").glob("unit_grade_model-*/manifest.json"))
    assert artefacts, "no artefact manifest was written"
    manifest = json.loads(artefacts[-1].read_text(encoding="utf-8"))
    assert manifest["feature_schema"] and "label" not in manifest["feature_schema"]
    assert manifest["label_mapping"] == {"0": "HGG", "1": "LGG"}
    assert manifest["preprocessing"]["fit_on"].startswith("training split")
    assert manifest["preprocessing"]["type"] == "standard_scaler"
    assert len(manifest["preprocessing"]["mean"]) == len(manifest["feature_schema"])
    assert manifest["leakage_check"]["passed"] is True
    assert manifest["leakage_check"]["method"]
    assert manifest["caveats"], "an artefact without caveats would overstate itself"
    metrics = manifest["metrics"]
    assert metrics["evaluated_model"] in {"baseline_logistic_regression", "improved_mlp"}
    assert metrics["evaluation_split"]
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert set(manifest["model_comparison"]) == {"baseline_logistic_regression", "improved_mlp"}
    for name, entry in manifest["model_comparison"].items():
        assert entry["validation"] is not None, name
        assert entry["test"] is not None, name
    assert manifest["metrics"]["class_distribution"]
    assert all(value > 0 for value in manifest["preprocessing"]["scale"])
    assert manifest["metrics"]["splits"] if "splits" in manifest["metrics"] else True
    assert set(manifest["splits"]) == {"train", "validation", "test"}

    index = json.loads((data_dir / "models" / "evaluation_results.json").read_text(encoding="utf-8"))
    assert index["classification"]["state"] == "evaluated"
    assert index["classification"]["model_id"] == "unit_grade_model"
    assert index["classification"]["produced_by"] == "scripts/train_classification.py"
    assert index["classification"]["comparison"]


def test_trainer_refuses_a_table_without_patient_identifiers(tmp_path):
    from train_classification import main

    table = tmp_path / "no_patients.csv"
    table.write_text("case_id,label,firstorder_mean\nc1,HGG,1.0\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        main(["--features", str(table), "--data-dir", str(tmp_path / "data"), "--model-id", "nope"])
    assert not (tmp_path / "data" / "models" / "classification").exists()


def test_served_classifier_reports_metrics_and_refuses_incomplete_vectors(tmp_path):
    from dataclasses import dataclass

    from train_classification import main

    table = _table(tmp_path / "features.csv", patients=16)
    data_dir = tmp_path / "data"
    assert main(
        [
            "--features", str(table),
            "--data-dir", str(data_dir),
            "--model-id", "served_grade_model",
            "--target-description", "unit-test label, synthetic rows",
        ]
    ) == 0

    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
    from app.config import DevelopmentConfig
    from app.services.classification_service import ClassificationService

    @dataclass
    class Config(DevelopmentConfig):
        CLASSIFIER_ARTIFACT_DIR: object = data_dir / "models" / "classification"

    service = ClassificationService(Config)
    status = service.status()
    assert status["available"] is True
    assert status["model_id"] == "served_grade_model"
    assert status["feature_count"] == len(status["feature_schema"])
    assert status["metrics"]["evaluated_model"]
    assert status["caveats"]

    outcome = service.classify({name: 1.0 for name in status["feature_schema"]}, explain=True).to_dict()
    assert outcome["available"] is True
    assert outcome["predicted_label"] in {"HGG", "LGG"}
    assert pytest.approx(sum(outcome["scores"]), abs=1e-6) == 1.0
    assert outcome["feature_contributions"]

    incomplete = service.classify(
        {name: 1.0 for name in status["feature_schema"][:-2]}, explain=False
    ).to_dict()
    assert incomplete["available"] is False
    assert incomplete["missing_features"]


def test_classification_status_is_unavailable_without_a_valid_artefact(tmp_path):
    from dataclasses import dataclass

    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
    from app.config import DevelopmentConfig
    from app.services.classification_service import ClassificationService

    @dataclass
    class Config(DevelopmentConfig):
        CLASSIFIER_ARTIFACT_DIR: object = tmp_path / "empty"

    (tmp_path / "empty").mkdir(parents=True, exist_ok=True)
    payload = ClassificationService(Config).status()
    assert payload["available"] is False
    assert payload["reason"]
    assert payload["required_artifacts"]


def test_classification_endpoint_exposes_the_model_card(client):
    response = client.get("/api/classification")
    assert response.status_code == 200
    payload = response.get_json()
    assert "status" in payload and "label_semantics" in payload and "xai" in payload
    assert payload["label_semantics"]["derived_from_mask"] is False
    assert payload["disclaimer"]
    assert payload["training"]["recommended"].endswith("train_classification.py")
