"""What the platform is allowed to say about where a model came from.

Requirement in the delivery: the interface and the API must distinguish

* implemented but not trained,
* trained on a synthetic fixture,
* trained on the real dataset,
* evaluated on a held-out test split,

and a fixture-trained model must never be presented as trained on real data. These tests pin
that behaviour down, including the default: **no recorded provenance is never promoted to
"trained on real data"**.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.ml import training_provenance as provenance
from app.services.classification_service import ClassificationService
from app.services.model_health_service import ModelHealthService

SYNTHETIC_MANIFEST = {
    "generated_at": "2026-01-01T00:00:00Z",
    "produced_by": "notebooks/brain_tumor_training_kaggle_t4.ipynb",
    "dataset": {
        "name": "Synthetic BraTS-format fixture",
        "synthetic": True,
        "split_method": "patient_level_stratified_by_grade",
        "split_seed": 7,
    },
    "models": {
        "segnet_brats_4ch": {
            "state": "trained_and_evaluated",
            "trained_on": "synthetic_fixture",
            "evaluated_on": "held-out patient-level test split",
            "exported_file": "checkpoints/segnet_brats_4ch_real_brats.pt",
        }
    },
}

REAL_MANIFEST = {
    "generated_at": "2026-01-02T00:00:00Z",
    "produced_by": "notebooks/brain_tumor_training_kaggle_t4.ipynb",
    "dataset": {"name": "BraTS 2020 training data", "synthetic": False},
    "models": {
        "segnet_brats_4ch": {
            "state": "trained_and_evaluated",
            "trained_on": "real_brats2020",
            "evaluated_on": "held-out patient-level test split",
        },
        "raunet_brats_4ch": {"state": "not_trained", "trained_on": "not_trained"},
        "unet_dynamic_512_real_brats": {
            "state": "trained_not_evaluated",
            "trained_on": "real_brats2020",
            "evaluated_on": None,
        },
    },
}


def sandbox_config(app_config, **overrides):
    """The session test config with a few attributes replaced (services accept the class)."""
    return type("SandboxOverride", (app_config,), dict(overrides))


def _write(data_dir: Path, payload: dict) -> None:
    models = data_dir / "models"
    models.mkdir(parents=True, exist_ok=True)
    (models / provenance.MANIFEST_NAME).write_text(json.dumps(payload), encoding="utf-8")


def test_state_is_derived_from_what_was_recorded():
    assert provenance.state_from_flags(trained=False, synthetic=True, evaluated_on=None) == provenance.STATE_NOT_TRAINED
    assert provenance.state_from_flags(trained=True, synthetic=True, evaluated_on="test") == provenance.STATE_SYNTHETIC
    assert provenance.state_from_flags(trained=True, synthetic=False, evaluated_on="test") == provenance.STATE_REAL
    assert (
        provenance.state_from_flags(trained=True, synthetic=False, evaluated_on=None)
        == provenance.STATE_TRAINED_NOT_EVALUATED
    )
    # An unknown dataset is its own state: it is not silently called real.
    assert provenance.state_from_flags(trained=True, synthetic=None, evaluated_on="test") == provenance.STATE_UNVERIFIED


def test_missing_manifest_claims_nothing(tmp_path):
    described = provenance.describe("segnet_brats_4ch", provenance.load_artifact_manifest(tmp_path))
    assert described["state"] == provenance.STATE_UNKNOWN
    assert described["label"] == "No provenance recorded"
    assert "never trained or" in described["note"]
    assert described["synthetic"] is None


def test_synthetic_manifest_is_reported_as_synthetic(tmp_path):
    _write(tmp_path, SYNTHETIC_MANIFEST)
    described = provenance.describe("segnet_brats_4ch", provenance.load_artifact_manifest(tmp_path), data_dir=tmp_path)
    assert described["state"] == provenance.STATE_SYNTHETIC
    assert described["synthetic"] is True
    assert "not results" in described["serving_note"]


def test_real_manifest_is_reported_as_real(tmp_path):
    _write(tmp_path, REAL_MANIFEST)
    manifest = provenance.load_artifact_manifest(tmp_path)
    real = provenance.describe("segnet_brats_4ch", manifest, data_dir=tmp_path)
    assert real["state"] == provenance.STATE_REAL
    untrained = provenance.describe("raunet_brats_4ch", manifest, data_dir=tmp_path)
    assert untrained["state"] == provenance.STATE_NOT_TRAINED
    # `unet_dynamic_512` is registered under the export name, not the shipped file name.
    partial = provenance.describe("unet_dynamic_512", manifest, data_dir=tmp_path)
    assert partial["state"] == provenance.STATE_TRAINED_NOT_EVALUATED


def test_dataset_that_cannot_be_verified_is_not_called_real(tmp_path):
    payload = json.loads(json.dumps(REAL_MANIFEST))
    payload["dataset"]["synthetic"] = None
    payload["dataset"]["provenance_note"] = "volume count too small to identify the release"
    _write(tmp_path, payload)
    described = provenance.describe("segnet_brats_4ch", provenance.load_artifact_manifest(tmp_path), data_dir=tmp_path)
    assert described["state"] == provenance.STATE_UNVERIFIED
    assert "not presented as trained on real data" in described["serving_note"]


def test_health_report_carries_the_provenance(app_config, tmp_path):
    from app.ml.registry import ModelRegistry

    _write(tmp_path, SYNTHETIC_MANIFEST)
    config = sandbox_config(app_config, DATA_DIR=tmp_path)
    service = ModelHealthService(ModelRegistry(config), config)
    report = service.check("segnet_brats_4ch").to_dict()
    assert report["training"]["state"] == provenance.STATE_SYNTHETIC
    summary = service.summary()
    assert summary["models"][0]["training"]["state"] in set(provenance.LABELS)


def test_health_reports_no_provenance_when_none_is_recorded(app_config, tmp_path):
    from app.ml.registry import ModelRegistry

    config = sandbox_config(app_config, DATA_DIR=tmp_path)
    service = ModelHealthService(ModelRegistry(config), config)
    report = service.check("segnet_brats_4ch").to_dict()
    assert report["training"]["state"] == provenance.STATE_UNKNOWN


def test_api_exposes_the_training_provenance(client):
    payload = client.get("/api/models").get_json()
    assert payload["models"], "the registry lists no model"
    for entry in payload["models"]:
        training = entry["health"]["training"]
        assert training["state"] in set(provenance.LABELS)
        assert training["label"]
        assert training["serving_note"]


def _write_classifier(directory: Path, *, dataset: dict | None, evaluated: bool = True) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "model.joblib").write_bytes(b"placeholder")
    manifest = {
        "model_id": "radiomics_grade_classifier",
        "version": "test",
        "kind": "sklearn",
        "model_file": "model.joblib",
        "feature_schema": ["firstorder_mean"],
        "label_mapping": {"0": "HGG", "1": "LGG"},
        "preprocessing": {"type": "standard_scaler", "mean": [0.0], "scale": [1.0]},
        "trainer": "notebooks/brain_tumor_training_kaggle_t4.ipynb",
        "trained_at": "2026-01-01T00:00:00Z",
        "metrics": {"evaluation_split": "held-out test split (patient level)"} if evaluated else {},
        "feature_sources": {"mask_source": "reference mask"},
    }
    if dataset is not None:
        manifest["dataset"] = dataset
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


@pytest.mark.parametrize(
    "dataset,expected",
    [
        ({"name": "Synthetic BraTS-format fixture", "synthetic": True}, provenance.STATE_SYNTHETIC),
        ({"name": "BraTS 2020 training data", "synthetic": False}, provenance.STATE_REAL),
        ({"name": "Unverified HDF5 tree", "synthetic": None}, provenance.STATE_UNVERIFIED),
        (None, provenance.STATE_UNVERIFIED),
    ],
)
def test_classifier_provenance(app_config, tmp_path, dataset, expected):
    directory = tmp_path / "classification" / "radiomics_grade_classifier-test"
    _write_classifier(directory, dataset=dataset)
    config = sandbox_config(app_config, CLASSIFIER_ARTIFACT_DIR=tmp_path / "classification")
    service = ClassificationService(config)
    training = service.training_provenance()
    assert training["state"] == expected, training
    assert training["serving_note"]
    if expected == provenance.STATE_SYNTHETIC:
        assert "synthetic fixture" in training["serving_note"]
        assert training["synthetic"] is True
        assert service.status()["training"]["state"] == provenance.STATE_SYNTHETIC


def test_classifier_provenance_without_artefact(app_config, tmp_path):
    config = sandbox_config(app_config, CLASSIFIER_ARTIFACT_DIR=tmp_path / "missing")
    training = ClassificationService(config).training_provenance()
    assert training["state"] == provenance.STATE_NOT_TRAINED
    assert "no prediction" in training["serving_note"].lower()


def test_untrained_classifier_is_reported_as_trained_but_unscored(app_config, tmp_path):
    directory = tmp_path / "classification" / "radiomics_grade_classifier-test"
    _write_classifier(directory, dataset={"name": "BraTS 2020", "synthetic": False}, evaluated=False)
    config = sandbox_config(app_config, CLASSIFIER_ARTIFACT_DIR=tmp_path / "classification")
    training = ClassificationService(config).training_provenance()
    assert training["state"] == provenance.STATE_TRAINED_NOT_EVALUATED
    assert "no metric is claimed" in training["serving_note"]
