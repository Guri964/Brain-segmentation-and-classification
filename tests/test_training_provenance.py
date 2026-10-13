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
    "classification_dataset": {
        "name": "Brain Tumor MRI Dataset",
        "synthetic": False,
        "classes": ["glioma", "meningioma", "pituitary", "notumor"],
    },
    "models": {
        "segnet_brats_4ch": {
            "state": "trained_and_evaluated",
            "task": "segmentation",
            "trained_on": "real_brats2020",
            "evaluated_on": "held-out patient-level test split",
        },
        "brain_tumor_mri_classifier": {
            "state": "trained_and_evaluated",
            "task": "multiclass_classification",
            "classes": ["glioma", "meningioma", "pituitary", "notumor"],
            "trained_on": "real_brain_tumor_mri",
            "evaluated_on": "the dataset's own Testing directory",
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


def test_each_artefact_carries_its_task_and_label_space(tmp_path):
    _write(tmp_path, REAL_MANIFEST)
    manifest = provenance.load_artifact_manifest(tmp_path)
    segmentation = provenance.describe("segnet_brats_4ch", manifest, data_dir=tmp_path)
    assert segmentation["task"] == "segmentation"
    assert segmentation["state"] == provenance.STATE_REAL

    multiclass = provenance.describe("brain_tumor_mri_classifier", manifest, data_dir=tmp_path)
    assert multiclass["task"] == "multiclass_classification"
    assert multiclass["classes"] == ["glioma", "meningioma", "pituitary", "notumor"]
    # the second dataset's own block decides, not the BraTS one
    assert multiclass["dataset"] == "Brain Tumor MRI Dataset"
    assert multiclass["state"] == provenance.STATE_REAL


def test_second_dataset_marked_synthetic_stays_synthetic(tmp_path):
    payload = json.loads(json.dumps(REAL_MANIFEST))
    payload["classification_dataset"]["synthetic"] = True
    payload["classification_dataset"]["name"] = "Synthetic image-classification fixture"
    payload["models"]["brain_tumor_mri_classifier"]["trained_on"] = "synthetic_fixture"
    _write(tmp_path, payload)
    described = provenance.describe(
        "brain_tumor_mri_classifier", provenance.load_artifact_manifest(tmp_path), data_dir=tmp_path
    )
    assert described["state"] == provenance.STATE_SYNTHETIC
    assert "not results" in described["serving_note"]


def test_unverified_second_dataset_is_not_called_real(tmp_path):
    payload = json.loads(json.dumps(REAL_MANIFEST))
    payload["classification_dataset"]["synthetic"] = None
    _write(tmp_path, payload)
    described = provenance.describe(
        "brain_tumor_mri_classifier", provenance.load_artifact_manifest(tmp_path), data_dir=tmp_path
    )
    assert described["state"] == provenance.STATE_UNVERIFIED
    assert "not presented as trained on real data" in described["serving_note"]


def test_legacy_checkpoint_does_not_borrow_the_trained_classifiers_provenance(tmp_path):
    """`radiomics_mlp_v1` is the shipped radiomics_cnn_model.pth, not the trained artefact."""
    _write(tmp_path, REAL_MANIFEST)
    described = provenance.describe("radiomics_mlp_v1", provenance.load_artifact_manifest(tmp_path), data_dir=tmp_path)
    assert described["state"] == provenance.STATE_UNKNOWN
    assert described["trained_on"] is None


def test_models_index_lists_every_artefact(tmp_path):
    models = tmp_path / "models"
    models.mkdir(parents=True)
    (models / provenance.INDEX_NAME).write_text(
        json.dumps(
            {
                "artefacts": [
                    {"name": "segnet_brats_4ch", "task": "segmentation", "served_by_application": True},
                    {"name": "brain_tumor_mri_classifier", "task": "multiclass_classification", "served_by_application": False},
                ]
            }
        ),
        encoding="utf-8",
    )
    entries = provenance.load_models_index(tmp_path)
    assert [entry["name"] for entry in entries] == ["segnet_brats_4ch", "brain_tumor_mri_classifier"]
    assert entries[1]["served_by_application"] is False


def test_models_index_falls_back_to_installed_manifests(tmp_path):
    """Without an index, whatever classifiers are installed are still identifiable."""
    directory = tmp_path / "models" / "classification_multiclass" / "brain_tumor_mri_classifier-x"
    directory.mkdir(parents=True)
    (directory / "manifest.json").write_text(
        json.dumps(
            {
                "model_id": "brain_tumor_mri_classifier",
                "task": "multiclass_classification",
                "classes": ["glioma", "meningioma", "pituitary", "notumor"],
                "dataset": {"name": "Brain Tumor MRI Dataset", "synthetic": True},
            }
        ),
        encoding="utf-8",
    )
    entries = provenance.load_models_index(tmp_path)
    assert len(entries) == 1
    assert entries[0]["task"] == "multiclass_classification"
    assert entries[0]["dataset_synthetic"] is True
    assert entries[0]["served_by_application"] is False


def test_classification_status_reports_the_other_trained_artefacts(app_config, tmp_path):
    directory = tmp_path / "classification" / "radiomics_grade_classifier-test"
    _write_classifier(directory, dataset={"name": "BraTS 2020", "synthetic": False})
    models = tmp_path / "models"
    models.mkdir(parents=True, exist_ok=True)
    (models / provenance.INDEX_NAME).write_text(
        json.dumps({"artefacts": [{"name": "brain_tumor_mri_classifier", "task": "multiclass_classification"}]}),
        encoding="utf-8",
    )
    config = sandbox_config(
        app_config,
        CLASSIFIER_ARTIFACT_DIR=tmp_path / "classification",
        DATA_DIR=tmp_path,
    )
    status = ClassificationService(config).status()
    assert status["trained_artefacts"][0]["name"] == "brain_tumor_mri_classifier"
    assert status["trained_artefacts"][0]["task"] == "multiclass_classification"
