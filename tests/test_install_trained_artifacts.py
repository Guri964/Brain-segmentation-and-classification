"""Tests for scripts/install_trained_artifacts.py.

The notebook's export is installed by that script, so it has to be conservative: the files that
ship with the repository must never be replaced, an existing file with different content must
not be silently overwritten, and everything installed must be recorded with a hash.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "install_trained_artifacts.py"


def load_module():
    spec = importlib.util.spec_from_file_location("install_trained_artifacts", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["install_trained_artifacts"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def fake_export(tmp_path: Path) -> Path:
    """A miniature version of what the notebook writes into brain_mri_artifacts/."""
    source = tmp_path / "brain_mri_artifacts"
    (source / "checkpoints").mkdir(parents=True)
    (source / "data" / "models" / "classification" / "radiomics_grade_classifier-20240101T000000").mkdir(parents=True)
    (
        source / "data" / "models" / "classification_multiclass" / "brain_tumor_mri_classifier-20240101T000000"
    ).mkdir(parents=True)
    (source / "reports").mkdir()
    for name, payload in (
        ("segnet_brats_4ch_real_brats.pt", b"segnet"),
        ("raunet_brats_4ch.pt", b"raunet"),
        ("unet_dynamic_512_real_brats.pt", b"unet"),
        ("brain_tumor_mri_cnn.pt", b"multiclass-cnn"),
    ):
        (source / "checkpoints" / name).write_bytes(payload)
    (source / "data" / "models" / "evaluation_results.json").write_text(json.dumps({"state": "evaluated"}), encoding="utf-8")
    (source / "data" / "models" / "artifacts_manifest.json").write_text(
        json.dumps(
            {
                "produced_by": "notebooks/brain_tumor_training_kaggle_t4.ipynb",
                "generated_at": "2024-01-01T00:00:00+00:00",
                "dataset": {"synthetic": False, "name": "BraTS 2020", "split_method": "patient_level_stratified_by_grade"},
                "classification_dataset": {
                    "name": "Brain Tumor MRI Dataset",
                    "synthetic": False,
                    "classes": ["glioma", "meningioma", "pituitary", "notumor"],
                    "split_source": "the dataset's own Training/ and Testing/ directories",
                },
                "models": {
                    "segnet_brats_4ch": {
                        "state": "trained_and_evaluated",
                        "task": "segmentation",
                        "trained_on": "real_brats2020",
                    },
                    "brain_tumor_mri_classifier": {
                        "state": "trained_and_evaluated",
                        "task": "multiclass_classification",
                        "classes": ["glioma", "meningioma", "pituitary", "notumor"],
                        "trained_on": "real_brain_tumor_mri",
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    (source / "data" / "models" / "classification" / "radiomics_grade_classifier-20240101T000000" / "manifest.json").write_text(
        json.dumps({"model_id": "radiomics_grade_classifier"}), encoding="utf-8"
    )
    (source / "data" / "models" / "models_index.json").write_text(
        json.dumps(
            {
                "generated_at": "2024-01-01T00:00:00+00:00",
                "artefacts": [
                    {
                        "name": "segnet_brats_4ch",
                        "task": "segmentation",
                        "label_space": "tumour region (binary mask)",
                        "dataset": "BraTS 2020",
                        "dataset_synthetic": False,
                        "state": "compatible",
                        "served_by_application": True,
                    },
                    {
                        "name": "brain_tumor_mri_classifier",
                        "task": "multiclass_classification",
                        "label_space": "glioma / meningioma / pituitary / notumor",
                        "dataset": "Brain Tumor MRI Dataset",
                        "dataset_synthetic": False,
                        "state": "evaluated",
                        "served_by_application": False,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    (source / "data" / "models" / "classification_multiclass" / "brain_tumor_mri_classifier-20240101T000000" / "manifest.json").write_text(
        json.dumps(
            {
                "model_id": "brain_tumor_mri_classifier",
                "task": "multiclass_classification",
                "classes": ["glioma", "meningioma", "pituitary", "notumor"],
            }
        ),
        encoding="utf-8",
    )
    (source / "data" / "models" / "classification_multiclass" / "brain_tumor_mri_classifier-20240101T000000" / "stacked_ensemble.joblib").write_bytes(b"stack")
    (source / "reports" / "comparison_table.csv").write_text("model,dice\n", encoding="utf-8")
    (source / "README.md").write_text("how to install\n", encoding="utf-8")
    return source


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_installs_export_without_touching_shipped_files(fake_export: Path, tmp_path: Path):
    module = load_module()
    data_dir = tmp_path / "data"
    shipped = REPO_ROOT / "segnet_model.pth"
    shipped_before = digest(shipped) if shipped.exists() else None

    assert module.main(["--source", str(fake_export), "--data-dir", str(data_dir)]) == 0

    for name in ("segnet_brats_4ch_real_brats.pt", "raunet_brats_4ch.pt", "unet_dynamic_512_real_brats.pt"):
        assert (data_dir / "models" / name).read_bytes() in {b"segnet", b"raunet", b"unet"}
    assert (data_dir / "models" / "evaluation_results.json").exists()
    assert (data_dir / "models" / "artifacts_manifest.json").exists()
    assert (data_dir / "models" / "classification" / "radiomics_grade_classifier-20240101T000000" / "manifest.json").exists()
    assert (data_dir / "models" / "reports" / "comparison_table.csv").exists()

    record = json.loads((data_dir / "models" / "install_record.json").read_text(encoding="utf-8"))
    assert record["dry_run"] is False
    states = {entry["state"] for entry in record["installed"]}
    assert states == {"installed"}
    assert record["provenance"]["synthetic"] is False
    assert record["environment"]["BRAIN_MODEL_SEGNET"].endswith("segnet_brats_4ch_real_brats.pt")
    assert record["environment"]["BRAIN_MODEL_RAUNET"].endswith("raunet_brats_4ch.pt")
    for entry in record["installed"]:
        if entry["state"] == "installed":
            assert entry["sha256"] == digest(Path(entry["destination"]))

    if shipped_before is not None:
        assert digest(shipped) == shipped_before, "the shipped SegNet checkpoint was modified"


def test_dry_run_writes_nothing(fake_export: Path, tmp_path: Path):
    module = load_module()
    data_dir = tmp_path / "data"
    assert module.main(["--source", str(fake_export), "--data-dir", str(data_dir), "--dry-run"]) == 0
    assert not (data_dir / "models").exists()


def test_existing_different_file_is_kept_until_forced(fake_export: Path, tmp_path: Path):
    module = load_module()
    data_dir = tmp_path / "data"
    module.main(["--source", str(fake_export), "--data-dir", str(data_dir)])

    # same content again -> recognised as already installed, not copied over
    assert module.main(["--source", str(fake_export), "--data-dir", str(data_dir)]) == 0
    record = json.loads((data_dir / "models" / "install_record.json").read_text(encoding="utf-8"))
    assert {entry["state"] for entry in record["installed"]} == {"already_installed"}

    # different content -> refused, the destination is untouched
    (fake_export / "checkpoints" / "raunet_brats_4ch.pt").write_bytes(b"raunet-v2")
    target = data_dir / "models" / "raunet_brats_4ch.pt"
    assert module.main(["--source", str(fake_export), "--data-dir", str(data_dir)]) == 0
    assert target.read_bytes() == b"raunet"
    record = json.loads((data_dir / "models" / "install_record.json").read_text(encoding="utf-8"))
    raunet = next(entry for entry in record["installed"] if entry["file"] == "raunet_brats_4ch.pt")
    assert raunet["state"] == "kept_existing"

    # --force replaces it and keeps a backup
    assert module.main(["--source", str(fake_export), "--data-dir", str(data_dir), "--force"]) == 0
    assert target.read_bytes() == b"raunet-v2"
    record = json.loads((data_dir / "models" / "install_record.json").read_text(encoding="utf-8"))
    raunet = next(entry for entry in record["installed"] if entry["file"] == "raunet_brats_4ch.pt")
    assert raunet["state"] == "installed"
    backup = Path(raunet["backup"])
    assert backup.exists() and backup.read_bytes() == b"raunet"


def test_rejects_something_that_is_not_the_export(tmp_path: Path):
    module = load_module()
    not_an_export = tmp_path / "random"
    not_an_export.mkdir()
    with pytest.raises(SystemExit):
        module.main(["--source", str(not_an_export), "--data-dir", str(tmp_path / "data")])


def test_multiclass_artefacts_are_installed_and_labelled(tmp_path, fake_export):
    """The second dataset's artefacts land next to the first one's, labelled with task and dataset."""
    module = load_module()
    data_dir = tmp_path / "data"
    report = module.build_report(fake_export, data_dir, force=False, dry_run=False, logger=module.configure_logging(False))

    installed = {Path(entry["destination"]).name: entry for entry in report["installed"]}
    assert "brain_tumor_mri_cnn.pt" in installed
    multiclass_root = data_dir / "models" / "classification_multiclass" / "brain_tumor_mri_classifier-20240101T000000"
    assert (multiclass_root / "manifest.json").exists()
    assert (multiclass_root / "stacked_ensemble.joblib").read_bytes() == b"stack"
    assert report["classification_multiclass_versions"] == ["brain_tumor_mri_classifier-20240101T000000"]
    assert report["classification_versions"] == ["radiomics_grade_classifier-20240101T000000"]

    # the index travels with the artefacts and says what each one is
    assert (data_dir / "models" / "models_index.json").exists()
    by_name = {entry["name"]: entry for entry in report["artefacts"]}
    assert by_name["brain_tumor_mri_classifier"]["task"] == "multiclass_classification"
    assert by_name["brain_tumor_mri_classifier"]["served_by_application"] is False
    assert by_name["segnet_brats_4ch"]["task"] == "segmentation"

    # provenance keeps the two datasets apart
    provenance = report["provenance"]
    assert provenance["dataset"]["name"] == "BraTS 2020"
    assert provenance["classification_dataset"]["name"] == "Brain Tumor MRI Dataset"
    assert provenance["classification_dataset"]["classes"] == ["glioma", "meningioma", "pituitary", "notumor"]
    assert provenance["models"]["brain_tumor_mri_classifier"]["task"] == "multiclass_classification"
    assert provenance["models"]["segnet_brats_4ch"]["task"] == "segmentation"

    # and the CNN does not become the model the API serves
    assert "brain_tumor_mri_cnn" not in {var.replace("BRAIN_MODEL_", "") for var in report["environment"]}
    assert "BRAIN_CLASSIFIER_DIR" in report["environment"]


def test_missing_multiclass_directory_is_reported_not_silently_skipped(tmp_path, fake_export):
    """An export from a run that skipped the second dataset says so instead of pretending."""
    import shutil

    shutil.rmtree(fake_export / "data" / "models" / "classification_multiclass")
    (fake_export / "data" / "models" / "models_index.json").unlink()
    module = load_module()
    report = module.build_report(fake_export, tmp_path / "data", force=False, dry_run=True, logger=module.configure_logging(False))
    assert "data/models/classification_multiclass" in report["missing"]
    assert "artefacts" not in report
