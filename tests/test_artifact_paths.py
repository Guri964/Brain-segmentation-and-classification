"""Central artefact-path resolution.

Weights are never committed, so every model file has to be found through one search order and
the failure mode has to be actionable: the operator is told which file is missing and which
directory it is looked for in. These tests pin that behaviour down without needing the real
checkpoints (a temporary ``BASE_DIR`` is used for the layout tests).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from app import config as config_module


@pytest.fixture()
def fake_project(tmp_path, monkeypatch):
    """A throwaway project layout with the three documented locations."""

    root = tmp_path / "project"
    (root / "models" / "weights").mkdir(parents=True)
    (root / "data" / "models").mkdir(parents=True)
    monkeypatch.setattr(config_module, "BASE_DIR", root)
    monkeypatch.delenv("BRAIN_WEIGHTS_DIR", raising=False)
    return root


def test_weights_directory_defaults_to_the_project_models_folder(fake_project):
    assert config_module.weights_directory() == fake_project / "models" / "weights"


def test_weights_directory_honours_the_environment_override(fake_project, tmp_path, monkeypatch):
    elsewhere = tmp_path / "somewhere-else"
    monkeypatch.setenv("BRAIN_WEIGHTS_DIR", str(elsewhere))
    assert config_module.weights_directory() == elsewhere
    # a blank value must not be treated as an override
    monkeypatch.setenv("BRAIN_WEIGHTS_DIR", "   ")
    assert config_module.weights_directory() == fake_project / "models" / "weights"


def test_search_order_is_weights_then_repo_root_then_training_output(fake_project):
    candidates = [str(path) for path in config_module.artifact_candidates("segnet_model.pth")]
    assert candidates == [
        str(fake_project / "models" / "weights" / "segnet_model.pth"),
        str(fake_project / "segnet_model.pth"),
        str(fake_project / "data" / "models" / "segnet_model.pth"),
    ]


def test_a_name_with_a_subdirectory_is_only_looked_for_in_the_weights_directory(fake_project):
    candidates = [str(path) for path in config_module.artifact_candidates("sub/dir/w.bin")]
    assert candidates == [str(fake_project / "models" / "weights" / "sub" / "dir" / "w.bin")]


def test_first_existing_candidate_wins(fake_project):
    weights_file = fake_project / "models" / "weights" / "segnet_model.pth"
    root_file = fake_project / "segnet_model.pth"
    training_file = fake_project / "data" / "models" / "segnet_model.pth"

    training_file.write_bytes(b"training")
    assert config_module.artifact_path("segnet_model.pth") == training_file
    assert config_module.resolve_artifact("segnet_model.pth") == training_file

    root_file.write_bytes(b"root")
    assert config_module.artifact_path("segnet_model.pth") == root_file

    weights_file.write_bytes(b"weights")
    assert config_module.artifact_path("segnet_model.pth") == weights_file


def test_missing_artifact_resolves_to_the_documented_placement_directory(fake_project):
    """The returned path is not a guess: it is where the operator is told to put the file."""

    path = config_module.artifact_path("brain_tumor_xception_final.keras")
    assert path == fake_project / "models" / "weights" / "brain_tumor_xception_final.keras"
    assert not path.exists()
    assert config_module.resolve_artifact("brain_tumor_xception_final.keras") is None


def test_artifact_status_is_serialisable_and_lists_every_location_searched(fake_project):
    status = config_module.artifact_status("metadata.json", expected="class order")
    assert status["name"] == "metadata.json"
    assert status["expected"] == "class order"
    assert status["present"] is False
    assert status["resolved"] is None
    assert status["weights_dir"] == str(fake_project / "models" / "weights")
    assert len(status["searched"]) == 3

    (fake_project / "models" / "weights" / "metadata.json").write_text("{}")
    refreshed = config_module.artifact_status("metadata.json")
    assert refreshed["present"] is True
    assert refreshed["resolved"] == str(fake_project / "models" / "weights" / "metadata.json")


def test_model_paths_all_resolve_through_the_central_helper(fake_project):
    paths = config_module.BaseConfig.MODEL_PATHS
    assert set(paths) == {
        "segnet_brats_4ch",
        "unet_dynamic_512",
        "raunet_brats_4ch",
        "radiomics_mlp_v1",
        "brain_tumor_xception",
    }
    assert paths["brain_tumor_xception"].endswith("brain_tumor_xception_final.keras")
    assert paths["segnet_brats_4ch"].endswith("segnet_model.pth")
    assert paths["radiomics_mlp_v1"].endswith("radiomics_cnn_model.pth")
    # absolute, so a service never depends on the process working directory
    assert all(Path(value).is_absolute() for value in paths.values())


def test_the_classifier_model_card_reports_the_resolved_location():
    """Whatever the deployment looks like, the card must name the file and the directory."""

    from app.config import TestingConfig
    from app.services.image_classification_service import describe_image_classifier

    summary = describe_image_classifier(TestingConfig)
    assert summary["model_id"] == "brain_tumor_xception"
    assert summary["model_path"].endswith("brain_tumor_xception_final.keras")
    assert summary["metadata_path"].endswith("metadata.json")
    assert summary["state"] in {
        "ready",
        "checkpoint_missing",
        "metadata_missing",
        "framework_missing",
    }


def test_weights_are_not_promised_to_the_repository():
    """The shipped .gitignore must keep every weight format out of git."""

    root = Path(__file__).resolve().parent.parent
    patterns = (root / ".gitignore").read_text().splitlines()
    for pattern in ("/models/weights/", "*.pth", "*.pt", "*.keras", "*.h5", "*.hdf5", "*.ckpt"):
        assert pattern in patterns, f"{pattern} missing from .gitignore"


def test_no_weight_file_is_tracked_by_git():
    """Git must not track any checkpoint; the local file, if present, stays on disk."""

    import subprocess

    root = Path(__file__).resolve().parent.parent
    try:
        listing = subprocess.run(
            ["git", "ls-files"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):  # pragma: no cover - no git here
        pytest.skip("git is not available in this environment")
    tracked = [
        line
        for line in listing.splitlines()
        if line.lower().endswith((".pth", ".pt", ".keras", ".h5", ".hdf5", ".ckpt"))
    ]
    assert tracked == [], f"checkpoints are still tracked: {tracked}"
