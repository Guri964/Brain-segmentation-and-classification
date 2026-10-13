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
    (source / "reports").mkdir()
    for name, payload in (
        ("segnet_brats_4ch_real_brats.pt", b"segnet"),
        ("raunet_brats_4ch.pt", b"raunet"),
        ("unet_dynamic_512_real_brats.pt", b"unet"),
    ):
        (source / "checkpoints" / name).write_bytes(payload)
    (source / "data" / "models" / "evaluation_results.json").write_text(json.dumps({"state": "evaluated"}), encoding="utf-8")
    (source / "data" / "models" / "artifacts_manifest.json").write_text(
        json.dumps(
            {
                "produced_by": "notebooks/brain_tumor_training_kaggle_t4.ipynb",
                "generated_at": "2024-01-01T00:00:00+00:00",
                "dataset": {"synthetic": False},
                "models": {"segnet_brats_4ch": {"state": "trained_and_evaluated", "trained_on": "real_brats2020"}},
            }
        ),
        encoding="utf-8",
    )
    (source / "data" / "models" / "classification" / "radiomics_grade_classifier-20240101T000000" / "manifest.json").write_text(
        json.dumps({"model_id": "radiomics_grade_classifier"}), encoding="utf-8"
    )
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
