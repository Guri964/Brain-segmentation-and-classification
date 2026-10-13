"""What each checkpoint was trained on, read from what the training run recorded.

The platform has to be able to answer a plain question: *was this model trained on the real
dataset, on a synthetic fixture, or not at all?* The answer cannot be guessed from a file name,
so it is written by the training run and read back here:

* `scripts/install_trained_artifacts.py` installs the Kaggle notebook's export and writes
  `<data-dir>/models/artifacts_manifest.json` (the notebook's own provenance manifest),
* the same file is produced by `scripts/train_classification.py`, `scripts/train_segnet.py` and
  friends through their run records when they are pointed at `<data-dir>/models/`,
* when nothing is recorded the answer is `no_provenance_recorded`, with the remedy. It is never
  promoted to "trained on real data" by default.

Nothing here trains or loads a model: it only reads a JSON file.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

MANIFEST_NAME = "artifacts_manifest.json"
INDEX_NAME = "models_index.json"
INSTALL_RECORD_NAME = "install_record.json"

# registry model id -> key used in the provenance manifest
MODEL_ID_TO_MANIFEST_KEY = {
    "segnet_brats_4ch": "segnet_brats_4ch",
    "unet_dynamic_512": "unet_dynamic_512_real_brats",
    "raunet_brats_4ch": "raunet_brats_4ch",
    # Deliberately absent: `radiomics_mlp_v1` is the *shipped* radiomics_cnn_model.pth, not the
    # classifier a training run produced. Mapping it to the trained artefact would report the
    # trained model's provenance for a checkpoint that was never trained here. Its provenance
    # comes from the classifier artefact itself, which /api/classification reads.
    # trained on the second dataset (multiclass tumour type), identified but not served
    "brain_tumor_mri_cnn": "brain_tumor_mri_cnn",
    "brain_tumor_mri_classifier": "brain_tumor_mri_classifier",
}

STATE_REAL = "trained_on_real_data"
STATE_SYNTHETIC = "trained_on_synthetic_fixture"
STATE_TRAINED_NOT_EVALUATED = "trained_not_evaluated"
STATE_NOT_TRAINED = "not_trained"
STATE_UNKNOWN = "no_provenance_recorded"

STATE_UNVERIFIED = "trained_on_unverified_dataset"

LABELS = {
    STATE_REAL: "Trained on the real dataset",
    STATE_SYNTHETIC: "Trained on a synthetic fixture",
    STATE_TRAINED_NOT_EVALUATED: "Trained, not evaluated on a held-out split",
    STATE_UNVERIFIED: "Trained on a dataset whose provenance was not established",
    STATE_NOT_TRAINED: "Not trained",
    STATE_UNKNOWN: "No provenance recorded",
}

# What the interface has to show next to a prediction, for every state.
SERVING_NOTES = {
    STATE_REAL: "Trained and evaluated on the real dataset by the training run recorded above.",
    STATE_SYNTHETIC: (
        "Trained on a synthetic fixture: the images, masks and labels it was fitted on are "
        "generated. Its predictions, and every metric recorded with it, describe that fixture "
        "only and are not results."
    ),
    STATE_TRAINED_NOT_EVALUATED: (
        "Trained, but the training run recorded no evaluation on a held-out split, so no metric "
        "is claimed for it."
    ),
    STATE_UNVERIFIED: (
        "The training run could not establish whether the data was the real dataset or a "
        "fixture, so this model is not presented as trained on real data."
    ),
    STATE_NOT_TRAINED: "No trained weights exist for this model.",
    STATE_UNKNOWN: "Nothing is recorded about this model's training data, so nothing is claimed.",
}


def state_from_flags(*, trained: bool, synthetic: bool | None, evaluated_on: str | None) -> str:
    """The single place where a state is derived from what a training run recorded."""
    if not trained:
        return STATE_NOT_TRAINED
    if synthetic is True:
        return STATE_SYNTHETIC
    if synthetic is False:
        return STATE_REAL if evaluated_on else STATE_TRAINED_NOT_EVALUATED
    return STATE_UNVERIFIED


def _read_json(path: Path) -> dict | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def manifest_path(data_dir: Path | str) -> Path:
    return Path(data_dir) / "models" / MANIFEST_NAME


def load_artifact_manifest(data_dir: Path | str) -> dict | None:
    """Return the provenance manifest, or None when the deployment has never recorded one."""
    return _read_json(manifest_path(data_dir))


def models_index_path(data_dir: Path | str) -> Path:
    return Path(data_dir) / "models" / INDEX_NAME


def load_models_index(data_dir: Path | str) -> list[dict]:
    """Every artefact the training run exported, with its task, label space and dataset.

    Read from `models_index.json` (written by the notebook) and, when that file is absent, from
    the manifests of whatever classifiers are installed. An empty list means "nothing recorded".
    """
    payload = _read_json(models_index_path(data_dir))
    if payload and isinstance(payload.get("artefacts"), list):
        return [entry for entry in payload["artefacts"] if isinstance(entry, dict)]
    discovered: list[dict] = []
    for directory in sorted((Path(data_dir) / "models").glob("classification*/*/manifest.json")):
        manifest = _read_json(directory) or {}
        discovered.append(
            {
                "name": manifest.get("model_id", directory.parent.name),
                "task": manifest.get("task", "classification"),
                "label_space": " / ".join(manifest.get("classes", []) or list((manifest.get("label_mapping") or {}).values())),
                "dataset": (manifest.get("dataset") or {}).get("name"),
                "dataset_synthetic": (manifest.get("dataset") or {}).get("synthetic"),
                "state": "installed",
                "file": str(directory.parent),
                "served_by_application": "classification/" in str(directory),
            }
        )
    return discovered


def _dataset_block(manifest: dict, entry: dict) -> dict:
    """The dataset block that belongs to this entry: the second dataset has its own."""
    if str(entry.get("task", "")).startswith("multiclass"):
        return manifest.get("classification_dataset") or {}
    return manifest.get("dataset") or {}


def _state_of(entry: dict, dataset: dict) -> str:
    declared = str(entry.get("state", "")).lower()
    if declared in {"not_trained", "missing"} or entry.get("trained_on") == "not_trained":
        return STATE_NOT_TRAINED
    synthetic = dataset.get("synthetic") if isinstance(dataset, dict) else None
    if synthetic is None:
        # A model may only be *called* trained on real data when the dataset block says so.
        # A `trained_on: real_brats2020` string on its own is not enough: if the run could not
        # establish the dataset, the state stays "unverified".
        if "synthetic" in str(entry.get("trained_on") or "").lower():
            synthetic = True
    if synthetic is None and not entry:
        return STATE_UNKNOWN
    return state_from_flags(
        trained=bool(entry),
        synthetic=synthetic if isinstance(synthetic, bool) else None,
        evaluated_on=entry.get("evaluated_on"),
    )


def describe(model_id: str, manifest: dict | None, *, data_dir: Path | str | None = None) -> dict[str, Any]:
    """Provenance of one registered model, honest about what is unknown."""
    if not manifest:
        return {
            "state": STATE_UNKNOWN,
            "label": LABELS[STATE_UNKNOWN],
            "trained_on": None,
            "evaluated_on": None,
            "dataset": None,
            "synthetic": None,
            "recorded_at": None,
            "source": None,
            "serving_note": SERVING_NOTES[STATE_UNKNOWN],
            "note": (
                "No provenance manifest was found, so nothing is claimed about this checkpoint. It "
                "may be one of the files that ship with the repository, which were never trained or "
                "evaluated here. Record provenance with scripts/install_trained_artifacts.py after a "
                "training run."
            ),
        }
    key = MODEL_ID_TO_MANIFEST_KEY.get(model_id, model_id)
    models = manifest.get("models") or {}
    entry = models.get(key) or models.get(model_id)
    dataset = _dataset_block(manifest, entry) if entry else (manifest.get("dataset") or {})
    if entry is None:
        return {
            "state": STATE_UNKNOWN,
            "label": LABELS[STATE_UNKNOWN],
            "trained_on": None,
            "evaluated_on": None,
            "dataset": dataset.get("name"),
            "synthetic": dataset.get("synthetic"),
            "recorded_at": manifest.get("generated_at"),
            "source": str(manifest_path(data_dir)) if data_dir else None,
            "serving_note": SERVING_NOTES[STATE_UNKNOWN],
            "note": (
                f"The provenance manifest ({manifest.get('produced_by') or 'unknown run'}) has no "
                f"entry for '{key}', so this checkpoint is not covered by it."
            ),
        }
    state = _state_of(entry, dataset)
    return {
        "state": state,
        "label": LABELS[state],
        "task": entry.get("task"),
        "classes": entry.get("classes"),
        "trained_on": entry.get("trained_on"),
        "evaluated_on": entry.get("evaluated_on"),
        "dataset": dataset.get("name"),
        "synthetic": dataset.get("synthetic"),
        "recorded_at": manifest.get("generated_at"),
        "produced_by": manifest.get("produced_by"),
        "source": str(manifest_path(data_dir)) if data_dir else None,
        "split_method": dataset.get("split_method"),
        "split_seed": dataset.get("split_seed"),
        "splits": dataset.get("splits"),
        "exported_file": entry.get("exported_file"),
        "best_validation_dice": entry.get("best_validation_dice"),
        "serving_note": SERVING_NOTES[state],
        "note": entry.get("note"),
    }


def summary(model_ids, manifest: dict | None, *, data_dir: Path | str | None = None) -> dict:
    return {
        "manifest": str(manifest_path(data_dir)) if data_dir else None,
        "recorded": bool(manifest),
        "produced_by": (manifest or {}).get("produced_by"),
        "generated_at": (manifest or {}).get("generated_at"),
        "models": {model_id: describe(model_id, manifest, data_dir=data_dir) for model_id in model_ids},
    }
