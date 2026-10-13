from __future__ import annotations
from flask import Blueprint, current_app, jsonify
from ..utils.io import read_json
evaluation_bp = Blueprint("evaluation", __name__, url_prefix="/api/evaluation")
def _services():
    return current_app.extensions["brain_services"]
@evaluation_bp.get("")
def index():
    services = _services()
    evaluation = services["evaluation"]
    return jsonify(
        {
            "evaluation": evaluation.stored_results(),
            "model_selection": evaluation.model_selection(),
            "segmentation": evaluation.segmentation_summary(),
            "classification": evaluation.classification_summary(),
            "runtime": evaluation.runtime_summary(),
            "model_health": services["health"].summary(),
            "training_runs": _training_runs(services),
            "protocol": {
                "splits": [
                    "Training split: used to fit parameters.",
                    "Validation split: used for model selection, early stopping and ensemble calibration.",
                    "Reserved test split: evaluated once, after selection, and never used for tuning.",
                ],
                "principles": [
                    "Patient-level splitting whenever patient identifiers are available; slice-level "
                    "splitting leaks data between splits and is reported as a limitation when it "
                    "cannot be avoided.",
                    "Every metric records the dataset version, preprocessing, checkpoint and seed.",
                    "Ensemble weights are estimated on validation data only; a calibration file that "
                    "claims to come from the test split is rejected by the API.",
                    "Ablations differ by exactly one component, so the effect of each addition can be "
                    "attributed.",
                ],
                "scripts": {
                    "train_segnet": "scripts/train_segnet.py - reproduces the four-channel SegNet.",
                    "train_unet": "scripts/train_unet.py - trains RAUNet (or the single-channel baseline).",
                    "evaluate": "scripts/evaluate_models.py - evaluates the reserved test split.",
                    "ablations": "scripts/run_ablations.py - runs the component ablations.",
                    "calibrate": "scripts/calibrate_ensemble.py - estimates fusion weights on validation data.",
                    "classifier": "scripts/train_classifier.py - trains and records the radiomics classifier.",
                },
            },
        }
    )
def _training_runs(services) -> list[dict]:
    import glob
    from pathlib import Path
    runs = []
    for path in sorted(glob.glob(str(Path(current_app.config["BRAIN_DATA_DIR"]) / "models" / "training" / "*.json"))):
        payload = read_json(Path(path))
        if payload:
            runs.append(payload)
    return runs
