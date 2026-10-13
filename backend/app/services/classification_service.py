from __future__ import annotations
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable
import numpy as np
from ..utils.io import read_json, utc_now
logger = logging.getLogger(__name__)
MANIFEST_NAME = "manifest.json"
@dataclass
class ClassificationOutcome:
    available: bool
    reason: str
    model_id: str | None = None
    model_version: str | None = None
    trainer: str | None = None
    label_mapping: dict[str, str] = field(default_factory=dict)
    predicted_label: str | None = None
    predicted_index: int | None = None
    scores: list[float] = field(default_factory=list)
    calibrated: bool = False
    calibration_method: str | None = None
    metrics: dict | None = None
    feature_contributions: list[dict] = field(default_factory=list)
    missing_features: list[str] = field(default_factory=list)
    statement: str = ""
    target_description: str | None = None
    def to_dict(self) -> dict:
        return {
            "available": self.available,
            "reason": self.reason,
            "model_id": self.model_id,
            "model_version": self.model_version,
            "trainer": self.trainer,
            "label_mapping": self.label_mapping,
            "predicted_label": self.predicted_label,
            "predicted_index": self.predicted_index,
            "scores": self.scores,
            "calibrated": self.calibrated,
            "calibration_method": self.calibration_method,
            "metrics": self.metrics,
            "feature_contributions": self.feature_contributions,
            "missing_features": self.missing_features,
            "statement": self.statement,
            "target_description": self.target_description,
        }
class ClassificationService:
    def __init__(self, config, registry=None):
        self.config = config
        self.registry = registry
        self._artefact: dict | None = None
        self._loaded = False
    @property
    def artifact_dir(self) -> Path:
        return Path(self.config.CLASSIFIER_ARTIFACT_DIR)
    def discover(self) -> dict | None:
        if self._loaded:
            return self._artefact
        self._loaded = True
        directory = self.artifact_dir
        if not directory.exists():
            self._artefact = None
            return None
        manifests = sorted(directory.glob(f"*{MANIFEST_NAME}"))
        if not manifests:
            self._artefact = None
            return None
        manifest_path = manifests[-1]
        payload = read_json(manifest_path)
        if not payload:
            self._artefact = None
            return None
        payload["_manifest_path"] = str(manifest_path)
        payload["_directory"] = str(manifest_path.parent)
        self._artefact = payload
        return payload
    def status(self) -> dict:
        directory = self.artifact_dir
        legacy_path = Path(self.config.MODEL_PATHS["radiomics_mlp_v1"])
        legacy_note = None
        if legacy_path.exists():
            legacy_note = (
                "radiomics_cnn_model.pth is present but is not used: its first layer expects 21 "
                "radiomics features, the repository ships features_df.csv with 11 columns, and no "
                "feature schema, scaler, label mapping or evaluation result was saved next to it. "
                "Serving predictions from it would mean silently re-ordering or padding features, "
                "which would produce arbitrary numbers."
            )
        artefact = self.discover()
        if artefact is None:
            return {
                "available": False,
                "reason": (
                    "No validated classifier artefact was found in "
                    f"{directory}. Train one with scripts/train_classifier.py, which writes the "
                    "model, the ordered feature schema, the label mapping and the held-out metrics "
                    "into that directory."
                ),
                "required_artifacts": [
                    f"{directory}/<name>.joblib  (or .pt) - the trained model",
                    f"{directory}/{MANIFEST_NAME} - feature schema, label mapping, metrics, seed",
                ],
                "legacy_checkpoint_note": legacy_note,
                "requirements": self.data_requirements(),
            }
        return {
            "available": True,
            "model_id": artefact.get("model_id", "radiomics_classifier"),
            "model_version": artefact.get("version"),
            "trained_at": artefact.get("trained_at"),
            "label_mapping": artefact.get("label_mapping", {}),
            "feature_count": len(artefact.get("feature_schema", [])),
            "metrics": artefact.get("metrics", {}),
            "calibration": artefact.get("calibration"),
            "target_description": artefact.get("target_description"),
            "legacy_checkpoint_note": legacy_note,
        }
    @staticmethod
    def data_requirements() -> list[str]:
        return [
            "Labelled radiomics table: one row per case with the feature columns and a class label column.",
            "A defined label column with its meaning recorded (for example BraTS grade: HGG/LGG).",
            "Patient-level identifiers so the split can be made per patient rather than per slice.",
            "A held-out test split that is untouched during model selection, for the reported metrics.",
            "At least 10 cases per class per feature is a common rule of thumb; with ~57 features that "
            "means a few hundred labelled cases before a classifier can be trusted.",
        ]
    def classify(
        self,
        features: dict[str, Any],
        *,
        explain: bool = True,
        feature_names: list[str] | None = None,
    ) -> ClassificationOutcome:
        artefact = self.discover()
        if artefact is None:
            return ClassificationOutcome(
                available=False,
                reason=self.status()["reason"],
                statement=(
                    "Classification is disabled because no validated model with a recorded feature "
                    "schema and labels is available. No prediction is shown, and none was invented."
                ),
            )
        schema: list[str] = list(artefact.get("feature_schema", []))
        if not schema:
            return ClassificationOutcome(
                available=False,
                reason=(
                    "The classifier artefact does not contain a feature schema, so the input vector "
                    "cannot be assembled in the order the model was trained with."
                ),
                model_id=artefact.get("model_id"),
                trainer="scripts/train_classifier.py",
            )
        missing = [name for name in schema if name not in features or features[name] is None]
        if missing:
            return ClassificationOutcome(
                available=False,
                reason=(
                    f"{len(missing)} of the {len(schema)} features required by the classifier could "
                    "not be computed for this analysis, so no prediction is produced."
                ),
                model_id=artefact.get("model_id"),
                missing_features=missing[:50],
                statement="A prediction from an incomplete feature vector would be meaningless.",
            )
        try:
            predictor, metadata = self._load_predictor(artefact)
        except Exception as exc:
            logger.exception("Failed to load the classifier artefact")
            return ClassificationOutcome(
                available=False,
                reason=f"The classifier artefact could not be loaded: {type(exc).__name__}: {exc}",
                model_id=artefact.get("model_id"),
            )
        vector = np.asarray([float(features[name]) for name in schema], dtype=np.float32).reshape(1, -1)
        scaled = self._apply_preprocessing(vector, artefact)
        scores = np.asarray(predictor(scaled)).ravel().astype(float)
        if scores.size == 0:
            return ClassificationOutcome(
                available=False,
                reason="The classifier returned no scores.",
                model_id=artefact.get("model_id"),
            )
        if scores.min() < 0 or not np.isclose(scores.sum(), 1.0, atol=1e-3):
            scores = _softmax(scores)
        label_mapping = artefact.get("label_mapping", {})
        predicted_index = int(np.argmax(scores))
        predicted_label = label_mapping.get(str(predicted_index), str(predicted_index))
        calibration = artefact.get("calibration") or {}
        metrics = artefact.get("metrics") or {}
        contributions = []
        if explain:
            contributions = self._feature_contributions(
                artefact, predictor, scaled, schema, predicted_index
            )
        return ClassificationOutcome(
            available=True,
            reason="A validated classifier artefact with a recorded feature schema was used.",
            model_id=artefact.get("model_id", "radiomics_classifier"),
            model_version=artefact.get("version"),
            trainer="scripts/train_classifier.py",
            label_mapping={str(k): str(v) for k, v in label_mapping.items()},
            predicted_label=predicted_label,
            predicted_index=predicted_index,
            scores=[float(s) for s in scores],
            calibrated=bool(calibration.get("applied")),
            calibration_method=calibration.get("method"),
            metrics=metrics,
            feature_contributions=contributions,
            statement=(
                "Predicted class according to the trained model, reported together with the "
                "validation/test metrics recorded at training time. This is a dataset-defined label "
                "(for example the tumour grade recorded in the dataset), not a clinical diagnosis, "
                "and it is not derived from the segmentation mask."
            ),
            target_description=artefact.get("target_description"),
        )
    def _load_predictor(self, artefact: dict) -> tuple[Callable, dict]:
        directory = Path(artefact["_directory"])
        model_file = artefact.get("model_file")
        if not model_file:
            raise FileNotFoundError("the manifest does not name a model file")
        path = directory / model_file
        if not path.exists():
            raise FileNotFoundError(f"{path} is missing")
        kind = artefact.get("kind", "sklearn")
        if kind == "sklearn":
            import joblib
            model = joblib.load(path)
            def predict(vector: np.ndarray) -> np.ndarray:
                if hasattr(model, "predict_proba"):
                    return model.predict_proba(vector)[0]
                decision = np.asarray(model.decision_function(vector)).ravel()
                return _softmax(decision)
            return predict, {"kind": kind}
        if kind == "torch":
            import torch
            from ..ml.classifier_models import build_classifier
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            architecture = artefact.get("architecture", "mlp")
            model = build_classifier(
                architecture, input_dim=len(artefact["feature_schema"]), **artefact.get("architecture_kwargs", {})
            )
            model.load_state_dict(checkpoint.get("state_dict", checkpoint))
            model.eval()
            def predict(vector: np.ndarray) -> np.ndarray:
                with torch.no_grad():
                    logits = model(torch.from_numpy(vector.astype(np.float32)))
                return _softmax(logits.numpy().ravel())
            return predict, {"kind": kind, "architecture": architecture}
        raise ValueError(f"unsupported classifier kind '{kind}'")
    @staticmethod
    def _apply_preprocessing(vector: np.ndarray, artefact: dict) -> np.ndarray:
        preprocessing = artefact.get("preprocessing") or {}
        mean = preprocessing.get("mean")
        scale = preprocessing.get("scale")
        if preprocessing.get("type") == "standard_scaler" and mean and scale:
            return (vector - np.asarray(mean, dtype=np.float32)) / np.asarray(scale, dtype=np.float32)
        return vector
    def _feature_contributions(
        self, artefact: dict, predictor: Callable, scaled: np.ndarray, schema: list[str], predicted_index: int
    ) -> list[dict]:
        weights = artefact.get("coefficients")
        contributions: list[dict] = []
        if weights:
            try:
                if isinstance(weights, dict):
                    array = np.asarray(weights[str(predicted_index)], dtype=np.float32)
                else:
                    array = np.asarray(weights, dtype=np.float32)[predicted_index]
                values = scaled.ravel()
                for name, value, weight in zip(schema, values, array):
                    contributions.append(
                        {
                            "feature": name,
                            "value": float(value),
                            "weight": float(weight),
                            "contribution": float(value * weight),
                            "method": "linear coefficient x standardised value",
                        }
                    )
                contributions.sort(key=lambda item: abs(item["contribution"]), reverse=True)
                return contributions[:25]
            except Exception:
                logger.warning("Stored coefficients could not be applied; using permutation importance")
        rng = np.random.default_rng(0)
        baseline = float(np.asarray(predictor(scaled)).ravel()[predicted_index])
        quantiles = artefact.get("feature_quantiles") or []
        values = scaled.ravel()
        for index, name in enumerate(schema):
            reference = None
            if index < len(quantiles) and quantiles[index]:
                reference = float(rng.choice(np.asarray(quantiles[index], dtype=np.float32)))
            if reference is None:
                continue
            perturbed = values.copy()
            perturbed[index] = reference
            score = float(np.asarray(predictor(perturbed.reshape(1, -1))).ravel()[predicted_index])
            contributions.append(
                {
                    "feature": name,
                    "value": float(values[index]),
                    "reference_value": reference,
                    "contribution": baseline - score,
                    "method": "single-feature substitution against the training quantiles",
                }
            )
        contributions.sort(key=lambda item: abs(item["contribution"]), reverse=True)
        return contributions[:25]
def _softmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).ravel()
    shifted = values - values.max()
    exponent = np.exp(shifted)
    return exponent / exponent.sum()
