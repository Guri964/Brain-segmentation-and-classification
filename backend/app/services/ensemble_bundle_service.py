"""The stacked ensemble of ``FINAL ENSEMBLE NB.ipynb``, served as the notebook defined it.

Pipeline reproduced (cells 26-32)
---------------------------------
::

    four-channel slice (H, W, 4)  ->  46 handcrafted features (radiomics_notebook.py)
                                  ->  StandardScaler (radiomics_scaler.pkl)
                                  ->  RadiomicsCNN (radiomics_cnn.pth)      -> p(notebook target)
    three segmentation members (segnet.pth, unet_best.pth, raunet.pth)
                                  ->  per member: max probability, mean probability, Dice
                                  ->  [3 x 3 stats, radiomics prediction]   (10 columns)
                                  ->  StandardScaler (meta_scaler.pkl)
                                  ->  GradientBoostingClassifier (meta_learner.pkl)
                                  ->  prediction

What this service insists on before it will run
----------------------------------------------
The meta-learner was fitted on a fixed 10-column matrix, so every one of its inputs has to be
reproducible, and each artefact has to agree with the next one:

* the saved feature schema (``radiomics_feature_names.csv``) must be a permutation of the 46
  names the extractor produces, and the scaler and the CNN must both be 46 wide;
* the meta-scaler and the meta-learner must both be 10 wide;
* all three segmentation members must be loadable, because the notebook's matrix has a slot for
  each of them.  A missing member is reported, never silently zero-filled - imputing a member's
  statistics would feed the meta-learner a distribution it was never trained on;
* the input must be the four-channel 240x240 z-scored tensor the members were trained on.

Three honest substitutions, each reported in the payload
--------------------------------------------------------
1. **The Dice column.** In the notebook the members' Dice was computed against the ground-truth
   mask, which does not exist at serving time.  Each member's Dice slot is therefore filled with
   its Dice against the ensemble consensus mask, and the payload says so
   (``dice_reference: "ensemble consensus"``).  The number is not the notebook's number.
2. **The feature mask.** The notebook extracted features from the ground-truth mask; this service
   uses the consensus mask produced by the three members.  The implication is stated with every
   result: features (and therefore the radiomics branch) are conditioned on predicted masks.
3. **Nothing else.** No number is invented: if an artefact, a member or a contract check is
   missing or fails, the call raises with the precise reason instead of returning a score.

The target is *not* a grade
---------------------------
See :data:`app.ml.radiomics_notebook.TARGET_SEMANTICS`: the notebook's labels are
``(mask pixel count > 100)``.  This service labels its output as that proxy and carries the
leakage note with every prediction.
"""

from __future__ import annotations

import csv
import hashlib
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..errors import DependencyMissingError, ModelUnavailableError, ValidationError
from ..ml import radiomics_notebook
from ..ml.radiomics_notebook import LEAKAGE_NOTE, TARGET_SEMANTICS, feature_order

logger = logging.getLogger(__name__)

#: The three segmentation members whose statistics the meta-learner consumes, in the notebook's
#: column order (SegNet, U-Net, RAUNet).
SEGMENTATION_MEMBERS: tuple[str, ...] = (
    "segnet_ensemble_240",
    "unet_ensemble_240",
    "raunet_ensemble_240",
)

#: Statistics collected per member, in the notebook's order.
MEMBER_STATS: tuple[str, ...] = ("dice", "max_prob", "mean_prob")

#: 3 members x 3 statistics + the radiomics prediction.
META_FEATURE_COUNT = len(SEGMENTATION_MEMBERS) * len(MEMBER_STATS) + 1

#: 240 x 240 four-channel, per-channel z-score - the tensor the notebook trained on.
INPUT_CHANNELS = 4
INPUT_SIZE = 240

#: Metrics exactly as the notebook's own outputs printed them.  They are recorded here so the
#: API can show them *as reported* without ever recomputing or restating them as verified.
REPORTED_METRICS = {
    "stacked_ensemble_test_accuracy": 0.9633,
    "stacked_ensemble_validation_accuracy": 0.9733,
    "radiomics_cnn_test_accuracy": 0.9417,
    "radiomics_cnn_validation_accuracy": 0.94,
    "segnet_validation_dice": 0.7717,
    "unet_validation_dice": 0.8235,
    "raunet_validation_dice": 0.7585,
    "member_validation_accuracy": {
        "SegNet": 0.28,
        "UNet": 0.7567,
        "RAUNet": 0.6767,
        "Radiomics-CNN": 0.94,
        "Stacked Ensemble": 0.9733,
    },
    "test_split_size": 300,
}

METRICS_PROVENANCE = (
    "These numbers are the ones the notebook's own run printed (cells 28-31) on a slice-level "
    "random split of 2,000 BraTS 2020 slices. They have not been re-measured here: the split, "
    "the ground-truth masks and the radiologist labels are not shipped with the weights. They "
    "are reported as the training run's record, not as a property of this deployment."
)

INSTALL_NOTE = (
    "Copy segnet.pth, unet_best.pth, raunet.pth, radiomics_cnn.pth, radiomics_scaler.pkl, "
    "radiomics_feature_names.csv, meta_scaler.pkl and meta_learner.pkl from the extracted "
    "brain_mri_ensemble_artifacts folder into models/weights/ (or point BRAIN_WEIGHTS_DIR at "
    "that folder). Nothing is downloaded automatically."
)


@dataclass
class MemberStatistics:
    """One segmentation member's row of the meta matrix."""

    model_id: str
    dice_vs_consensus: float
    max_probability: float
    mean_probability: float
    foreground_fraction: float
    inference_seconds: float | None
    reused: bool


class EnsembleBundleService:
    """Load, verify and run the notebook's stacked ensemble."""

    def __init__(self, config, registry, segmentation_service, *, loader=None) -> None:
        self.config = config
        self.registry = registry
        self.segmentation = segmentation_service
        self._loader = loader  # injection point for tests; defaults to joblib.load
        self._artifacts: dict | None = None
        self._artifact_error: str | None = None
        self._checked_at: str | None = None

    # ------------------------------------------------------------------ artefacts
    def artifact_specs(self) -> list[dict]:
        """Every file the ensemble needs, with the path the resolver produced."""

        return [
            {
                "key": "radiomics_cnn",
                "path": self.config.MODEL_PATHS["radiomics_ensemble_cnn"],
                "role": "46-feature classifier head",
            },
            {
                "key": "radiomics_scaler",
                "path": self.config.RADIOMICS_ENSEMBLE_SCALER_PATH,
                "role": "StandardScaler for the 46 features",
            },
            {
                "key": "radiomics_feature_schema",
                "path": self.config.RADIOMICS_ENSEMBLE_SCHEMA_PATH,
                "role": "feature names and column order",
            },
            {
                "key": "meta_scaler",
                "path": self.config.META_SCALER_PATH,
                "role": "StandardScaler for the 10 meta features",
            },
            {
                "key": "meta_learner",
                "path": self.config.META_LEARNER_PATH,
                "role": "GradientBoostingClassifier (the stacked head)",
            },
        ]

    def artifact_status(self) -> list[dict]:
        entries = []
        for spec in self.artifact_specs():
            path = Path(spec["path"])
            exists = path.is_file()
            entries.append(
                {
                    "key": spec["key"],
                    "role": spec["role"],
                    "file": path.name,
                    "path": str(path),
                    "present": exists,
                    "size_bytes": path.stat().st_size if exists else None,
                }
            )
        return entries

    def missing_artifacts(self) -> list[str]:
        return [entry["file"] for entry in self.artifact_status() if not entry["present"]]

    def _joblib(self):
        try:
            import joblib  # noqa: PLC0415 - optional dependency, reported when absent
        except Exception as exc:  # pragma: no cover - environment dependent
            raise DependencyMissingError(
                "Loading the ensemble's scalers and meta-learner needs joblib.",
                details={"dependency": "joblib", "install": "pip install joblib"},
            ) from exc
        return joblib

    def _file_digest(self, path: Path, *, limit: int = 1 << 20) -> str:
        """A short digest of the artefact, so a payload can name the exact bytes it used."""

        digest = hashlib.sha256()
        with path.open("rb") as handle:
            digest.update(handle.read(limit))
        return digest.hexdigest()[:16]

    def load_artifacts(self, *, force: bool = False) -> dict:
        """Read the five head artefacts and verify every cross-artefact contract."""

        if self._artifacts is not None and not force:
            return self._artifacts
        if self._artifact_error is not None and not force:
            raise ModelUnavailableError(self._artifact_error)

        missing = self.missing_artifacts()
        if missing:
            self._artifact_error = (
                "The stacked ensemble needs every one of its artefacts; missing: "
                + ", ".join(missing)
                + ". "
                + INSTALL_NOTE
            )
            raise ModelUnavailableError(
                self._artifact_error,
                details={
                    "missing_artifacts": missing,
                    "searched": [entry["path"] for entry in self.artifact_status()],
                },
            )

        joblib = self._joblib()
        specs = {entry["key"]: entry for entry in self.artifact_status()}
        # The scalers and the meta-learner are joblib objects; the CNN is a torch state dict
        # and is loaded (and shape-checked) by the registry, not here.
        try:
            radiomics_scaler = joblib.load(specs["radiomics_scaler"]["path"])
            meta_scaler = joblib.load(specs["meta_scaler"]["path"])
            meta_learner = joblib.load(specs["meta_learner"]["path"])
        except Exception as exc:  # noqa: BLE001 - reported, never guessed around
            self._artifact_error = (
                f"the ensemble artefacts could not be read: {type(exc).__name__}: {exc}"
            )
            raise ModelUnavailableError(self._artifact_error) from exc

        cnn_state = self.registry.load_model("radiomics_ensemble_cnn")
        if not cnn_state.is_ready:
            self._artifact_error = (
                f"radiomics_cnn.pth is not usable ({cnn_state.status}): {cnn_state.message}"
            )
            raise ModelUnavailableError(self._artifact_error)

        schema_order, schema_error = self._read_schema(specs["radiomics_feature_schema"]["path"])
        if schema_error:
            self._artifact_error = schema_error
            raise ModelUnavailableError(schema_error)

        expected_names = set(feature_order())
        unexpected = [name for name in schema_order if name not in expected_names]
        absent = [name for name in expected_names if name not in schema_order]
        if unexpected or absent:
            self._artifact_error = (
                "radiomics_feature_names.csv does not describe the feature set this build "
                f"extracts: {len(unexpected)} unknown name(s)"
                + (f" ({', '.join(unexpected[:5])})" if unexpected else "")
                + f" and {len(absent)} missing name(s)"
                + (f" ({', '.join(absent[:5])})" if absent else "")
                + ". The scaler and the CNN were fitted on that exact set, so the miss is "
                "reported instead of being worked around."
            )
            raise ModelUnavailableError(self._artifact_error)

        # ---- contract: 46 features everywhere -------------------------------------------
        scaler_dim = getattr(radiomics_scaler, "n_features_in_", len(schema_order))
        cnn_dim = self._first_linear_input_dim(self.registry, "radiomics_ensemble_cnn")
        problems = []
        if scaler_dim != len(schema_order):
            problems.append(
                f"radiomics_scaler.pkl expects {scaler_dim} features but the saved schema lists "
                f"{len(schema_order)}"
            )
        if cnn_dim is not None and cnn_dim != len(schema_order):
            problems.append(
                f"radiomics_cnn.pth expects {cnn_dim} input features but the saved schema lists "
                f"{len(schema_order)}"
            )
        # ---- contract: 10 meta features --------------------------------------------------
        meta_scaler_dim = getattr(meta_scaler, "n_features_in_", None)
        meta_learner_dim = getattr(meta_learner, "n_features_in_", None)
        if meta_scaler_dim is not None and meta_scaler_dim != META_FEATURE_COUNT:
            problems.append(
                f"meta_scaler.pkl expects {meta_scaler_dim} meta features; the notebook built "
                f"{META_FEATURE_COUNT}"
            )
        if meta_learner_dim is not None and meta_learner_dim != META_FEATURE_COUNT:
            problems.append(
                f"meta_learner.pkl expects {meta_learner_dim} meta features; the notebook built "
                f"{META_FEATURE_COUNT}"
            )
        if problems:
            self._artifact_error = (
                "the ensemble artefacts disagree with each other, so no prediction can be "
                "produced: " + "; ".join(problems) + "."
            )
            raise ModelUnavailableError(self._artifact_error, details={"problems": problems})

        # ``classes_`` is a NumPy array; casting here keeps every payload JSON-serialisable.
        # numpy integer scalars are not JSON serialisable, so the class list is cast here once
        classes = [str(value) for value in np.asarray(getattr(meta_learner, "classes_", [])).tolist()]
        self._artifacts = {
            "radiomics_cnn": cnn_state.module,
            "radiomics_scaler": radiomics_scaler,
            "meta_scaler": meta_scaler,
            "meta_learner": meta_learner,
            "schema_order": schema_order,
            "digests": {
                key: self._file_digest(Path(entry["path"]))
                for key, entry in specs.items()
            },
            "sizes": {key: entry["size_bytes"] for key, entry in specs.items()},
            "meta_classes": classes,
        }
        self._checked_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        return self._artifacts

    @staticmethod
    def _read_schema(path: str | Path) -> tuple[list[str], str | None]:
        """Read the saved feature names (the CSV's header row)."""

        try:
            with Path(path).open(newline="", encoding="utf-8") as handle:
                reader = csv.reader(handle)
                header = next(reader, None)
        except Exception as exc:  # noqa: BLE001
            return [], f"{Path(path).name} could not be read: {type(exc).__name__}: {exc}"
        if not header:
            return [], f"{Path(path).name} has no header row, so the feature order is unknown."
        names = [name.strip() for name in header if name and not name.startswith("Unnamed:")]
        return names, None

    def _first_linear_input_dim(self, registry, model_id: str) -> int | None:
        """The input width of a classifier's first Linear layer, read from the checkpoint."""

        try:
            loaded = registry.load_model(model_id)
        except Exception:  # noqa: BLE001
            return None
        if loaded.module is None:
            return None
        for module in loaded.module.modules():
            if hasattr(module, "in_features") and hasattr(module, "out_features"):
                return int(module.in_features)
        return None

    # ---------------------------------------------------------------------- status
    def status(self, *, force: bool = False) -> dict:
        """Everything the interface may say about the stacked ensemble, without running it."""

        if force:
            self._artifacts = None
            self._artifact_error = None
        artifacts = self.artifact_status()
        members = []
        for model_id in SEGMENTATION_MEMBERS:
            loaded = self.registry.load_model(model_id)
            members.append(
                {
                    "model_id": model_id,
                    "display_name": loaded.descriptor.display_name,
                    "status": loaded.status,
                    "available": loaded.is_ready,
                    "message": loaded.message,
                    "path": loaded.descriptor.path,
                    "architecture": loaded.descriptor.architecture,
                    "parameter_count": loaded.parameter_count,
                }
            )
        missing = [entry["file"] for entry in artifacts if not entry["present"]]
        missing_members = [m["model_id"] for m in members if not m["available"]]
        state = "ready"
        reason = (
            "All five head artefacts are present and the three segmentation members load; the "
            "ten meta features can be reproduced exactly."
        )
        contracts: dict = {}
        if missing:
            state = "artifacts_missing"
            reason = (
                "The stacked ensemble needs all of its artefacts; missing: "
                + ", ".join(missing)
                + ". "
                + INSTALL_NOTE
            )
        elif missing_members:
            state = "members_missing"
            reason = (
                "The meta-learner consumes one row per segmentation member and was fitted on "
                "three, so all three checkpoints have to be present. Missing: "
                + ", ".join(missing_members)
                + "."
            )
        else:
            try:
                loaded = self.load_artifacts(force=force)
                contracts = {
                    "radiomics_features": len(loaded["schema_order"]),
                    "meta_features": META_FEATURE_COUNT,
                    "meta_classes": loaded["meta_classes"],
                    "artifact_digests": loaded["digests"],
                }
            except ModelUnavailableError as exc:
                state = "incompatible"
                reason = exc.message
                contracts = exc.details
        return {
            "available": state == "ready",
            "state": state,
            "reason": reason,
            "task": "stacked_ensemble",
            "task_description": (
                "The notebook's stacked ensemble: three segmentation members contribute their "
                "maximum, mean and consensus-Dice statistics, the 46-feature radiomics CNN "
                "contributes one binary score, and a GradientBoostingClassifier combines the ten "
                "into the notebook's target."
            ),
            "members": members,
            "artifacts": artifacts,
            "contracts": contracts,
            "input_requirements": {
                "analysis_type": "four_channel",
                "channels": INPUT_CHANNELS,
                "spatial_size": [INPUT_SIZE, INPUT_SIZE],
                "normalisation": "per-channel z-score over the resized 240x240 slice",
                "modality_order": ["t1", "t1ce", "t2", "flair"],
            },
            "meta_features": [
                f"{model_id}.{stat}" for model_id in SEGMENTATION_MEMBERS for stat in MEMBER_STATS
            ]
            + ["radiomics_prediction"],
            "target_semantics": TARGET_SEMANTICS,
            "leakage_note": LEAKAGE_NOTE,
            "reported_metrics": REPORTED_METRICS,
            "metrics_provenance": METRICS_PROVENANCE,
            "dice_slot_note": (
                "In the notebook each member's Dice was measured against the ground-truth mask. "
                "No ground truth exists at serving time, so this build fills the slot with the "
                "member's Dice against the consensus mask and says so in every payload. The "
                "value is therefore not the notebook's Dice."
            ),
            "feature_mask_note": (
                "The notebook extracted the 46 features from the ground-truth mask; here they "
                "come from the consensus mask the three members produce, so the radiomics branch "
                "is conditioned on predicted masks."
            ),
            "distinct_from_grade_and_type_models": (
                "This is neither the BraTS HGG/LGG grade classifier nor the four-class Xception "
                "tumour-type classifier. It predicts the notebook's mask-size proxy, and its "
                "output is never combined with, or substituted for, either of the other two."
            ),
            "install_note": INSTALL_NOTE,
            "checked_at": self._checked_at,
            "disclaimer": getattr(self.config, "DISCLAIMER", ""),
        }

    # ------------------------------------------------------------------- inference
    def predict(
        self,
        tensor,
        *,
        threshold: float = 0.5,
        member_results: dict | None = None,
        consensus_mask: np.ndarray | None = None,
        display_reference: np.ndarray | None = None,
    ) -> dict:
        """Run the notebook's pipeline on one preprocessed four-channel slice.

        ``tensor`` must be the ``(1, 4, 240, 240)`` z-scored tensor.  ``member_results`` may
        carry segmentation results the caller has already computed (the analysis service does),
        which are reused instead of being recomputed.
        """

        loaded = self.load_artifacts()
        if tensor is None:
            raise ValidationError("The stacked ensemble needs the preprocessed four-channel slice.")
        shape = tuple(int(value) for value in np.asarray(tensor.shape))
        if len(shape) != 4 or shape[1] != INPUT_CHANNELS:
            raise ValidationError(
                "The stacked ensemble needs a (1, 4, H, W) tensor; received "
                f"{list(shape)}. It consumes the notebook's four-channel input, not a single "
                "grayscale image.",
                details={"required": [1, INPUT_CHANNELS, INPUT_SIZE, INPUT_SIZE]},
            )
        if shape[2] != INPUT_SIZE or shape[3] != INPUT_SIZE:
            raise ValidationError(
                f"The stacked ensemble was trained on {INPUT_SIZE}x{INPUT_SIZE} slices; the "
                f"preprocessed tensor is {shape[2]}x{shape[3]}. Re-run the analysis with "
                "analysis_type=four_channel so the input matches, rather than resizing silently "
                "here.",
                details={"required": [INPUT_SIZE, INPUT_SIZE], "received": [shape[2], shape[3]]},
            )

        member_results = dict(member_results or {})
        missing_members = [m for m in SEGMENTATION_MEMBERS if m not in member_results]
        if missing_members:
            unavailable = [
                model_id
                for model_id in missing_members
                if not self.registry.load_model(model_id).is_ready
            ]
            if unavailable:
                raise ModelUnavailableError(
                    "The meta-learner was fitted on three segmentation members and cannot be fed "
                    "a different matrix; unavailable: "
                    + ", ".join(unavailable)
                    + ". "
                    + INSTALL_NOTE,
                    details={"missing_members": unavailable, "members_required": list(SEGMENTATION_MEMBERS)},
                )

        started = time.perf_counter()
        # 1. members ---------------------------------------------------------------------
        member_probabilities: dict[str, np.ndarray] = {}
        member_payloads: list[dict] = []
        for model_id in SEGMENTATION_MEMBERS:
            result = member_results.get(model_id)
            reused = result is not None
            if result is None:
                result = self.segmentation.predict(model_id, _TensorWrapper(tensor), threshold=threshold)
            probability = np.asarray(result.probability, dtype=np.float32)
            member_probabilities[model_id] = probability
            member_payloads.append(
                {
                    "model_id": model_id,
                    "display_name": result.model_display_name,
                    "reused": reused,
                    "threshold": float(result.threshold),
                    "foreground_fraction": float(result.foreground_fraction),
                    "max_probability": float(probability.max()),
                    "mean_probability": float(probability.mean()),
                    "inference_seconds": result.inference_seconds,
                }
            )

        # 2. consensus mask and per-member Dice against it --------------------------------
        stack = np.stack([member_probabilities[mid] for mid in SEGMENTATION_MEMBERS], axis=0)
        consensus_probability = stack.mean(axis=0)
        consensus = (
            np.asarray(consensus_mask).astype(bool)
            if consensus_mask is not None
            else consensus_probability > threshold
        )
        if not consensus.any():
            raise ModelUnavailableError(
                "The three members produced an empty consensus mask at threshold "
                f"{threshold}, so the 46 radiomics features are undefined (the notebook returns "
                "None for an empty mask as well). Lower the threshold or re-check the input; no "
                "feature is invented to keep the pipeline running.",
                details={"threshold": float(threshold), "members": list(SEGMENTATION_MEMBERS)},
            )

        statistics: list[MemberStatistics] = []
        for model_id, payload in zip(SEGMENTATION_MEMBERS, member_payloads):
            probability = member_probabilities[model_id]
            binary = probability > threshold
            intersection = float(np.logical_and(binary, consensus).sum())
            union = float(binary.sum() + consensus.sum())
            dice = (2.0 * intersection / union) if union > 0 else 0.0
            payload["dice_vs_consensus"] = dice
            statistics.append(
                MemberStatistics(
                    model_id=model_id,
                    dice_vs_consensus=dice,
                    max_probability=float(probability.max()),
                    mean_probability=float(probability.mean()),
                    foreground_fraction=float(binary.mean()),
                    inference_seconds=payload["inference_seconds"],
                    reused=payload["reused"],
                )
            )

        # 3. radiomics branch -------------------------------------------------------------
        image_hwc = np.asarray(tensor, dtype=np.float32)[0].transpose(1, 2, 0)  # (H, W, 4)
        features = radiomics_notebook.extract_features(image_hwc, consensus)
        if features is None:  # pragma: no cover - consensus is non-empty by construction
            raise ModelUnavailableError(
                "No pixel was inside the consensus mask, so no radiomics feature exists."
            )
        schema_order = loaded["schema_order"]
        ordered = radiomics_notebook.vector(features, schema_order)
        scaled = np.asarray(
            loaded["radiomics_scaler"].transform(ordered.reshape(1, -1)), dtype=np.float32
        )
        radiomics_logits = self._run_radiomics_cnn(scaled)          # (1, 2)
        radiomics_probabilities = _softmax(radiomics_logits)[0]     # this sample's two classes
        radiomics_score = float(radiomics_probabilities[1])
        radiomics_prediction = int(radiomics_score > 0.5)

        # 4. meta matrix ------------------------------------------------------------------
        meta_row = []
        for stat in statistics:
            meta_row.extend([stat.dice_vs_consensus, stat.max_probability, stat.mean_probability])
        meta_row.append(float(radiomics_prediction))
        meta_matrix = np.asarray([meta_row], dtype=np.float64)
        if meta_matrix.shape[1] != META_FEATURE_COUNT:
            raise ModelUnavailableError(
                f"built {meta_matrix.shape[1]} meta features but the learner was fitted on "
                f"{META_FEATURE_COUNT}."
            )
        meta_scaled = np.asarray(loaded["meta_scaler"].transform(meta_matrix), dtype=np.float64)
        meta_learner = loaded["meta_learner"]
        prediction = int(meta_learner.predict(meta_scaled)[0])
        probabilities = None
        if hasattr(meta_learner, "predict_proba"):
            probabilities = [float(value) for value in meta_learner.predict_proba(meta_scaled)[0]]

        return {
            "model_id": "stacked_ensemble_brats",
            "display_name": "Stacked ensemble (FINAL ENSEMBLE NB.ipynb)",
            "available": True,
            "task": "stacked_ensemble",
            "state": "completed",
            "prediction": prediction,
            "class_labels": ["small_lesion_proxy", "large_lesion_proxy"],
            "class_probabilities": probabilities,
            "meta_features": {
                name: float(value)
                for name, value in zip(
                    [
                        f"{model_id}.{stat}"
                        for model_id in SEGMENTATION_MEMBERS
                        for stat in MEMBER_STATS
                    ]
                    + ["radiomics_prediction"],
                    meta_row,
                )
            },
            "meta_features_scaled": [float(value) for value in meta_scaled[0]],
            "meta_feature_count": META_FEATURE_COUNT,
            "radiomics": {
                "feature_count": len(schema_order),
                "feature_names_from": Path(self.config.RADIOMICS_ENSEMBLE_SCHEMA_PATH).name,
                "raw_feature_digest": _digest_vector(ordered),
                "scaled_feature_digest": _digest_vector(scaled[0]),
                "logits": [float(v) for v in radiomics_logits[0]],
                "probabilities": [float(v) for v in radiomics_probabilities],
                "score": radiomics_score,
                "prediction": radiomics_prediction,
                "scaler": Path(self.config.RADIOMICS_ENSEMBLE_SCALER_PATH).name,
                "cnn": Path(self.config.MODEL_PATHS["radiomics_ensemble_cnn"]).name,
            },
            "members": member_payloads,
            "member_statistics": [
                {
                    "model_id": stat.model_id,
                    "dice_vs_consensus": stat.dice_vs_consensus,
                    "max_probability": stat.max_probability,
                    "mean_probability": stat.mean_probability,
                    "foreground_fraction": stat.foreground_fraction,
                    "inference_seconds": stat.inference_seconds,
                    "reused_from_analysis": stat.reused,
                }
                for stat in statistics
            ],
            "consensus": {
                "threshold": float(threshold),
                "foreground_pixels": int(consensus.sum()),
                "foreground_fraction": float(consensus.mean()),
                "probability_mean": float(consensus_probability.mean()),
                "probability_max": float(consensus_probability.max()),
            },
            "dice_reference": "ensemble consensus",
            "dice_note": (
                "Each member's Dice is measured against the consensus mask of the other members' "
                "average, not against a ground-truth segmentation (none exists at serving time)."
            ),
            "feature_mask_source": "consensus mask of the three members",
            "target_semantics": TARGET_SEMANTICS,
            "leakage_note": LEAKAGE_NOTE,
            "reported_metrics": REPORTED_METRICS,
            "metrics_provenance": METRICS_PROVENANCE,
            "artifact_digests": loaded["digests"],
            "artifact_sizes": loaded["sizes"],
            "meta_classes": loaded["meta_classes"],
            "inference_seconds": round(time.perf_counter() - started, 4),
            "label_semantics": TARGET_SEMANTICS,
            "disclaimer": getattr(self.config, "DISCLAIMER", ""),
        }

    def _run_radiomics_cnn(self, scaled: np.ndarray) -> np.ndarray:
        """One forward pass through the notebook's radiomics CNN (cached in the registry)."""

        import torch

        loaded = self.registry.load_model("radiomics_ensemble_cnn")
        if not loaded.is_ready or loaded.module is None:
            raise ModelUnavailableError(
                f"radiomics_cnn.pth is not usable ({loaded.status}): {loaded.message}"
            )
        module = loaded.module
        module.eval()
        with torch.no_grad():
            output = module(torch.from_numpy(scaled).to(loaded.device))
        tensor_out = output[0] if isinstance(output, (tuple, list)) else output
        logits = np.asarray(tensor_out.detach().cpu().numpy(), dtype=np.float64)
        # One sample is a single row; keep the batch dimension explicit so callers can always
        # index [0] to get this sample's two logits.
        return logits.reshape(1, -1)


class _TensorWrapper:
    """Minimal stand-in for ``PreprocessedInput`` so a raw tensor can drive the segmentation.

    ``SegmentationService`` moves ``.tensor`` onto the device itself, so this holds a
    ``torch.Tensor``: a NumPy array is converted once here rather than crashing inside the
    forward pass.
    """

    def __init__(self, tensor) -> None:
        import torch

        if isinstance(tensor, torch.Tensor):
            self.tensor = tensor
        else:
            self.tensor = torch.from_numpy(np.ascontiguousarray(tensor, dtype=np.float32))

    @property
    def channels(self) -> int:
        return int(self.tensor.shape[1])

    @property
    def height(self) -> int:
        return int(self.tensor.shape[2])

    @property
    def width(self) -> int:
        return int(self.tensor.shape[3])

    def summary(self) -> dict:
        return {"tensor_shape": list(self.tensor.shape), "source": "stacked ensemble"}


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max(axis=-1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=-1, keepdims=True)


def _digest_vector(values: np.ndarray) -> str:
    """A stable, short digest of a feature vector: lets a reviewer compare runs."""

    return hashlib.sha256(np.asarray(values, dtype=np.float64).round(10).tobytes()).hexdigest()[:16]


__all__ = [
    "INPUT_SIZE",
    "MEMBER_STATS",
    "META_FEATURE_COUNT",
    "METRICS_PROVENANCE",
    "REPORTED_METRICS",
    "SEGMENTATION_MEMBERS",
    "EnsembleBundleService",
]
