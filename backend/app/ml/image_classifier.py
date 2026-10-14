"""TensorFlow/Keras image classifier: the four-class tumour-type model.

Source
------
`firstnb.ipynb` (supplied notebook, kept out of git with the other weight artefacts)
trains ``Xception(weights="imagenet", include_top=False, input_shape=(299, 299, 3))`` with a
``GlobalAveragePooling2D`` head, ``Dropout(0.4)`` and a 4-way softmax over
``["Glioma", "Meningioma", "Pituitary", "No Tumor"]``, and writes two files:

* ``brain_tumor_xception_final.keras`` — the trained model;
* ``metadata.json`` — architecture, framework version, input size, class order,
  preprocessing, augmentation, training configuration, reported metrics and hardware.

Contract served here (taken from those two files, never assumed)
----------------------------------------------------------------
* input          ``(299, 299, 3)`` RGB, float32
* preprocessing  ``keras.applications.xception.preprocess_input`` — implemented in NumPy as
                 ``x / 127.5 - 1`` so the scaling is identical without importing TensorFlow;
                 any other recorded preprocessing is refused rather than guessed
* output         4 softmax probabilities, ordered by ``class_to_index`` from the metadata
* labels         tumour *type*, not grade, not malignancy and not stage

Why the metadata is required
----------------------------
The checkpoint alone does not say which index means which class. Serving index 3 as
"No Tumor" when the training run meant "Pituitary" would be a silent, invisible error, so a
missing or inconsistent ``metadata.json`` disables this model with an explanation instead of
producing a label. Nothing is downloaded, nothing is retrained and no prediction is invented.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ..errors import DependencyMissingError, ModelUnavailableError, ValidationError
from ..utils.io import read_json, utc_now

logger = logging.getLogger(__name__)

MODEL_ID = "brain_tumor_xception"
DISPLAY_NAME = "Tumour type (Xception, four classes)"

#: Preprocessing names that all mean "scale [0, 255] RGB to [-1, 1]".
XCEPTION_NAMES = {
    "keras.applications.xception.preprocess_input",
    "tf.keras.applications.xception.preprocess_input",
    "xception.preprocess_input",
    "xception",
}

SUPPORTED_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

LABEL_SEMANTICS = (
    "The four classes are tumour types recorded in the image dataset "
    "(glioma, meningioma, pituitary, no tumour). They are not a grade, not a malignancy "
    "verdict, not a clinical stage and not a prognosis, and the model does not see the "
    "BraTS patient-grade labels."
)

METRICS_PROVENANCE = (
    "These numbers are the metrics the training notebook recorded for this checkpoint. They "
    "have not been re-measured by this deployment: the test split it was scored on is not "
    "shipped with the weights, so they are reported as historical values, not as a "
    "verification of the artefact in place."
)


def _import_tensorflow():
    """Import TensorFlow lazily, with an actionable message when it is absent."""

    try:
        import tensorflow as tf  # noqa: PLC0415 - deliberate lazy import (heavy, optional)
    except Exception as exc:  # pragma: no cover - depends on the environment
        raise DependencyMissingError(
            "Serving the Keras image classifier needs TensorFlow in the API environment.",
            details={
                "dependency": "tensorflow",
                "import_error": f"{type(exc).__name__}: {exc}",
                "install": "pip install tensorflow-cpu   # or tensorflow for a GPU host",
                "alternative": (
                    "The pytorch models and every non-classification endpoint work without "
                    "TensorFlow; only /api/classification/image needs it."
                ),
            },
        ) from exc
    return tf


def xception_preprocess(array: np.ndarray) -> np.ndarray:
    """``keras.applications.xception.preprocess_input`` for float RGB in ``[0, 255]``.

    The Keras function scales to ``[-1, 1]`` with ``x / 127.5 - 1`` and writes the result
    back as float32. Implementing it here keeps serving independent of TensorFlow for
    everything except the forward pass, and keeps the preprocessing testable on its own.
    """

    values = np.asarray(array, dtype=np.float32)
    return np.ascontiguousarray(values / 127.5 - 1.0)


def is_probability_vector(values: np.ndarray) -> bool:
    """True when ``values`` already looks like a probability distribution."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        return False
    if not np.all(np.isfinite(array)):
        return False
    if array.min() < -1e-6 or array.max() > 1 + 1e-6:
        return False
    return abs(float(array.sum()) - 1.0) <= 1e-3


def softmax(values: np.ndarray) -> np.ndarray:
    """Numerically stable softmax, used only when a model returned logits."""

    array = np.asarray(values, dtype=np.float64)
    shifted = array - np.max(array)
    exponentials = np.exp(shifted)
    return exponentials / np.sum(exponentials)


class KerasImageClassifier:
    """Loads ``brain_tumor_xception_final.keras`` + ``metadata.json`` and serves one image.

    Everything is lazy: constructing the service costs nothing, the framework is imported and
    the model is read only on the first prediction, and the loaded model is cached afterwards
    so an HTTP request never pays the load twice.
    """

    def __init__(
        self,
        config,
        *,
        model_path: str | Path | None = None,
        metadata_path: str | Path | None = None,
        loader: Callable[[str], Any] | None = None,
    ) -> None:
        self.config = config
        self._model: Any | None = None
        self._explicit_model_path = Path(model_path) if model_path else None
        self._metadata_path = Path(metadata_path) if metadata_path else None
        self._loader = loader
        self._metadata: dict | None = None
        self._metadata_error: str | None = None
        self._framework: dict | None = None
        self._load_seconds: float | None = None
        self._loaded_at: str | None = None

    # ------------------------------------------------------------------ paths
    @property
    def model_path(self) -> Path:
        """Where the ``.keras`` checkpoint is read from, resolved on every call.

        Precedence: an explicit path given to the constructor, then ``BRAIN_MODEL_XCEPTION``,
        then the central search order (``models/weights`` -> repository root -> ``data/models``).
        Resolving lazily means dropping the file into the weights directory takes effect on the
        next request instead of requiring a restart, and moving the directory through
        ``BRAIN_WEIGHTS_DIR`` always wins over the legacy locations.
        """

        if self._explicit_model_path is not None:
            return self._explicit_model_path
        override = os.environ.get("BRAIN_MODEL_XCEPTION")
        if override and override.strip():
            return Path(override).expanduser()
        from ..config import artifact_path  # local import: avoids a circular import

        return artifact_path("brain_tumor_xception_final.keras")

    @property
    def metadata_path(self) -> Path:
        """``metadata.json``: the declared path, next to the model, or in the weights directory.

        Deliberately *not* a generic search over the repository: a stray ``metadata.json``
        somewhere else must never be mistaken for this model's class ordering.
        """

        if self._metadata_path is not None:
            return self._metadata_path
        override = os.environ.get("BRAIN_MODEL_XCEPTION_METADATA")
        if override and override.strip():
            return Path(override).expanduser()
        beside_model = self.model_path.parent / "metadata.json"
        if beside_model.exists():
            return beside_model
        from ..config import weights_directory  # local import: avoids a circular import

        return weights_directory() / "metadata.json"

    # --------------------------------------------------------------- metadata
    def read_metadata(self) -> dict | None:
        """The validated metadata, or ``None`` (with ``metadata_error`` set)."""

        if self._metadata is not None or self._metadata_error is not None:
            return self._metadata
        path = self.metadata_path
        try:
            if not path.is_file():
                self._metadata_error = (
                    f"{path} was not found. The checkpoint does not record which output index "
                    "means which class, so the classifier cannot be served without it."
                )
                return None
            payload = read_json(path)
        except Exception as exc:  # noqa: BLE001 - reported, never raised from a status call
            self._metadata_error = f"{path} could not be read: {type(exc).__name__}: {exc}"
            return None
        if not isinstance(payload, dict):
            self._metadata_error = f"{path} is not a JSON object."
            return None
        problem = self._validate_metadata(payload)
        if problem:
            self._metadata_error = problem
            return None
        self._metadata = payload
        return payload

    @staticmethod
    def _validate_metadata(payload: dict) -> str | None:
        """Return a human-readable problem with the metadata, or ``None`` when it is usable."""

        classes = payload.get("classes")
        mapping = payload.get("class_to_index")
        if not isinstance(classes, list) or len(classes) < 2:
            return "metadata 'classes' must list at least two class names."
        if not isinstance(mapping, dict) or len(mapping) != len(classes):
            return "metadata 'class_to_index' must map every class name to an index."
        normalised = {str(name): int(index) for name, index in mapping.items()}
        if sorted(normalised.values()) != list(range(len(classes))):
            return (
                "metadata 'class_to_index' is not a 0..n-1 permutation: "
                f"{sorted(normalised.values())} for {len(classes)} classes."
            )
        if set(normalised) != {str(name) for name in classes}:
            return "metadata 'classes' and 'class_to_index' disagree about the class names."
        input_size = payload.get("input_size")
        if input_size is not None:
            size = list(input_size)
            if len(size) not in (2, 3):
                return f"metadata 'input_size' must have 2 or 3 entries, got {size}."
            if len(size) == 3 and int(size[2]) != 3:
                return (
                    f"metadata declares {int(size[2])} input channels; this serving path only "
                    "implements the three-channel RGB contract of the notebook."
                )
        preprocessing = str(payload.get("preprocessing") or "")
        if preprocessing and preprocessing not in XCEPTION_NAMES:
            return (
                f"metadata records preprocessing '{preprocessing}', which this build does not "
                "implement. Serving it with a different scaling would change the input "
                "distribution silently, so the model is refused instead."
            )
        return None

    @property
    def metadata_error(self) -> str | None:
        return self._metadata_error

    # ----------------------------------------------------------------- status
    def status(self, *, force: bool = False) -> dict:
        """Everything the interface may say about this model, without loading it."""

        if force:
            self._metadata = None
            self._metadata_error = None
            self._framework = None
        metadata = self.read_metadata()
        model_exists = self.model_path.is_file()
        # Probing the framework imports TensorFlow (~1.5 s, and it logs a banner), so it is
        # only done once the files that would be executed are actually present.
        framework = (
            self._framework_status()
            if model_exists
            else {"available": None, "version": None, "error": None, "probed": False}
        )
        state, reason = self._state(model_exists, metadata, framework)
        payload: dict[str, Any] = {
            "model_id": MODEL_ID,
            "display_name": DISPLAY_NAME,
            "task": "image_classification",
            "framework": "TensorFlow/Keras",
            "state": state,
            "available": state == "ready",
            "reason": reason,
            "model_file": self.model_path.name,
            "model_path": str(self.model_path),
            "model_file_present": model_exists,
            "metadata_file": str(self.metadata_path),
            "metadata_present": metadata is not None,
            "framework_available": framework["available"],
            "framework_version": framework["version"],
            "framework_error": framework["error"],
            "loaded": self._model is not None,
            "loaded_at": self._loaded_at,
            "load_seconds": self._load_seconds,
            "parameter_count": self._parameter_count(),
            "label_semantics": LABEL_SEMANTICS,
            "preprocessing": None,
            "preprocessing_implementation": (
                "RGB in [0, 255] -> x / 127.5 - 1 (identical to "
                "keras.applications.xception.preprocess_input)"
            ),
            "input_size": None,
            "classes": [],
            "class_to_index": {},
            "reported_metrics": None,
            "metrics_provenance": METRICS_PROVENANCE,
            "training_config": None,
            "hardware": None,
            "architecture": None,
            "framework_version_recorded": None,
            "python_version_recorded": None,
            "installed_at": None,
            "trainer": "firstnb.ipynb (Keras/TensorFlow notebook uploaded with the weights)",
            "searched": [str(self.model_path), str(self.metadata_path)],
            "accepted_formats": sorted(SUPPORTED_SUFFIXES),
        }
        if metadata is not None:
            payload.update(
                {
                    "input_size": metadata.get("input_size"),
                    "classes": list(metadata.get("classes") or []),
                    "class_to_index": metadata.get("class_to_index") or {},
                    "preprocessing": metadata.get("preprocessing"),
                    "architecture": metadata.get("architecture"),
                    "framework": metadata.get("framework") or payload["framework"],
                    "framework_version_recorded": metadata.get("framework_version"),
                    "python_version_recorded": metadata.get("python_version"),
                    "reported_metrics": metadata.get("metrics"),
                    "training_config": metadata.get("training_config"),
                    "hardware": metadata.get("hardware"),
                    "model_name": metadata.get("model_name"),
                    "augmentation": metadata.get("augmentation"),
                    "recorded_model_artifact": metadata.get("model_artifact"),
                    "installed_at": metadata.get("generated_at"),
                }
            )
        return payload

    def _framework_status(self) -> dict:
        if self._framework is not None:
            return self._framework
        try:
            tf = _import_tensorflow()
        except DependencyMissingError as exc:
            self._framework = {"available": False, "version": None, "error": exc.message}
        else:
            self._framework = {
                "available": True,
                "version": getattr(tf, "__version__", None),
                "error": None,
            }
        return self._framework

    def _state(self, model_exists: bool, metadata: dict | None, framework: dict) -> tuple[str, str]:
        if not model_exists:
            return (
                "checkpoint_missing",
                f"{self.model_path.name} is not installed. Place it - together with "
                f"metadata.json - in {self.model_path.parent} (see models/README.md).",
            )
        if metadata is None:
            return "metadata_missing", self._metadata_error or "metadata.json is not usable."
        if framework["available"] is False:
            return (
                "framework_missing",
                "TensorFlow is not importable in this environment, so the Keras checkpoint "
                "cannot be executed.",
            )
        return "ready", (
            "Checkpoint and metadata are present; the model is loaded on the first prediction."
        )

    def _parameter_count(self) -> int | None:
        if self._model is None:
            return None
        try:
            return int(self._model.count_params())
        except Exception:  # noqa: BLE001 - cosmetic information only
            return None

    # -------------------------------------------------------------- inference
    def ensure_loaded(self):
        """Load and validate the model once. Raises a structured error when it cannot be used."""

        if self._model is not None:
            return self._model, self._metadata or {}
        status = self.status()
        if not status["available"]:
            raise ModelUnavailableError(
                status["reason"],
                details={
                    "model_id": MODEL_ID,
                    "state": status["state"],
                    "model_path": status["model_path"],
                    "metadata_path": status["metadata_file"],
                    "expected_files": [
                        "brain_tumor_xception_final.keras",
                        "metadata.json",
                    ],
                },
            )
        metadata = self._metadata or {}
        started = time.perf_counter()
        tf = _import_tensorflow()
        loader = self._loader or (lambda path: tf.keras.models.load_model(path, compile=False))
        try:
            model = loader(str(self.model_path))
        except Exception as exc:  # noqa: BLE001 - the message is what the operator needs
            raise ModelUnavailableError(
                f"{self.model_path.name} could not be loaded: {type(exc).__name__}: {exc}",
                details={
                    "model_id": MODEL_ID,
                    "hint": (
                        "The file must be the .keras archive written by the training notebook. "
                        "Loading a Keras-2 .h5 file, a bare weights file or a pickled object "
                        "needs the conversion step described in models/README.md."
                    ),
                },
            ) from exc
        self._check_model_contract(model, metadata)
        self._model = model
        self._load_seconds = round(time.perf_counter() - started, 4)
        self._loaded_at = utc_now()
        logger.info(
            "Loaded %s from %s (%s parameters, %.2fs)",
            MODEL_ID,
            self.model_path,
            self._parameter_count(),
            self._load_seconds,
        )
        return self._model, metadata

    def _check_model_contract(self, model, metadata: dict) -> None:
        """The checkpoint must agree with the metadata before a single image is served."""

        input_shape = getattr(model, "input_shape", None)
        if isinstance(input_shape, list):  # multi-input model: not this contract
            input_shape = input_shape[0]
        output_shape = getattr(model, "output_shape", None)
        if isinstance(output_shape, list):
            output_shape = output_shape[0]
        expected_classes = len(metadata.get("classes") or [])
        declared_size = metadata.get("input_size")
        if output_shape is not None and expected_classes:
            actual_outputs = int(output_shape[-1])
            if actual_outputs != expected_classes:
                raise ModelUnavailableError(
                    f"The checkpoint outputs {actual_outputs} values but metadata.json declares "
                    f"{expected_classes} classes; the class mapping cannot be trusted.",
                    details={"model_id": MODEL_ID, "output_shape": list(output_shape)},
                )
        if input_shape is not None and declared_size:
            size = list(declared_size)
            actual = list(input_shape[1:])
            if len(size) == 3 and actual and tuple(actual) != tuple(int(v) for v in size):
                raise ModelUnavailableError(
                    f"The checkpoint expects input {actual} but metadata.json declares {size}.",
                    details={"model_id": MODEL_ID},
                )
            if len(size) == 2 and actual[:2] != [int(v) for v in size]:
                raise ModelUnavailableError(
                    f"The checkpoint expects input {actual} but metadata.json declares {size}.",
                    details={"model_id": MODEL_ID},
                )
        if input_shape is not None and int(input_shape[-1]) != 3:
            raise ModelUnavailableError(
                f"The checkpoint takes {int(input_shape[-1])} input channels; this serving path "
                "implements the three-channel RGB contract of the notebook.",
                details={"model_id": MODEL_ID, "input_shape": list(input_shape)},
            )

    def predict(self, image: np.ndarray, *, filename: str | None = None) -> dict:
        """Run one RGB image through the classifier and return probabilities + provenance."""

        model, metadata = self.ensure_loaded()
        rgb = np.asarray(image)
        if rgb.ndim != 3 or rgb.shape[-1] != 3:
            raise ValidationError(
                f"The image classifier needs an RGB image (H, W, 3); received shape {rgb.shape}.",
                details={"model_id": MODEL_ID},
            )
        size = metadata.get("input_size") or [299, 299, 3]
        height, width = int(size[0]), int(size[1])
        prepared = self._prepare(rgb, height, width)
        started = time.perf_counter()
        raw = np.asarray(model.predict(prepared[None, ...], verbose=0))[0]
        seconds = time.perf_counter() - started
        if is_probability_vector(raw):
            probabilities = np.asarray(raw, dtype=np.float64)
            transform = "none (the model returned a probability distribution)"
        else:
            probabilities = softmax(raw)
            transform = "softmax applied to the model output"
        classes = list(metadata.get("classes") or [])
        mapping = metadata.get("class_to_index") or {}
        ordered = [str(name) for name, _ in sorted(mapping.items(), key=lambda item: int(item[1]))]
        if len(ordered) != len(probabilities):
            raise ModelUnavailableError(
                f"metadata lists {len(ordered)} classes but the model returned "
                f"{len(probabilities)} values.",
                details={"model_id": MODEL_ID, "classes": ordered, "outputs": int(len(probabilities))},
            )
        index = int(np.argmax(probabilities))
        return {
            "model_id": MODEL_ID,
            "display_name": DISPLAY_NAME,
            "available": True,
            "task": "image_classification",
            "filename": filename,
            "predicted_index": index,
            "predicted_label": ordered[index],
            "probability": float(probabilities[index]),
            "classes": ordered,
            "probabilities": [float(value) for value in probabilities],
            "scores": [
                {"index": position, "label": label, "probability": float(probabilities[position])}
                for position, label in enumerate(ordered)
            ],
            "class_to_index": mapping,
            "classes_source": str(self.metadata_path),
            "output_transform": transform,
            "calibrated": False,
            "calibration_note": (
                "The values are the network's softmax output. They are not calibrated "
                "probabilities and they are not a measure of clinical certainty."
            ),
            "input": {
                "filename": filename,
                "original_shape": list(rgb.shape),
                "model_input_size": [height, width, 3],
                "resize": "bilinear (PIL) to the size recorded in metadata.json",
            },
            "preprocessing": metadata.get("preprocessing"),
            "preprocessing_implementation": "RGB [0, 255] -> x / 127.5 - 1",
            "inference_seconds": round(seconds, 4),
            "device": "see /api/system",
            "architecture": metadata.get("architecture"),
            "framework_version_recorded": metadata.get("framework_version"),
            "reported_metrics": metadata.get("metrics"),
            "metrics_provenance": METRICS_PROVENANCE,
            "training_config": metadata.get("training_config"),
            "label_semantics": LABEL_SEMANTICS,
            "disclaimer": getattr(self.config, "DISCLAIMER", ""),
        }

    @staticmethod
    def _prepare(rgb: np.ndarray, height: int, width: int) -> np.ndarray:
        """Resize to the model input and apply the Xception scaling."""

        from PIL import Image  # local import keeps module import cheap

        array = np.asarray(rgb)
        if array.dtype != np.uint8:
            if np.issubdtype(array.dtype, np.floating) and float(np.nanmax(array)) <= 1.0 + 1e-6:
                array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
            else:
                array = np.clip(array, 0, 255).astype(np.uint8)
        image = Image.fromarray(array.astype(np.uint8), mode="RGB")
        if image.size != (width, height):
            image = image.resize((width, height), Image.BILINEAR)
        return xception_preprocess(np.asarray(image, dtype=np.float32))
