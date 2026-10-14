"""Serving path for the four-class tumour-type image classifier.

Wraps :class:`app.ml.image_classifier.KerasImageClassifier` with the HTTP-facing concerns:
decoding an uploaded image, validating what may be uploaded, and turning every failure into a
structured error a caller can act on.  Nothing here fabricates a prediction: when the
checkpoint, its metadata or TensorFlow is missing, the service reports the exact missing piece
and the endpoints answer with an error instead of a label.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path

import numpy as np

from ..errors import UnsupportedInputError, ValidationError
from ..ml.image_classifier import (
    DISPLAY_NAME,
    MODEL_ID,
    SUPPORTED_SUFFIXES,
    KerasImageClassifier,
    METRICS_PROVENANCE,
)
logger = logging.getLogger(__name__)


class ImageClassificationService:
    """Status and inference for the Keras image classifier."""

    def __init__(self, config, classifier: KerasImageClassifier | None = None):
        self.config = config
        self._classifier = classifier

    # ------------------------------------------------------------------ models
    @property
    def classifier(self) -> KerasImageClassifier:
        if self._classifier is None:
            self._classifier = KerasImageClassifier(self.config)
        return self._classifier

    def models(self, *, force: bool = False) -> list[dict]:
        """The catalogue the interface reads: one entry per image-classification model."""

        return [self.classifier.status(force=force)]

    def status(self, *, force: bool = False) -> dict:
        model = self.classifier.status(force=force)
        return {
            "available": bool(model["available"]),
            "state": model["state"],
            "reason": model["reason"],
            "models": [model],
            "accepted_formats": sorted(SUPPORTED_SUFFIXES),
            "task": "image_classification",
            "task_description": (
                "Tumour-type classification from a single MRI image: the model consumes the "
                "image itself (299 x 299 x 3 RGB), not radiomics features and not a mask."
            ),
            "distinct_from_grade_model": (
                "This is a different task from the BraTS HGG/LGG grade classifier: different "
                "dataset, different label space, different input. Predictions from the two are "
                "never combined."
            ),
            "metrics_provenance": METRICS_PROVENANCE,
            "install_note": (
                "Place brain_tumor_xception_final.keras and metadata.json in "
                f"{self.classifier.model_path.parent} (or point BRAIN_WEIGHTS_DIR at another "
                "directory). The weights are not shipped with this repository and are never "
                "downloaded automatically."
            ),
            "disclaimer": getattr(self.config, "DISCLAIMER", ""),
        }

    # --------------------------------------------------------------- inference
    def classify_upload(self, filename: str | None, data: bytes) -> dict:
        """Decode one uploaded image and classify it."""

        if not data:
            raise ValidationError("The uploaded file is empty.")
        name = Path(str(filename or "")).name
        suffix = Path(name).suffix.lower()
        if suffix not in SUPPORTED_SUFFIXES:
            raise UnsupportedInputError(
                f"'{name or 'upload'}' is not an accepted image format for the image "
                f"classifier. Accepted formats: {', '.join(sorted(SUPPORTED_SUFFIXES))}.",
                details={"filename": name, "accepted": sorted(SUPPORTED_SUFFIXES)},
            )
        array = self._decode(data, name)
        payload = self.classifier.predict(array, filename=name or None)
        payload["upload"] = {
            "filename": name or None,
            "bytes": len(data),
            "decoded_shape": list(array.shape),
        }
        return payload

    @staticmethod
    def _decode(data: bytes, name: str) -> np.ndarray:
        """Bytes -> RGB ``uint8`` array, with an honest error when the file is not an image."""

        from PIL import Image, UnidentifiedImageError

        try:
            with Image.open(io.BytesIO(data)) as handle:
                handle.load()
                converted = handle.convert("RGB")
                array = np.asarray(converted, dtype=np.uint8)
        except UnidentifiedImageError as exc:
            raise UnsupportedInputError(
                f"'{name}' could not be decoded as an image.",
                details={"filename": name},
            ) from exc
        except Exception as exc:  # noqa: BLE001 - malformed image data
            raise ValidationError(
                f"'{name}' could not be read: {type(exc).__name__}: {exc}",
                details={"filename": name},
            ) from exc
        if array.ndim != 3 or array.shape[2] != 3 or min(array.shape[:2]) < 16:
            raise ValidationError(
                f"'{name}' decoded to shape {list(array.shape)}, which is not a usable image; "
                "it must be at least 16 x 16 pixels.",
                details={"filename": name, "decoded_shape": list(array.shape)},
            )
        return array

    # ------------------------------------------------------------------ report
    def summary(self) -> dict:
        """A compact form for /api/capabilities."""

        model = self.classifier.status()
        return {
            "available": model["available"],
            "state": model["state"],
            "model_id": MODEL_ID,
            "display_name": DISPLAY_NAME,
            "reason": model["reason"],
            "classes": model.get("classes") or [],
            "input_size": model.get("input_size"),
            "preprocessing": model.get("preprocessing"),
            "reported_metrics": model.get("reported_metrics"),
            "metrics_provenance": METRICS_PROVENANCE,
            "model_path": model["model_path"],
            "metadata_path": model["metadata_file"],
        }


def describe_image_classifier(config) -> dict:
    """Convenience helper for scripts and diagnostics (no service instance needed)."""

    return ImageClassificationService(config).summary()


__all__ = ["ImageClassificationService", "describe_image_classifier"]
