from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Sequence
import numpy as np
logger = logging.getLogger(__name__)
@dataclass
class UncertaintyResult:
    available: bool
    method: str
    n_passes: int
    mean_dispersion: float | None = None
    max_dispersion: float | None = None
    p95_dispersion: float | None = None
    high_uncertainty_fraction: float | None = None
    per_pixel_map: np.ndarray | None = None
    boundary_dispersion: float | None = None
    interior_dispersion: float | None = None
    ensemble_disagreement: dict | None = None
    notes: list[str] = field(default_factory=list)
    unavailable_reason: str | None = None
    def summary(self) -> dict:
        return {
            "available": self.available,
            "method": self.method,
            "n_passes": self.n_passes,
            "mean_dispersion": self.mean_dispersion,
            "max_dispersion": self.max_dispersion,
            "p95_dispersion": self.p95_dispersion,
            "high_uncertainty_fraction": self.high_uncertainty_fraction,
            "boundary_dispersion": self.boundary_dispersion,
            "interior_dispersion": self.interior_dispersion,
            "ensemble_disagreement": self.ensemble_disagreement,
            "notes": self.notes,
            "unavailable_reason": self.unavailable_reason,
        }
METHOD_DESCRIPTIONS = {
    "test_time_augmentation": (
        "The image is segmented four times (identity, horizontal flip, vertical flip, 90 degree "
        "rotation) and each prediction is mapped back to the original frame. The reported "
        "dispersion is the standard deviation across those four probability maps, so it measures "
        "how sensitive the decision is to small geometric perturbations."
    ),
    "monte_carlo_dropout": (
        "Dropout layers are switched on during inference and the network is evaluated repeatedly. "
        "The standard deviation across passes is a Bayesian approximation of model uncertainty. "
        "This is only available for architectures that contain dropout layers - the reused "
        "checkpoints do not, and the API reports the method as unavailable for them."
    ),
    "ensemble_disagreement": (
        "The standard deviation between the probability maps of independently trained models. It "
        "captures model uncertainty, but it cannot detect cases where all models are wrong in the "
        "same way."
    ),
}
class UncertaintyService:
    def __init__(self, config):
        self.config = config
    def from_probability_stack(
        self, stack: Sequence[np.ndarray], *, method: str, threshold: float = 0.5
    ) -> UncertaintyResult:
        if stack is None or len(stack) < 2:
            return UncertaintyResult(
                available=False,
                method=method,
                n_passes=len(stack) if stack else 0,
                unavailable_reason=(
                    "At least two predictions are required to estimate dispersion; only one is "
                    "available for this configuration."
                ),
                notes=[METHOD_DESCRIPTIONS.get(method, "")] if method in METHOD_DESCRIPTIONS else [],
            )
        array = np.stack([np.asarray(map_, dtype=np.float32) for map_ in stack], axis=0)
        dispersion = array.std(axis=0)
        mean_probability = array.mean(axis=0)
        mask = mean_probability > threshold
        interior = _erode(mask)
        boundary = mask & ~interior
        high_uncertainty_fraction = float((dispersion > 0.1).mean())
        notes = [METHOD_DESCRIPTIONS.get(method, "")]
        if high_uncertainty_fraction > 0.2:
            notes.append(
                "A large share of pixels has a dispersion above 0.1; the segmentation of this "
                "study should be treated as unstable."
            )
        return UncertaintyResult(
            available=True,
            method=method,
            n_passes=len(stack),
            mean_dispersion=float(dispersion.mean()),
            max_dispersion=float(dispersion.max()),
            p95_dispersion=float(np.percentile(dispersion, 95)),
            high_uncertainty_fraction=high_uncertainty_fraction,
            per_pixel_map=dispersion.astype(np.float32),
            boundary_dispersion=float(dispersion[boundary].mean()) if boundary.any() else None,
            interior_dispersion=float(dispersion[interior].mean()) if interior.any() else None,
            notes=[note for note in notes if note],
        )
    @staticmethod
    def classification_confidence(
        probabilities: Sequence[float], *, calibrated: bool = False, method: str | None = None
    ) -> dict:
        probs = np.asarray(probabilities, dtype=float)
        if probs.size == 0:
            return {"available": False, "reason": "The classifier returned no scores."}
        predicted_index = int(np.argmax(probs))
        return {
            "available": True,
            "scores": probs.tolist(),
            "predicted_index": predicted_index,
            "model_score_of_prediction": float(probs[predicted_index]),
            "margin": float(probs[predicted_index] - np.partition(probs, -2)[-2]) if probs.size > 1 else None,
            "calibrated": bool(calibrated),
            "calibration_method": method,
            "statement": (
                "This value is the model's output score after softmax."
                + (
                    f" It has been calibrated on held-out data using {method}."
                    if calibrated
                    else " It has NOT been calibrated, so it must be read as a ranking score "
                    "rather than as a probability that the label is present."
                )
            ),
        }
def _erode(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask).astype(bool)
    if mask.ndim != 2 or not mask.any():
        return np.zeros_like(mask)
    padded = np.pad(mask, 1)
    eroded = mask.copy()
    for shift, axis in ((-1, 0), (1, 0), (-1, 1), (1, 1)):
        rolled = np.roll(padded, shift, axis=axis)
        sl = [slice(None), slice(None)]
        sl[axis] = 0 if shift == 1 else -1
        rolled[tuple(sl)] = False
        eroded &= rolled[1:-1, 1:-1]
    return eroded
def dispersion_to_heatmap(dispersion: np.ndarray, *, scale: float | None = None) -> np.ndarray:
    array = np.asarray(dispersion, dtype=np.float32)
    reference = float(scale if scale is not None else np.percentile(array, 99))
    if reference <= 1e-8:
        return np.zeros_like(array)
    return np.clip(array / reference, 0.0, 1.0)
