from __future__ import annotations
import logging
import time
from dataclasses import dataclass, field
import numpy as np
import torch
import torch.nn.functional as F
from ..errors import InferenceError, ModelUnavailableError
from ..ml.registry import LoadedModel, ModelRegistry
from ..utils.images import resize_array
from .preprocessing_service import PreprocessedInput
logger = logging.getLogger(__name__)
TTA_TRANSFORMS = ("identity", "hflip", "vflip", "rot90")
@dataclass
class SegmentationResult:
    model_id: str
    model_display_name: str
    device: str
    threshold: float
    probability: np.ndarray
    mask: np.ndarray
    foreground_pixels: int
    foreground_fraction: float
    probability_mean_in_mask: float | None
    input_shape: list[int]
    output_semantics: str
    inference_seconds: float
    tta_used: bool
    mc_dropout_used: bool
    n_passes: int
    warnings: list[str] = field(default_factory=list)
    extra: dict = field(default_factory=dict)
    def resized(self, size: tuple[int, int]) -> "SegmentationResult":
        if self.probability.shape == size:
            return self
        probability = resize_array(self.probability, size)
        return SegmentationResult(
            model_id=self.model_id,
            model_display_name=self.model_display_name,
            device=self.device,
            threshold=self.threshold,
            probability=probability.astype(np.float32),
            mask=(probability > self.threshold),
            foreground_pixels=int((probability > self.threshold).sum()),
            foreground_fraction=float((probability > self.threshold).mean()),
            probability_mean_in_mask=self.probability_mean_in_mask,
            input_shape=self.input_shape,
            output_semantics=self.output_semantics,
            inference_seconds=self.inference_seconds,
            tta_used=self.tta_used,
            mc_dropout_used=self.mc_dropout_used,
            n_passes=self.n_passes,
            warnings=list(self.warnings),
            extra=dict(self.extra),
        )
    def summary(self) -> dict:
        payload = {
            "model_id": self.model_id,
            "model_display_name": self.model_display_name,
            "device": self.device,
            "threshold": self.threshold,
            "input_shape": self.input_shape,
            "output_semantics": self.output_semantics,
            "foreground_pixels": self.foreground_pixels,
            "foreground_fraction": round(self.foreground_fraction, 6),
            "probability_mean_in_mask": self.probability_mean_in_mask,
            "probability_max": float(self.probability.max()),
            "probability_mean": float(self.probability.mean()),
            "inference_seconds": round(self.inference_seconds, 4),
            "tta_used": self.tta_used,
            "mc_dropout_used": self.mc_dropout_used,
            "n_passes": self.n_passes,
            "warnings": self.warnings,
        }
        payload.update(self.extra)
        return payload
class SegmentationService:
    def __init__(self, registry: ModelRegistry, config):
        self.registry = registry
        self.config = config
    def require_model(self, model_id: str) -> LoadedModel:
        loaded = self.registry.load_model(model_id)
        if not loaded.is_ready:
            raise ModelUnavailableError(
                f"Model '{model_id}' is not available ({loaded.status}): {loaded.message}",
                details={"model_id": model_id, "status": loaded.status},
            )
        return loaded
    def predict(
        self,
        model_id: str,
        preprocessed: PreprocessedInput,
        *,
        threshold: float | None = None,
        tta: bool = False,
        mc_dropout: bool = False,
        mc_samples: int | None = None,
    ) -> SegmentationResult:
        loaded = self.require_model(model_id)
        descriptor = loaded.descriptor
        spec = descriptor.input_spec
        tensor = preprocessed.tensor
        if tensor.ndim != 4:
            raise InferenceError(f"Expected a 4-D tensor, received shape {list(tensor.shape)}")
        if tensor.shape[1] != spec.channels:
            raise InferenceError(
                f"{model_id} expects {spec.channels} input channel(s); the preprocessor produced "
                f"{tensor.shape[1]}. The input pipeline and the model are not compatible."
            )
        expected = (spec.height, spec.width)
        if (tensor.shape[2], tensor.shape[3]) != expected:
            tensor = F.interpolate(
                tensor, size=expected, mode="bilinear", align_corners=False
            )
        threshold = descriptor.fixed_threshold if threshold is None else float(threshold)
        device = loaded.device
        module = loaded.module
        assert module is not None
        warnings: list[str] = []
        if mc_dropout and not descriptor.supports_mc_dropout:
            warnings.append(
                f"{model_id} does not contain dropout layers; Monte-Carlo dropout is not applicable."
            )
            mc_dropout = False
        started = time.perf_counter()
        n_passes = 0
        predictions: list[np.ndarray] = []
        if mc_dropout:
            samples = int(mc_samples or getattr(self.config, "MC_DROPOUT_SAMPLES", 8))
            predictions.extend(self._mc_dropout_passes(module, tensor, device, samples))
            n_passes += samples
        elif tta:
            predictions.extend(self._tta_passes(module, tensor, device))
            n_passes += len(TTA_TRANSFORMS)
        else:
            predictions.append(self._forward_probability(module, tensor, device))
            n_passes += 1
        stacked = np.stack(predictions, axis=0)
        probability = stacked.mean(axis=0).astype(np.float32)
        spread = stacked.std(axis=0).astype(np.float32) if len(predictions) > 1 else np.zeros_like(probability)
        inference_seconds = time.perf_counter() - started
        mask = probability > threshold
        foreground = int(mask.sum())
        mean_in_mask = float(probability[mask].mean()) if foreground else None
        if foreground == 0:
            warnings.append(
                "The model produced an empty mask at the configured threshold. This can mean "
                "no abnormality is present, or that the input does not match the model's training "
                "distribution."
            )
        if probability.mean() > 0.5:
            warnings.append(
                "More than half of the image is above the threshold; the input may be outside the "
                "model's expected intensity distribution."
            )
        return SegmentationResult(
            model_id=model_id,
            model_display_name=descriptor.display_name,
            device=str(device),
            threshold=threshold,
            probability=probability,
            mask=mask,
            foreground_pixels=foreground,
            foreground_fraction=float(foreground / mask.size),
            probability_mean_in_mask=mean_in_mask,
            input_shape=list(tensor.shape),
            output_semantics=descriptor.output_semantics,
            inference_seconds=inference_seconds,
            tta_used=bool(tta and not mc_dropout),
            mc_dropout_used=bool(mc_dropout),
            n_passes=n_passes,
            warnings=warnings,
            extra={
                "predictive_spread_mean": float(spread.mean()) if len(predictions) > 1 else None,
                "predictive_spread_max": float(spread.max()) if len(predictions) > 1 else None,
            },
        )
    def _forward_probability(self, module, tensor: torch.Tensor, device) -> np.ndarray:
        module.eval()
        with torch.no_grad():
            outputs = module(tensor.to(device))
        logits = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
        semantics = getattr(module, "output_semantics", "probability")
        if semantics == "logit":
            probabilities = torch.sigmoid(logits)
        else:
            probabilities = logits
        return probabilities[0, 0].detach().cpu().numpy().astype(np.float32)
    def _tta_passes(self, module, tensor: torch.Tensor, device) -> list[np.ndarray]:
        results = []
        for name in TTA_TRANSFORMS:
            augmented = self._apply_tta(tensor, name)
            prediction = self._forward_probability(module, augmented, device)
            results.append(self._invert_tta(prediction, name))
        return results
    def _mc_dropout_passes(
        self, module, tensor: torch.Tensor, device, samples: int
    ) -> list[np.ndarray]:
        module.eval()
        enabled = []
        for submodule in module.modules():
            if isinstance(submodule, torch.nn.Dropout2d) or isinstance(submodule, torch.nn.Dropout):
                submodule.train()
                enabled.append(submodule)
        if not enabled:
            module.eval()
            return [self._forward_probability(module, tensor, device)]
        predictions = []
        try:
            for _ in range(max(1, samples)):
                predictions.append(self._forward_probability(module, tensor, device))
        finally:
            module.eval()
        return predictions
    @staticmethod
    def _apply_tta(tensor: torch.Tensor, name: str) -> torch.Tensor:
        if name == "identity":
            return tensor
        if name == "hflip":
            return torch.flip(tensor, dims=[3])
        if name == "vflip":
            return torch.flip(tensor, dims=[2])
        if name == "rot90":
            return torch.rot90(tensor, k=1, dims=[2, 3])
        return tensor
    @staticmethod
    def _invert_tta(array: np.ndarray, name: str) -> np.ndarray:
        if name == "identity":
            return array
        if name == "hflip":
            return np.flip(array, axis=1).copy()
        if name == "vflip":
            return np.flip(array, axis=0).copy()
        if name == "rot90":
            return np.rot90(array, k=-1).copy()
        return array
