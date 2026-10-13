from __future__ import annotations
import logging
from pathlib import Path
from dataclasses import dataclass, field
import numpy as np
import torch
from ..ml.registry import ModelRegistry
from ..ml.training_provenance import describe as describe_provenance
from ..ml.training_provenance import load_artifact_manifest
from ..utils.io import utc_now
logger = logging.getLogger(__name__)
REFERENCE_PHANTOM_VERSION = "phantom-v1"
def brainai_reference_phantom_4ch(size: int = 256, seed: int = 20240101) -> np.ndarray:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    centre = size / 2.0
    head = (((yy - centre) / (size * 0.37)) ** 2 + ((xx - centre) / (size * 0.30)) ** 2) <= 1.0
    wm = (((yy - centre) / (size * 0.31)) ** 2 + ((xx - centre) / (size * 0.26)) ** 2) <= 1.0
    lesion = (
        ((yy - (centre - size * 0.04)) / (size * 0.10)) ** 2
        + ((xx - (centre + size * 0.05)) / (size * 0.086)) ** 2
    ) <= 1.0
    noise = rng.normal(0, 0.03, (size, size)).astype(np.float32)
    t1 = np.where(wm, 0.62, 0.0) + np.where(lesion, -0.10, 0.0)
    t1ce = np.where(wm, 0.58, 0.0) + np.where(lesion, 0.45, 0.0)
    t2 = np.where(wm, 0.46, 0.0) + np.where(lesion, 0.55, 0.0)
    flair = np.where(wm, 0.34, 0.0) + np.where(lesion, 0.62, 0.0)
    stacked = np.stack([t1, t1ce, t2, flair], axis=-1).astype(np.float32) + noise[..., None]
    stacked = np.where(head[..., None], stacked, 0.0).astype(np.float32)
    normalised = (stacked - stacked.mean()) / max(stacked.std(), 1e-6)
    return normalised.transpose(2, 0, 1).astype(np.float32)
def reference_phantom_1ch(size: int = 512, seed: int = 20240101) -> np.ndarray:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    centre = size / 2.0
    head = (((yy - centre) / (size * 0.36)) ** 2 + ((xx - centre) / (size * 0.29)) ** 2) <= 1.0
    tissue = 0.45 + 0.05 * np.sin(xx / 17.0) + 0.04 * np.cos(yy / 13.0)
    lesion = np.exp(
        -(((yy - (centre - size * 0.05)) ** 2 + (xx - (centre + size * 0.06)) ** 2) / (size * 0.02) ** 2)
    )
    image = np.where(head, tissue + 0.35 * lesion, 0.0)
    image = image + rng.normal(0, 0.015, (size, size))
    return np.clip(image, 0.0, 1.0).astype(np.float32)[None]
@dataclass
class HealthReport:
    model_id: str
    status: str
    verdict: str
    device: str
    threshold: float
    reference: str
    output_semantics: str
    checks: list[dict] = field(default_factory=list)
    message: str = ""
    recommendation: str = ""
    checked_at: str = utc_now()
    phantom_version: str = REFERENCE_PHANTOM_VERSION
    training: dict = field(default_factory=dict)
    @property
    def responsive(self) -> bool:
        return self.verdict == "responsive"
    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "status": self.status,
            "verdict": self.verdict,
            "device": self.device,
            "threshold": self.threshold,
            "reference": self.reference,
            "output_semantics": self.output_semantics,
            "checks": self.checks,
            "message": self.message,
            "recommendation": self.recommendation,
            "checked_at": self.checked_at,
            "phantom_version": self.phantom_version,
            "responsive": self.responsive,
            "training": self.training,
        }
class ModelHealthService:
    def __init__(self, registry: ModelRegistry, config):
        self.registry = registry
        self.config = config
        self._cache: dict[str, HealthReport] = {}
        # Read once: what the training run recorded about each checkpoint. Absent file means
        # "no provenance recorded", never an assumption of good provenance.
        self._provenance = load_artifact_manifest(getattr(config, "DATA_DIR", Path("data")))
    def check(self, model_id: str, *, force: bool = False) -> HealthReport:
        if model_id in self._cache and not force:
            return self._cache[model_id]
        loaded = self.registry.load_model(model_id)
        descriptor = loaded.descriptor
        threshold = descriptor.fixed_threshold
        if not loaded.is_ready or loaded.module is None:
            report = HealthReport(
                model_id=model_id,
                status=loaded.status,
                verdict="not_loaded",
                device=str(loaded.device),
                threshold=threshold,
                reference="none",
                output_semantics=descriptor.output_semantics,
                message=loaded.message,
                recommendation=(
                    f"The model cannot be exercised until the checkpoint is available. "
                    f"Produce it with: {descriptor.trainer}"
                ),
                training=self._training_for(model_id),
            )
            self._cache[model_id] = report
            return report
        spec = descriptor.input_spec
        sources = self._reference_inputs(spec)
        module = loaded.module
        checks: list[dict] = []
        verdicts: list[str] = []
        for label, array in sources:
            tensor = torch.from_numpy(array[None]).to(loaded.device)
            try:
                with torch.no_grad():
                    output = module(tensor)
                logits = output[0] if isinstance(output, (list, tuple)) else output
                probabilities = (
                    torch.sigmoid(logits) if descriptor.output_semantics == "logit" else logits
                )
                probability = probabilities[0, 0].detach().cpu().numpy().astype(np.float32)
                raw = logits[0, 0].detach().cpu().numpy().astype(np.float32)
            except Exception as exc:
                logger.exception("Health check forward pass failed for %s", model_id)
                checks.append(
                    {
                        "input": label,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                verdicts.append("failed")
                continue
            foreground_fraction = float((probability > threshold).mean())
            spread = float(probability.max() - probability.min())
            entry = {
                "input": label,
                "input_shape": list(tensor.shape),
                "raw_min": float(raw.min()),
                "raw_max": float(raw.max()),
                "raw_mean": float(raw.mean()),
                "probability_min": float(probability.min()),
                "probability_max": float(probability.max()),
                "probability_mean": float(probability.mean()),
                "probability_spread": spread,
                "foreground_fraction_at_threshold": foreground_fraction,
                "threshold": threshold,
            }
            if spread < 1e-7:
                entry["verdict"] = "degenerate"
                entry["explanation"] = (
                    "The model produced a numerically constant output for this input, so it "
                    "cannot separate foreground from background."
                )
            elif foreground_fraction <= 0.0:
                entry["verdict"] = "no_foreground"
                entry["explanation"] = (
                    "No pixel reaches the decision threshold. Two readings are possible: there is "
                    "nothing to segment, or the checkpoint is not responding to this input "
                    "distribution."
                )
            elif foreground_fraction >= 0.999:
                entry["verdict"] = "saturated"
                entry["explanation"] = "Essentially every pixel is above the threshold."
            else:
                entry["verdict"] = "responsive"
                entry["explanation"] = (
                    "The model separates the reference input into a foreground region and a "
                    "background at the configured threshold."
                )
            checks.append(entry)
            verdicts.append(entry["verdict"])
        priority = ["failed", "degenerate", "saturated", "no_foreground", "responsive"]
        overall = next((v for v in priority if v in verdicts), "failed")
        message, recommendation = self._describe(overall, checks, descriptor)
        report = HealthReport(
            model_id=model_id,
            status=loaded.status,
            verdict=overall,
            device=str(loaded.device),
            threshold=threshold,
            reference=sources[0][0] if sources else "none",
            output_semantics=descriptor.output_semantics,
            checks=checks,
            message=message,
            recommendation=recommendation,
            training=self._training_for(model_id),
        )
        self._cache[model_id] = report
        return report

    def _training_for(self, model_id: str) -> dict:
        """What the training run recorded about this checkpoint (see training_provenance)."""
        return describe_provenance(model_id, self._provenance, data_dir=getattr(self.config, "DATA_DIR", None))
    def _reference_inputs(self, spec) -> list[tuple[str, np.ndarray]]:
        if spec.channels == 4:
            phantom = brainai_reference_phantom_4ch(spec.height)
            return [
                (f"BraTS-like phantom ({REFERENCE_PHANTOM_VERSION}, {spec.height}px)", phantom),
            ]
        if spec.channels == 1:
            phantom = reference_phantom_1ch(spec.height)
            return [
                (f"grayscale phantom ({REFERENCE_PHANTOM_VERSION}, {spec.height}px)", phantom),
            ]
        return []
    @staticmethod
    def _describe(verdict: str, checks: list[dict], descriptor) -> tuple[str, str]:
        if verdict == "responsive":
            return (
                "The checkpoint loads, runs and produces a usable probability map on the "
                "reference input.",
                "No action required. Evaluation against labelled data is still necessary before "
                "any accuracy claim.",
            )
        if verdict == "failed":
            return (
                "The forward pass failed; see the per-input error in the checks list.",
                f"Re-export the checkpoint using {descriptor.trainer}.",
            )
        if verdict == "degenerate":
            return (
                "The model returned a constant output for the reference input.",
                f"Retrain with {descriptor.trainer} and verify the new checkpoint with this check.",
            )
        if verdict == "saturated":
            return (
                "The model marked essentially the whole reference image as foreground.",
                "Check the checkpoint and the input normalisation before using its output.",
            )
        return (
            "The checkpoint loads and runs, but every output value stays below the decision "
            "threshold, so the produced mask is empty for the reference input. A checkpoint that "
            "loads is therefore not necessarily a checkpoint that works, and this model cannot be "
            "reported as a functioning segmenter until it has been retrained and re-checked.",
            "Retrain the model on the project dataset with scripts/train_segnet.py (or "
            "scripts/train_unet.py) and re-run this check. Until then, treat any mask this model "
            "produces as invalid rather than as evidence that no abnormality is present.",
        )
    def healthy_segmentation_models(self) -> list[str]:
        return [
            model_id
            for model_id in self.registry.segmentation_model_ids()
            if (loaded := self.registry.load_model(model_id)).is_ready
            and self.check(model_id).verdict == "responsive"
        ]
    def summary(self) -> dict:
        reports = [self.check(model_id).to_dict() for model_id in self.registry.descriptors]
        return {
            "phantom_version": REFERENCE_PHANTOM_VERSION,
            "generated_at": utc_now(),
            "models": reports,
            "responsive_models": [r["model_id"] for r in reports if r["responsive"]],
        }
