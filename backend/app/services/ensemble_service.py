from __future__ import annotations
import logging
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
from ..errors import ModelUnavailableError, ValidationError
from ..ml.registry import ModelRegistry
from ..utils.io import read_json
from .segmentation_service import SegmentationResult, SegmentationService
logger = logging.getLogger(__name__)
STRATEGIES = ("equal", "calibrated", "confidence_aware")
@dataclass
class EnsembleResult:
    available: bool
    strategy: str
    weights: dict[str, float]
    weight_source: str
    members: list[str]
    threshold: float
    probability: np.ndarray | None = None
    mask: np.ndarray | None = None
    foreground_pixels: int = 0
    foreground_fraction: float = 0.0
    member_results: dict[str, SegmentationResult] = field(default_factory=dict)
    disagreement: dict | None = None
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    def summary(self) -> dict:
        return {
            "available": self.available,
            "strategy": self.strategy,
            "weights": self.weights,
            "weight_source": self.weight_source,
            "members": self.members,
            "threshold": self.threshold,
            "foreground_pixels": self.foreground_pixels,
            "foreground_fraction": round(self.foreground_fraction, 6),
            "disagreement": self.disagreement,
            "reasons": self.reasons,
            "warnings": self.warnings,
        }
class EnsembleService:
    def __init__(self, registry: ModelRegistry, config, segmentation: SegmentationService, health=None):
        self.registry = registry
        self.config = config
        self.segmentation = segmentation
        self.health = health
    def _health_of(self, model_id: str) -> dict | None:
        if self.health is None:
            return None
        report = self.health.check(model_id)
        return {"verdict": report.verdict, "responsive": report.responsive, "message": report.message}
    def compatibility(self, model_ids: list[str] | None = None) -> dict:
        model_ids = model_ids or self.registry.segmentation_model_ids()
        statuses = []
        ready_by_spec: dict[tuple, list[str]] = {}
        for model_id in model_ids:
            loaded = self.registry.load_model(model_id)
            spec = loaded.descriptor.input_spec
            key = (spec.channels, spec.height, spec.width, spec.normalisation)
            health = self._health_of(model_id)
            statuses.append(
                {
                    "model_id": model_id,
                    "status": loaded.status,
                    "message": loaded.message,
                    "input_spec": spec.to_dict(),
                    "health": health,
                }
            )
            healthy_enough = health is None or health["responsive"]
            if loaded.is_ready and healthy_enough:
                ready_by_spec.setdefault(key, []).append(model_id)
        groups = []
        for key, members in ready_by_spec.items():
            groups.append(
                {
                    "channels": key[0],
                    "height": key[1],
                    "width": key[2],
                    "normalisation": key[3],
                    "members": members,
                    "ensemblable": len(members) > 1,
                }
            )
        reasons: list[str] = []
        ensemblable = [group for group in groups if group["ensemblable"]]
        if not ensemblable:
            ready = [entry["model_id"] for entry in statuses if entry["status"] == "ready"]
            missing = [entry["model_id"] for entry in statuses if entry["status"] == "missing"]
            unhealthy = [
                entry["model_id"]
                for entry in statuses
                if entry["status"] == "ready"
                and entry.get("health")
                and not entry["health"]["responsive"]
            ]
            if unhealthy:
                reasons.append(
                    "These checkpoints load but fail the reference health check, so they are "
                    "excluded from the ensemble: "
                    + ", ".join(f"{mid} ({self.health.check(mid).verdict})" for mid in unhealthy)
                    + ". Fusion with a non-responsive model would only dilute the partner model."
                )
            if len(ready) <= 1:
                reasons.append(
                    "Fewer than two segmentation checkpoints are usable; an ensemble needs at least two."
                )
            if missing:
                reasons.append(
                    "Training checkpoints are absent: " + ", ".join(missing) + ". "
                    "Run scripts/train_unet.py to produce an input-compatible partner for SegNet."
                )
            if len(groups) > 1 and all(len(group["members"]) == 1 for group in groups):
                reasons.append(
                    "The available models take different inputs ("
                    + "; ".join(
                        f"{group['members'][0]}: {group['channels']}x{group['height']}x{group['width']}"
                        for group in groups
                    )
                    + "). Fusing them would compare predictions made from different tensors, "
                    "so the ensemble is disabled rather than approximated."
                )
        return {
            "available": bool(ensemblable),
            "groups": groups,
            "model_status": statuses,
            "reasons": reasons,
        }
    def predict(
        self,
        member_predictions: dict[str, SegmentationResult],
        *,
        strategy: str = "equal",
        threshold: float = 0.5,
        spreads: dict[str, np.ndarray] | None = None,
        model_ids: list[str] | None = None,
    ) -> EnsembleResult:
        if strategy not in STRATEGIES:
            raise ValidationError(f"Unknown fusion strategy '{strategy}'.", details={"allowed": list(STRATEGIES)})
        members = list(member_predictions.keys()) if model_ids is None else list(model_ids)
        missing = [m for m in members if m not in member_predictions]
        if missing:
            raise ModelUnavailableError(
                "Cannot fuse: predictions are missing for " + ", ".join(missing),
                details={"missing": missing},
            )
        if len(members) < 2:
            return EnsembleResult(
                available=False,
                strategy=strategy,
                weights={},
                weight_source="none",
                members=members,
                threshold=threshold,
                reasons=[
                    "At least two segmentation models with identical input specifications "
                    "are required for an ensemble."
                ],
            )
        specs = []
        unresolved: list[str] = []
        for model_id in members:
            descriptor = self.registry.descriptors.get(model_id)
            if descriptor is None:
                unresolved.append(model_id)
                continue
            spec = descriptor.input_spec
            specs.append((spec.channels, spec.height, spec.width, spec.normalisation))
        shapes = {member_predictions[model_id].probability.shape for model_id in members}
        if len(shapes) != 1:
            return EnsembleResult(
                available=False,
                strategy=strategy,
                weights={},
                weight_source="none",
                members=members,
                threshold=threshold,
                reasons=[
                    "The member predictions have different resolutions ("
                    + ", ".join(str(shape) for shape in sorted(shapes))
                    + "); fusing them would compare maps that do not describe the same pixels."
                ],
            )
        if unresolved and specs and len(set(specs)) != 1:
            return EnsembleResult(
                available=False,
                strategy=strategy,
                weights={},
                weight_source="none",
                members=members,
                threshold=threshold,
                reasons=[
                    "The selected models do not share an input specification; "
                    "probability fusion is undefined for heterogeneous inputs."
                ],
            )
        weights, source, warnings = self._resolve_weights(
            members, strategy=strategy, spreads=spreads, member_predictions=member_predictions
        )
        stacked = np.stack(
            [member_predictions[model_id].probability for model_id in members], axis=0
        )
        weight_vector = np.array([weights[model_id] for model_id in members], dtype=np.float32)
        weight_vector = weight_vector / weight_vector.sum()
        if strategy == "confidence_aware" and spreads:
            probability = self._confidence_weighted(stacked, weight_vector, spreads, members)
        else:
            probability = np.tensordot(weight_vector, stacked, axes=1)
        probability = probability.astype(np.float32)
        mask = probability > threshold
        foreground = int(mask.sum())
        disagreement = self._disagreement(stacked, members, threshold)
        return EnsembleResult(
            available=True,
            strategy=strategy,
            weights=weights,
            weight_source=source,
            members=members,
            threshold=threshold,
            probability=probability,
            mask=mask,
            foreground_pixels=foreground,
            foreground_fraction=float(foreground / mask.size),
            member_results={model_id: member_predictions[model_id] for model_id in members},
            disagreement=disagreement,
            warnings=warnings,
        )
    def _confidence_weighted(
        self,
        stacked: np.ndarray,
        weight_vector: np.ndarray,
        spreads: dict[str, np.ndarray],
        members: list[str],
    ) -> np.ndarray:
        reliability = np.zeros_like(stacked)
        for index, model_id in enumerate(members):
            spread = spreads.get(model_id)
            if spread is None:
                reliability[index] = 1.0
                continue
            reliability[index] = 1.0 / (1.0 + spread.astype(np.float32) / (spread.mean() + 1e-6))
        combined = reliability * weight_vector[:, None, None]
        combined = combined / (combined.sum(axis=0, keepdims=True) + 1e-8)
        return (combined * stacked).sum(axis=0)
    def _resolve_weights(
        self,
        members: list[str],
        *,
        strategy: str,
        spreads: dict[str, np.ndarray] | None,
        member_predictions: dict[str, SegmentationResult],
    ) -> tuple[dict[str, float], str, list[str]]:
        warnings: list[str] = []
        if strategy == "equal":
            weights = {model_id: 1.0 / len(members) for model_id in members}
            return weights, "equal_weights_default", warnings
        calibration_path = Path(self.config.ENSEMBLE_CALIBRATION_PATH)
        calibration = read_json(calibration_path, default=None)
        if not calibration:
            weights = {model_id: 1.0 / len(members) for model_id in members}
            warnings.append(
                "No validation calibration file was found at "
                f"{calibration_path}; falling back to equal weights. Run "
                "scripts/calibrate_ensemble.py on the validation split to enable the "
                f"'{strategy}' strategy."
            )
            return weights, "equal_weights_default", warnings
        if str(calibration.get("split", "")).lower() == "test":
            raise ValidationError(
                "Refusing to use an ensemble calibration file computed on the test split. "
                "Fusion weights must be estimated on training/validation data only."
            )
        stored = {k: float(v) for k, v in (calibration.get("weights") or {}).items()}
        available = {model_id: stored.get(model_id) for model_id in members}
        if any(value is None for value in available.values()):
            weights = {model_id: 1.0 / len(members) for model_id in members}
            warnings.append(
                "The calibration file does not cover the current model set "
                f"({sorted(available)}); using equal weights."
            )
            return weights, "equal_weights_default", warnings
        total = sum(available.values()) or 1.0
        weights = {model_id: float(value) / total for model_id, value in available.items()}
        source = f"validation_calibration:{calibration_path.name}:{calibration.get('metric', 'dice')}"
        if strategy == "confidence_aware" and not spreads:
            warnings.append(
                "Confidence-aware fusion was requested but no predictive spread is available "
                "(enable test-time augmentation or Monte-Carlo dropout); the calibrated global "
                "weights are used instead."
            )
        return weights, source, warnings
    @staticmethod
    def _disagreement(stacked: np.ndarray, members: list[str], threshold: float) -> dict:
        mean_map = stacked.mean(axis=0)
        pairwise = {}
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                a = stacked[i] > threshold
                b = stacked[j] > threshold
                intersection = np.logical_and(a, b).sum()
                union = np.logical_or(a, b).sum()
                dice_agreement = (
                    float(2 * intersection / (a.sum() + b.sum())) if (a.sum() + b.sum()) else 1.0
                )
                pairwise[f"{members[i]}_vs_{members[j]}"] = {
                    "dice_agreement": dice_agreement,
                    "iou_agreement": float(intersection / union) if union else 1.0,
                    "only_in_first": int(np.logical_and(a, ~b).sum()),
                    "only_in_second": int(np.logical_and(~a, b).sum()),
                }
        return {
            "mean_pairwise_dice_agreement": float(
                np.mean([v["dice_agreement"] for v in pairwise.values()])
            )
            if pairwise
            else None,
            "probability_std_mean": float(stacked.std(axis=0).mean()),
            "probability_std_max": float(stacked.std(axis=0).max()),
            "pixels_with_majority_disagreement": int(
                ((mean_map > threshold) != (stacked > threshold).all(axis=0)).sum()
            ),
            "pairwise": pairwise,
            "note": (
                "These values measure agreement between models, not accuracy against a "
                "reference segmentation. Accuracy requires labelled data."
            ),
        }
