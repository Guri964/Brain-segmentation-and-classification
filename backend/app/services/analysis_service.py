from __future__ import annotations
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable
import numpy as np
from ..errors import ModelUnavailableError, ValidationError
from ..utils.images import (
    encode_png,
    make_heatmap_overlay,
    make_overlay,
    resize_array,
    to_uint8,
)
from ..utils.io import ensure_within, new_analysis_id, safe_filename, utc_now, write_bytes_atomic, write_json
from ..utils.metrics import agreement_metrics
logger = logging.getLogger(__name__)
@dataclass
class AnalysisRequest:
    files: list[tuple[str, bytes]]
    analysis_type: str = "four_channel"
    model_ids: list[str] | None = None
    scan_identifier: str | None = None
    notes: str | None = None
    threshold: float = 0.5
    enable_ensemble: bool = True
    ensemble_strategy: str = "equal"
    enable_uncertainty: bool = False
    enable_explainability: bool = True
    enable_radiomics: bool = True
    radiomics_mask: str = "best"
    modality_names: list[str] | None = None
    spacing: tuple[float, ...] | None = None
class AnalysisService:
    def __init__(
        self,
        config,
        repository,
        registry,
        preprocessing,
        segmentation,
        ensemble,
        radiomics,
        explainability,
        uncertainty,
        classification,
        health,
        report=None,
    ):
        self.config = config
        self.repository = repository
        self.registry = registry
        self.preprocessing = preprocessing
        self.segmentation = segmentation
        self.ensemble = ensemble
        self.radiomics = radiomics
        self.explainability = explainability
        self.uncertainty = uncertainty
        self.classification = classification
        self.health = health
        self.report = report
    @property
    def upload_dir(self) -> Path:
        return Path(self.config.DATA_DIR) / self.config.UPLOAD_DIR_NAME
    @property
    def artifact_dir(self) -> Path:
        return Path(self.config.DATA_DIR) / self.config.ARTIFACT_DIR_NAME
    def payload_path(self, analysis_id: str) -> Path:
        return Path(self.config.DATA_DIR) / self.config.ARTIFACT_DIR_NAME / analysis_id / "analysis.json"
    def load_payload(self, analysis_id: str) -> dict | None:
        from ..utils.io import read_json
        return read_json(self.payload_path(analysis_id), default=None)
    def summary_from_database(self, analysis_id: str) -> dict | None:
        full = self.repository.get_full_analysis(analysis_id)
        if full is None:
            return None
        models = {}
        for row in full.get("segmentation") or []:
            models[row["model_id"]] = {
                "available": True,
                "status": "completed",
                "summary": {
                    "model_id": row["model_id"],
                    "threshold": row["threshold"],
                    "foreground_pixels": row["foreground_pixels"],
                    "foreground_fraction": row["foreground_fraction"],
                    "probability_mean": row.get("probability_mean"),
                    "probability_max": row.get("probability_max"),
                    "inference_seconds": row.get("inference_seconds"),
                },
                "artifacts": row.get("artifact_paths") or {},
                "health": {"verdict": "not_checked", "message": "Reconstructed from the database."},
                "descriptor": self.registry.load_model(row["model_id"]).to_dict(),
            }
        radiomics_rows = full.get("radiomics") or []
        radiomics = {"available": False, "unavailable_reason": "No radiomics record was stored."}
        if radiomics_rows:
            last = radiomics_rows[-1]
            radiomics = {
                "available": True,
                "backend": last["backend"],
                "dimensionality": last["dimensionality"],
                "feature_count": last["feature_count"],
                "features": last["features"],
                "categories": last.get("categories") or {},
                "quality": last.get("quality") or {},
                "warnings": last.get("warnings") or [],
                "spacing_available": bool(last.get("spacing_available")),
                "mask_source": last.get("mask_source"),
            }
        return {
            "analysis_id": analysis_id,
            "scan_identifier": full.get("scan_identifier"),
            "analysis_type": full.get("analysis_type"),
            "created_at": full.get("created_at"),
            "device": full.get("device"),
            "threshold": self.config.SEGMENTATION_THRESHOLD,
            "models": models,
            "ensemble": {"available": False, "reasons": ["The stored ensemble record was not available."]},
            "radiomics": radiomics,
            "classification": {
                "available": bool((full.get("classification") or {}).get("available")),
                "predicted_label": (full.get("classification") or {}).get("predicted_label"),
                "scores": (full.get("classification") or {}).get("scores") or [],
                "reason": (full.get("classification") or {}).get("reason"),
                "model_id": (full.get("classification") or {}).get("model_id"),
                "metrics": (full.get("classification") or {}).get("metrics") or {},
                "feature_contributions": (full.get("classification") or {}).get("contributions") or [],
            },
            "explainability": [
                {
                    "model_id": row["model_id"],
                    "available": True,
                    "method": row["method"],
                    "target_layer": row["target_layer"],
                    "target_score": row["target_score"],
                    "artifacts": {"heatmap": row.get("heatmap_path"), "overlay": row.get("overlay_path")},
                    "warnings": row.get("warnings") or [],
                }
                for row in full.get("explainability") or []
            ],
            "uncertainty": (full.get("uncertainty") or {}).get("payload"),
            "reports": full.get("reports") or [],
            "reconstructed": True,
        }
    def resolve_payload(self, analysis_id: str) -> dict | None:
        return self.load_payload(analysis_id) or self.summary_from_database(analysis_id)
    def plan(self, request: AnalysisRequest) -> dict:
        model_ids = request.model_ids or self._default_models(request.analysis_type)
        steps = []
        for model_id in model_ids:
            loaded = self.registry.load_model(model_id)
            report = self.health.check(model_id)
            steps.append(
                {
                    "step": "segmentation",
                    "model_id": model_id,
                    "status": "ready" if loaded.is_ready else loaded.status,
                    "runnable": loaded.is_ready,
                    "health": report.verdict,
                    "message": loaded.message if not loaded.is_ready else report.message,
                    "input_spec": loaded.descriptor.input_spec.to_dict(),
                }
            )
        compatibility = self.ensemble.compatibility(model_ids) if request.enable_ensemble else None
        return {
            "analysis_type": request.analysis_type,
            "requested_models": model_ids,
            "steps": steps,
            "ensemble": compatibility,
            "radiomics": {
                "runnable": request.enable_radiomics,
                "backend": self.radiomics.backend_status(),
                "note": (
                    "Radiomics requires a non-empty mask. If every model returns an empty mask, "
                    "no region-based feature can be computed and the analysis will say so."
                ),
            },
            "uncertainty": {
                "runnable": request.enable_uncertainty,
                "note": (
                    "Test-time augmentation works for any model. Monte-Carlo dropout requires an "
                    "architecture with dropout layers; the reused checkpoints have none."
                ),
            },
            "explainability": {
                "runnable": request.enable_explainability,
                "targets": [
                    {
                        "model_id": model_id,
                        "target_layer": self.registry.load_model(model_id).descriptor.model_id
                        and self.explainability._target_layer_name(self.registry.load_model(model_id)),
                    }
                    for model_id in model_ids
                ],
            },
            "classification": self.classification.status(),
        }
    def _default_models(self, analysis_type: str) -> list[str]:
        if analysis_type == "single_channel":
            return ["unet_dynamic_512"]
        ready = [
            model_id
            for model_id in self.registry.segmentation_model_ids()
            if self.registry.load_model(model_id).is_ready
        ]
        four_channel = [
            model_id
            for model_id in ready
            if self.registry.load_model(model_id).descriptor.input_spec.channels == 4
        ]
        return four_channel or ready
    def run(
        self,
        request: AnalysisRequest,
        *,
        progress: Callable[[str, str, int], None] | None = None,
    ) -> dict:
        analysis_id = new_analysis_id()
        scan_identifier = request.scan_identifier or analysis_id.split("-")[-1].upper()
        def emit(stage: str, message: str, percent: int) -> None:
            if progress is not None:
                try:
                    progress(stage, message, percent)
                except Exception:
                    logger.debug("progress callback failed", exc_info=True)
        model_ids = request.model_ids or self._default_models(request.analysis_type)
        if not model_ids:
            raise ModelUnavailableError(
                "No segmentation checkpoint is available for this analysis type."
            )
        stored_uploads = self._store_uploads(analysis_id, request.files)
        self.repository.create_analysis(
            analysis_id=analysis_id,
            scan_identifier=scan_identifier,
            analysis_type=request.analysis_type,
            model_ids=model_ids,
            input_manifest={"uploads": stored_uploads},
            preprocessing={"analysis_type": request.analysis_type},
            device=str(self.registry.device),
            notes=request.notes,
            status="running",
        )
        try:
            result = self._execute(analysis_id, scan_identifier, request, model_ids, stored_uploads, emit)
        except Exception as exc:
            logger.exception("Analysis %s failed", analysis_id)
            self.repository.update_status(
                analysis_id, "failed", error_message=f"{type(exc).__name__}: {exc}", completed=True
            )
            raise
        self.repository.update_status(analysis_id, "completed", completed=True)
        result["analysis_id"] = analysis_id
        result["status"] = "completed"
        return result
    def _execute(
        self,
        analysis_id: str,
        scan_identifier: str,
        request: AnalysisRequest,
        model_ids: list[str],
        stored_uploads: list[dict],
        emit: Callable[[str, str, int], None],
    ) -> dict:
        artifact_root = ensure_within(self.artifact_dir, Path(analysis_id))
        artifact_root.mkdir(parents=True, exist_ok=True)
        emit("preprocessing", "Validating and normalising the input", 5)
        descriptors = [self.registry.load_model(model_id).descriptor for model_id in model_ids]
        primary_spec = descriptors[0].input_spec
        payloads = [(entry["filename"], entry["_bytes"]) for entry in stored_uploads]
        if primary_spec.channels == 1:
            if len(payloads) != 1:
                raise ValidationError(
                    "The single-channel pipeline expects exactly one image; received "
                    f"{len(payloads)}."
                )
            preprocessed = self.preprocessing.preprocess_single_channel(
                payloads[0][1], payloads[0][0], size=primary_spec.height
            )
        else:
            preprocessed = self.preprocessing.preprocess_four_channel(
                payloads, size=primary_spec.height, modality_names=request.modality_names
            )
        preprocessing_artifacts = {}
        for index, (label, array) in enumerate(preprocessed.display_stack):
            path = artifact_root / f"input_{index}_{safe_filename(label.lower())}.png"
            write_bytes_atomic(path, encode_png(to_uint8(array)))
            preprocessing_artifacts[label] = path.name
        display_reference = preprocessed.display_stack[0][1]
        emit("segmentation", "Running segmentation models", 25)
        model_payloads: dict[str, dict] = {}
        probabilities: dict[str, np.ndarray] = {}
        spreads: dict[str, np.ndarray] = {}
        member_results = {}
        for model_id in model_ids:
            loaded = self.registry.load_model(model_id)
            if not loaded.is_ready:
                model_payloads[model_id] = {
                    "model_id": model_id,
                    "available": False,
                    "status": loaded.status,
                    "message": loaded.message,
                    "descriptor": loaded.to_dict(),
                    "is_reference_validated": False,
                }
                continue
            segmentation = self.segmentation.predict(
                model_id, preprocessed, threshold=request.threshold
            )
            member_results[model_id] = segmentation
            probabilities[model_id] = segmentation.probability
            artefacts = self._write_segmentation_artifacts(
                artifact_root, model_id, display_reference, segmentation
            )
            health = self.health.check(model_id)
            model_payloads[model_id] = {
                "available": True,
                "status": "completed",
                "descriptor": loaded.to_dict(),
                "summary": segmentation.summary(),
                "artifacts": artefacts,
                "health": {
                    "verdict": health.verdict,
                    "message": health.message,
                    "checks": health.checks,
                },
                "is_reference_validated": health.responsive,
            }
            self.repository.add_segmentation_result(
                analysis_id,
                {
                    "model_id": model_id,
                    "display_name": segmentation.model_display_name,
                    "threshold": segmentation.threshold,
                    "foreground_pixels": segmentation.foreground_pixels,
                    "image_pixels": int(segmentation.mask.size),
                    "foreground_fraction": segmentation.foreground_fraction,
                    "probability_mean": float(segmentation.probability.mean()),
                    "probability_max": float(segmentation.probability.max()),
                    "inference_seconds": segmentation.inference_seconds,
                    "tta_used": segmentation.tta_used,
                    "mc_dropout_used": segmentation.mc_dropout_used,
                    "metrics": {},
                    "artifact_paths": artefacts,
                    "warnings": segmentation.warnings,
                },
            )
        emit("ensemble", "Fusing model probabilities", 45)
        ensemble_outcome = self._run_ensemble(
            request, model_ids, member_results, artifact_root, display_reference, preprocessed.tensor
        )
        uncertainty_payload = None
        if request.enable_uncertainty and member_results:
            emit("uncertainty", "Estimating predictive uncertainty", 55)
            uncertainty_payload = self._run_uncertainty(
                request, model_ids, member_results, preprocessed, spreads
            )
            if uncertainty_payload is not None:
                self.repository.add_uncertainty(
                    analysis_id,
                    {
                        "method": uncertainty_payload["method"],
                        "n_passes": uncertainty_payload.get("n_passes", 0),
                        "mean_dispersion": uncertainty_payload.get("mean_dispersion"),
                        "max_dispersion": uncertainty_payload.get("max_dispersion"),
                        "high_uncertainty_fraction": uncertainty_payload.get("high_uncertainty_fraction"),
                        "payload": uncertainty_payload,
                    },
                )
                if uncertainty_payload.get("artifact"):
                    uncertainty_payload["artifact_path"] = uncertainty_payload.pop("artifact")
        reference_mask, mask_source, reference_model = self._select_reference_mask(
            request, member_results, ensemble_outcome
        )
        radiomics_payload = None
        if request.enable_radiomics:
            emit("radiomics", "Extracting radiomics features", 65)
            radiomics_payload = self._run_radiomics(
                analysis_id, artifact_root, display_reference, reference_mask, mask_source, request
            )
        explainability_payload = []
        if request.enable_explainability:
            emit("explainability", "Generating Grad-CAM explanations", 75)
            explainability_payload = self._run_explainability(
                analysis_id, artifact_root, preprocessed.tensor, display_reference, member_results
            )
        emit("classification", "Checking classification availability", 85)
        if radiomics_payload and radiomics_payload.get("available") and request.enable_radiomics:
            outcome = self.classification.classify(radiomics_payload["features"])
        else:
            outcome = self.classification.classify({})
        classification_payload = outcome.to_dict()
        self.repository.add_classification(
            analysis_id,
            {
                "available": outcome.available,
                "model_id": outcome.model_id,
                "model_version": outcome.model_version,
                "predicted_label": outcome.predicted_label,
                "scores": outcome.scores,
                "calibrated": outcome.calibrated,
                "metrics": outcome.metrics or {},
                "contributions": outcome.feature_contributions,
                "reason": outcome.reason,
            },
        )
        comparison = self._comparison_payload(member_results, ensemble_outcome, reference_mask)
        manifest = {
            "analysis_id": analysis_id,
            "scan_identifier": scan_identifier,
            "created_at": utc_now(),
            "analysis_type": request.analysis_type,
            "device": str(self.registry.device),
            "preprocessing": preprocessed.summary(),
            "uploads": [{k: v for k, v in entry.items() if k != "_bytes"} for entry in stored_uploads],
            "models": {
                model_id: self.registry.load_model(model_id).to_dict() for model_id in model_ids
            },
            "threshold": request.threshold,
            "ensemble_strategy": request.ensemble_strategy,
            "phases": {
                "config": self.config.as_dict() if hasattr(self.config, "as_dict") else {},
            },
        }
        write_json(artifact_root / "manifest.json", manifest)
        payload = {
            "analysis_id": analysis_id,
            "scan_identifier": scan_identifier,
            "analysis_type": request.analysis_type,
            "created_at": manifest["created_at"],
            "device": manifest["device"],
            "threshold": request.threshold,
            "preprocessing": preprocessed.summary(),
            "inputs": preprocessing_artifacts,
            "models": model_payloads,
            "ensemble": ensemble_outcome,
            "uncertainty": uncertainty_payload,
            "reference_mask": {
                "source": mask_source,
                "model_id": reference_model,
                "foreground_pixels": int(reference_mask.sum()) if reference_mask is not None else 0,
            },
            "radiomics": radiomics_payload,
            "explainability": explainability_payload,
            "classification": classification_payload,
            "model_comparison": comparison,
            "artifacts_root": str(artifact_root.relative_to(Path(self.config.DATA_DIR))),
            "disclaimer": self.config.DISCLAIMER,
        }
        if self.report is not None:
            emit("report", "Rendering the PDF report", 92)
            try:
                report_info = self.report.generate(payload, artifact_root, display_reference)
                self.repository.add_report(
                    analysis_id,
                    {
                        "report_id": report_info["report_id"],
                        "path": report_info["relative_path"],
                        "size_bytes": report_info["size_bytes"],
                        "generator_version": report_info["generator_version"],
                    },
                )
                payload["report"] = report_info
            except Exception as exc:
                logger.exception("Report generation failed for %s", analysis_id)
                payload["report"] = {
                    "available": False,
                    "reason": f"Report generation failed: {type(exc).__name__}: {exc}",
                }
        emit("complete", "Analysis complete", 100)
        write_json(artifact_root / "analysis.json", payload)
        return payload
    def _store_uploads(self, analysis_id: str, files: list[tuple[str, bytes]]) -> list[dict]:
        if not files:
            raise ValidationError("No file was uploaded.")
        if len(files) > int(self.config.MAX_FILES_PER_ANALYSIS):
            raise ValidationError(
                f"At most {self.config.MAX_FILES_PER_ANALYSIS} files are accepted per analysis."
            )
        directory = ensure_within(self.upload_dir, Path(analysis_id))
        directory.mkdir(parents=True, exist_ok=True)
        stored = []
        total = 0
        for filename, payload in files:
            total += len(payload)
            if total > int(self.config.MAX_CONTENT_LENGTH):
                raise ValidationError("The upload exceeds the configured size limit.")
            safe = safe_filename(filename)
            path = directory / safe
            write_bytes_atomic(path, payload)
            stored.append(
                {
                    "filename": safe,
                    "original_filename": filename,
                    "size_bytes": len(payload),
                    "path": str(path.relative_to(Path(self.config.DATA_DIR))),
                    "_bytes": payload,
                }
            )
        return stored
    def _write_segmentation_artifacts(
        self, root: Path, model_id: str, display_reference: np.ndarray, segmentation
    ) -> dict:
        key = safe_filename(model_id)
        probability_u8 = to_uint8(segmentation.probability)
        mask_u8 = (segmentation.mask.astype(np.uint8) * 255)
        overlay = make_overlay(display_reference, segmentation.mask)
        probability_vis = resize_array(segmentation.probability, display_reference.shape)
        heat = np.repeat(to_uint8(probability_vis)[..., None], 3, axis=2)
        heat[:, :, 0] = to_uint8(probability_vis)
        paths = {
            "probability": root / f"prob_{key}.png",
            "mask": root / f"mask_{key}.png",
            "overlay": root / f"overlay_{key}.png",
        }
        write_json(root / f"segmentation_{key}.json", segmentation.summary())
        write_bytes_atomic(paths["probability"], encode_png(heat))
        write_bytes_atomic(paths["mask"], encode_png(mask_u8))
        write_bytes_atomic(paths["overlay"], encode_png(overlay))
        return {name: path.name for name, path in paths.items()}
    def _run_ensemble(
        self,
        request: AnalysisRequest,
        model_ids: list[str],
        member_results: dict,
        artifact_root: Path,
        display_reference: np.ndarray,
        tensor,
    ) -> dict:
        compatibility = self.ensemble.compatibility(model_ids)
        if not request.enable_ensemble:
            return {"available": False, "strategy": request.ensemble_strategy, "reasons": ["The ensemble was disabled for this analysis."]}
        if not compatibility["available"]:
            return {
                "available": False,
                "strategy": request.ensemble_strategy,
                "reasons": compatibility["reasons"],
                "model_status": compatibility["model_status"],
                "weights": {},
                "members": [],
            }
        eligible = [
            group["members"]
            for group in compatibility["groups"]
            if group["ensemblable"]
        ]
        members = max(eligible, key=len) if eligible else []
        predictions = {model_id: member_results[model_id] for model_id in members if model_id in member_results}
        if len(predictions) < 2:
            return {
                "available": False,
                "strategy": request.ensemble_strategy,
                "reasons": ["Fewer than two models produced a prediction for this input."],
                "weights": {},
                "members": list(predictions),
            }
        spreads = {}
        if request.enable_uncertainty:
            for model_id in predictions:
                loaded = self.registry.load_model(model_id)
                if loaded.module is None:
                    continue
                try:
                    passes = self.segmentation._tta_passes(loaded.module, tensor, loaded.device)
                    spreads[model_id] = np.stack(
                        passes + [member_results[model_id].probability], axis=0
                    ).std(axis=0).astype(np.float32)
                except Exception:
                    logger.debug("TTA spread unavailable for %s", model_id, exc_info=True)
        outcome = self.ensemble.predict(
            predictions,
            strategy=request.ensemble_strategy,
            threshold=request.threshold,
            spreads=spreads or None,
            model_ids=list(predictions),
        )
        payload = outcome.summary()
        if outcome.available and outcome.mask is not None:
            overlay = make_overlay(display_reference, outcome.mask)
            mask_u8 = outcome.mask.astype(np.uint8) * 255
            write_bytes_atomic(artifact_root / "ensemble_mask.png", encode_png(mask_u8))
            write_bytes_atomic(artifact_root / "ensemble_overlay.png", encode_png(overlay))
            payload["artifacts"] = {
                "mask": "ensemble_mask.png",
                "overlay": "ensemble_overlay.png",
            }
            probability = to_uint8(outcome.probability)
            write_bytes_atomic(
                artifact_root / "ensemble_probability.png",
                encode_png(np.repeat(probability[..., None], 3, axis=2)),
            )
            payload["artifacts"]["probability"] = "ensemble_probability.png"
        return payload
    def _run_uncertainty(
        self,
        request: AnalysisRequest,
        model_ids: list[str],
        member_results: dict,
        preprocessed,
        spreads: dict,
    ) -> dict | None:
        eligible = [model_id for model_id in model_ids if model_id in member_results]
        if not eligible:
            return None
        primary = eligible[0]
        loaded = self.registry.load_model(primary)
        module = loaded.module
        if module is None:
            return None
        passes = []
        if loaded.descriptor.supports_mc_dropout:
            predictions = self.segmentation._mc_dropout_passes(
                module, preprocessed.tensor, loaded.device, int(self.config.MC_DROPOUT_SAMPLES)
            )
            method = "monte_carlo_dropout"
            passes.extend(predictions)
        else:
            passes.extend(self.segmentation._tta_passes(module, preprocessed.tensor, loaded.device))
            method = "test_time_augmentation"
        passes.append(member_results[primary].probability)
        outcome = self.uncertainty.from_probability_stack(
            passes, method=method, threshold=request.threshold
        )
        payload = outcome.summary()
        payload["model_id"] = primary
        if outcome.per_pixel_map is not None:
            artifact = Path(self.config.DATA_DIR) / self.config.ARTIFACT_DIR_NAME / "uncertainty_map.png"
            write_bytes_atomic(
                artifact,
                encode_png(np.repeat(to_uint8(outcome.per_pixel_map)[..., None], 3, axis=2)),
            )
            payload["artifact"] = str(artifact.relative_to(Path(self.config.DATA_DIR)))
        return payload
    def _select_reference_mask(self, request, member_results, ensemble_outcome):
        if request.radiomics_mask == "ensemble" and ensemble_outcome.get("available") and ensemble_outcome.get("mask") is not None:
            return ensemble_outcome["mask"], "ensemble", "+".join(ensemble_outcome.get("members", []))
        if request.radiomics_mask in {"best", "primary"} and member_results:
            preferred_order = [
                model_id
                for model_id in member_results
                if self.health.check(model_id).responsive
            ] or list(member_results)
            for model_id in preferred_order:
                if request.radiomics_mask == "primary":
                    break
                if member_results[model_id].foreground_pixels > 0:
                    return member_results[model_id].mask, f"model:{model_id}", model_id
            first = preferred_order[0]
            return member_results[first].mask, f"model:{first}", first
        return None, "none", None
    def _run_radiomics(
        self,
        analysis_id: str,
        artifact_root: Path,
        display_reference: np.ndarray,
        reference_mask: np.ndarray | None,
        mask_source: str,
        request: AnalysisRequest,
    ) -> dict:
        if reference_mask is None:
            return {
                "available": False,
                "unavailable_reason": (
                    "No segmentation mask was produced, so no region of interest exists for the "
                    "radiomics extraction."
                ),
            }
        image = np.asarray(display_reference, dtype=np.float32)
        mask = np.asarray(reference_mask)
        if mask.shape != image.shape:
            mask = resize_array(mask.astype(np.float32), image.shape, nearest=True) > 0.5
        outcome = self.radiomics.extract(
            image, mask, spacing=request.spacing, mask_source=mask_source
        )
        payload = outcome.to_dict()
        if outcome.available:
            csv_path = artifact_root / "radiomics_features.csv"
            _write_feature_csv(csv_path, outcome.features)
            json_path = artifact_root / "radiomics_features.json"
            write_json(json_path, outcome.to_dict())
            payload["artifacts"] = {"csv": csv_path.name, "json": json_path.name}
            self.repository.add_radiomics(
                analysis_id,
                {
                    "mask_source": mask_source,
                    "backend": outcome.backend,
                    "dimensionality": outcome.dimensionality,
                    "feature_count": outcome.feature_count,
                    "spacing_available": outcome.spacing_available,
                    "features": outcome.features,
                    "categories": outcome.categories,
                    "quality": outcome.quality,
                    "warnings": outcome.warnings,
                    "artifact_path": str(csv_path.relative_to(Path(self.config.DATA_DIR))),
                },
            )
        return payload
    def _run_explainability(
        self,
        analysis_id: str,
        artifact_root: Path,
        tensor,
        display_reference: np.ndarray,
        member_results: dict,
    ) -> list[dict]:
        entries = []
        for model_id, segmentation in member_results.items():
            try:
                payload = self.explainability.explain_segmentation(
                    model_id,
                    tensor,
                    display_image=display_reference,
                    predicted_mask=segmentation.mask,
                    include_activation_map=False,
                )
            except Exception as exc:
                logger.warning("Explainability failed for %s: %s", model_id, exc)
                entries.append(
                    {
                        "model_id": model_id,
                        "available": False,
                        "reason": f"{type(exc).__name__}: {exc}",
                    }
                )
                continue
            key = safe_filename(model_id)
            heat_path = artifact_root / f"gradcam_{key}.png"
            overlay_path = artifact_root / f"gradcam_overlay_{key}.png"
            write_bytes_atomic(heat_path, encode_png(payload["images"]["heatmap_rgb"]))
            write_bytes_atomic(
                overlay_path,
                encode_png(
                    make_heatmap_overlay(display_reference, payload["images"]["heatmap_gray"], alpha=0.5)
                ),
            )
            entry = payload.copy()
            entry.pop("images", None)
            entry.update(
                {
                    "available": True,
                    "artifacts": {"heatmap": heat_path.name, "overlay": overlay_path.name},
                }
            )
            entries.append(entry)
            self.repository.add_explainability(
                analysis_id,
                {
                    "model_id": model_id,
                    "method": entry["method"],
                    "target_layer": entry["target_layer"],
                    "target_score": entry["target_score"],
                    "heatmap_path": heat_path.name,
                    "overlay_path": overlay_path.name,
                    "metadata": {
                        "activation_shape": entry["activation_shape"],
                        "heatmap_mean": entry["heatmap_mean"],
                        "heatmap_max": entry["heatmap_max"],
                    },
                    "warnings": entry["warnings"],
                },
            )
        return entries
    def _comparison_payload(self, member_results, ensemble_outcome, reference_mask) -> dict:
        available = {model_id: result for model_id, result in member_results.items()}
        payload: dict[str, Any] = {
            "state": "unlabelled",
            "note": (
                "This scan has no reference segmentation, so no accuracy metric can be computed. "
                "The values below describe the models' outputs and how much the models agree with "
                "each other - not whether they are correct."
            ),
            "models": [
                {
                    "model_id": model_id,
                    "foreground_pixels": result.foreground_pixels,
                    "foreground_fraction": result.foreground_fraction,
                    "probability_mean": float(result.probability.mean()),
                    "probability_max": float(result.probability.max()),
                    "inference_seconds": result.inference_seconds,
                    "health_verdict": self.health.check(model_id).verdict,
                }
                for model_id, result in available.items()
            ],
        }
        if len(available) >= 2:
            pairs = {}
            keys = list(available)
            for i in range(len(keys)):
                for j in range(i + 1, len(keys)):
                    a = available[keys[i]].resized(available[keys[j]].mask.shape)
                    pairs[f"{keys[i]}_vs_{keys[j]}"] = agreement_metrics(a.mask, available[keys[j]].mask)
            payload["pairwise_agreement"] = pairs
        if ensemble_outcome.get("available"):
            payload["ensemble"] = {
                "weights": ensemble_outcome.get("weights"),
                "weight_source": ensemble_outcome.get("weight_source"),
                "disagreement": ensemble_outcome.get("disagreement"),
            }
        if reference_mask is not None:
            if ensemble_outcome.get("available") and ensemble_outcome.get("mask") is not None:
                payload["mask_agreement_with_ensemble"] = {
                    model_id: agreement_metrics(
                        result.resized(ensemble_outcome["mask"].shape).mask, ensemble_outcome["mask"]
                    )
                    for model_id, result in available.items()
                }
        return payload
def _write_feature_csv(path: Path, features: dict) -> None:
    import csv
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["feature", "value"])
        for name, value in features.items():
            writer.writerow([name, "" if value is None else value])
