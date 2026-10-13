from __future__ import annotations
import logging
import os
from pathlib import Path
from flask import Flask, jsonify
from .config import BaseConfig, get_config
from .database.repository import AnalysisRepository
from .errors import register_error_handlers
from .logging_config import configure_logging
from .ml.registry import ModelRegistry, resolve_device
from .routes import register_blueprints
from .services.analysis_service import AnalysisService
from .services.assistant_service import AssistantService
from .services.classification_service import ClassificationService
from .services.ensemble_service import EnsembleService
from .services.evaluation_service import EvaluationService
from .services.image_classification_service import ImageClassificationService
from .services.explainability_service import ExplainabilityService
from .services.model_health_service import ModelHealthService
from .services.preprocessing_service import (
    build_input_for_model,
    preprocess_four_channel,
    preprocess_single_channel,
)
from .services.radiomics_service import RadiomicsService
from .services.report_service import ReportService
from .services.segmentation_service import SegmentationService
from .services.uncertainty_service import UncertaintyService
from .utils.io import utc_now
logger = logging.getLogger(__name__)
__version__ = "2.0.0"
class _PreprocessingFacade:
    preprocess_single_channel = staticmethod(preprocess_single_channel)
    preprocess_four_channel = staticmethod(preprocess_four_channel)
    build_input_for_model = staticmethod(build_input_for_model)
def create_app(config_object=None, *, env: str | None = None) -> Flask:
    config = config_object or get_config(env)
    config.init_directories()
    configure_logging(config.LOG_LEVEL, Path(config.LOG_DIR))
    app = Flask(__name__, static_folder=None)
    app.config["BRAIN_ENV"] = config.ENV_NAME
    app.config["BRAIN_DATA_DIR"] = str(config.DATA_DIR)
    app.config["BRAIN_UPLOAD_DIR"] = config.UPLOAD_DIR_NAME
    app.config["BRAIN_ARTIFACT_DIR"] = str(Path(config.DATA_DIR) / config.ARTIFACT_DIR_NAME)
    app.config["BRAIN_REPORT_DIR"] = config.REPORT_DIR_NAME
    app.config["BRAIN_MAX_UPLOAD_BYTES"] = int(config.MAX_CONTENT_LENGTH)
    app.config["BRAIN_MAX_FILES_PER_ANALYSIS"] = int(config.MAX_FILES_PER_ANALYSIS)
    app.config["BRAIN_SEG_THRESHOLD"] = float(config.SEGMENTATION_THRESHOLD)
    app.config["BRAIN_RADIOMICS_DISCRETIZATION"] = config.RADIOMICS_DISCRETIZATION
    app.config["BRAIN_RADIOMICS_BIN_COUNT"] = int(config.RADIOMICS_BIN_COUNT)
    app.config["BRAIN_RADIOMICS_BIN_WIDTH"] = float(config.RADIOMICS_BIN_WIDTH)
    app.config["BRAIN_DISCLAIMER"] = config.DISCLAIMER
    app.config["BRAIN_CONFIG_SUMMARY"] = {
        "env": config.ENV_NAME,
        "device_preference": config.DEVICE_PREFERENCE,
        "torch_threads": config.TORCH_THREADS,
        "model_cache_enabled": config.MODEL_CACHE_ENABLED,
        "max_upload_bytes": config.MAX_CONTENT_LENGTH,
        "radiomics_backend": config.RADIOMICS_BACKEND,
        "assistant_llm_enabled": config.ASSISTANT_LLM_ENABLED,
    }
    app.config["MAX_CONTENT_LENGTH"] = int(config.MAX_CONTENT_LENGTH)
    app.config["JSON_SORT_KEYS"] = False
    app.config["TESTING"] = config.TESTING
    device = resolve_device(getattr(config, "DEVICE_PREFERENCE", "auto"))
    registry = ModelRegistry(config, device=device)
    repository = AnalysisRepository(config.database_path())
    migration = repository.initialise()
    logger.info(
        "Database ready at %s (schema version %s, migrations applied: %s)",
        migration["database_path"],
        migration["schema_version"],
        migration["applied"] or "none",
    )
    preprocessing = _PreprocessingFacade()
    segmentation = SegmentationService(registry, config)
    health_service = ModelHealthService(registry, config)
    ensemble = EnsembleService(registry, config, segmentation, health=health_service)
    radiomics = RadiomicsService(config)
    explainability = ExplainabilityService(registry, config)
    uncertainty = UncertaintyService(config)
    classification = ClassificationService(config, registry=registry)
    image_classification = ImageClassificationService(config)
    evaluation = EvaluationService(registry, config, segmentation=segmentation, ensemble=ensemble)
    report = ReportService(config, repository=repository)
    assistant = AssistantService(config, repository=repository)
    analysis = AnalysisService(
        config=config,
        repository=repository,
        registry=registry,
        preprocessing=preprocessing,
        segmentation=segmentation,
        ensemble=ensemble,
        radiomics=radiomics,
        explainability=explainability,
        uncertainty=uncertainty,
        classification=classification,
        health=health_service,
        report=report,
    )
    app.extensions["brain_services"] = {
        "config": config,
        "registry": registry,
        "repository": repository,
        "preprocessing": preprocessing,
        "segmentation": segmentation,
        "health": health_service,
        "ensemble": ensemble,
        "radiomics": radiomics,
        "explainability": explainability,
        "uncertainty": uncertainty,
        "classification": classification,
        "image_classification": image_classification,
        "evaluation": evaluation,
        "report": report,
        "assistant": assistant,
        "analysis": analysis,
        "clock": utc_now,
    }
    register_error_handlers(app)
    register_blueprints(app)
    _register_cors(app, config)
    _register_frontend_hosting(app, config)
    @app.get("/")
    def index():
        return jsonify(
            {
                "service": "Brain Abnormality Prediction API",
                "version": __version__,
                "environment": config.ENV_NAME,
                "documentation": "docs/API.md",
                "frontend": {
                    "development": "http://localhost:5173 (Vite dev server, proxies /api)",
                    "production": "built assets are served from this origin at /",
                },
                "endpoints": [
                    "/api/health",
                    "/api/system",
                    "/api/capabilities",
                    "/api/analysis-options",
                    "/api/analysis",
                    "/api/models",
                    "/api/history",
                    "/api/reports",
                    "/api/evaluation",
                    "/api/radiomics/catalogue",
                    "/api/assistant/status",
                ],
                "disclaimer": config.DISCLAIMER,
            }
        )
    if getattr(config, "MODEL_CACHE_ENABLED", True) and not config.TESTING:
        try:
            loaded = registry.preload()
            logger.info(
                "Model cache warm: %s",
                ", ".join(f"{mid}={m.status}" for mid, m in loaded.items()),
            )
        except Exception:
            logger.exception("Model preloading failed; models will be loaded on demand")
    return app
def _register_cors(app: Flask, config) -> None:
    allowed = {origin.strip() for origin in config.CORS_ALLOWED_ORIGINS if origin.strip()}
    @app.after_request
    def _add_cors_headers(response):
        origin = request_origin()
        if origin and origin in allowed:
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Vary"] = "Origin"
            response.headers["Access-Control-Allow-Headers"] = "Content-Type, Accept"
            response.headers["Access-Control-Allow-Methods"] = "GET, POST, DELETE, OPTIONS"
        return response
    @app.route("/api/<path:_path>", methods=["OPTIONS"])
    def _preflight(_path):
        from flask import Response
        return Response(status=204)
def request_origin() -> str | None:
    from flask import request
    return request.headers.get("Origin")
def _register_frontend_hosting(app: Flask, config) -> None:
    dist_candidates = [
        Path(config.__class__.__module__ and Path(__file__).resolve().parent.parent.parent)
        / "frontend"
        / "dist",
        Path(os.environ.get("BRAIN_FRONTEND_DIST", "")) if os.environ.get("BRAIN_FRONTEND_DIST") else None,
    ]
    dist = next((path for path in dist_candidates if path and path.exists()), None)
    if dist is None:
        return
    from flask import send_from_directory
    logger.info("Serving built frontend from %s", dist)
    @app.get("/")
    def _frontend_index():
        return send_from_directory(dist, "index.html")
    @app.get("/<path:asset_path>")
    def _frontend_assets(asset_path: str):
        candidate = dist / asset_path
        if candidate.exists() and candidate.is_file():
            return send_from_directory(dist, asset_path)
        return send_from_directory(dist, "index.html")
