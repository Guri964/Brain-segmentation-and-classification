from __future__ import annotations
import platform
import sys
import torch
from flask import Blueprint, current_app, jsonify
health_bp = Blueprint("health", __name__, url_prefix="/api")
@health_bp.get("/health")
def health():
    services = current_app.extensions["brain_services"]
    repository = services["repository"]
    return jsonify(
        {
            "status": "ok",
            "service": "brain-abnormality-api",
            "environment": current_app.config["BRAIN_ENV"],
            "time": services["clock"](),
            "database": {
                "path": str(repository.database_path),
                "schema_version": repository.schema_version(),
            },
        }
    )
@health_bp.get("/system")
def system():
    services = current_app.extensions["brain_services"]
    registry = services["registry"]
    return jsonify(
        {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "torch_threads": torch.get_num_threads(),
            "selected_device": str(registry.device),
            "config": current_app.config["BRAIN_CONFIG_SUMMARY"],
        }
    )
@health_bp.get("/capabilities")
def capabilities():
    services = current_app.extensions["brain_services"]
    registry = services["registry"]
    health_service = services["health"]
    ensemble = services["ensemble"]
    classification = services["classification"]
    radiomics = services["radiomics"]
    assistant = services["assistant"]
    model_health = health_service.summary()
    return jsonify(
        {
            "models": registry.status_report(),
            "model_health": model_health,
            "ensemble": ensemble.compatibility(),
            "radiomics": radiomics.backend_status(),
            "classification": classification.status(),
            "assistant": assistant.status(),
            "explainability": services["explainability"].available_methods(),
            "evaluation": services["evaluation"].stored_results()["state"],
            "disclaimer": current_app.config["BRAIN_DISCLAIMER"],
        }
    )
