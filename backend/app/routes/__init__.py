from __future__ import annotations
from flask import Flask
from .analysis import analysis_bp
from .assistant import assistant_bp
from .classification import classification_bp
from .evaluation import evaluation_bp
from .health import health_bp
from .history import history_bp
from .models import models_bp
from .radiomics import radiomics_bp
from .reports import reports_bp
ALL_BLUEPRINTS = (
    health_bp,
    analysis_bp,
    models_bp,
    radiomics_bp,
    reports_bp,
    history_bp,
    evaluation_bp,
    assistant_bp,
    classification_bp,
)
def register_blueprints(app: Flask) -> None:
    for blueprint in ALL_BLUEPRINTS:
        app.register_blueprint(blueprint)
