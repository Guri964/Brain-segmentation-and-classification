from __future__ import annotations
import logging
from flask import Flask, jsonify, request
from werkzeug.exceptions import HTTPException
logger = logging.getLogger(__name__)
class AppError(Exception):
    status_code = 400
    code = "application_error"
    def __init__(self, message: str, *, details: dict | None = None, status_code: int | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}
        if status_code is not None:
            self.status_code = status_code
    def to_dict(self) -> dict:
        return {"error": {"code": self.code, "message": self.message, "details": self.details}}
class ValidationError(AppError):
    status_code = 422
    code = "validation_error"
class UnsupportedInputError(AppError):
    status_code = 415
    code = "unsupported_input"
class NotFoundError(AppError):
    status_code = 404
    code = "not_found"
class ModelUnavailableError(AppError):
    status_code = 409
    code = "model_unavailable"
class DependencyMissingError(AppError):
    status_code = 503
    code = "dependency_missing"
class PayloadTooLargeError(AppError):
    status_code = 413
    code = "payload_too_large"
class InferenceError(AppError):
    status_code = 500
    code = "inference_error"
def register_error_handlers(app: Flask) -> None:
    @app.errorhandler(AppError)
    def _handle_app_error(exc: AppError):
        logger.warning("%s: %s", exc.code, exc.message)
        return jsonify(exc.to_dict()), exc.status_code
    @app.errorhandler(HTTPException)
    def _handle_http_error(exc: HTTPException):
        code = (exc.name or "http_error").lower().replace(" ", "_")
        return (
            jsonify({"error": {"code": code, "message": exc.description, "details": {}}}),
            exc.code or 500,
        )
    @app.errorhandler(Exception)
    def _handle_unexpected(exc: Exception):
        logger.exception("Unhandled error on %s %s", request.method, request.path)
        return (
            jsonify(
                {
                    "error": {
                        "code": "internal_error",
                        "message": "Unexpected server error.",
                        "details": {"type": type(exc).__name__},
                    }
                }
            ),
            500,
        )
