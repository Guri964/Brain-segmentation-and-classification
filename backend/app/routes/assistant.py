from __future__ import annotations
from flask import Blueprint, current_app, jsonify, request
from ..errors import NotFoundError, ValidationError
assistant_bp = Blueprint("assistant", __name__, url_prefix="/api/assistant")
def _services():
    return current_app.extensions["brain_services"]
@assistant_bp.get("/status")
def status():
    return jsonify(_services()["assistant"].status())
@assistant_bp.post("/ask")
def ask():
    body = request.get_json(silent=True) or {}
    question = (body.get("question") or "").strip()
    if not question:
        raise ValidationError("A 'question' field is required.")
    if len(question) > 2000:
        raise ValidationError("Questions are limited to 2000 characters.")
    analysis_id = body.get("analysis_id")
    services = _services()
    analysis = None
    if analysis_id:
        analysis = services["analysis"].resolve_payload(analysis_id)
        if analysis is None:
            raise NotFoundError(f"No analysis exists with identifier '{analysis_id}'.")
    answer = services["assistant"].answer(question, analysis)
    if analysis_id:
        repository = services["repository"]
        repository.add_assistant_message(analysis_id, "user", question)
        repository.add_assistant_message(analysis_id, "assistant", answer.answer, answer.sources)
    payload = answer.to_dict()
    payload["analysis_id"] = analysis_id
    return jsonify(payload)
@assistant_bp.get("/history/<analysis_id>")
def history(analysis_id: str):
    services = _services()
    if services["repository"].get_analysis(analysis_id) is None:
        raise NotFoundError(f"No analysis exists with identifier '{analysis_id}'.")
    return jsonify({"analysis_id": analysis_id, "messages": services["repository"].get_assistant_messages(analysis_id)})
