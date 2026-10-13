from __future__ import annotations
from flask import Blueprint, current_app, jsonify, request
from ..database.repository import ANALYSIS_STATUSES
history_bp = Blueprint("history", __name__, url_prefix="/api/history")
def _services():
    return current_app.extensions["brain_services"]
@history_bp.get("")
def list_history():
    services = _services()
    repository = services["repository"]
    args = request.args
    result = repository.list_analyses(
        search=args.get("search"),
        status=args.get("status"),
        analysis_type=args.get("analysis_type"),
        model_id=args.get("model_id"),
        date_from=args.get("date_from"),
        date_to=args.get("date_to"),
        limit=int(args.get("limit", 25)),
        offset=int(args.get("offset", 0)),
        order=args.get("order", "created_at"),
        direction=args.get("direction", "desc"),
    )
    result["filters"] = {
        "statuses": list(ANALYSIS_STATUSES),
        "analysis_types": ["four_channel", "single_channel"],
        "models": list(services["registry"].descriptors),
    }
    return jsonify(result)
@history_bp.get("/statistics")
def statistics():
    services = _services()
    repository = services["repository"]
    payload = repository.statistics()
    payload["status_counts"] = repository.counts_by_status()
    payload["models"] = {
        "ready": len(services["registry"].available_segmentation_models()),
        "total": len(services["registry"].descriptors),
    }
    payload["evaluation_state"] = services["evaluation"].stored_results()["state"]
    return jsonify(payload)
@history_bp.get("/<analysis_id>/export")
def export_analysis(analysis_id: str):
    services = _services()
    full = services["repository"].get_full_analysis(analysis_id)
    if full is None:
        from ..errors import NotFoundError
        raise NotFoundError(f"No analysis exists with identifier '{analysis_id}'.")
    import io
    import json
    from flask import send_file
    payload = json.dumps(full, indent=2, default=str).encode("utf-8")
    return send_file(
        io.BytesIO(payload),
        mimetype="application/json",
        as_attachment=True,
        download_name=f"{analysis_id}.json",
    )
