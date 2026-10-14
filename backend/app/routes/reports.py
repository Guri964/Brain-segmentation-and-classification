from __future__ import annotations
import logging
from pathlib import Path
from flask import Blueprint, current_app, jsonify, request, send_file
from ..errors import NotFoundError
reports_bp = Blueprint("reports", __name__, url_prefix="/api/reports")
logger = logging.getLogger(__name__)
def _services():
    return current_app.extensions["brain_services"]
@reports_bp.get("")
def list_reports():
    services = _services()
    limit = int(request.args.get("limit", 100))
    return jsonify({"items": services["repository"].list_reports(limit=limit)})
@reports_bp.get("/<report_id>")
def get_report(report_id: str):
    services = _services()
    stored = services["repository"].get_report(report_id)
    if stored is None:
        raise NotFoundError(f"No report exists with identifier '{report_id}'.")
    path = Path(current_app.config["BRAIN_DATA_DIR"]) / stored["path"]
    if not path.exists():
        raise NotFoundError("The report file is no longer on disk.")
    return send_file(path, as_attachment=True, download_name=f"{report_id}.pdf")
@reports_bp.post("/analysis/<analysis_id>")
def regenerate(analysis_id: str):
    services = _services()
    repository = services["repository"]
    full = repository.get_full_analysis(analysis_id)
    if full is None:
        raise NotFoundError(f"No analysis exists with identifier '{analysis_id}'.")
    analysis_root = Path(current_app.config["BRAIN_ARTIFACT_DIR"]) / analysis_id
    if not analysis_root.exists():
        raise NotFoundError("The artefacts for this analysis are no longer on disk.")
    from ..utils.images import decode_image_bytes, to_uint8
    manifest_uploads = (full.get("input_manifest") or {}).get("uploads") or []
    if not manifest_uploads:
        raise NotFoundError("The stored upload for this analysis is missing.")
    upload_path = Path(current_app.config["BRAIN_DATA_DIR"]) / manifest_uploads[0]["path"]
    if not upload_path.exists():
        raise NotFoundError("The stored upload for this analysis is missing.")
    display = to_uint8(decode_image_bytes(upload_path.read_bytes()).array)
    payload = {
        "analysis_id": analysis_id,
        "scan_identifier": full.get("scan_identifier"),
        "analysis_type": full.get("analysis_type"),
        "created_at": full.get("created_at"),
        "device": full.get("device"),
        "threshold": full.get("preprocessing", {}).get("threshold", 0.5),
        "preprocessing": full.get("preprocessing") or {},
        "models": {
            entry["model_id"]: {
                "status": "completed",
                "summary": entry.get("metrics") or {},
                "artifacts": entry.get("artifact_paths") or {},
                "descriptor": services["registry"].load_model(entry["model_id"]).to_dict(),
            }
            for entry in (full.get("segmentation") or [])
        },
        "ensemble": {"available": False, "reasons": ["Regenerated reports contain the stored per-model output only."]},
        "radiomics": _radiomics_from_rows(full.get("radiomics") or []),
        "classification": _classification_from_row(full.get("classification")),
        "explainability": [
            {
                "model_id": entry["model_id"],
                "available": True,
                "method": entry["method"],
                "target_layer": entry["target_layer"],
                "target_score": entry["target_score"],
                "artifacts": {"overlay": entry.get("overlay_path")},
                "warnings": entry.get("warnings") or [],
            }
            for entry in (full.get("explainability") or [])
        ],
        "uncertainty": (full.get("uncertainty") or {}).get("payload"),
        "model_comparison": {"note": "Regenerated report: see the original analysis for agreement statistics.", "models": []},
    }
    info = services["report"].generate(payload, analysis_root, display)
    repository.add_report(
        analysis_id,
        {
            "report_id": info["report_id"] + "-" + info["generated_at"].replace(":", "").replace("-", "")[:15],
            "path": info["relative_path"],
            "size_bytes": info["size_bytes"],
            "generator_version": info["generator_version"],
        },
    )
    return jsonify(info), 201
def _radiomics_from_rows(rows: list[dict]) -> dict:
    if not rows:
        return {"available": False, "unavailable_reason": "No radiomics record was stored."}
    entry = rows[-1]
    return {
        "available": True,
        "backend": entry.get("backend"),
        "dimensionality": entry.get("dimensionality"),
        "feature_count": entry.get("feature_count"),
        "features": entry.get("features") or {},
        "categories": entry.get("categories") or {},
        "quality": entry.get("quality") or {},
        "warnings": entry.get("warnings") or [],
        "spacing_available": bool(entry.get("spacing_available")),
    }
def _classification_from_row(row: dict | None) -> dict:
    if not row:
        return {"available": False, "reason": "No classification record was stored."}
    return {
        "available": bool(row.get("available")),
        "model_id": row.get("model_id"),
        "model_version": row.get("model_version"),
        "predicted_label": row.get("predicted_label"),
        "scores": row.get("scores") or [],
        "calibrated": bool(row.get("calibrated")),
        "metrics": row.get("metrics") or {},
        "feature_contributions": row.get("contributions") or [],
        "reason": row.get("reason"),
    }
