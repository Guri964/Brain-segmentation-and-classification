from __future__ import annotations
import logging
from pathlib import Path
from flask import Blueprint, current_app, jsonify, request, send_file
from ..errors import NotFoundError, PayloadTooLargeError, UnsupportedInputError, ValidationError
from ..services.analysis_service import AnalysisRequest
from ..utils.io import ensure_within, human_size, safe_filename
from ..utils.images import encode_png, to_uint8
analysis_bp = Blueprint("analysis", __name__, url_prefix="/api")
logger = logging.getLogger(__name__)
ALLOWED_SUFFIXES = {".png", ".jpg", ".jpeg", ".npz"}
def _services():
    return current_app.extensions["brain_services"]
def _parse_files(files_field: str = "files"):
    uploads = request.files.getlist(files_field)
    if not uploads:
        uploads = [f for key in request.files for f in request.files.getlist(key)]
    if not uploads:
        raise ValidationError(
            "No file was included. Send the images as multipart/form-data under the field 'files'."
        )
    payloads = []
    total = 0
    limit = int(current_app.config["BRAIN_MAX_UPLOAD_BYTES"])
    for upload in uploads:
        filename = safe_filename(upload.filename or "upload.png")
        suffix = Path(filename).suffix.lower()
        if suffix not in ALLOWED_SUFFIXES:
            raise UnsupportedInputError(
                f"'{filename}' has an unsupported extension '{suffix}'.",
                details={"allowed": sorted(ALLOWED_SUFFIXES)},
            )
        data = upload.read()
        if not data:
            raise ValidationError(f"'{filename}' is empty.")
        total += len(data)
        if total > limit:
            raise PayloadTooLargeError(
                f"The upload exceeds the configured limit of {human_size(limit)}.",
                details={"received_bytes": total, "limit_bytes": limit},
            )
        payloads.append((filename, data))
    max_files = int(current_app.config["BRAIN_MAX_FILES_PER_ANALYSIS"])
    if len(payloads) > max_files:
        raise ValidationError(f"At most {max_files} files may be uploaded per analysis.")
    return payloads
@analysis_bp.post("/analysis/plan")
def plan():
    body = request.get_json(silent=True) or {}
    analysis_type = body.get("analysis_type", "four_channel")
    if analysis_type not in {"four_channel", "single_channel"}:
        raise ValidationError("analysis_type must be 'four_channel' or 'single_channel'.")
    services = _services()
    plan_payload = services["analysis"].plan(
        AnalysisRequest(
            files=[],
            analysis_type=analysis_type,
            model_ids=body.get("model_ids"),
            enable_ensemble=bool(body.get("enable_ensemble", True)),
            enable_radiomics=bool(body.get("enable_radiomics", True)),
            enable_uncertainty=bool(body.get("enable_uncertainty", False)),
            enable_explainability=bool(body.get("enable_explainability", True)),
            ensemble_strategy=body.get("ensemble_strategy", "equal"),
            radiomics_mask=body.get("radiomics_mask", "best"),
            enable_stacked_ensemble=bool(body.get("enable_stacked_ensemble", False)),
        )
    )
    return jsonify(plan_payload)
@analysis_bp.post("/analysis")
def create_analysis():
    form = request.form
    analysis_type = form.get("analysis_type", "four_channel")
    if analysis_type not in {"four_channel", "single_channel"}:
        raise ValidationError("analysis_type must be 'four_channel' or 'single_channel'.")
    model_ids = [value for value in form.getlist("model_ids") if value] or None
    modality_names = [value for value in form.getlist("modality_names") if value] or None
    spacing = None
    if form.get("spacing"):
        try:
            spacing = tuple(float(part) for part in form["spacing"].split(","))
        except ValueError as exc:
            raise ValidationError("spacing must be a comma separated list of numbers.") from exc
    payloads = _parse_files()
    files = _validate_against_type(payloads, analysis_type)
    request_object = AnalysisRequest(
        files=files,
        analysis_type=analysis_type,
        model_ids=model_ids,
        scan_identifier=(form.get("scan_identifier") or "").strip() or None,
        notes=(form.get("notes") or "").strip() or None,
        threshold=float(form.get("threshold", current_app.config["BRAIN_SEG_THRESHOLD"])),
        enable_ensemble=_as_bool(form.get("enable_ensemble"), True),
        ensemble_strategy=form.get("ensemble_strategy", "equal"),
        enable_uncertainty=_as_bool(form.get("enable_uncertainty"), False),
        enable_explainability=_as_bool(form.get("enable_explainability"), True),
        enable_radiomics=_as_bool(form.get("enable_radiomics"), True),
        radiomics_mask=form.get("radiomics_mask", "best"),
        modality_names=modality_names,
        spacing=spacing,
        enable_stacked_ensemble=_as_bool(form.get("enable_stacked_ensemble"), False),
    )
    services = _services()
    result = services["analysis"].run(request_object)
    return jsonify(_public_payload(result)), 201
@analysis_bp.get("/analysis/<analysis_id>")
def get_analysis(analysis_id: str):
    services = _services()
    repository = services["repository"]
    stored = repository.get_full_analysis(analysis_id)
    if stored is None:
        raise NotFoundError(f"No analysis exists with identifier '{analysis_id}'.")
    payload = services["analysis"].load_payload(analysis_id)
    response = {
        "metadata": {
            key: value
            for key, value in stored.items()
            if key not in {"segmentation", "radiomics", "classification", "explainability", "uncertainty", "reports", "input_manifest"}
        },
        "stored_records": {
            "segmentation": stored["segmentation"],
            "radiomics": [
                {key: value for key, value in row.items() if key != "features"} | {"feature_count": row.get("feature_count")}
                for row in stored["radiomics"]
            ],
            "classification": stored["classification"],
            "explainability": stored["explainability"],
            "uncertainty": {
                key: value
                for key, value in (stored["uncertainty"] or {}).items()
                if key != "payload"
            }
            if stored["uncertainty"]
            else None,
            "reports": stored["reports"],
        },
        "payload_available": payload is not None,
        "payload_source": "artifacts/analysis.json" if payload is not None else "database_reconstruction",
    }
    if payload is not None:
        response["result"] = payload
    else:
        response["result"] = services["analysis"].summary_from_database(analysis_id)
    return jsonify(response)
@analysis_bp.delete("/analysis/<analysis_id>")
def delete_analysis(analysis_id: str):
    services = _services()
    repository = services["repository"]
    stored = repository.get_analysis(analysis_id)
    if stored is None:
        raise NotFoundError(f"No analysis exists with identifier '{analysis_id}'.")
    removed_files = 0
    for stem in (
        Path(current_app.config["BRAIN_DATA_DIR"]) / current_app.config["BRAIN_UPLOAD_DIR"],
        Path(current_app.config["BRAIN_DATA_DIR"]) / current_app.config["BRAIN_ARTIFACT_DIR"],
        Path(current_app.config["BRAIN_DATA_DIR"]) / current_app.config["BRAIN_REPORT_DIR"] / analysis_id,
    ):
        directory = Path(stem) / analysis_id
        if directory.exists():
            for item in directory.rglob("*"):
                if item.is_file():
                    item.unlink(missing_ok=True)
                    removed_files += 1
            for item in sorted(directory.rglob("*"), reverse=True):
                if item.is_dir():
                    item.rmdir()
            directory.rmdir()
    repository.delete_analysis(analysis_id)
    return jsonify(
        {
            "deleted": True,
            "analysis_id": analysis_id,
            "files_removed": removed_files,
            "note": "Analysis record, uploads, artefacts and reports were removed.",
        }
    )
@analysis_bp.get("/analysis/<analysis_id>/artifacts/<path:filename>")
def get_artifact(analysis_id: str, filename: str):
    services = _services()
    if services["repository"].get_analysis(analysis_id) is None:
        raise NotFoundError(f"No analysis exists with identifier '{analysis_id}'.")
    root = ensure_within(Path(current_app.config["BRAIN_ARTIFACT_DIR"]), Path(analysis_id))
    candidate = ensure_within(root, Path(filename))
    if not candidate.exists() or not candidate.is_file():
        raise NotFoundError(f"Artifact '{filename}' does not exist for this analysis.")
    return send_file(candidate, max_age=0, conditional=True)
@analysis_bp.get("/analysis/<analysis_id>/inputs/<int:index>")
def get_input_image(analysis_id: str, index: int):
    services = _services()
    stored = services["repository"].get_analysis(analysis_id)
    if stored is None:
        raise NotFoundError(f"No analysis exists with identifier '{analysis_id}'.")
    manifest = stored.get("input_manifest") or {}
    uploads = manifest.get("uploads") or []
    if index < 0 or index >= len(uploads):
        raise NotFoundError(f"Input index {index} is out of range for this analysis.")
    path = Path(current_app.config["BRAIN_DATA_DIR"]) / uploads[index]["path"]
    if not path.exists():
        raise NotFoundError("The stored upload is no longer on disk.")
    return send_file(path, max_age=0, conditional=True)
def _validate_against_type(payloads, analysis_type: str) -> list[tuple[str, bytes]]:
    suffixes = [Path(name).suffix.lower() for name, _ in payloads]
    has_npz = any(suffix == ".npz" for suffix in suffixes)
    has_image = any(suffix in {".png", ".jpg", ".jpeg"} for suffix in suffixes)
    if analysis_type == "single_channel":
        if has_npz or len(payloads) != 1:
            raise ValidationError(
                "The single-channel pipeline expects exactly one PNG or JPEG image. "
                "Received "
                f"{len(payloads)} file(s).",
                details={"received": [name for name, _ in payloads]},
            )
        return payloads
    if has_npz and len(payloads) > 1:
        raise ValidationError("Upload either one .npz volume or four modality images, not both.")
    if has_npz:
        return payloads
    if has_image and len(payloads) != 4:
        raise UnsupportedInputError(
            "The four-channel pipeline needs one image per MRI modality (T1, T1ce, T2, FLAIR) - "
            f"that is 4 files, but {len(payloads)} were supplied. A single image must not be "
            "duplicated into four channels, because the model would then be reading sequences "
            "that were never acquired.",
            details={"received": [name for name, _ in payloads]},
        )
    return payloads
def _as_bool(value, default: bool) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}
def _public_payload(result: dict) -> dict:
    def clean(node):
        if isinstance(node, dict):
            return {key: clean(value) for key, value in node.items() if not key.startswith("_")}
        if isinstance(node, list):
            return [clean(item) for item in node]
        return node
    return clean(result)
@analysis_bp.get("/analysis/<analysis_id>/preview/<int:index>")
def preview_input(analysis_id: str, index: int):
    services = _services()
    stored = services["repository"].get_analysis(analysis_id)
    if stored is None:
        raise NotFoundError(f"No analysis exists with identifier '{analysis_id}'.")
    manifest = stored.get("input_manifest") or {}
    uploads = manifest.get("uploads") or []
    if index < 0 or index >= len(uploads):
        raise NotFoundError(f"Input index {index} is out of range.")
    path = Path(current_app.config["BRAIN_DATA_DIR"]) / uploads[index]["path"]
    if not path.exists():
        raise NotFoundError("The stored upload is no longer on disk.")
    from ..utils.images import decode_image_bytes
    decoded = decode_image_bytes(path.read_bytes())
    return send_file(
        __import__("io").BytesIO(encode_png(to_uint8(decoded.array))),
        mimetype="image/png",
        max_age=0,
    )
