from __future__ import annotations
import io
import logging
from pathlib import Path
import numpy as np
from flask import Blueprint, current_app, jsonify, request, send_file
from ..errors import NotFoundError, UnsupportedInputError, ValidationError
from ..ml.radiomics_native import FEATURE_CATEGORIES, DiscretisationSettings
from ..services.radiomics_service import FEATURE_DESCRIPTIONS, feature_tooltip
from ..utils.images import decode_image_bytes, resize_array
radiomics_bp = Blueprint("radiomics", __name__, url_prefix="/api/radiomics")
logger = logging.getLogger(__name__)
def _services():
    return current_app.extensions["brain_services"]
@radiomics_bp.get("/catalogue")
def catalogue():
    services = _services()
    service = services["radiomics"]
    return jsonify(
        {
            "backend": service.backend_status(),
            "categories": [
                {
                    "key": key,
                    "label": key.replace("firstorder", "first-order").replace("glcm", "GLCM")
                    .replace("glrlm", "run-length").replace("diagnostics", "diagnostics")
                    .title(),
                    "description": description,
                }
                for key, description in FEATURE_CATEGORIES.items()
            ],
            "descriptions": FEATURE_DESCRIPTIONS,
            "discretisation": DiscretisationSettings(
                method=current_app.config["BRAIN_RADIOMICS_DISCRETIZATION"],
                bin_count=int(current_app.config["BRAIN_RADIOMICS_BIN_COUNT"]),
                bin_width=float(current_app.config["BRAIN_RADIOMICS_BIN_WIDTH"]),
            ).as_dict(),
            "notes": [
                "Values are computed from the image and mask actually supplied; nothing is defaulted.",
                "For a 2-D slice, areas are reported in pixels and no physical volume is computed.",
                "Shape and texture values describe the segmented region and inherit its errors.",
            ],
        }
    )
@radiomics_bp.get("/analysis/<analysis_id>")
def analysis_features(analysis_id: str):
    services = _services()
    stored = services["repository"].get_analysis(analysis_id)
    if stored is None:
        raise NotFoundError(f"No analysis exists with identifier '{analysis_id}'.")
    full = services["repository"].get_full_analysis(analysis_id)
    rows = full.get("radiomics") or []
    if not rows:
        return (
            jsonify(
                {
                    "available": False,
                    "unavailable_reason": (
                        "No radiomics record exists for this analysis. The region of interest was "
                        "probably empty, or the extraction was disabled."
                    ),
                }
            ),
            200,
        )
    entry = rows[-1]
    features = entry.get("features") or {}
    return jsonify(
        {
            "available": True,
            "analysis_id": analysis_id,
            "backend": entry.get("backend"),
            "dimensionality": entry.get("dimensionality"),
            "mask_source": entry.get("mask_source"),
            "feature_count": entry.get("feature_count"),
            "spacing_available": bool(entry.get("spacing_available")),
            "features": features,
            "categories": entry.get("categories") or {},
            "quality": entry.get("quality") or {},
            "warnings": entry.get("warnings") or [],
            "descriptions": {name: feature_tooltip(name) for name in features},
        }
    )
@radiomics_bp.post("/extract")
def extract():
    services = _services()
    if "image" not in request.files:
        raise ValidationError("An 'image' file is required.")
    image_file = request.files["image"]
    mask_file = request.files.get("mask")
    decoded = decode_image_bytes(image_file.read())
    image = decoded.array
    mask_source = "uploaded_mask"
    if mask_file is not None and mask_file.filename:
        mask_decoded = decode_image_bytes(mask_file.read())
        mask = mask_decoded.array
        if mask.shape != image.shape:
            mask = resize_array(mask, image.shape, nearest=True)
        mask = mask > 0.5
        mask_source = "uploaded_mask"
    else:
        model_id = request.form.get("model_id") or _default_model(services)
        if model_id is None:
            raise UnsupportedInputError(
                "No mask was uploaded and no segmentation model is available to generate one."
            )
        from ..services.analysis_service import AnalysisRequest
        segmentation = services["segmentation"]
        loaded = services["registry"].load_model(model_id)
        if not loaded.is_ready:
            raise UnsupportedInputError(
                f"Model '{model_id}' is not available ({loaded.status}).",
                details={"model_id": model_id},
            )
        if loaded.descriptor.input_spec.channels == 1:
            preprocessed = services["preprocessing"].preprocess_single_channel(
                image_file.read() or _reencode(decoded.array),
                image_file.filename or "upload.png",
                size=loaded.descriptor.input_spec.height,
            )
        else:
            raise UnsupportedInputError(
                "Standalone extraction for a four-channel model requires the four modality images; "
                "run a full analysis instead, or upload a mask together with the image.",
                details={"model_id": model_id, "required_channels": 4},
            )
        result = segmentation.predict(model_id, preprocessed)
        mask = resize_array(result.mask.astype(np.float32), image.shape, nearest=True) > 0.5
        mask_source = f"model:{model_id}"
    spacing = None
    if request.form.get("spacing"):
        try:
            spacing = tuple(float(part) for part in request.form["spacing"].split(","))
        except ValueError as exc:
            raise ValidationError("spacing must be comma separated numbers.") from exc
    outcome = services["radiomics"].extract(image, mask, spacing=spacing, mask_source=mask_source)
    payload = outcome.to_dict()
    if outcome.available:
        payload["descriptions"] = {name: feature_tooltip(name) for name in outcome.features}
    del AnalysisRequest
    return jsonify(payload)
@radiomics_bp.get("/analysis/<analysis_id>/features.csv")
def download_features(analysis_id: str):
    services = _services()
    full = services["repository"].get_full_analysis(analysis_id)
    if full is None:
        raise NotFoundError(f"No analysis exists with identifier '{analysis_id}'.")
    rows = full.get("radiomics") or []
    if not rows or not rows[-1].get("artifact_path"):
        raise NotFoundError("This analysis has no exported feature table.")
    path = Path(current_app.config["BRAIN_DATA_DIR"]) / rows[-1]["artifact_path"]
    if not path.exists():
        raise NotFoundError("The exported feature table is no longer on disk.")
    return send_file(path, as_attachment=True, download_name=f"{analysis_id}_radiomics.csv")
def _reencode(array: np.ndarray) -> bytes:
    from ..utils.images import encode_png
    from ..utils.images import to_uint8
    return encode_png(to_uint8(array))
def _default_model(services):
    for model_id in services["registry"].segmentation_model_ids():
        loaded = services["registry"].load_model(model_id)
        if loaded.is_ready and loaded.descriptor.input_spec.channels == 1:
            return model_id
    for model_id in services["registry"].segmentation_model_ids():
        if services["registry"].load_model(model_id).is_ready:
            return model_id
    return None
