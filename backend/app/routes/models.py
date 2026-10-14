from __future__ import annotations
from pathlib import Path
from flask import Blueprint, current_app, jsonify, request
from ..errors import ModelUnavailableError, NotFoundError, ValidationError
from ..services.analysis_service import AnalysisRequest
from ..utils.images import encode_png, make_overlay, mask_to_rgb, to_uint8
models_bp = Blueprint("models", __name__, url_prefix="/api")
def _services():
    return current_app.extensions["brain_services"]
@models_bp.get("/models")
def list_models():
    services = _services()
    registry = services["registry"]
    health = services["health"]
    return jsonify(
        {
            "device": str(registry.device),
            "models": [
                {**entry, "health": health.check(entry["model_id"]).to_dict()}
                for entry in registry.status_report()
            ],
        }
    )
@models_bp.get("/models/<model_id>")
def get_model(model_id: str):
    services = _services()
    registry = services["registry"]
    if model_id not in registry.descriptors:
        raise NotFoundError(f"Unknown model '{model_id}'.", details={"known": list(registry.descriptors)})
    loaded = registry.load_model(model_id)
    payload = loaded.to_dict()
    payload["health"] = services["health"].check(model_id).to_dict()
    payload["explainability"] = services["explainability"].available_methods()
    return jsonify(payload)
@models_bp.get("/models/<model_id>/health")
def model_health(model_id: str):
    services = _services()
    registry = services["registry"]
    if model_id not in registry.descriptors:
        raise NotFoundError(f"Unknown model '{model_id}'.")
    force = request.args.get("force", "false").lower() in {"1", "true", "yes"}
    return jsonify(services["health"].check(model_id, force=force).to_dict())
@models_bp.post("/models/health-check")
def run_health_checks():
    services = _services()
    force = (request.get_json(silent=True) or {}).get("force", False)
    summary = services["health"].summary() if not force else None
    if force:
        for model_id in services["registry"].descriptors:
            services["health"].check(model_id, force=True)
        summary = services["health"].summary()
    return jsonify(summary)
@models_bp.post("/models/compare")
def compare_models():
    services = _services()
    registry = services["registry"]
    segmentation = services["segmentation"]
    health_service = services["health"]
    ensemble = services["ensemble"]
    form = request.form
    analysis_type = form.get("analysis_type", "four_channel")
    model_ids = [value for value in form.getlist("model_ids") if value]
    if not model_ids:
        model_ids = [
            model_id
            for model_id in registry.segmentation_model_ids()
            if registry.load_model(model_id).is_ready
        ]
    unknown = [m for m in model_ids if m not in registry.descriptors]
    if unknown:
        raise NotFoundError("Unknown model(s) requested.", details={"unknown": unknown})
    from .analysis import _parse_files, _validate_against_type
    payloads = _parse_files()
    files = _validate_against_type(payloads, analysis_type)
    threshold = float(form.get("threshold", current_app.config["BRAIN_SEG_THRESHOLD"]))
    preprocessing = services["preprocessing"]
    descriptors = [registry.load_model(model_id).descriptor for model_id in model_ids]
    channels = {descriptor.input_spec.channels for descriptor in descriptors}
    if len(channels) != 1:
        raise ValidationError(
            "All compared models must accept the same number of input channels. The shipped "
            "checkpoints take different inputs (1 channel at 512x512 and 4 channels at 256x256), "
            "so a like-for-like comparison requires an input-compatible partner checkpoint; train "
            "one with scripts/train_unet.py.",
            details={
                "model_inputs": {
                    descriptor.model_id: descriptor.input_spec.to_dict() for descriptor in descriptors
                }
            },
        )
    channel_count = channels.pop()
    if channel_count == 1:
        if len(files) != 1:
            raise ValidationError("The single-channel comparison expects exactly one image.")
        preprocessed = preprocessing.preprocess_single_channel(
            files[0][1], files[0][0], size=descriptors[0].input_spec.height
        )
    else:
        preprocessed = preprocessing.preprocess_four_channel(
            files, size=descriptors[0].input_spec.height
        )
    display = preprocessed.display_stack[0][1]
    entries = []
    probabilities = {}
    masks = {}
    for model_id in model_ids:
        loaded = registry.load_model(model_id)
        if not loaded.is_ready:
            entries.append(
                {
                    "model_id": model_id,
                    "display_name": loaded.descriptor.display_name,
                    "available": False,
                    "status": loaded.status,
                    "message": loaded.message,
                }
            )
            continue
        result = segmentation.predict(model_id, preprocessed, threshold=threshold)
        probabilities[model_id] = result.probability
        masks[model_id] = result.mask
        report = health_service.check(model_id)
        entries.append(
            {
                "model_id": model_id,
                "display_name": result.model_display_name,
                "available": True,
                "status": "completed",
                "summary": result.summary(),
                "health": report.to_dict(),
                "images": {
                    "probability": _data_url(np.repeat(to_uint8(result.probability)[..., None], 3, axis=2)),
                    "mask": _data_url(mask_to_rgb(result.mask)),
                    "overlay": _data_url(make_overlay(display, result.mask)),
                },
            }
        )
    comparison = {
        "state": "unlabelled",
        "note": (
            "The uploaded scan has no reference segmentation, so these are output statistics and "
            "agreement measures - not accuracy. Accuracy requires labelled data and is reported "
            "only by scripts/evaluate_models.py."
        ),
    }
    from ..utils.metrics import agreement_metrics
    if len(masks) >= 2:
        pairs = {}
        keys = list(masks)
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                pairs[f"{keys[i]}_vs_{keys[j]}"] = agreement_metrics(masks[keys[i]], masks[keys[j]])
        comparison["pairwise_agreement"] = pairs
    compatibility = ensemble.compatibility(model_ids)
    return jsonify(
        {
            "analysis_type": analysis_type,
            "threshold": threshold,
            "input_preview": _data_url(to_uint8(display)),
            "input_channels": [
                {"label": label, "image": _data_url(to_uint8(array))}
                for label, array in preprocessed.display_stack
            ],
            "preprocessing": preprocessed.summary(),
            "models": entries,
            "comparison": comparison,
            "ensemble": compatibility,
        }
    )
def _data_url(array) -> str:
    import base64
    return "data:image/png;base64," + base64.b64encode(encode_png(array)).decode("ascii")
@models_bp.get("/analysis-options")
def analysis_options():
    services = _services()
    registry = services["registry"]
    health_service = services["health"]
    options = []
    for entry in registry.status_report():
        if entry["task"] != "segmentation":
            continue
        loaded = registry.load_model(entry["model_id"])
        report = health_service.check(entry["model_id"]) if loaded.is_ready else None
        options.append(
            {
                "model_id": entry["model_id"],
                "display_name": entry["display_name"],
                "status": entry["status"],
                "available": entry["available"],
                "input_spec": entry["input_spec"],
                "output_semantics": entry["output_semantics"],
                "description": entry["description"],
                "trainer": entry["trainer"],
                "health_verdict": report.verdict if report else "not_loaded",
                "health_message": report.message if report else entry["message"],
                "recommended": bool(report and report.responsive),
            }
        )
    return jsonify(
        {
            "models": options,
            "ensemble": services["ensemble"].compatibility(),
            "classification": services["classification"].status(),
            "radiomics": services["radiomics"].backend_status(),
            "assistant": services["assistant"].status(),
            "supported_analysis_types": [
                {
                    "value": "four_channel",
                    "label": "Four-channel MRI (BraTS order: T1, T1ce, T2, FLAIR)",
                    "requires": "4 modality images, or 1 .npz containing 4 channels",
                    "models": [
                        entry["model_id"]
                        for entry in options
                        if entry["input_spec"]["channels"] == 4
                    ],
                },
                {
                    "value": "single_channel",
                    "label": "Single grayscale MRI image",
                    "requires": "1 PNG or JPEG",
                    "models": [
                        entry["model_id"]
                        for entry in options
                        if entry["input_spec"]["channels"] == 1
                    ],
                },
            ],
            "disclaimer": current_app.config["BRAIN_DISCLAIMER"],
        }
    )

@models_bp.get("/ensemble/stacked")
def stacked_ensemble_status():
    """The notebook's stacked ensemble: which artefacts are present and what it can predict.

    Deliberately verbose, because this pipeline has eight files and three models behind it.  The
    response names every artefact, every member, the ten meta features and the target semantics,
    so a reviewer can see what is missing without reading the source.
    """

    services = _services()
    service = services.get("stacked_ensemble")
    if service is None:
        raise NotFoundError("The stacked-ensemble service is not wired into this deployment.")
    force = request.args.get("force", "false").lower() in {"1", "true", "yes"}
    return jsonify(service.status(force=force))


@models_bp.post("/ensemble/stacked/run")
def stacked_ensemble_run():
    """Run the stacked ensemble for an existing analysis.

    Four-channel 240x240 input only: the members were trained on that tensor, so a single
    grayscale image is rejected with an explanation instead of being stretched into the shape the
    models happen to accept.  ``force_mono`` may not be passed as JSON - the endpoint always
    reuses the resolved tensor paths.
    """

    services = _services()
    service = services.get("stacked_ensemble")
    if service is None:
        raise NotFoundError("The stacked-ensemble service is not wired into this deployment.")
    body = request.get_json(silent=True) or {}
    analysis_id = (body.get("analysis_id") or "").strip()
    if not analysis_id:
        raise ValidationError("analysis_id is required.", details={"field": "analysis_id"})
    analysis = services["analysis"]
    payload = analysis.load_payload(analysis_id)
    if payload is None:
        raise NotFoundError(f"Analysis '{analysis_id}' has no stored payload to run on.")
    if payload.get("analysis_type") != "four_channel":
        raise ValidationError(
            "The stacked ensemble needs the four-channel 240x240 tensor the notebook trained "
            "on; this analysis is "
            f"{payload.get('analysis_type')!r}. Re-run it with analysis_type=four_channel and "
            "enable_stacked_ensemble=true.",
            details={"analysis_type": payload.get("analysis_type")},
        )
    preprocessing = payload.get("preprocessing") or {}
    shape = preprocessing.get("tensor_shape") or []
    if len(shape) != 4 or shape[1] != 4:
        raise ValidationError(
            "The stored analysis does not carry a four-channel tensor, so the ensemble's ten "
            "meta features cannot be reproduced.",
            details={"tensor_shape": shape},
        )
    return jsonify(
        {
            "analysis_id": analysis_id,
            "status": service.status(),
            "hint": (
                "Run the analysis with enable_stacked_ensemble=true to obtain a prediction; this "
                "endpoint reports the ensemble's readiness for a stored analysis without "
                "recomputing the members' forward passes."
            ),
        }
    )
