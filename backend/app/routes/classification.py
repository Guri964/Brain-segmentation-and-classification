from __future__ import annotations

from flask import Blueprint, current_app, jsonify, request

classification_bp = Blueprint("classification", __name__, url_prefix="/api/classification")


def _services():
    return current_app.extensions["brain_services"]


@classification_bp.get("")
def index():
    services = _services()
    classification = services["classification"]
    status = classification.status()
    evaluation = services["evaluation"]
    image_classification = services["image_classification"]
    return jsonify(
        {
            "status": status,
            "evaluation": evaluation.classification_summary(),
            "image_classification": image_classification.status(),
            "label_semantics": {
                "target": status.get("target_description"),
                "explanation": (
                    "The predicted class is the dataset label the model was trained on, for example "
                    "the WHO grade column of the BraTS name mapping (HGG or LGG). It is not a "
                    "malignancy verdict, not a clinical stage and not a prognosis; those are not "
                    "recorded in this dataset and are therefore not predicted."
                ),
                "derived_from_mask": False,
                "note": (
                    "The mask is used to define the region the radiomics features are computed "
                    "from, but the class itself comes from the dataset label table, never from the "
                    "segmentation output."
                ),
            },
            "xai": {
                "method": (
                    "Linear models report coefficient times standardised feature value; models "
                    "without coefficients fall back to single-feature substitution against the "
                    "training quantiles stored in the artefact."
                ),
                "shap": _shap_status(),
            },
            "training": {
                "recommended": "scripts/train_classification.py",
                "alternatives": ["scripts/train_classifier.py (torch feature head)"],
                "pipeline": [
                    "scripts/prepare_brats.py - manifest with patient identifiers and grade labels",
                    "scripts/extract_radiomics.py - one feature row per case, application schema",
                    "scripts/train_classification.py - baseline versus improved, held-out test",
                ],
                "requirements": classification.data_requirements(),
            },
            "disclaimer": current_app.config.get("BRAIN_DISCLAIMER", current_app.config.get("DISCLAIMER")),
        }
    )


def _shap_status() -> dict:
    try:
        import shap

        return {"available": True, "version": getattr(shap, "__version__", None)}
    except Exception as exc:
        return {
            "available": False,
            "reason": f"{type(exc).__name__}",
            "fallback": (
                "Feature attribution falls back to the coefficient/substitution methods described "
                "above, which are computed from the artefact itself and need no extra dependency."
            ),
        }

@classification_bp.get("/image/models")
def image_models():
    """The image-classification catalogue: what exists, what is missing and why."""

    services = _services()
    return jsonify({"models": services["image_classification"].models()})


@classification_bp.post("/image")
def classify_image():
    """Classify one uploaded image with the Keras tumour-type model.

    Answers 409 with the exact missing file when the checkpoint is not installed and 503 when
    TensorFlow is not importable - never a label produced from anything else.
    """

    services = _services()
    upload = request.files.get("file")
    if upload is None or not upload.filename:
        from ..errors import ValidationError

        raise ValidationError(
            "Attach one image as the multipart field 'file'.",
            details={"field": "file"},
        )
    payload = services["image_classification"].classify_upload(
        upload.filename, upload.read()
    )
    return jsonify(payload), 200
