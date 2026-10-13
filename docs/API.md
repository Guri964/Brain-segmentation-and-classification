# API reference

Base URL: the same origin that serves the dashboard, under `/api`. In development Vite
proxies `/api` to the Flask process. Every response is JSON except the PDF report, the
CSV export and the PNG artefacts.

Errors always use the same envelope:

```json
{
  "error": {
    "code": "validation_error",
    "message": "The four supplied channels are not all different: t1 == t1ce, t1 == t2, ...",
    "details": { "filename": "scan.png" }
  }
}
```

| Status | Code | Meaning |
|---|---|---|
| 400 | `bad_request` | Malformed request |
| 404 | `not_found` | Unknown analysis, model or report |
| 409 | `model_unavailable` | The requested model cannot run (not loaded, wrong input) |
| 413 | `payload_too_large` | Upload exceeds `BRAIN_MAX_UPLOAD_BYTES` |
| 415 | `unsupported_input` | File type or channel count not supported by the chosen pipeline |
| 422 | `validation_error` | The request is well formed but semantically invalid |
| 500 | `inference_error` | The model ran but failed |
| 503 | `dependency_missing` | An optional dependency is required for this request |

---

## System and capabilities

### `GET /api/health`

```json
{ "status": "ok", "environment": "development", "time": "2026-10-13T07:38:11+00:00",
  "database": { "path": "…/data/app.db", "schema_version": 2 } }
```

### `GET /api/system`

Python and platform versions, torch version, `cuda_available`, `cuda_device_name`,
`torch_threads`, `selected_device`, and the effective configuration (`config`).

### `GET /api/capabilities`

The single source of truth for what the deployment can do; the dashboard renders its
availability panel from this response.

```json
{
  "models": [ { "model_id": "unet_dynamic_512", "input_spec": {...}, "status": "ready", "available": true, ... } ],
  "model_health": { "phantom_version": "phantom-v1", "generated_at": "...", "models": [ { "verdict": "responsive", "responsive": true, ... } ], "responsive_models": ["unet_dynamic_512"] },
  "ensemble": { "available": false, "groups": [ { "channels": 1, "height": 512, "width": 512, "members": ["unet_dynamic_512"], "ensemblable": false } ], "reasons": ["…"] },
  "radiomics": { "selected_backend": "native", "pyradiomics_available": false, "note": "…" },
  "classification": { "available": false, "reason": "…", "requirements": ["…"] },
  "assistant": { "backend": "local", "note": "…", "suggested_questions": ["…"] },
  "explainability": { "segmentation": [ { "model_id": "…", "available": true, "method": "grad_cam", "target_layer": "conv9_2" } ], "classification": {...}, "disclaimer": "…" },
  "evaluation": "not_evaluated",
  "disclaimer": "Research prototype. Outputs are AI-generated and are not intended to replace professional medical diagnosis."
}
```

### `GET /api/analysis-options`

The pipelines, model choices and defaults the analysis form should offer.

---

## Analysis

### `POST /api/analysis` — multipart/form-data

| Field | Required | Notes |
|---|---|---|
| `files` | yes | One image for `single_channel`; four distinct images (or one four-channel `.npz`) for `four_channel` |
| `analysis_type` | no | `single_channel` (default) or `four_channel` |
| `model_ids` | no | Repeatable; defaults to every usable model for the pipeline |
| `scan_identifier` | no | Pseudonymous study code; no identifying data is accepted or stored |
| `notes` | no | Free text, stored with the analysis |
| `threshold` | no | Mask threshold, default `0.5` |
| `enable_ensemble` | no | `true`/`false`, default `true` |
| `ensemble_strategy` | no | `equal`, `calibrated` or `confidence_aware` |
| `enable_radiomics` | no | Default `true` |
| `enable_explainability` | no | Default `true` |
| `enable_uncertainty` | no | Default `false`; costs extra passes |
| `radiomics_mask` | no | `best`, `primary` or `ensemble` |

Returns **201** with the canonical analysis payload (the same object stored as
`data/artifacts/<analysis_id>/analysis.json`):

```json
{
  "analysis_id": "an-20261013-e8ccde6b6e",
  "scan_identifier": "DEMO-001",
  "analysis_type": "single_channel",
  "created_at": "2026-10-13T07:38:22+00:00",
  "device": "cpu",
  "threshold": 0.5,
  "preprocessing": { "pipeline": "single_channel_512", "tensor_shape": [1,1,512,512], "channels": 1,
                     "spatial_size": [512,512], "normalisation": "[0,1] intensity scaling (image / 255), bilinear resize",
                     "metadata": { ... } },
  "inputs": { "Grayscale MRI": "input_0_grayscale_mri.png" },
  "models": { "unet_dynamic_512": { "available": true, "status": "completed", "summary": { ... },
                                    "artifacts": { "mask": "mask_unet_dynamic_512.png", "overlay": "overlay_unet_dynamic_512.png", "probability": "prob_unet_dynamic_512.png" },
                                    "health": { "verdict": "responsive", "message": "…" },
                                    "descriptor": { ... }, "is_reference_validated": true } },
  "ensemble": { "available": false, "strategy": "equal", "weights": {}, "members": [], "reasons": ["Fewer than two segmentation checkpoints are usable; an ensemble needs at least two."] },
  "uncertainty": { "available": true, "method": "test_time_augmentation", "n_passes": 5, "mean_dispersion": 0.0153, ... },
  "reference_mask": { "source": "model:unet_dynamic_512", "model_id": "unet_dynamic_512", "foreground_pixels": 5556 },
  "region_measurements": {
    "detected": true, "pixel_area": 5556, "image_coverage": 0.0212, "image_coverage_percent": 2.12,
    "bounding_box": { "x": 121, "y": 96, "width": 137, "height": 148 },
    "centroid": { "x": 191.42, "y": 168.07 },
    "perimeter_px": 712.43, "equivalent_diameter_px": 84.11,
    "units": "pixels", "image_size": { "width": 512, "height": 512 },
    "mask_source": "model:unet_dynamic_512",
    "note": "Measurements describe the region the segmentation model produced. …"
  },
  "radiomics": { "available": true, "backend": "native", "dimensionality": "2D", "feature_count": 68, "spacing_available": false, "features": { ... }, "categories": { ... }, "quality": { ... }, "warnings": ["…"] },
  "explainability": [ { "model_id": "unet_dynamic_512", "available": true, "method": "grad_cam", "target_layer": "conv9_2", "heatmap_mean": 0.28, "artifacts": { "heatmap": "gradcam_unet_dynamic_512.png", "overlay": "gradcam_overlay_unet_dynamic_512.png" } } ],
  "classification": { "available": false, "reason": "No validated classifier artefact was found in …" },
  "model_comparison": { "state": "unlabelled", "note": "…", "models": [ ... ] },
  "artifacts_root": "artifacts/an-20261013-e8ccde6b6e",
  "report": { "available": true, "report_id": "an-20261013-e8ccde6b6e-report", "relative_path": "reports/…/…-report.pdf", "size_bytes": 618715, "generator_version": "reportgen-1.0" },
  "disclaimer": "Research prototype. …"
}
```

`region_measurements` is measured from the reference mask on every analysis
(`backend/app/ml/region_metrics.py`): pixel area, image coverage, the tightest bounding box
(`x`, `y`, `width`, `height`), the centroid, the boundary length and the equivalent diameter,
each in **pixels**. When nothing crosses the threshold, `detected` is `false` and the geometry
is `null` — never `0`. The same block is rendered in the PDF report and quoted by the
assistant; `GET /api/radiomics/analysis/<id>` additionally returns `roi_centroid` (per mask
axis, same order as `roi_bounding_box`) inside `quality`.

### `POST /api/analysis/plan`

Same body as the analysis request (files excluded). Returns what *would* run, with a
`runnable` flag, the health verdict and the reason for each step, plus the ensemble
compatibility, radiomics, explainability, uncertainty and classification availability.
The dashboard shows this before the user commits to a run.

### `GET /api/analysis/<analysis_id>`

```json
{ "metadata": { "analysis_id": "...", "status": "completed", "duration_seconds": 2.57, "model_ids": ["unet_dynamic_512"], ... },
  "stored_records": { "segmentation": [...], "radiomics": [...], "classification": {...}, "explainability": [...], "uncertainty": {...}, "reports": [...] },
  "payload_available": true,
  "payload_source": "artifacts/analysis.json",
  "result": { ... the canonical payload ... } }
```

`result` is `null` only when neither the artefact nor the database can reconstruct it;
`payload_source` says which of the two was used.

### `DELETE /api/analysis/<analysis_id>`

Removes the database rows, the uploaded input, the derived artefacts and the report.

### Artefacts

| Endpoint | Returns |
|---|---|
| `GET /api/analysis/<id>/artifacts/<filename>` | A PNG artefact (`mask_*.png`, `overlay_*.png`, `prob_*.png`, `gradcam_*.png`, `ensemble_*.png`, `input_*.png`) |
| `GET /api/analysis/<id>/inputs/<index>` | The original uploaded file |
| `GET /api/analysis/<id>/preview/<index>` | A PNG preview of an uploaded input |

Filenames are validated against the analysis directory; traversal attempts are
rejected with `not_found`.

---

## Models

| Endpoint | Purpose |
|---|---|
| `GET /api/models` | Every registered checkpoint with its descriptor, status, parameter count and load time |
| `GET /api/models/<model_id>` | One descriptor plus its health report and its explainability status |
| `GET /api/models/<model_id>/health?force=true` | Re-runs the deterministic reference check for one model |
| `POST /api/models/health-check` `{"force": false}` | Runs the reference check for every model and returns the summary |
| `POST /api/models/compare` | Runs several models on the same upload in one request and returns each model's probability, mask and overlay as data URLs |

A health report contains `verdict` (`responsive`, `no_foreground`, `saturated`,
`degenerate`, `failed`, `not_loaded`), `responsive`, the per-check numbers
(raw min/max/mean, probability min/max/mean/spread, foreground fraction at threshold)
and a `recommendation`. Verdicts come from deterministic phantoms, not from held-out
data, so they prove that a checkpoint *works*, never that it is *accurate*.

---

## Radiomics

| Endpoint | Purpose |
|---|---|
| `GET /api/radiomics/catalogue` | Backend status, feature categories, definitions and discretisation settings |
| `GET /api/radiomics/analysis/<analysis_id>` | The stored feature table for an analysis, with categories, quality flags and per-feature descriptions |
| `GET /api/radiomics/analysis/<analysis_id>/features.csv` | The same table as CSV |
| `POST /api/radiomics/extract` | Ad-hoc extraction from an uploaded image plus mask |

Feature values are numbers or explicit `null`. Shape features that need physical
spacing (`shape_equivalent_diameter_mm`, `shape_sphericity`, …) are `null` when the
input carries no pixel spacing, and `spacing_available` says so; pixel-domain features
are always present.

---

## History

| Endpoint | Purpose |
|---|---|
| `GET /api/history?search=&status=&analysis_type=&model_id=&date_from=&date_to=&limit=&offset=&order=&direction=` | Paginated, filtered list plus `filters` (valid statuses, types and models) |
| `GET /api/history/statistics` | Totals, recent items, status counts, model counts, evaluation state |
| `GET /api/history/<analysis_id>/export` | Full JSON export of the stored analysis |

---

## Reports

| Endpoint | Purpose |
|---|---|
| `GET /api/reports` | Every stored report with its analysis, scan identifier, size and creation time |
| `GET /api/reports/<report_id>` | The PDF (`application/pdf`) |
| `POST /api/reports/analysis/<analysis_id>` | Regenerates the report from the stored payload; returns **201** with the new metadata |

The report has fourteen sections, renders unavailable values as text rather than as
zero, and always ends with the limitations and disclaimer pages.

---

## Evaluation

### `GET /api/evaluation`

```json
{
  "evaluation": { "state": "not_evaluated", "reason": "No evaluation artefact was found …",
                  "expected_artifact": "…/data/models/evaluation_results.json",
                  "produced_by": "scripts/evaluate_models.py" },
  "model_selection": { "state": "not_evaluated", "produced_by": "scripts/evaluate_models.py --select", ... },
  "segmentation": { "state": "not_evaluated", "models": [ { "model_id": "…", "metrics": null, "status": "not_evaluated" } ] },
  "classification": { "state": "not_evaluated", "reason": "…" },
  "runtime": { "state": "measured", "models": [ { "model_id": "…", "seconds_per_image": 0.52, "parameter_count": 1940945 } ] },
  "model_health": { "phantom_version": "phantom-v1", "models": [ ... ], "responsive_models": ["unet_dynamic_512"] },
  "training_runs": [],
  "protocol": { "splits": ["…"], "principles": ["…"], "scripts": { "evaluate": "…" } }
}
```

When the artefact exists, `evaluation.segmentation` holds per-case metrics, aggregates
with bootstrap confidence intervals and a comparison table, and `runtime` is populated
from the same run. `scripts/evaluate_models.py` writes that file; the API never
computes a metric from unlabelled predictions.

---

## Assistant

| Endpoint | Purpose |
|---|---|
| `GET /api/assistant/status` | Which backend is active, whether a key is present, suggested questions and the disclaimer |
| `POST /api/assistant/ask` `{"analysis_id": "…", "question": "…"}` | An answer grounded in the stored analysis |
| `GET /api/assistant/history/<analysis_id>` | Stored conversation for that analysis |

```json
{ "answer": "Segmentation assigns a tumour/abnormality probability to every pixel …",
  "topic": "segmentation",
  "sources": ["models.unet_dynamic_512.summary.foreground_pixels", "…"],
  "grounding": { "…": "…" },
  "backend": "local",
  "suggestions": ["…"],
  "disclaimers": ["This is a research prototype. …"],
  "scope": "Answers describe this analysis only. Educational statements are marked as such.",
  "analysis_id": "an-20261013-e8ccde6b6e" }
```

Questions about malignancy, clinical grading, staging, prognosis, survival, biopsy or
treatment are refused (`topic: "out_of_scope"`) before any backend runs. Questions about the
stored dataset label ("which class did the classifier predict?") are answered from the stored
classification record (`topic: "classification"`), and the answer states that the label is the
dataset's own annotation and not a clinical diagnosis; when no validated artefact exists the
answer says that no class was predicted instead of producing one. If a language-model backend is
enabled it receives a reduced, non-identifying summary; its key is read from the environment and
is never stored, logged or returned.

---

## Tumour classification

### `GET /api/classification`

```json
{
  "status": {
    "available": true,
    "model_id": "radiomics_grade_classifier",
    "version": "radiomics_grade_classifier-20261013T091500",
    "label_mapping": { "0": "HGG", "1": "LGG" },
    "feature_count": 68,
    "feature_schema": ["firstorder_mean", "…"],
    "preprocessing": { "type": "standard_scaler", "fit_on": "training split only", "imputation": "training-split median" },
    "metrics": { "accuracy": 0.0, "macro_f1": 0.0, "roc_auc": 0.0, "per_class": {}, "confusion_matrix": [], "evaluated_model": "…", "evaluation_split": "held-out test split", "samples": 0 },
    "model_comparison": { "baseline_logistic_regression": { "validation": {}, "test": {} }, "improved_mlp": { "validation": {}, "test": {} } },
    "splits": { "train": { "HGG": 0, "LGG": 0 } },
    "caveats": ["…"],
    "trainer": "scripts/train_classification.py"
  },
  "evaluation": { "state": "evaluated", "comparison": { "…": "…" } },
  "label_semantics": {
    "target": "BraTS 2020 HGG vs LGG (WHO grade group)",
    "explanation": "The predicted class is the dataset label … not a malignancy verdict, not a clinical stage and not a prognosis",
    "derived_from_mask": false,
    "note": "The mask defines the region the features are measured in; the class comes from the label table."
  },
  "xai": { "method": "coefficient x standardised value, or single-feature substitution", "shap": { "available": false, "reason": "ModuleNotFoundError", "fallback": "…" } },
  "training": { "recommended": "scripts/train_classification.py", "pipeline": ["…"], "requirements": ["…"] },
  "disclaimer": "Research prototype. …"
}
```

When no servable artefact exists, `status.available` is `false`, `status.reason` names the
directory that was searched, `status.required_artifacts` lists what an artefact must contain, and
`status.legacy_checkpoint_note` explains why the shipped `radiomics_cnn_model.pth` is not used
(21 declared features, 11 available, no scaler, no feature order). The response never contains a
placeholder label.

Per-analysis results appear inside `GET /api/analysis/<analysis_id>`:

```json
{
  "classification": {
    "available": true,
    "model_id": "radiomics_grade_classifier",
    "predicted_label": "HGG",
    "predicted_index": 0,
    "scores": [0.61, 0.39],
    "calibrated": false,
    "metrics": { "accuracy": 0.0, "macro_f1": 0.0 },
    "evaluated_on": "held-out test split",
    "model_comparison": { "…": "…" },
    "caveats": ["…"],
    "feature_contributions": [
      { "feature": "glcm_contrast", "value": 12.4, "weight": 0.08, "contribution": 0.42, "method": "linear coefficient x standardised value" }
    ],
    "missing_features": [],
    "statement": "Predicted class according to the trained model …"
  }
}
```

An incomplete vector is refused with `available: false`, a `reason` and the `missing_features`
list: the platform never pads, re-orders or drops features to force a prediction.

---

### `GET /api/classification/image/models`

The image-classification catalogue: one model card per Keras artefact, with `state`
(`ready`, `checkpoint_missing`, `metadata_missing`, `framework_missing`, `incompatible`,
`error`), the exact paths searched, the class list and order read from `metadata.json`, the
input size, the recorded preprocessing, the metrics the training run reported (labelled as
reported) and the trained parameter count once the model has been loaded.

```json
{
  "models": [
    {
      "model_id": "brain_tumor_xception",
      "display_name": "Tumour type (Xception, four classes)",
      "task": "image_classification",
      "state": "checkpoint_missing",
      "available": false,
      "reason": "brain_tumor_xception_final.keras is not installed. Place it - together with metadata.json - in /…/models/weights (see models/README.md).",
      "model_path": "/…/models/weights/brain_tumor_xception_final.keras",
      "metadata_file": "/…/models/weights/metadata.json",
      "framework_available": null,
      "classes": [],
      "reported_metrics": null,
      "metrics_provenance": "These numbers are the metrics the training notebook recorded…"
    }
  ]
}
```

### `POST /api/classification/image` — multipart/form-data

| Field | Required | Notes |
| --- | --- | --- |
| `file` | yes | One MRI image: `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tif`, `.tiff`, `.webp`; RGB or grayscale (converted), at least 16 × 16 |

Runs the Keras Xception model over the image itself (not radiomics features, not a mask) and
returns the four softmax values in the order `metadata.json` declares, as
`predicted_index`, `predicted_label`, `probability`, `probabilities` and a `scores` list. The
response also carries `classes_source` (the metadata file the order came from), `calibrated:
false` with an explanation, the resize that was applied, the preprocessing name and its
implementation, `inference_seconds`, the recorded metrics of the training run with a provenance
sentence, and `label_semantics` (tumour type, **not** grade, malignancy or stage).

Errors: `422` when no file is attached, `415` when the suffix is not an accepted image format or
the bytes cannot be decoded, `409` when the checkpoint, its metadata or a compatible
preprocessing declaration is missing, `503` when TensorFlow is not installed. A prediction is
never returned without the model, and the endpoint never falls back to another model.

## Static assets

`GET /` returns the dashboard shell when `frontend/dist` exists, and a JSON index
describing the API otherwise. Unknown non-API paths fall back to the shell so client
routing works on refresh.
