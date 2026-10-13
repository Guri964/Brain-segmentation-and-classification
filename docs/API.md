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
  "radiomics": { "available": true, "backend": "native", "dimensionality": "2D", "feature_count": 68, "spacing_available": false, "features": { ... }, "categories": { ... }, "quality": { ... }, "warnings": ["…"] },
  "explainability": [ { "model_id": "unet_dynamic_512", "available": true, "method": "grad_cam", "target_layer": "conv9_2", "heatmap_mean": 0.28, "artifacts": { "heatmap": "gradcam_unet_dynamic_512.png", "overlay": "gradcam_overlay_unet_dynamic_512.png" } } ],
  "classification": { "available": false, "reason": "No validated classifier artefact was found in …" },
  "model_comparison": { "state": "unlabelled", "note": "…", "models": [ ... ] },
  "artifacts_root": "artifacts/an-20261013-e8ccde6b6e",
  "report": { "available": true, "report_id": "an-20261013-e8ccde6b6e-report", "relative_path": "reports/…/…-report.pdf", "size_bytes": 618715, "generator_version": "reportgen-1.0" },
  "disclaimer": "Research prototype. …"
}
```

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

Questions about malignancy, grade, staging, prognosis, survival, biopsy or treatment
are refused (`topic: "out_of_scope"`) before any backend runs. If a language-model
backend is enabled it receives a reduced, non-identifying summary; its key is read
from the environment and is never stored, logged or returned.

---

## Static assets

`GET /` returns the dashboard shell when `frontend/dist` exists, and a JSON index
describing the API otherwise. Unknown non-API paths fall back to the shell so client
routing works on refresh.
