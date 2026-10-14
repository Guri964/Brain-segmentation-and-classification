# IMPLEMENTATION PLAN

## 0. Scope decision taken after the audit

The audit established three hard constraints that shape everything below:

1. **No labelled dataset is present** (no BraTS slices, no masks, no grade labels, no splits).
   Therefore no training, no accuracy metric and no validated classifier can be produced in this
   environment. All training/evaluation code is delivered as reproducible entry points, and every
   user-visible metric is either a real measurement on real inputs or the literal string
   `Not evaluated`.
2. **The two usable checkpoints are not input-compatible** (1 x 512 x 512 vs 4 x 256 x 256).
   The ensemble therefore cannot be produced from the shipped artifacts alone. It is fully
   implemented, unit-tested on synthetic tensors, and enabled the moment an input-compatible
   partner checkpoint exists (`scripts/train_unet.py` produces it).
3. **The radiomics classifier checkpoint is incompatible with the shipped feature table**
   (21 expected inputs vs 11 columns) and has no label mapping. It is preserved, documented and
   gated off.

## 1. Phase 1 - Audit (done)

Deliverables: `docs/PROJECT_AUDIT.md`, `docs/CURRENT_ARCHITECTURE.md`, this file.
Status: complete; evidence recorded per checkpoint, per defect and per reuse decision.

## 2. Phase 2 - Model and data pipeline

| Task | File(s) | Status |
| --- | --- | --- |
| Re-implement `SegNet` and `DynamicUNet` with identical parameter names/keys | `backend/app/ml/architectures/segnet.py`, `dynamic_unet.py` | to implement |
| Checkpoint registry with architecture probing, `input_spec`, `output_semantics`, checksum, device mapping, lazy caching | `backend/app/ml/checkpoint_registry.py` | to implement |
| New architecture: residual attention U-Net with attention gates, deep supervision, boundary-aware loss (`RAUNet`) | `backend/app/ml/architectures/attention_unet.py`, `losses.py` | to implement |
| Preprocessing: single-channel `[0,1]` pipeline and four-channel z-score pipeline, with validation | `services/preprocessing_service.py` | to implement |
| Segmentation inference for both models, TTA, MC-dropout | `services/segmentation_service.py` | to implement |
| Ensemble: equal weights, validation-calibrated weights, confidence-aware fusion | `services/ensemble_service.py` | to implement |
| Radiomics: IBSI-aligned native extractor (2-D and 3-D) with optional PyRadiomics backend | `services/radiomics_service.py`, `ml/radiomics_native.py` | to implement |
| Uncertainty: TTA variance, MC-dropout, ensemble disagreement, calibration (ECE) | `services/uncertainty_service.py` | to implement |
| Explainability: correct Grad-CAM for each architecture, permutation attribution for tabular models | `ml/gradcam.py`, `services/explainability_service.py` | to implement |
| Classification: gated service, explicit incompatibility report, training entry point | `services/classification_service.py`, `scripts/train_classifier.py` | to implement |

## 3. Phase 3 - Backend

Flask application factory with blueprints, structured JSON errors, request-size limits, safe
file handling, environment-based configuration, logging, and SQLite persistence with a
migration runner. Endpoints are listed in `docs/API.md`.

Order: `config` -> `errors`/`logging` -> `utils` (images, metrics, io) -> `database` ->
`ml` -> `services` -> `routes` -> `run.py`.

## 4. Phase 4 - Frontend

React + TypeScript + Tailwind dashboard replacing Streamlit, eleven pages, dark/light theme,
toast notifications, skeletons, empty/error states, responsive viewers, Recharts visualisations
and a connected assistant. No inference logic in the browser; every value displayed comes from
an API response.

## 5. Phase 5 - Research improvements (only where technically justified)

Selected: (a) RAUNet with attention gates + deep supervision + boundary-aware objective;
(b) confidence-aware ensemble fusion with validation-derived reliability; (c) segmentation-guided
radiomics + deep-feature fusion classifier; (d) uncertainty estimation via TTA/MC-dropout/ensemble
disagreement. Each has a baseline, an ablation switch (`--no-attention`, `--no-deep-supervision`,
`--loss bce_dice`), and an evaluation entry point. **No improvement is claimed as proven**: the
ablation runners exist, the numbers do not, because the data does not. See
`docs/ALGORITHM_NOVELTY.md`.

## 6. Phase 6 - Integration and verification

* `pytest` suite covering checkpoint loading, preprocessing, tensor shapes, both segmentation
  models, fusion maths, radiomics on analytic phantoms, Grad-CAM, uncertainty, database CRUD,
  report generation and every API route.
* End-to-end run: upload -> preprocess -> inference -> radiomics -> Grad-CAM -> persist -> PDF.
* CPU verification in this environment (no GPU available); CUDA paths are written and tested for
  device-selection logic only.

## 7. Phase 7 - Documentation

`README.md`, `HOW_TO_RUN.md`, `docs/API.md`, `docs/DATASET.md`, `docs/LIMITATIONS.md`,
`docs/ALGORITHM_NOVELTY.md`, updated `backend/requirements.txt`, updated `install.bat`.

## 8. Open questions / blockers requiring the client

1. **The four DOCX/PDF documents are not in the repository.** They must be supplied (or placed in
   `docs/source/`) so that wording, required deliverables and any figure the client expects can be
   cross-checked against the implementation. Everything implemented so far is traceable to the
   written task specification, the notebook and the code.
2. **Dataset and labels.** Provide BraTS 2020 (or the company's own four-modality cohort) with
   patient identifiers and reference masks, plus grade labels if grade classification is required.
   Until then the segmentation metrics page reports "Not evaluated" and the classifier is disabled.
3. **Which single-channel dataset trained `UNet-[16, 32, 64, 128, 256].pt`?** The repository has no
   record of it, so the U-Net's input distribution (modality, intensity window) is documented as
   "grayscale 512 x 512, `[0,1]`, trained elsewhere". If the original dataset is available it
   should be supplied for a fair re-evaluation.
4. **Should the new four-channel model replace SegNet or complement it?** Default: complement
   (ensemble partner). Confirm whether the company wants the ensembled output to be the primary
   deliverable.

## 9. Proposed implementation order (as executed)

1. `backend/app/{config,errors,logging,utils}` and the Python environment check.
2. `backend/app/ml/architectures` + registry + checkpoint verification test.
3. `services/preprocessing` + `services/segmentation` + tests.
4. `services/ensemble`, `services/uncertainty` + tests.
5. `services/radiomics` + phantom tests.
6. `services/explainability`, `services/classification` + tests.
7. `database` + `services/analysis` orchestrator + `report_service`.
8. `routes` + API tests.
9. `frontend` build + API integration.
10. `scripts` (training/evaluation/ablation/calibration) + smoke tests.
11. Documentation and final end-to-end verification.
