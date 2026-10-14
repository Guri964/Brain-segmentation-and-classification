# Requirements traceability

The client supplied four PDFs. They are now part of the repository (they were added by the
client on `main` and are also available to this branch):

| File | Content |
| --- | --- |
| `1.pdf` | "Brain Tumor Detection and Analysis Using Deep Learning, Radiomics, and Explainable Artificial Intelligence" — early overview |
| `2.pdf` | "A Deep Ensemble-Driven Framework Integrating Radiomics and Segmentation-Based Attention Mechanisms…" — conference paper |
| `3.pdf` | **Project design document** — "A Deep Stacked Ensemble Framework for Brain Abnormality Prediction Using Radiomics" (sections 1–8) |
| `4.pdf` | **Modifications required** — items NO 1 – NO 8 and the frontend request |

This file is the audit trail for those two documents: for every requirement, where it lives
in the code, how a reviewer can reach it, which test covers it, and what the honest
limitation is. Nothing here claims a result the repository cannot produce; every limit is
stated in the row where it applies.

Status legend: **implemented** (reachable and tested), **implemented with a documented
limit** (works, but the row says exactly what it refuses to claim), **blocked by missing
data** (the code refuses to run rather than guessing).

---

## A. Project design document (`3.pdf`)

| § | Requirement | Implementation | Reachable at | Test |
| --- | --- | --- | --- | --- |
| 1 | Functional mapping of the supplied files: `final.py`, `model.py`, `bts/model.py`, `bts/classifier.py`, `bts/loss.py`, notebook, `features_df.csv`, checkpoints, `install.bat`/`req.txt` | The supplied code is kept unchanged and reused as the *source* of the architecture definitions (`backend/app/ml/architectures/segnet.py`, `dynamic_unet.py`). The Streamlit app (`final.py`) is superseded by the React/Tailwind application, and the legacy checkpoints are loaded by `backend/app/ml/registry.py` with their real input specifications. | `GET /api/models`, `GET /api/models/<model_id>` | `tests/test_registry_and_health.py` |
| 3 | Layered architecture: input → preprocessing → segmentation → ensemble → radiomics → explainability → application | One service per layer: `preprocessing_service`, `segmentation_service`, `ensemble_service`, `radiomics_service`, `explainability_service`, `uncertainty_service`, `classification_service`, `report_service`. | `POST /api/analysis` (all layers in one call), `POST /api/models/compare`, `POST /api/radiomics/extract` | `tests/test_api.py`, `tests/test_inference_and_ensemble.py` |
| 4 | End-to-end workflow: metadata merge → tumour slices → splits → normalisation/binarisation → per-model training → probability averaging → radiomics → neural feature model → Grad-CAM | Notebook (`notebooks/brain_tumor_training_kaggle_t4.ipynb`, generated from `notebooks/_notebook_source/`) for training; scripts under `scripts/`; the same algorithms are served by the API. Ensemble = arithmetic mean of sigmoid probabilities (`P = (P_SegNet + P_U-Net)/2`). | notebook, `POST /api/analysis` with `enable_ensemble=true` | `tests/test_kaggle_notebook.py`, `tests/test_inference_and_ensemble.py` |
| 5.1 | Dataset preparation and label mapping (BraTS 2020 HDF5, `meta_data.csv` + `name_mapping.csv`, HGG=0/LGG=1) | `scripts/prepare_brats.py`, `scripts/make_splits.py`, notebook sections; the split is **patient-level**, not slice-level (slice-level splitting leaks between train and validation). | `python scripts/prepare_brats.py --help` | `tests/test_brats_pipeline.py` |
| 5.2 | Image/mask preprocessing: four-channel tensor, resize to 256, mean/standard-deviation normalisation, binary mask (all non-zero labels = abnormal tissue) | `read_slice` (whole-tumour mask `mask > 0`), `resize2d` (bilinear image, nearest mask), `zscore_channels`, tensors `(S, 256, 256, 4)` with the channel order T1, T1ce, T2, FLAIR preserved. | notebook cache; `POST /api/analysis` with `analysis_type=four_channel` | `tests/test_preprocessing.py`, `tests/test_data_pipeline.py` |
| 5.3 | SegNet: 2 encoder stages (64, 128), bilinear decoder, one-channel logits, sigmoid applied outside the model | `backend/app/ml/architectures/segnet.py`; the shipped `segnet_model.pth` is loaded with `output_semantics="logit"`. | `GET /api/models/segnet_brats_4ch` | `tests/test_registry_and_health.py` |
| 5.4 | U-Net variants: MONAI U-Net (4 channel, 16/32/64/128) in the supplied notebook; Dynamic U-Net `[16,32,64,128,256]` in the application | The notebook trains **two** U-Net members rather than the MONAI one: `unet_dynamic_512_real_brats` (the Dynamic U-Net that the application already loads, retrained on 4-channel BraTS slices so training and deployment share one architecture) and `raunet_brats_4ch` (residual attention U-Net, the research model). The application serves the shipped `UNet-[16, 32, 64, 128, 256].pt` as `unet_dynamic_512` (1 channel, 512 × 512). | `GET /api/models/unet_dynamic_512`, `GET /api/models/raunet_brats_4ch` | `tests/test_registry_and_health.py`, `tests/test_kaggle_notebook.py` |
| 5.5 | Composite objective `L = α·Dice + (1−α)·BCE`, α = 0.6 | `backend/app/ml/losses.py` (`BCEDiceLoss`), used by both the notebook trainer and `scripts/train_*.py`. | notebook cell "Composite loss" | `tests/test_kaggle_notebook.py` |
| 5.6 | Deep ensemble mask: sigmoid → arithmetic mean → threshold | `backend/app/services/ensemble_service.py` (weights, member compatibility, disagreement metrics, per-pair agreement). | `POST /api/analysis` → `ensemble` block, `POST /api/models/compare` | `tests/test_inference_and_ensemble.py` |
| 5.7 | PyRadiomics extraction from the ensemble mask through SimpleITK, all feature families enabled, NaNs → 0 | `backend/app/services/radiomics_service.py` (PyRadiomics when installed) with a documented native fallback (`backend/app/ml/radiomics_native.py`) that computes the same families in NumPy. Spacing is honoured when supplied; without spacing, features are pixel-based and that is stated. | `GET /api/radiomics/analysis/<id>`, `.../features.csv`, `GET /api/radiomics/catalogue` | `tests/test_radiomics.py` |
| 5.8 | Compact fully connected model over the radiomics vector (`radiomics_cnn_model.pth`, 21 → 128 → 64 → 2) | **Blocked by missing data.** The checkpoint expects 21 features; `features_df.csv` ships 11 columns, the notebook's `labels_subset` is undefined, and no feature schema, scaler or metric was saved. The registry reports it as incompatible and the serving path refuses it. | `GET /api/capabilities` → `classification.legacy_checkpoint_note` | `tests/test_classification_pipeline.py`, `tests/test_registry_and_health.py` |
| 5.9 | Grad-CAM on the final convolutional layer: hooks, channel weights by global averaging, ReLU, resize, colour map, blend | `backend/app/ml/gradcam.py` + `explainability_service` with a per-model target layer and layer semantics (SegNet `encoder.4`, U-Net `conv9_2`, RAUNet attention block). | `POST /api/analysis` with `enable_explainability=true`, UI page *Explainability* | `tests/test_explainability_uncertainty.py` |
| 5.10 | Deployment: CPU/GPU detection, saved weights, upload, two analysis tabs (segmentation overlay, Grad-CAM) | Superseded by the React application: 12 pages, no manual model wiring, plus everything Streamlit did not expose (ensemble, radiomics, classification, uncertainty, PDF report). | `frontend/` (`npm run dev`), `backend/run.py` | `tests/test_frontend_integration.py` |
| 6 | Table 2 — "what the notebook does vs what the application exposes" | Every gap listed in Table 2 is closed: four-channel inference, SegNet inference, ensemble, radiomics, radiomics model wiring (refused with a reason because the artefact is unusable), Grad-CAM, interactive visualisation. The only remaining difference is deliberate: the *single-channel* upload path uses the one model that legitimately accepts a grayscale PNG (duplicating one channel into four is refused). | UI *Segmentation*, *Model comparison*, *Radiomics*, *Classification*, *Explainability* pages | `tests/test_api.py`, `tests/test_inference_and_ensemble.py` |
| 7 | Recommended final integrated operation | Shipped: same four-channel representation, both segmentation networks, fused mask, radiomics from that mask, a clearly defined supervised classifier (enabled only with a validated artefact), and a UI that shows the decision/grade, confidence, fused segmentation, radiomics and Grad-CAM. | end-to-end run | `tests/test_api.py::test_single_channel_analysis_end_to_end` |

### Deliberate deviations from the design document (and why)

| Document | Repository | Reason |
| --- | --- | --- |
| "limits loading to 2,000 slices", `train_test_split` over slices | full patient volumes, patient-level stratified splits | Slice-level splitting puts slices of the same patient in different splits; every reported metric would then be optimistic. |
| Grade label described as "tumour / no tumour" in the notebook comment | HGG/LGG grade group only | The BraTS `Grade` column is the WHO grade group, not a malignancy verdict, stage or prognosis. |
| Physical measurements from a 2-D slice | pixel units unless spacing is supplied | A 2-D slice has no voxel spacing in these HDF5 files; reporting cm² would be an invention. |
| `features_df.csv` used as a feature table | kept as an archival artefact, not used for training | The extraction geometry in the notebook stacks the four modalities on the z axis and replicates the 2-D mask, so the values are not radiomics of a lesion (see `docs/PROJECT_AUDIT.md` §3.4). |
| MONAI U-Net (notebook cell 25: 4 channels, 16/32/64/128, `num_res_units=1`) | Dynamic U-Net (`unet_dynamic_512_real_brats`) + residual attention U-Net | The application can only serve architectures whose definition it ships and whose checkpoint it can load strictly. Training a 4-channel Dynamic U-Net keeps the ensemble member loadable by `backend/app/ml/registry.py` instead of introducing an artefact nothing can execute. MONAI is therefore not a dependency of this repository. |

---

## B. Modifications required (`4.pdf`)

| Item | Requirement | Implementation | Evidence |
| --- | --- | --- | --- |
| NO 1 | Tumour **classification**: which type, and "what stage / malignant or not" | Implemented for the label that actually exists — the WHO **grade group** (HGG/LGG) — with an explicit refusal of malignancy, clinical stage and prognosis, plus a 4-class image-classification path for the second dataset. The serving path stays disabled until a validated artefact (feature schema + scaler + held-out metrics) exists. | `docs/CLASSIFICATION.md`; `GET /api/classification`; UI *Tumour classification* (the page states "No class is reported as malignant, benign or staged"); `tests/test_classification_pipeline.py` |
| NO 1b | Tumour **type** classification from the image (`firstnb.ipynb`: Xception, four classes glioma / meningioma / pituitary / no tumour) | Served by `backend/app/ml/image_classifier.py` + `image_classification_service.py`: the class order is read from `metadata.json` (never hard-coded), the preprocessing is the Xception scaling verified bit-identical to `keras.applications.xception.preprocess_input`, the checkpoint's output width and input shape must agree with the metadata before an image is scored, and the recorded metrics are labelled as the training run's. The two artefact files are not in this deployment, so the model reports `checkpoint_missing` and the endpoint answers 409 instead of a label. | `POST /api/classification/image`, `GET /api/classification/image/models`, UI *Classification* page -> "Tumour type from the image" | `tests/test_image_classifier.py` (state machine, metadata contract, preprocessing, serving path incl. a subprocess end-to-end run), `tests/test_artifact_paths.py` |
| NO 2 | Explainable AI or chatbot that explains the finding and the reasons | Grounded assistant (`/api/assistant/ask`, `/api/assistant/history/<id>`): answers are assembled from the stored payload of that analysis (region measurements, radiomics values, model status, uncertainty, limitations), refuses out-of-scope questions, and never invents a reason. Grad-CAM answers "which pixels made this mask", not "why is this a tumour". | UI *Clinical support*; `tests/test_assistant.py`; `docs/LIMITATIONS.md` |
| NO 3 | Radiomics dashboard: shape / intensity / texture / diagnostics visible in the UI, with charts | Per-family feature tables with definitions and units, per-category charts, ROI panel (voxel count, extent, bounding box, **centroid**, relative size, border contact), CSV/JSON export, and an honest "no spacing supplied → pixels" note. | UI *Radiomics*; `GET /api/radiomics/*`; `tests/test_radiomics.py` |
| NO 4 | Actually use the SegNet + U-Net ensemble: `MRI → U-Net / SegNet → probability → average → final mask`, shown side by side | `ensemble_service` runs every compatible member on the identical preprocessed tensor, averages probabilities with explicit weights, reports per-model outputs, per-pair Dice/IoU agreement and the disagreement map, and refuses incompatible pairs with a reason. The UI has side-by-side mask, overlay and probability panels plus the ensemble panel. | UI *Model comparison*; `POST /api/models/compare`; `tests/test_inference_and_ensemble.py` |
| NO 5 | Uncertainty estimation ("confidence" and "segmentation confidence") | Uncertainty is computed, never asserted: ensemble disagreement, test-time augmentation and MC-dropout dispersion, with a per-pixel dispersion map. The UI shows it only when at least two members or a stochastic method produced it; otherwise it says why not. The word "confidence" is never used for an uncalibrated softmax. | UI *Segmentation* / *Model comparison*; `tests/test_explainability_uncertainty.py` |
| NO 6 | Tumour/abnormality measurements: pixel area, bounding box, width, height, image coverage, centroid, perimeter, shape descriptors | `region_measurements` is computed from the reference mask on every analysis by `backend/app/ml/region_metrics.py`: detected flag, pixel area, coverage (% of image), bounding box (x, y, width, height), centroid (x, y), boundary length, equivalent diameter, pointing at the mask source. Shape descriptors (circularity, elongation, sphericity where defined) come from the radiomics shape family. | UI *Segmentation* → "Abnormal region analysis", UI *Radiomics* → ROI panel; PDF report §3; `tests/test_region_measurements.py` |
| NO 7 | Automatic medical-style report with a download and a research-prototype disclaimer | `report_service` renders a PDF per analysis: inputs, model status and provenance, segmentation panels, region measurements, ensemble, radiomics table, classification (or the reason it is unavailable), Grad-CAM, uncertainty, evaluation state, limitations and the disclaimer. Downloadable from the UI and `/api/reports/<report_id>`. | UI *Reports*; `tests/test_reports.py` |
| NO 8 | "If possible, add a database" | SQLite schema with migrations (schema version 2) recording analyses, segmentation results, radiomics, classification, explainability, uncertainty, reports, assistant messages and model versions; history, statistics and JSON export endpoints back the *History* page. | `backend/app/database/`; `tests/test_database.py`; UI *History* |
| — | "Instead of Streamlit use some other frontend like Tailwind with a good interactive UI" | React 18 + TypeScript + Tailwind 3 + Recharts single-page application, dark/light, keyboard accessible, served by Flask in production. | `tests/test_frontend_integration.py`; `npm run build` |
| — | "In the image the text is also inverted… make sure all the content in the conference matches the project" | Not a code defect: the interface has no inverted text. The conference material must be corrected by its authors; this file exists so the text can be checked against the implementation row by row. | this document |

---

## C. What this audit changed in the code

1. **Region measurements (`NO 6`)** — added `backend/app/ml/region_metrics.py` and wired it into
   the analysis payload (`region_measurements`), the radiomics quality block (`roi_centroid`),
   the PDF report (a dedicated table) and the assistant answer. The UI gained the "Abnormal
   region analysis" card on the *Segmentation* page and the centroid row on the *Radiomics*
   page. `tests/test_region_measurements.py` pins the arithmetic to hand-computed examples
   (a 10 × 10 square at (30, 10) must report area 100, coverage 1 %, box 10 × 10, centroid
   (34.5, 14.5)), proves the perimeter equals the radiomics shape estimator, and proves an
   empty mask is reported as "not detected", never as a zero-sized region.
2. **Assistant routing** — "how large…" questions now answer with the measurements topic
   (previously the word "segment" won and the answer was about the mask).
3. **Notebook** — regenerated so the embedded `radiomics_service.py` and the new
   `region_metrics.py` match the repository (`tests/test_kaggle_notebook.py` enforces this).
4. **Central artefact resolution (`T2`)** — `backend/app/config.py` now resolves every weight
   through one helper (`models/weights/` → repository root → `data/models/`, `BRAIN_WEIGHTS_DIR`
   override) and `MODEL_PATHS` gained `brain_tumor_xception`. A missing file resolves to the
   documented placement directory so the error names where to put it.
5. **Four-class image classifier (`NO 1b`)** — new `backend/app/ml/image_classifier.py`,
   `backend/app/services/image_classification_service.py`, the two endpoints above, the
   `/api/capabilities` summary, and the UI panel `ImageClassifierPanel.tsx` on the
   *Classification* page. It refuses to run without its checkpoint, its metadata or a
   preprocessing it implements.
6. **Weights are untracked (`T3`)** — `.gitignore` now covers `/models/weights/`, `*.pth`,
   `*.pt`, `*.keras`, `*.h5`, `*.hdf5`, `*.ckpt`, `*.safetensors`; the three checkpoints that
   were committed at the repository root (`segnet_model.pth`,
   `UNet-[16, 32, 64, 128, 256].pt`, `radiomics_cnn_model.pth`) were removed from the index with
   `git rm --cached`, which left the local files exactly where they were. `models/README.md`
   documents the expected files and their contracts.
7. **Audit by execution** — `scripts/audit_integration.py` loads each checkpoint through the
   registry, runs one forward pass on the documented input and prints what it found; see
   `docs/PROJECT_AUDIT.md` §10 for the results in this workspace.
8. **TensorFlow isolation** — the TensorFlow-dependent tests run in a subprocess
   (`tests/tf_image_classifier_driver.py`) because importing TensorFlow and then `torch.compile`
   crashes this environment; see `docs/LIMITATIONS.md` §16.

Test suite: 240 tests, all passing (CPU only; 10 skip without TensorFlow). Frontend: `tsc --noEmit` clean,
`npm run build` = 888.37 kB JS (252.13 kB gzip) + 38.49 kB CSS.

---

## D. What is still not possible, and why

| Item | Reason |
| --- | --- |
| Any real-data accuracy, Dice, F1 or calibrated confidence | No BraTS 2020 data and no trained artefact exist in this repository. Training must run on Kaggle (`HOW_TO_RUN.md` §5); the application reports "not evaluated" until then. |
| Classification predictions in the UI | Requires a validated classifier artefact (model + ordered feature schema + scaler + held-out metrics). The shipped `radiomics_cnn_model.pth` is incompatible (21 features expected, 11 available, undefined labels) and is refused. |
| A four-class prediction | Requires `brain_tumor_xception_final.keras` + `metadata.json` in `models/weights/` and TensorFlow in the API environment. Neither is present here, so the endpoint reports the missing file and predicts nothing. |
| Ensemble in the default deployment | Needs at least two compatible checkpoints for the same input specification. With only `unet_dynamic_512` usable for a grayscale upload, the ensemble reports exactly that reason instead of fusing one model with itself. |
| Malignancy, clinical stage or prognosis | Not present in the labels. The dataset provides the WHO grade group; anything else would be a fabricated clinical claim. |
| Physical (mm/cm) measurements from an HDF5 slice | The slices carry no voxel spacing; the API accepts spacing when it exists and labels pixel units otherwise. |
