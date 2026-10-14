# Brain abnormality research platform

Segmentation, quantitative radiomics, explainable AI and uncertainty estimation for
brain MRI, delivered as a Flask API with a React + TypeScript dashboard.

**Research prototype. Outputs are AI-generated and are not intended to replace
professional medical diagnosis.**

The guiding rule of this codebase: a value that was not measured is reported as
unavailable, with the reason, and never as a zero. There are no demo datasets, no
placeholder metrics and no fabricated predictions anywhere in this repository.

---

## 1. What the platform does

| Capability | State in this deployment | Where it is implemented |
|---|---|---|
| Single-channel segmentation (512×512 U-Net) | Working | `backend/app/services/segmentation_service.py` |
| Four-channel segmentation (SegNet) | Checkpoint loads but produces an empty mask; **needs retraining** | `scripts/train_segnet.py` |
| Attention U-Net (RAUNet, four channel) | Not trained; registered and reported as unavailable | `scripts/train_unet.py --architecture raunet` |
| Probability ensemble (P = Σ wᵢ·Pᵢ) | Implemented; runs when two models share an input spec, refused otherwise | `backend/app/services/ensemble_service.py` |
| Quantitative radiomics | Working (native extractor, 68 features on a 2-D slice) | `backend/app/services/radiomics_service.py`, `backend/app/ml/radiomics_native.py` |
| Region measurements (area, coverage, bounding box, centroid, perimeter) | Working on every analysis, in pixel units; "not detected" when the mask is empty | `backend/app/ml/region_metrics.py` |
| Grad-CAM explanations | Working for the single-channel U-Net | `backend/app/ml/gradcam.py`, `explainability_service.py` |
| Feature attribution | Implemented; shown when a classifier artefact exists | `backend/app/services/classification_service.py` |
| Uncertainty (TTA, MC dropout, ensemble disagreement) | Working (test-time augmentation) | `backend/app/services/uncertainty_service.py` |
| BraTS 2020 ingestion | Working: inspector, patient-level manifest builder, lazy HDF5/NIfTI loading | `scripts/inspect_brats.py`, `scripts/prepare_brats.py`, `scripts/_training.py` |
| Radiomics feature table | Working: one labelled row per case in the application's own schema | `scripts/extract_radiomics.py` |
| Tumour classification | Pipeline complete and served; **no artefact installed here**, so no class is predicted | `scripts/train_classification.py`, `backend/app/services/classification_service.py` |
| Tumour **type** from the image (Keras Xception, four classes) | Serving path complete; the `.keras` checkpoint and its `metadata.json` are not installed in this deployment, so no class is predicted | `backend/app/ml/image_classifier.py`, `POST /api/classification/image` |
| Stacked ensemble of `FINAL ENSEMBLE NB.ipynb` (three 4×240×240 members + 46-feature radiomics CNN → gradient-boosted head) | Pipeline complete and served end-to-end; **its eight artefacts are not installed here**, so it reports `artifacts_missing` and predicts nothing. Predicts the notebook's mask-size proxy, never a grade | `backend/app/services/ensemble_bundle_service.py`, `backend/app/ml/radiomics_notebook.py`, `GET /api/ensemble/stacked` |
| SQLite persistence + history + search | Working | `backend/app/database/` |
| PDF report (14 sections) | Working | `backend/app/services/report_service.py` |
| Model evaluation module | Working; reports "Not evaluated" until the test split has been run | `backend/app/services/evaluation_service.py`, `scripts/evaluate_models.py` |
| Contextual assistant | Working (local, grounded; LLM optional) | `backend/app/services/assistant_service.py` |

## 2. Quick start

```bash
# 1. Python environment
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r backend/requirements.txt

# 2. Frontend build (optional: the API also serves JSON on its own)
cd frontend && npm install && npm run build && cd ..

# 3. Verify the checkpoints before using them (and see every artefact's real state)
python scripts/check_models.py
python scripts/audit_integration.py     # per-artefact audit: found, loaded, executed

# 4. Run the application
python backend/run.py            # http://localhost:5000
```

Development mode with hot reload:

```bash
cd frontend && npm run dev       # http://localhost:5173, proxies /api to :5000
```

Model weights are **not** in this repository. Put them in `models/weights/` (or point
`BRAIN_WEIGHTS_DIR` elsewhere) - see [models/README.md](models/README.md) for the expected file
names, the input contract of each artefact and the one checkpoint that cannot be served. The
notebook's stacked ensemble needs its eight files **together**; [HOW_TO_RUN.md](HOW_TO_RUN.md) §3
lists them with the exact copy commands for Windows and Linux/macOS.

Full instructions, including GPU setup and dataset preparation, are in
[HOW_TO_RUN.md](HOW_TO_RUN.md).

## 3. Documentation

| Document | Contents |
|---|---|
| [HOW_TO_RUN.md](HOW_TO_RUN.md) | Installation, CPU/GPU setup, running, troubleshooting |
| [docs/API.md](docs/API.md) | Every endpoint with its request and response shape |
| [docs/DATASET.md](docs/DATASET.md) | Manifest format, folder layouts, labelling requirements, splits |
| [docs/ALGORITHM_NOVELTY.md](docs/ALGORITHM_NOVELTY.md) | Baseline, the modifications, and the experiments that would justify each claim |
| [docs/BRAFTS2020.md](docs/BRAFTS2020.md) | BraTS 2020 layout, grade labels, inspection, manifest building, synthetic fixture |
| [docs/CLASSIFICATION.md](docs/CLASSIFICATION.md) | Grade classifier: pipeline, splits, artefact contract, API, verified reproductions |
| [docs/LIMITATIONS.md](docs/LIMITATIONS.md) | What the platform cannot do, and why |
| [docs/PROJECT_AUDIT.md](docs/PROJECT_AUDIT.md) | Audit of the original application (17 defects) |
| [docs/CURRENT_ARCHITECTURE.md](docs/CURRENT_ARCHITECTURE.md) | As-found versus target architecture |
| [docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md) | Phase plan and open questions |
| [docs/DELIVERY.md](docs/DELIVERY.md) | What was delivered: changes, structure, algorithms, results, tests, setup, limitations |
| [docs/REQUIREMENTS_TRACEABILITY.md](docs/REQUIREMENTS_TRACEABILITY.md) | The supplied design document and modifications list (`3.pdf`, `4.pdf`) mapped row by row to the implementation, the endpoint, the UI page, the test and the honest limitation |

## 4. Repository layout

```
backend/
  run.py                    entry point
  requirements.txt
  app/
    __init__.py             application factory
    config.py               environment-driven configuration
    errors.py               error hierarchy and handlers
    logging_config.py
    routes/                 HTTP layer only (9 blueprints, 33 routes)
    services/               all business logic
    ml/
      architectures/        SegNet, DynamicUNet, RAUNet
      registry.py           checkpoint loading, caching, device selection
      radiomics_native.py   extractor used when PyRadiomics is unavailable
      gradcam.py            input-anchored Grad-CAM
      losses.py
      classifier_models.py
    database/               schema + migrations + repository
    schemas/                request validation
    utils/                  io, image and metric helpers
frontend/
  src/
    pages/                  13 dashboard pages
    components/             shared UI, charts, image viewer
    services/               API client and formatting
    hooks/                  analysis state
    types/                  API type mirror
scripts/
  check_models.py           reference verification of every checkpoint
  make_splits.py            patient-level train/validation/test split
  train_segnet.py           four-channel SegNet
  train_unet.py             single-channel U-Net / four-channel RAUNet
  train_classifier.py       torch feature-head classifier
  train_classification.py   baseline-vs-improved sklearn classifier + servable artefact
  inspect_brats.py          read-only BraTS tree inspection (layout, keys, masks, grades)
  prepare_brats.py          BraTS -> patient-level manifest with leakage check
  extract_radiomics.py      one labelled radiomics row per case
  make_brats_fixture.py     synthetic fixture that mimics the Kaggle HDF5 layout
  make_classification_fixture.py  synthetic image-classification fixture (Training/<class>, Testing/<class>)
  evaluate_models.py        reserved test split evaluation
  run_ablations.py          one-component-at-a-time ablations
  calibrate_ensemble.py     validation-only fusion weights
  extract_deep_features.py  encoder embeddings for the fusion experiment
  setup_env.py              environment report / installer
  install_trained_artifacts.py  install the Kaggle export without overwriting shipped files
notebooks/
  brain_tumor_training_kaggle_t4.ipynb   the T4 notebook that trains and exports every model
  build_kaggle_notebook.py               regenerates the notebook from notebooks/_notebook_source/
tests/                      pytest suite (152 tests, CPU-only)
docs/
data/                       runtime storage (git-ignored)
```

## 5. What is shipped versus what has to be produced

**Shipped and reused**

* `UNet-[16, 32, 64, 128, 256].pt` — DynamicUNet, 1 channel, 512×512, internal
  sigmoid. Verified as responsive: it loads, runs and produces a structured mask on a
  deterministic reference phantom in ~0.5 s on CPU.
* `segnet_model.pth` — SegNet(4→1), 256×256, logits. Verified as **not responsive**:
  for every input available without the training data, its sigmoid output stays near
  zero and the mask is empty. It is registered, reported honestly in the interface,
  and excluded from ensembles. Retraining is required before it can appear in any
  reported result.
* `radiomics_cnn_model.pth` — requires 21 standardised features; only 11 columns
  exist in `features_df.csv` and no feature schema, scaler or label mapping was saved
  with it. Predictions are therefore **disabled**, and the interface explains exactly
  what is missing.
* `features_df.csv` — used only to demonstrate the classification data requirement,
  never to train.

**Produced by the scripts, when the real dataset is supplied**

* Retrained SegNet and/or RAUNet checkpoints with a validated training run.
* `data/models/evaluation_results.json` — test-split metrics with bootstrap intervals.
* `data/models/model_selection.json` — which model was chosen, on which split, why.
* `data/models/ensemble_calibration.json` — validation-only fusion weights.
* `data/models/ablation_results.json` — one-component-difference ablations.
* `data/models/classification/manifest.json` — the classifier artefact and its
  feature schema.

## 6. Honest current results

| Item | Result |
|---|---|
| Reference check (`scripts/check_models.py`) | `unet_dynamic_512` responsive; `segnet_brats_4ch` `no_foreground`; `raunet_brats_4ch` missing; `radiomics_mlp_v1` incompatible |
| Test-split segmentation metrics | **Not evaluated** — no labelled test split has been supplied |
| Classification metrics | **Not evaluated** — no labelled feature table has been supplied |
| Ensemble | Not available in this deployment: the two loaded checkpoints take different inputs (4×256×256 vs 1×512×512) |
| Radiomics | 68 features extracted from the predicted region on a 2-D slice (~0.2 s), pixel-domain units only (no pixel spacing in the test input) |
| Runtime | 0.46–0.58 s per 512×512 single-channel image on 2 CPU cores, no GPU |
| Test suite | 152 tests, all passing, CPU-only (includes executing every cell of the Kaggle notebook end to end against the BraTS and image-classification fixtures) |
| Two datasets, two tasks | Segmentation and the HGG/LGG grade experiment use BraTS 2020; the multiclass tumour-type model uses the Brain Tumor MRI Dataset. Artefacts carry `task`, `label space` and `dataset` (see `data/models/models_index.json`), and the stacked ensemble refuses members from a different dataset or label space |
| Model training provenance | Reported per model from what the training run recorded: *trained on real data* / *trained on a synthetic fixture* / *trained, not evaluated* / *trained on an unverified dataset* / *no provenance recorded*. With no manifest installed, the shipped checkpoints are reported as **no provenance recorded** — never as trained on real data |
| Data pipeline | The notebook trains from a per-patient cache through a bounded payload store: the cache is memory-mappable (one `.npy` per member) and legacy `.npz` archives are converted in place after verification. Cell 9 measures both layouts on your data; cell 18 prints batch preparation, CPU->GPU transfer, forward, backward+step and the whole step separately, with `torch.cuda.synchronize()` and a real `nvidia-smi` sample. `docs/DELIVERY.md` §K has the measured root cause and the before/after numbers |
| Kaggle T4 notebook | `notebooks/brain_tumor_training_kaggle_t4.ipynb` — trains SegNet / U-Net / RAUNet and the HGG-vs-LGG classifier on the real BraTS 2020 mirror, **and** a multiclass tumour-type model trained from scratch on the Brain Tumor MRI Dataset (`masoudnickparvar/brain-tumor-mri-dataset`, classes read from its directories), then evaluates once on each held-out split and exports `brain_mri_artifacts/` for `scripts/install_trained_artifacts.py`. Its 38 code cells are executed by the test suite against both fixtures; the metrics it produces depend on the run and are written into the artefacts, not into this repository |

Nothing in the table above is estimated or extrapolated. Where a number could not be
measured, the table says so.

## 7. Safety properties enforced by the code

* A 1-channel image can never be silently broadcast into a 4-channel model;
  `preprocess_four_channel` refuses four identical uploads and reports which channels
  matched.
* No metric is produced without a reference standard; the comparison page labels
  inter-model similarity as *agreement*, never as accuracy.
* An empty predicted region yields an unavailable measurement, not a zero.
* The assistant refuses questions about malignancy, grade, staging, prognosis,
  survival and treatment before any backend runs, and says why.
* Ensemble weights from the test split are rejected; calibration records the split it
  used.
* Grad-CAM heatmaps carry an explicit statement that a saliency map is not a
  diagnosis and not a lesion outline.

## 8. Licence and attribution

Released for research use. The SegNet, U-Net and radiomics architectures and the
original training notebook come from the existing company project; the platform
structure, services, scripts, tests and dashboard were built on top of it.
