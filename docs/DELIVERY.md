# Delivery summary

Final state of the transformation of the brain MRI project into a research platform.
Every number below was produced by a command in this repository on the delivery machine
(CPU only: no GPU, 2 cores, 3 GB RAM, Python 3.11, torch 2.4.1, Node 22).

## A. Summary of changes

| Area | Before | Now |
| --- | --- | --- |
| Interface | Streamlit script (`final.py`, 168 lines with hard-coded metrics) | React 18 + TypeScript + Tailwind dashboard, 12 routes, dark and light themes, Recharts, Framer Motion |
| Backend | three Gradio functions inside the Streamlit file | Flask application factory, 8 blueprints, 32 routes, 13 services, schema migrations, structured error handling |
| Persistence | none | SQLite (`data/app.db`, schema v2): analyses, scan metadata, segmentation rows, radiomics rows, classification rows, explainability artefacts, uncertainty rows, reports, assistant messages |
| Radiomics | absent from the product path (a CSV in the repository root) | PyRadiomics-backed extraction with a native fallback extractor, 68 two-dimensional features, category grouping, CSV export, pixel-vs-physical distinction |
| Explainability | absent | Grad-CAM from the correct layer for each segmentation checkpoint, heat-map and overlay artefacts, radiomics feature attribution |
| Uncertainty | absent | test-time augmentation dispersion, Monte-Carlo dropout and ensemble disagreement, each shown only where technically valid |
| Clinical support | a hard-coded consulting string | grounded assistant (retrieval over the stored analysis, optional environment-configured LLM, safety refusals) plus a decision-support page with explicit research language |
| Reports | none | 14-section PDF written from stored values, unavailable distinguished from zero |
| Evaluation | none | evaluation module reading real artefacts, "Not evaluated" when absent, producing scripts named in the interface |
| Training | Colab notebook with two hard-wired paths | patient-level split builder, two training scripts with seeding, checkpointing, early stopping, LR scheduling, resume, logs; evaluation, calibration and ablation scripts |
| Tests | none | 58 tests, CPU-only, covering preprocessing, inference, ensemble, radiomics, XAI, uncertainty, database, reports, API and frontend contract |
| Docs | none | README, HOW_TO_RUN, API reference, dataset guide, limitations, algorithm notes, audit and architecture documents |

Two pre-existing defects were confirmed rather than hidden: the SegNet checkpoint answers
"no foreground" for every input available here (see G), and `radiomics_cnn_model.pth`
receives 11 of the 21 features it was built for, so classification stays disabled.

## B. New structure

```
backend/
  app/
    __init__.py            application factory, service wiring, static serving
    config.py              environment-driven configuration and limits
    errors.py              error hierarchy and JSON handlers
    logging_config.py      level, format, rotating file log
    routes/                health, system, analysis, models, radiomics, history,
                           reports, evaluation, assistant   (8 blueprints, 32 routes)
    services/              preprocessing, segmentation, ensemble, radiomics, uncertainty,
                           explainability, classification, evaluation, analysis,
                           report, assistant, model_health, dataset
    ml/
      architectures/       segnet.py, dynamic_unet.py, attention_unet.py (RAUNet)
      registry.py          descriptor-driven loading, caching, device selection
      losses.py            bce, dice, bce_dice, focal_tversky, boundary
      gradcam.py           gradient-weighted activation mapping
      radiomics_native.py  dependency-free fallback extractor
      classifier_models.py feature-learner head for the trained artefact
    database/              schema.py (numbered migrations), repository.py (one API)
    utils/                 io, images, metrics (Dice, IoU, HD95, ASD, bootstrap CI)
  run.py                   development server
frontend/
  src/pages/               12 routed pages
  src/components/          ui kit, image viewer, charts, providers
  src/services/            typed API client, formatting rules
  src/types/api.ts         every response shape used by the interface
scripts/                   13 command-line entry points (see HOW_TO_RUN)
tests/                     10 modules, 58 tests
docs/                      audit, architecture, plan, API, dataset, limitations,
                           algorithm notes, delivery summary, source materials
```

## C. Algorithms and architectures

1. **SegNet** (VGG16-style encoder with max-pooling indices, decoder with the original
   SegNet design), four-channel input, 151,105 parameters, single logit map, sigmoid applied
   at serving time.
2. **U-Net**: `DynamicUNet` with an encoder of widths 16/32/64/128/256, a bottleneck,
   transposed convolutions and skip connections, single-channel input at 512x512,
   1,940,945 parameters, sigmoid inside the network.
3. **RAUNet** (new, `attention_unet.py`): residual blocks, additive attention gates on every
   skip connection and deep supervision with auxiliary heads. It is an order of magnitude
   larger than the shipped U-Net and is the candidate for the novelty claim; it cannot be
   trained here (no dataset, no GPU) and is therefore reported as untrained.
4. **Ensemble**: `P = sum(w_i * p_i) / sum(w_i)` over compatible member probability maps with
   a documented threshold. Weights come from `equal` (baseline), `calibrated` (validation split
   only) or `confidence_aware` (predictive-variance weighting). The service refuses members
   whose channel count, spatial size or normalisation differ.
5. **Preprocessing**: explicit per-model contract (channels, size, interpolation,
   normalisation, dtype, tensor shape). One-channel inputs are never duplicated into a
   four-channel model; a mismatch is refused with a structured error.
6. **Radiomics**: 68 two-dimensional features from the predicted mask with square-pixel
   spacing handling, bin-count discretisation, overlap diagnostics and a record of whether the
   region touches the field of view.
7. **Explainability**: Grad-CAM computed from the layer that actually carries the segmentation
   decision (no shared path between the two models), radiomics attribution by feature weight,
   and a statement that a heat map is not a diagnosis.
8. **Uncertainty**: TTA dispersion (five passes), Monte-Carlo dropout, and ensemble
   disagreement, each gated on the number of technically valid passes.
9. **Metrics**: Dice, IoU, precision, recall, sensitivity, specificity, HD95 and average
   surface distance implemented in `utils/metrics.py` with pure NumPy (no SciPy dependency),
   with percentile bootstrap confidence intervals; they appear only when a reference mask
   exists.

## D. Models reused as shipped

| Artefact | Status in this delivery |
| --- | --- |
| `UNet-[16,32,64,128,256].pt` | Loads and runs; used for every demonstrated analysis (single-channel 512x512, 0.28-0.58 s per image on two CPU cores) |
| `segnet_model.pth` | Loads but is degenerate: for the reference phantom, a real slice and scaled inputs the logit range stays around -29 to -54, i.e. sigmoid below 1e-12, so no pixel ever crosses the threshold. Kept, reported, and excluded from metric claims |
| `radiomics_cnn_model.pth` | Input head expects 21 radiomics features; only 11 exist in the pipeline schema, and no scaler/feature-order artefact was shipped. Classification therefore reports "disabled" with the missing requirements listed |

No weights were modified, retrained or replaced, and no sample or Hugging Face weights were
introduced.

## E. Models retrained

None. Training requires the client dataset, which is not present on this machine; inventing
results was not an option. Instead the full pipeline is committed and verified for syntax and
contract:

| Script | What it will do when the dataset is available |
| --- | --- |
| `scripts/make_splits.py` | Patient-level train/validation/test split from a JSON manifest, leak-checked, frozen to disk |
| `scripts/train_segnet.py` | Retrains SegNet (4-channel) with seed, checkpointing, early stopping, LR scheduling, resume and logs |
| `scripts/train_unet.py` | Retrains the U-Net, including the RAUNet variant and the ablation switches |
| `scripts/train_classifier.py` | Trains the radiomics classifier on labelled CSV rows and writes a versioned artefact with its feature schema and calibration |
| `scripts/evaluate_models.py` | Metrics on the reserved test split, writes `evaluation_results.json`, `model_selection.json` and per-run files |
| `scripts/calibrate_ensemble.py` | Fusion weights from the validation split only (`test_split_used: false` is recorded) |
| `scripts/run_ablations.py` | One-component-difference ablations for the novelty claims |
| `scripts/extract_deep_features.py` | Encoder embeddings for the deep-feature fusion experiment |

## F. Dataset and label requirements

* Segmentation: 2-D grayscale slices or 4-channel stacks with matching binary masks, supplied
  through a JSON manifest (`docs/DATASET.md` has the schema, a generator snippet and a leakage
  checklist). Patient-level grouping is mandatory; slice-level splitting is refused.
* Classification: one row per patient or slice with a label column, the 21 named radiomics
  features (or a superset from which they can be selected), and a documented label vocabulary.
  Without labels the classifier stays disabled.
* Nothing in this repository is trained on example or demo data. `data/manifest.example.json`
  is an empty template.

## G. Actual evaluation results

| Item | Result |
| --- | --- |
| Test-split segmentation metrics | **Not evaluated** - no dataset on this machine; the interface shows "Not evaluated" and names `scripts/evaluate_models.py` |
| Classification metrics | **Not evaluated**, and the classifier is disabled: 11 of 21 required features exist, no scaler artefact |
| Reference-verification checks | Produced by `scripts/check_models.py`: `unet_dynamic_512` responsive, `segnet_brats_4ch` not responsive (all outputs at the background floor), `raunet_brats_4ch` missing, `radiomics_mlp_v1` incompatible |
| Inference runtime (measured) | U-Net 0.28 s for the single slice above, 0.46-0.58 s per image across repeated runs on two CPU cores |
| End-to-end analysis (measured) | 2.81 s for one slice: preprocess, segmentation, 68 radiomics features, Grad-CAM, five-pass TTA uncertainty, report (PDF, 618,715 bytes), database write |
| Ensembling | Not available on this deployment: the two shipped checkpoints disagree on channels and on spatial size, so no fused probability map can exist. The interface says so instead of faking one |
| Uncertainty (measured) | TTA over five passes: mean dispersion 0.0153, p95 0.0156, 3.99% of pixels in the high-uncertainty band |

## H. Test results

| Command | Result |
| --- | --- |
| `/home/user/.venv/bin/python -m pytest tests -q` | **58 passed** in 19.4 s (CPU only) |
| `cd frontend && npx tsc --noEmit` | **0 errors** under `strict`, `noUnusedLocals`, `noUnusedParameters` |
| `cd frontend && npm run build` | Success: 864.22 kB JavaScript (245.92 kB gzipped), 37.65 kB CSS |
| `git status --porcelain` | clean after commit |

Coverage: checkpoint loading and registry behaviour, preprocessing contracts (including the
refusal to duplicate one channel into a four-channel model), inference and ensemble semantics,
radiomics with and without spacing, Grad-CAM and uncertainty gating, database migrations and
repository API, report generation, every API route including error paths, assistant safety
refusals, and a frontend/backend contract test that checks every endpoint the interface calls
exists in the Flask URL map.

## I. Setup

```bash
python -m venv .venv
.venv/bin/pip install -r backend/requirements.txt          # add requirements-dev.txt for tests
.venv/bin/python scripts/setup_env.py                      # environment report
.venv/bin/python scripts/check_models.py                   # reference verification of the checkpoints
.venv/bin/python -m pytest tests -q                        # 58 tests

cd frontend && npm install && npm run build && cd ..
.venv/bin/python backend/run.py                            # http://127.0.0.1:5000
```

GPU: install a CUDA build of torch for your driver (`pip install torch==2.4.1 --index-url
https://download.pytorch.org/whl/cu121`) and set `DEVICE_PREFERENCE=cuda`; the platform detects
the device at start-up and reports it in `/api/system` and in Settings. Everything above was
verified on CPU. Full details, environment variables and troubleshooting: `HOW_TO_RUN.md`.

## J. Remaining limitations

1. No metric is reported for this deployment because there is no labelled dataset here; the
   interface distinguishes "not available", "not evaluated" and a measured zero.
2. The shipped SegNet checkpoint produces no foreground on any input available here; it must be
   retrained before it can take part in a reported result or in an ensemble.
3. The shipped classification artefact cannot be used: the feature schema does not match and no
   scaler was shipped. Classification stays disabled until `scripts/train_classifier.py` runs on
   labelled features.
4. The demonstrated pipeline is two-dimensional. Slice-level masks are not a tumour volume and
   the radiomics features describe a region of interest on one slice.
5. Physical size is reported only when the image carries pixel spacing; a PNG never does, so
   pixel counts stay pixel counts.
6. Radiomics is computed from a predicted mask, so its accuracy is bounded by the segmentation.
7. Grad-CAM explains the network, not the biology; heat maps are artefacts of model sensitivity.
8. Ensemble fusion weights are uncalibrated on this deployment because no validation split
   exists; `equal` remains the documented default.
9. RAUNet and the deep-feature fusion experiment are implemented but untrained, so no novelty
   claim is made; `docs/ALGORITHM_NOVELTY.md` lists the experiment each claim needs.
10. Single process, SQLite, synchronous inference: appropriate for research use on one machine,
    not a multi-user service.
11. This is a research prototype. Outputs are AI-generated and are not intended to replace
    professional medical diagnosis.
