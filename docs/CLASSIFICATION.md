# Tumour classification (dataset label): pipeline, artefact and honest limits

The platform predicts the label the dataset records for a case; for BraTS that is the WHO grade
group (`HGG` / `LGG`). It is a **dataset label**, not a diagnosis: no malignancy verdict, no
clinical stage and no prognosis is predicted anywhere, because the dataset carries none of those
annotations. The class never comes from the segmentation mask — the mask only defines the region
the radiomics features are measured in.

## 1. Pipeline

```
BraTS HDF5/NIfTI  ->  prepare_brats.py       manifest with patient_id + grade
                          |
                          v
                   extract_radiomics.py      one feature row per case, application schema
                          |
                          v
                   train_classification.py   baseline vs improved, patient-level splits
                          |
                          v
            data/models/classification/<model_id>-<stamp>/
                   |-- <model_id>.joblib     sklearn estimator
                   \-- manifest.json         schema, labels, scaler, metrics, caveats
                          |
                          v
      ClassificationService -> /api/classification, /api/analysis/<id>, PDF section 6

Same artefact, produced on a GPU with the real dataset:
notebooks/brain_tumor_training_kaggle_t4.ipynb -> brain_mri_artifacts.zip
                   -> scripts/install_trained_artifacts.py --source brain_mri_artifacts --data-dir data
```

Commands (run end to end against the synthetic fixture in `scripts/make_brats_fixture.py`):

```bash
/home/user/.venv/bin/python scripts/prepare_brats.py --root <brats-root> \
  --out data/manifests/brats2020.json --grade-table <name_mapping.csv> --splits

/home/user/.venv/bin/python scripts/extract_radiomics.py \
  --data data/manifests/brats2020.json --out data/manifests/brats_features.csv

/home/user/.venv/bin/python scripts/train_classification.py \
  --features data/manifests/brats_features.csv --split-column split \
  --model-id radiomics_grade_classifier \
  --target-description "BraTS 2020 HGG vs LGG (WHO grade group)"
```

## 2. What `train_classification.py` does

1. Reads the feature table (CSV or JSON). Identifier columns (`case_id`, `patient_id`, `label`,
   `split`, `notes`, …) never enter the feature set; only numeric columns do, in table order.
2. Rejects rows with no patient identifier or no label and reports why. A table without patient
   identifiers is refused outright: a patient-level split would be impossible and a slice-level
   split leaks.
3. Takes the split from the `split` column only when that column declares all three splits
   completely; otherwise it assigns a seeded, patient-level, label-stratified split and records
   the method.
4. Imputes with the **training-split median** and standardises with the **training-split mean and
   standard deviation**. Validation and test rows inherit those statistics; they never influence
   them.
5. Fits two models on the same training rows: a logistic-regression baseline (balanced class
   weights by default) and an improved model (`mlp`, `gradient_boosting` or `random_forest`).
6. Selects by **validation macro-F1**, then reports both models on the held-out test split. Both
   scores and the selection are written to disk so the comparison is auditable; nothing is tuned
   on the test rows.
7. Writes the artefact, an evaluation file beside it, a run record under
   `data/models/classification/`, and merges a `classification` block into
   `data/models/evaluation_results.json`, which `/api/evaluation` serves.

Recorded metrics: accuracy, macro precision/recall/F1, weighted F1, per-class
precision/recall/F1/support, confusion matrix with its class order, ROC-AUC (one-vs-rest, macro
for multiclass), sample count, class distribution, and the split each number came from. When there
are few patients per class the sample is small and the intervals are wide; the artefact says so in
its caveats instead of implying precision it does not have.

## 3. Artefact contract

`data/models/classification/<model_id>-<stamp>/manifest.json`:

| Field | Meaning |
| --- | --- |
| `model_id`, `version`, `kind`, `model_file` | Identity and the `.joblib` file it loads |
| `target_description` | Human-readable meaning of the label |
| `label_mapping` | `{"0": "HGG", "1": "LGG"}` — the index order the estimator uses |
| `feature_schema` | **Ordered** feature names; a vector in another order is refused |
| `preprocessing` | `standard_scaler` with `mean`, `scale`, `median`, `imputation`, `fit_on: training split only` |
| `coefficients` | Linear coefficients, when the estimator has them |
| `feature_quantiles` | Training quantiles used by the fallback attribution method |
| `metrics` | Test metrics, plus the selected model and the split they were measured on |
| `model_comparison` | Validation and test metrics for baseline and improved |
| `splits`, `leakage_check` | Class distribution per split, patient overlap, `passed` |
| `seed`, `trained_at`, `code_revision`, `environment`, `trainer` | Reproducibility record |
| `caveats` | The statements the interface and the PDF are obliged to repeat |

`ClassificationService` also reports *what the artefact was trained on*, read from the `dataset`
block the training run writes (`training` in `/api/classification`, and in every analysis result):

| `dataset.synthetic` | State | What the interface shows |
| --- | --- | --- |
| `false` | `trained_on_real_data` | Trained on the real dataset |
| `true` | `trained_on_synthetic_fixture` | Trained on a synthetic fixture — predictions and recorded metrics describe that fixture only |
| `null` / absent | `trained_on_unverified_dataset` | The training run could not establish the data, so the model is not presented as trained on real data |
| no artefact | `not_trained` | No classifier is installed; no prediction is produced |

`ClassificationService` serves an artefact **only** if it carries a feature schema, a label
mapping and preprocessing statistics. Anything missing yields `available: false` with the reason
and the missing items — never a quietly padded or re-ordered vector. A vector that is short of one
feature, or that carries unknown names, is refused and the missing names are listed. The shipped
`radiomics_cnn_model.pth` stays unused for exactly this reason: it expects 21 named features, no
scaler or feature order was ever saved with it, and inventing the missing 10 would produce
arbitrary numbers.

### 2.1 The second classification task (multiclass tumour type)

The project also trains a classifier on a **different dataset with a different label space**:
the Brain Tumor MRI Dataset (`masoudnickparvar/brain-tumor-mri-dataset`), whose four classes are
read from its directories (glioma, meningioma, pituitary, no tumour). It is not HGG/LGG and it is
never mixed with the BraTS grade model:

* its artefacts are exported to `data/models/classification_multiclass/<version>/`, with a
  `task: multiclass_classification` field, the class list, the ordered feature schema, the fitted
  scaler and the recorded metrics;
* the stacked ensemble that combines its members (from-scratch CNN + two radiomics classifiers)
  is guarded by an explicit compatibility check: identical dataset id, identical class order,
  identical images per split, probabilities that sum to one. The notebook offers the BraTS grade
  classifier as a candidate on purpose so the guard demonstrates the refusal, with the reason
  printed and stored;
* `data/models/models_index.json` lists every artefact with its task, label space, dataset,
  synthetic flag and whether the API serves it, and `/api/classification` exposes the same list as
  `trained_artefacts`;
* the multiclass artefact is **not** served by the upload pipeline: its ROI is derived from a
  single grayscale slice (Otsu brain region) while the API measures radiomics inside the
  segmentation mask of a four-modality upload. Its manifest says so in
  `provenance_contract.feature_sources.serving_contract` instead of pretending the two are the
  same input.

### 3.1 The schema a served prediction needs

`analysis_service._run_radiomics()` extracts features from `display_reference`, which is
`preprocessed.display_stack[0][1]` — the **first** channel of a multi-channel upload, i.e. T1
(`preprocessing_service.MODALITY_ORDER_BRATS = ("t1", "t1ce", "t2", "flair")`), on a single plane,
with the mask the segmentation produced. Two consequences follow, and the Kaggle notebook obeys
both:

* feature names carry **no channel prefix** (`firstorder_mean`, not `ch3__firstorder_mean`), because
  the API only ever computes one plane's features. A model trained on prefixed names could never be
  fed;
* non-numeric extractor output (`shape_units`, `diagnostics_dimensionality`, …) is **not** part of the
  schema, and neither are the physical-unit features that are `None` without voxel spacing
  (`shape_area_mm2`, …). A feature the API cannot compute is refused with `missing_features`, so a
  schema must only contain values the extractor always produces.

The notebook's tables therefore hold one row per patient (the slice with the largest tumour area),
bare feature names, and a `mask_source` column distinguishing the BraTS reference mask from the
segmentation model's predicted mask. Both tables are trained and compared on the same
patient-level split, and the artefact records which one was selected. Section 16 of the notebook
measures the drift between the two mask sources instead of assuming it is small.

## 4. Explainability

* Linear models: contribution = coefficient × standardised feature value, per class.
* Estimators without coefficients: single-feature substitution against the training quantiles
  stored in the artefact.
* Every contribution records the method it used, because a coefficient-based number and a
  substitution-based number are not the same quantity.
* SHAP is used when the package is installed. `/api/classification` reports whether it is
  available (`shap.available`) and states the fallback that is used instead.

## 5. API and interface

| Endpoint | Returns |
| --- | --- |
| `GET /api/classification` | Model card: status, target semantics, evaluation summary, XAI method, training pipeline, requirements |
| `GET /api/analysis/<analysis_id>` | `result.classification`: label, scores, contributions, comparison, caveats, `evaluated_on` |
| `GET /api/evaluation` | `classification` block with the recorded comparison, or `not_evaluated` with the reason |
| `POST /api/reports/analysis/<id>` | PDF section 6: label, scores, per-class metrics, confusion matrix, baseline-vs-improved table, caveats |

The **Tumour Classification** page (`/analysis/classification`) shows the model card, the current
analysis prediction, class probabilities, the attribution chart and table, the feature values the
model consumed, the recorded per-class metrics and confusion matrix, the baseline-vs-improved
comparison, and the caveats. When no artefact exists the page says so and names the command that
produces one; it never renders a placeholder label. The assistant answers the same question for a
stored analysis (`topic: classification`) from the stored record only.

## 6. Reproductions that were actually performed

On the synthetic fixture (6 volumes, 24 slices, 12 manifest cases, 6 patients):

| Step | Outcome |
| --- | --- |
| `prepare_brats.py --splits` | train / validation / test each 4 cases · 2 patients · 1 HGG + 1 LGG, `patient_overlap: []` |
| `extract_radiomics.py` | 12 rows × 76 columns (68 radiomics features, backend `native`), HGG 6 / LGG 6 |
| `train_classification.py` | baseline and improved trained, selected by validation macro-F1, artefact + `evaluation_results.json` written, `/api/classification` serves it |
| `POST /api/analysis` (PNG phantom) | request completed with `classification.available: true`, 25 contributions, `missing_features: []` |
| PDF report | section 6 rendered with the metrics tables (118,418 bytes) |
| Tests | 20 new tests in `tests/test_brats_pipeline.py` and `tests/test_classification_pipeline.py` |

The same pipeline also runs end to end inside
`notebooks/brain_tumor_training_kaggle_t4.ipynb`, and the test suite executes every one of its
cells against the fixture (`tests/test_kaggle_notebook.py`), including the export that
`ClassificationService` then loads. On the fixture the predictions are correctly refused when the
segmentation model's predicted region is empty: no feature, no score.

**None of those numbers is a result.** The fixture labels are invented and its masks are blobs, so
every metric above describes the fixture. Real numbers require the real BraTS 2020 download (see
`docs/BRAFTS2020.md`) and a training run on it; nothing in the repository quotes a fixture metric
as if it were a finding.

## 7. Limits stated on every surface

1. Only the datasets own label (for BraTS, the grade group) is predicted — not malignancy, not
   stage, not prognosis, not survival.
2. Metrics come from one split; with few patients per class the uncertainty is large and is not
   estimated here.
3. A classifier trained on predicted masks inherits the segmentation error, including the shipped
   SegNet checkpoint defect.
4. Radiomics features describe a region of interest on a slice, not a tumour volume.
5. Physical units appear only when the input carries pixel spacing; a PNG never does.

---

## 8. The Keras image classifier: tumour type from the image (four classes)

`firstnb.ipynb` trains a second, unrelated classifier: `Xception(weights="imagenet",
include_top=False, input_shape=(299, 299, 3))` with a global-average-pooling head and a four-way
softmax over `glioma / meningioma / pituitary / no tumour`. It consumes the **image**, not
radiomics features and not a mask.

### 8.1 Contract served by this application

| Property | Value | Source of truth |
| --- | --- | --- |
| Checkpoint | `brain_tumor_xception_final.keras` | file name, overridable with `BRAIN_MODEL_XCEPTION` |
| Class order | `class_to_index` in `metadata.json` | metadata, never hard-coded: the code sorts by the recorded index |
| Input | `input_size` + 3 channels (299 × 299 × 3 for the supplied run) | metadata |
| Preprocessing | the recorded name must be an Xception name; the implementation is `x / 127.5 - 1` in NumPy (bit-identical to `keras.applications.xception.preprocess_input`) | metadata + `image_classifier.xception_preprocess` |
| Output | softmax over the declared classes; if the model returns non-normalised values they are softmaxed and the response says so | response `output_transform` |
| Resize | bilinear (PIL) to the metadata size | response `input.resize` |

### 8.2 What the serving path refuses to do

* It never guesses a class order. A missing `metadata.json`, a `class_to_index` that is not an
  exact `0..n-1` permutation of `classes`, fewer than two classes, a non-three-channel
  `input_size`, or a preprocessing name this build does not implement all disable the model with
  a message naming the problem.
* It never serves an artefact that disagrees with its metadata: the checkpoint's output width
  must equal the declared class count and its input shape must equal the declared size, otherwise
  the request fails with `409` before an image is scored.
* It never labels a prediction as a grade, a malignancy verdict or a stage. The four classes are
  dataset annotations of tumour **type**; the dataset's `Grade` column (HGG/LGG) belongs to the
  other classifier and the two are never combined.
* It never presents a softmax value as calibrated confidence, and it never presents the metrics
  recorded in `metadata.json` as measurements made by this deployment (see
  `docs/LIMITATIONS.md` §17).

### 8.3 Why it is disabled unless you install it

The two artefact files are not in this repository (weights are never committed) and the
application never downloads them. Until they are placed in `models/weights/`, the state is
`checkpoint_missing`, the endpoint answers `409`, the UI shows the missing-file panel with the
directory to use, and every other capability - segmentation, ensemble, radiomics, region
measurements, reporting, the assistant - works exactly as before. `models/README.md` documents
the expected files; `python scripts/audit_integration.py` reports the state of each of them.
