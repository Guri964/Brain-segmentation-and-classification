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

`ClassificationService` serves an artefact **only** if it carries a feature schema, a label
mapping and preprocessing statistics. Anything missing yields `available: false` with the reason
and the missing items — never a quietly padded or re-ordered vector. A vector that is short of one
feature, or that carries unknown names, is refused and the missing names are listed. The shipped
`radiomics_cnn_model.pth` stays unused for exactly this reason: it expects 21 named features, no
scaler or feature order was ever saved with it, and inventing the missing 10 would produce
arbitrary numbers.

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
