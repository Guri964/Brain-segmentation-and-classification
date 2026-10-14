# Limitations

This document lists what the platform cannot do and why. It is deliberately blunt: a
research tool is only useful when its boundaries are as visible as its outputs.

---

## 1. No diagnosis, grading, staging or prognosis

The models segment a region and describe its intensity and texture. They were not
trained, and cannot be used, to name a pathology, assign a histological grade, infer a
stage or estimate survival. The assistant refuses those questions explicitly
(`topic: "out_of_scope"`) and says which models would be needed instead.

## 2. No accuracy claim without reference labels

Dice, IoU, precision, recall, sensitivity, specificity, HD95 and average surface
distance are only defined against a reference segmentation. An uploaded scan has none,
so the analysis pages show output statistics and inter-model *agreement* instead, and
label them as such. Accuracy appears only on the Model Evaluation page after
`scripts/evaluate_models.py` has run on a reserved test split.

In this deployment no such artefact exists, so every metric reads **Not evaluated**,
with the script and file that would produce it named inline.

## 3. The shipped four-channel checkpoint does not work

`segnet_model.pth` loads and runs, but for every input available without the training
data its sigmoid output stays near zero and the mask is empty. The reference check
reports `no_foreground` and the checkpoint is excluded from ensembles. This is a
finding, not a bug in the platform: a checkpoint that loads is not a checkpoint that
works. Retraining is required, and `scripts/train_segnet.py` is the entry point.

## 4. The classifier checkpoint cannot be used as shipped

`radiomics_cnn_model.pth` expects 21 standardised features. The repository ships a
feature table with 11 columns and no feature schema, no standardisation parameters, no
label mapping and no evaluation result. Serving predictions from it would require
guessing the missing ten features and their order, so the classification branch stays
disabled and the interface states exactly what is missing.

## 5. Single images are a proxy for volumes

A PNG or JPEG export of one slice is not the acquisition. Slice selection, windowing,
resampling and any intensity scaling applied before the export cannot be recovered.
Radiomics computed on such a slice are comparable only with features extracted from
other images prepared the same way.

## 6. Physical units require spacing

Without pixel spacing, shape features that describe physical size cannot be computed.
The affected features are reported as `null` and `spacing_available: false`; pixel
counts are never silently converted into millimetres. A 2-D slice additionally has no
voxel depth, so no volume is reported at all.

## 7. Radiomics describe the mask, not the lesion

Every feature is computed inside the **predicted** region. If the model includes
peritumoral tissue, or misses part of the abnormality, the numbers describe that
region. The mask source is recorded with the feature table so the provenance is never
lost.

## 8. Uncertainty is only estimated where it is valid

Dispersion is computed from test-time augmentation, Monte-Carlo dropout (only for
architectures that actually contain dropout) or ensemble disagreement. When neither is
available the field reads unavailable with the reason. Dispersion measures how much the
model's own prediction moves, not how likely the result is to be correct, and it is
not a calibrated probability.

## 9. Heatmaps are not evidence

Grad-CAM highlights the input regions that influenced a model's output. It can be
noisy, it can highlight unrelated anatomy, and it does not establish that a region is
abnormal. For a saturated model the gradient vanishes and the map is empty — the
platform reports that rather than presenting a blank image as an explanation.

## 10. Ensembles need identical inputs

Probability fusion is only defined between models that consume the same tensor:
identical channel count, resolution and normalisation. A 4×256×256 model and a
1×512×512 model cannot be averaged, and the platform refuses instead of resizing one
into the other. Consequently the shipped four-channel checkpoint cannot be averaged with
anything, and the notebook ensemble's three members (4×240×240, per-channel z-score) never
enter the same group as the 4×256×256 or 1×512×512 checkpoints: the compatibility endpoint
groups by `(channels, height, width, normalisation)` and reports the groups it could not fuse.

Two ensembles exist and they are different things:

* the **probability ensemble** (`/api/analysis`, `enable_ensemble`) averages the sigmoid
  outputs of models that share an input contract. It needs at least two compatible models,
  which this deployment does not have, so it reports the reason instead of a fused mask;
* the **stacked ensemble** (`enable_stacked_ensemble`, `GET /api/ensemble/stacked`) is
  `FINAL ENSEMBLE NB.ipynb`'s own learner: three segmentation members' statistics plus a
  radiomics CNN feed a `GradientBoostingClassifier`. It is not a probability average and it
  is not a second opinion on the segmentation. Sections 18-21 describe what it can and
  cannot support.

## 11. Weights are only as good as the data behind them

Validation-calibrated fusion weights, thresholds and any future model selection are
estimated on a split of a particular cohort. They do not transfer to a different
scanner, protocol or population, and the artefacts record the dataset version so a
reported number can be traced to the data that produced it.

## 12. Evaluation protocol

Training, validation and test splits are separated at the patient level, but the
quality of the split is limited by the identifiers supplied. If a cohort contains the
same patient under two identifiers, no tool can detect it. The leakage checklist in
`docs/DATASET.md` exists for that reason.

## 13. Operational limits of this deployment

CPU-only inference (0.46–0.58 s per 512×512 image on 2 cores), SQLite for storage,
synchronous analyses, and no user accounts. Analyses are addressed by identifier and
the schema contains no patient-identifying column, but there is no authentication
layer: deploy this behind your own access control before it holds anything sensitive.

## 14. The classification label is a dataset annotation

The classifier predicts the label the dataset records — for BraTS that is the WHO grade group
(`HGG` / `LGG`), a patient-level annotation in `name_mapping.csv`. It is not a malignancy
verdict, not a clinical stage, not a prognosis and not a substitute for histopathology. The
platform enforces this in the code, not only in the documentation: the mask never supplies the
class, an unknown grade stays unlabelled instead of being coerced, and a label with no recorded
meaning cannot be served. Two consequences worth stating plainly:

* A model trained on one cohort's grade annotation learns that cohort's annotation practice,
  including any label noise in it. High accuracy on such a table is not evidence of clinical
  validity.
* Class balance matters. BraTS 2020 is dominated by HGG cases, so a classifier can look strong
  on accuracy while being uninformative on the minority class; the platform reports macro F1 and
  the per-class table next to accuracy for exactly that reason.

## 15. Models trained on synthetic fixtures are not models

The BraTS pipeline, the radiomics feature table and the classifier were all executed against a
synthetic fixture so that their plumbing could be verified without the dataset. Fixture labels
are invented and fixture masks are blobs, so a fixture-trained artefact must never be installed
in `data/models/classification/` for anything other than a demonstration, and no metric from it
may be quoted. The artefact manifest records the source table, which makes that traceable.

## 16. TensorFlow is optional, and it must not share a process with `torch.compile`

The four-class image classifier needs TensorFlow/Keras. Nothing else in the platform does, so
the import is lazy: `/api/capabilities`, the model card and every PyTorch endpoint work on an
environment where TensorFlow is not installed at all, and `POST /api/classification/image`
answers `503 dependency_missing` with the install hint only when a checkpoint is actually
present and a prediction is requested.

In this development environment, `import tensorflow` followed by `torch.compile` - which imports
`torch._dynamo` and then `triton` - **segfaults the interpreter** (an incompatibility between
the two native stacks, not an application bug; the installed `triton` 3.8 does not match
`torch` 2.4.1). Consequences, all of them honoured by this repository:

* the API never calls `torch.compile`, so serving a request cannot trigger it;
* the training notebook may compile, because it runs in its own process;
* the test suite runs the TensorFlow serving path in a **subprocess**
  (`tests/tf_image_classifier_driver.py`) instead of importing TensorFlow into the pytest
  process, which is also why the suite passes on a machine without TensorFlow.

If you hit this in your own environment, align the versions (`pip install "triton==3.0.0"` for
torch 2.4.x) or keep the two stacks in separate processes - never in the same training script.

## 17. The four-class classifier's metrics are the training run's, not this deployment's

`brain_tumor_xception` reports `test_accuracy 0.9091`, `macro_f1 0.9067`, `weighted_f1 0.9072`
and `best_val_accuracy 0.9392` **because its `metadata.json` records them**. The test split they
were computed on is not shipped with the weights, so this deployment cannot re-measure them and
does not present them as verified. Every surface that prints them (`/api/capabilities`, the
model card, the classification page, the PDF report) carries the provenance sentence saying so.
The values are also class-imbalanced averages of a specific split; they say nothing about
performance on another scanner, another protocol or another population.

## 18. The notebook ensemble's target is a mask-size proxy, not a grade

`FINAL ENSEMBLE NB.ipynb` builds its label as `(mask pixel count > 100)`
(`labels = (np.array([m.sum() for m in masks]) > 100).astype(int)`) on slices whose ground-truth
mask already has at least 50 positive pixels. That is an **image-level tumour-size proxy**. The
`Grade` column of `meta_data.csv` is never used to train it; only the notebook's final report
*prints* the two classes under the names `LGG`/`HGG`. The deployment therefore serves the labels
`small_lesion_proxy` / `large_lesion_proxy` and attaches the notebook's own semantics string to
every payload. It is **not** a WHO grade, not a malignancy verdict, not a tumour type, and it is
never combined with — or substituted for — the BraTS HGG/LGG grade classifier or the four-class
Xception tumour-type model. Those are three different tasks with three different label spaces.

## 19. The notebook ensemble's reported accuracy is an upper bound, and it is not ours

Its own run printed `stacked_ensemble_test_accuracy 0.9633` / validation `0.9733`. Two structural
properties of that run inflate it, and both are reported with every prediction:

1. **Target leakage.** The label comes from the ground-truth mask while the meta-features include
   each member's maximum and mean *predicted* probability — statistics of the very same mask. The
   target is partly reconstructible from the inputs;
2. **Split leakage.** The 70/15/15 split is over *slices*, not patients, so near-duplicate slices of
   one case appear on both sides of it.

The split, the masks and the radiologist labels are not shipped with the weights, so this
deployment cannot re-measure those numbers. They are displayed as the training run's record, with
the provenance sentence, and are never restated as the performance of this deployment. **No
accuracy improvement over the members is claimed here**, because no held-out evaluation was
performed to support one.

## 20. Two slots of the notebook's meta-matrix are filled differently at serving time

Both are stated in every payload:

* **Dice.** The notebook computed each member's Dice against the ground-truth mask. No ground truth
  exists for an uploaded scan, so each member's Dice slot holds its Dice against the ensemble's own
  consensus mask (`dice_reference: "ensemble consensus"`). The number is not the notebook's number
  and cannot be compared with it;
* **Feature mask.** The 46 radiomics features come from that same consensus mask instead of a
  ground-truth mask, so the radiomics branch — and therefore the meta-learner — is conditioned on
  predicted masks. On a scan where all three members fail, the consensus is empty and the pipeline
  stops with the reason instead of inventing a feature vector.

Nothing else is substituted. The ten meta columns are reproduced in the notebook's order, the
feature names/order come from the saved schema (never re-derived), and a mismatch in any artefact
width is reported as `incompatible` rather than worked around.

## 21. The stacked ensemble needs all eight artefacts and MONAI

`segnet.pth`, `unet_best.pth`, `raunet.pth`, `radiomics_cnn.pth`, `radiomics_scaler.pkl`,
`radiomics_feature_names.csv`, `meta_scaler.pkl` and `meta_learner.pkl` must all be present in
`models/weights/`. The second member was built by the notebook with `monai.nets.UNet`, so **MONAI**
must be installed in the API environment; without it that member reports `dependency_missing`. A
missing member is never zero-filled: the meta-learner was fitted on a fixed ten-column matrix, and
imputing a column would feed it a distribution it never saw. The record of the notebook's
`SegNet accuracy 28.00 / UNet 75.67 / RAUNet 67.67` row is likewise the run's own record — in that
table the value is the member's Dice, not a classification accuracy, which is exactly the kind of
misreading the provenance strings exist to prevent.

## 22. Not a medical device

No part of this software has been reviewed, cleared or approved as a medical device.
It is a research prototype. Outputs are AI-generated and are not intended to replace
professional medical diagnosis.
