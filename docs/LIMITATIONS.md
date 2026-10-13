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
into the other. Consequently, this deployment cannot build an ensemble.

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

## 14. Not a medical device

No part of this software has been reviewed, cleared or approved as a medical device.
It is a research prototype. Outputs are AI-generated and are not intended to replace
professional medical diagnosis.
