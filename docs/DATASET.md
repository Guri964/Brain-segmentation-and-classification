# Dataset preparation

The platform ships no data. Every metric, checkpoint and feature table in this
repository was produced from data that you supply, and each script refuses to run when
what it needs is missing rather than improvising.

---

## 1. What the platform accepts

| Input | Accepted by | Notes |
|---|---|---|
| PNG / JPEG | both pipelines | Explicitly supported and tested |
| `.npz` with four arrays | four-channel pipeline | One file carrying T1, T1ce, T2, FLAIR |
| NIfTI / NRRD / DICOM volumes | requires SimpleITK | Sliced and resampled before use; the slice selection is recorded in the analysis |

The four-channel pipeline **refuses four identical images**. Duplicating one channel is
the single most common way to produce a meaningless but plausible-looking result, so
the backend compares a spatial signature of the uploaded channels and reports which of
them matched.

---

## 2. Manifest format

Every training, calibration and evaluation script reads one JSON manifest. It is the
only place that knows where the imagery lives, so a run can be reproduced by keeping
the manifest, the seed and the code revision together.

```json
{
  "name": "brats-style-cohort",
  "version": "v1",
  "root": "/absolute/path/to/dataset",
  "cases": [
    {
      "case_id": "case-0001",
      "patient_id": "patient-0007",
      "images": ["0007/t1.png", "0007/t1ce.png", "0007/t2.png", "0007/flair.png"],
      "mask": "0007/seg.png",
      "label": 1
    },
    {
      "case_id": "case-0002",
      "patient_id": "patient-0008",
      "images": ["0008/slice.png"],
      "mask": "0008/mask.png"
    }
  ],
  "splits": { "train": [], "validation": [], "test": [] }
}
```

| Field | Required | Meaning |
|---|---|---|
| `name`, `version` | recommended | Recorded in every artefact; bump `version` when the data changes |
| `root` | optional | Relative paths in the manifest resolve against this directory |
| `cases[].case_id` | yes | Unique per case |
| `cases[].patient_id` | yes | The split is made per patient; the scripts exit without it |
| `cases[].images` | yes | One image for single-channel models, four co-registered modalities for four-channel models |
| `cases[].mask` | for training | Binary reference segmentation on the same grid as the images |
| `cases[].label` | for classification | Class label for the labelled radiomics table |
| `splits` | written by `make_splits.py` | Case identifiers per split |

A template with no data is at `data/manifest.example.json`.

### BraTS-style directory trees

If your data follows the usual layout — one folder per patient containing the four
modalities and a segmentation file — the manifest is mechanical:

```bash
python - <<'PY'
import json, pathlib
root = pathlib.Path("/data/brats")
cases = []
for patient in sorted(p for p in root.iterdir() if p.is_dir()):
    images = [str(pathlib.Path(patient.name) / f"{patient.name}_{mod}.png")
              for mod in ("t1", "t1ce", "t2", "flair")]
    cases.append({
        "case_id": patient.name,
        "patient_id": patient.name,
        "images": images,
        "mask": f"{patient.name}_seg.png",
    })
json.dump({"name": "brats-style", "version": "v1", "root": str(root), "cases": cases},
          open("data/manifest.json", "w"), indent=2)
print(len(cases), "cases")
PY

python scripts/make_splits.py --data data/manifest.json
```

---

## 3. Splits

`scripts/make_splits.py` assigns patients, not slices, and stratifies by label when
labels exist:

```bash
python scripts/make_splits.py --data data/manifest.json --train 0.7 --val 0.15 --test 0.15 --seed 20240101
```

The rules it enforces:

* a patient never appears in two splits — the scripts check this again at training
  time and exit if they find a leak;
* the assignment is deterministic for a given seed;
* the test split is written once and then left alone. `scripts/evaluate_models.py` is
  the only script that reads it, and `--select` refuses to run on anything but the
  validation split;
* cohorts too small to support a test split are moved into training with a warning
  rather than producing a metric computed on three patients.

---

## 4. Requirements per capability

| Capability | Needs |
|---|---|
| Segmentation inference | One image (single-channel) or four distinct modalities (four-channel) |
| Retraining a segmentation model | Image–mask pairs, patient identifiers, a training and a validation split |
| Dice / IoU / precision / recall / specificity / HD95 | Reference masks for the reserved test split |
| Radiomics | A non-empty region of interest; pixel spacing for physical units |
| Classification | A labelled feature table: one row per case, one column per feature, a label column with a recorded meaning, and patient identifiers. Around ten cases per class per feature is the usual rule of thumb — with ~60 features that means several hundred labelled cases before a metric is trustworthy |
| Deep-feature fusion | The segmentation checkpoint to freeze, plus the same labelled rows |
| Ensembling | Two or more checkpoints with the **same** channel count, resolution and normalisation |

### How a labelled radiomics table is built

1. Run analyses for every labelled case (the same slice-selection rule for all of
   them), export the feature tables, and concatenate them on `case_id`.
2. Add `label` and `patient_id` columns.
3. Train with `scripts/train_classifier.py --features table.csv --label-column label
   --patient-column patient_id`.

The script writes the ordered feature schema, the standardisation parameters fitted on
the training split only, the label mapping and the held-out metrics into
`data/models/classification/manifest.json`. The API activates the classification
branch only when that file exists, and it refuses to run the model when a feature
listed in the schema is missing from the request.

---

## 5. Leakage checklist

Before believing any number this platform reports, confirm:

- [ ] splits are per patient, not per slice;
- [ ] the test split was not read while choosing models, thresholds or weights;
- [ ] any preprocessing with learned parameters (standardisation, intensity
      normalisation) was fitted on the training split only;
- [ ] the same slice-selection rule was used for every case;
- [ ] augmentation was applied to training data only;
- [ ] `data/models/model_selection.json` names the validation split as the source;
- [ ] `data/models/ensemble_calibration.json` has `"test_split_used": false`.

The evaluation endpoint rejects a calibration file that claims to come from the test
split, and every artefact records the split it used, so these checks can be verified
rather than assumed.
