# BraTS 2020: what this repository expects and what it does with it

This document describes the contract between the BraTS 2020 training data and the scripts in
`scripts/`. Everything here was exercised against a **synthetic fixture** that reproduces the
real layout (see "Synthetic fixture" below). No real BraTS file was available on the delivery
machine, so no number produced from the fixture — Dice, accuracy, AUC, anything — is a result.

## 1. Where the data comes from

* BraTS 2020 training data, mirrored on Kaggle as `awsaf49/brats2020-training-data`
  (367 patients, one HDF5 file per 2-D slice, plus `name_mapping.csv`).
* Channel order inside each slice: **T1, T1ce, T2, FLAIR**, stored as `(240, 240, 155)` volumes
  in the original NIfTI release and as `(240, 240, 4)` per slice in the HDF5 mirror.
* Segmentation: `(3, 240, 240, 155)` in NIfTI; per-slice three-plane binary stacks in the
  mirror. The label meanings are **1 = NCR/NET, 2 = oedema, 4 = enhancing tumour**.
* `name_mapping.csv` carries `Grade` (`HGG` / `LGG`) and `BrATS_2020_subject_ID`. The grade is a
  **patient-level annotation**.

### Downloading it

```bash
# Kaggle CLI, once your own credentials are configured
kaggle datasets download -d awsaf49/brats2020-training-data -p ./data/brats
unzip ./data/brats/brats2020-training-data.zip -d ./data/brats
```

The delivery machine could not reach `kaggle.com` (network egress is restricted to PyPI, npm and
GitHub), so this download was never performed here. Every command below was instead verified
against the synthetic fixture, which is why no metric is claimed anywhere in this repository.

## 2. Expected layout

```
<root>/
  BraTS2020_training_data/content/data/
    volume_1/volume_1_slice_0.h5
    volume_1/volume_1_slice_1.h5
    ...
    volume_2/volume_2_slice_0.h5
  name_mapping.csv
```

The NIfTI release is also recognised: one directory per patient containing
`*_t1.nii.gz`, `*_t1ce.nii.gz`, `*_t2.nii.gz`, `*_flair.nii.gz` and `*_seg.nii.gz`.

## 3. Inspect before you train

```bash
/home/user/.venv/bin/python scripts/inspect_brats.py \
  --root ./data/brats \
  --grade-table ./data/brats/name_mapping.csv \
  --examples 3 \
  --json data/artifacts/brats_inspection.json
```

The inspector is read-only. It reports the layout it found, HDF5 dataset keys, image shapes and
channel order, how each mask is encoded, the patient identifiers it can derive, grade counts and
the input contract of every registered model, so you can see whether your copy can feed
`segnet_brats_4ch` (4 x 256 x 256), `unet_dynamic_512` (1 x 512 x 512) or neither. It exits with
code 2 when no recognised layout is found instead of guessing.

Two details it deliberately does not do: it never reorders channels (the stored order is
reported and preserved) and it never asserts which tumour sub-region a mask plane holds when the
file does not declare it. A one-hot mask without declared labels is reported as
`channel_labels_declared: false`, and only the union of the planes is used downstream. Pass
`--mask-channel-labels 1,2,4` to `prepare_brats.py` when you know the order.

## 4. Build a manifest

```bash
/home/user/.venv/bin/python scripts/prepare_brats.py \
  --root ./data/brats \
  --out data/manifests/brats2020.json \
  --grade-table ./data/brats/name_mapping.csv \
  --splits
```

* `case_id` is `{volume_key}_{slice_index}`, `patient_id` is the volume key, so a slice never
  crosses a split boundary.
* Slices whose mask has no foreground are dropped unless you pass `--slice-filter all`; they
  carry no supervision and would dominate the loss.
* `--max-slices-per-volume N` keeps the manifest small while you are smoke-testing.
* `--splits` writes `splits` plus `split_metadata` (method, ratios, per-split class distribution,
  patient overlap). The builder refuses to produce a manifest whose splits share a patient.
* Labels come from the grade table only. A case with no matching grade stays **unlabelled**
  rather than being guessed; how many cases that affects is logged and stored.

## 5. Train and evaluate

```bash
/home/user/.venv/bin/python scripts/train_segnet.py \
  --data data/manifests/brats2020.json \
  --size 256 --epochs 60 --batch-size 8 \
  --lazy-loading --model-id segnet_brats_4ch --data-dir data

/home/user/.venv/bin/python scripts/extract_radiomics.py \
  --data data/manifests/brats2020.json \
  --out data/manifests/brats_features.csv

/home/user/.venv/bin/python scripts/train_classification.py \
  --features data/manifests/brats_features.csv \
  --model-id radiomics_grade_classifier \
  --target-description "BraTS 2020 HGG vs LGG (WHO grade group)"
```

`--lazy-loading` reads each case from disk inside the data loader instead of holding the whole
dataset in memory; it is switched on automatically as soon as any training case comes from an
HDF5 slice or a NIfTI volume. `--val-cases` caps the validation pass while you are iterating.

## 6. Label semantics that the platform enforces

| Rule | Why |
| --- | --- |
| The class comes from the grade table, never from the mask | A mask cannot encode a grade; deriving one would be an invented label |
| `HGG` / `LGG` is described as the **WHO grade group** | The dataset records a grade annotation, not malignancy, stage or prognosis |
| Unknown grades stay `None` | `normalise_grade` returns `None` for anything that is not HGG/LGG; the case is excluded from classification rather than coerced |
| Whole tumour is "any label > 0" | The union is the supervised target for segmentation; sub-region identity is only claimed when the file declares it |
| Patient-level splits are mandatory | BraTS slices from one patient are near-duplicates; splitting by slice leaks |

## 7. Synthetic fixture (for pipeline checks only)

```bash
/home/user/.venv/bin/python scripts/make_brats_fixture.py \
  --out /tmp/brats_fixture --volumes 6 --slices-per-volume 4 --empty-every 3
```

It writes a layout identical to the Kaggle mirror (`volume_N/volume_N_slice_M.h5` with `image`
`(H, W, 4)` and a three-plane `mask`, plus `name_mapping.csv`) and prints:

> SYNTHETIC DATA — generated by scripts/make_brats_fixture.py. Labels are made up, masks are
> blobs and any metric computed from them is meaningless. Use it to check that a pipeline runs,
> never to report a number.

That warning exists because the temptation to quote a fixture number is real: the three-epoch
smoke run on this fixture reaches a validation Dice around 0.68, which says nothing at all about
BraTS.

## 8. Related documents

* `docs/DATASET.md` — generic manifest schema, leakage checklist.
* `docs/CLASSIFICATION.md` — the grade classifier: features, splits, artefact format, API.
* `docs/LIMITATIONS.md` — everything that cannot be claimed from a 2-D, CPU-only delivery.
