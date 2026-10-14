# Model weights

This directory is where the trained artefacts live. **Nothing in here is committed to git** —
weights are large, they are not reproducible from source, and they are the researcher's own
files. `.gitignore` covers `/models/weights/`, `*.pth`, `*.pt`, `*.keras`, `*.h5`, `*.hdf5`,
`*.ckpt` and `*.safetensors`; if a checkpoint was committed in the past it has been untracked
with `git rm --cached`, which leaves the file on disk untouched.

The application never downloads and never retrains a model on its own. When an artefact is
absent, the matching capability is reported as *not installed* with the exact path it looked for,
and the endpoint that needs it answers with an error instead of a prediction. Every other
function keeps working.

## Expected layout

```
models/
└── weights/
    ├── brain_tumor_xception_final.keras   # firstnb.ipynb  - four-class tumour type
    ├── metadata.json                      # same notebook - class order, input size, metrics
    ├── segnet_model.pth                   # Copy_of_Ensemble_Framework...ipynb - BraTS SegNet
    ├── UNet-[16, 32, 64, 128, 256].pt     # bts/model.py DynamicUNet - 512x512 grayscale
    ├── radiomics_cnn_model.pth            # radiomics feature classifier (see caveat below)
    │
    │   # FINAL ENSEMBLE NB.ipynb - the stacked ensemble. All eight files are required:
    ├── segnet.pth                         # member 1 - NotebookSegNet, 151,105 parameters
    ├── unet_best.pth                      # member 2 - MONAI U-Net, 1,604,024 parameters
    ├── raunet.pth                         # member 3 - NotebookRAUNet, 1,994,794 parameters
    ├── radiomics_cnn.pth                  # 46-feature classifier head, 54,082 parameters
    ├── radiomics_scaler.pkl               # StandardScaler for the 46 features
    ├── radiomics_feature_names.csv        # the 46 names, in the training column order
    ├── meta_scaler.pkl                    # StandardScaler for the 10 meta features
    └── meta_learner.pkl                   # GradientBoostingClassifier (the stacked head)
```

`segnet_best.pt` (the notebook's "best" checkpoint of the same SegNet) is optional: nothing reads
it, but copy it next to the others if you want the artefact folder to stay complete.

## Where the application looks

`backend/app/config.py` resolves every artefact through one search order:

1. `$BRAIN_WEIGHTS_DIR` if set, otherwise `PROJECT_ROOT/models/weights/`
2. the repository root (the historical location of `segnet_model.pth`, `UNet-[...].pt` and
   `radiomics_cnn_model.pth`)
3. `data/models/`

Individual artefacts can be pinned with `BRAIN_MODEL_SEGNET`, `BRAIN_MODEL_UNET`,
`BRAIN_MODEL_XCEPTION` (streaming, 4-class). `/api/capabilities` and
`/api/models/<model_id>/status` report which of these files were found, which path was used and
what is missing.

## Per-artefact notes

| Artefact | Task | Input contract | Served by |
| --- | --- | --- | --- |
| `brain_tumor_xception_final.keras` + `metadata.json` | Tumour **type** (glioma / meningioma / pituitary / no tumour) | RGB 299×299×3, `x / 127.5 - 1` | `POST /api/classification/image` |
| `segnet_model.pth` | Whole-tumour segmentation | 4 channels (T1, T1ce, T2, FLAIR), 256×256, global z-score, logits | `/api/analysis` with `analysis_type=four_channel` |
| `UNet-[16, 32, 64, 128, 256].pt` | Whole-tumour segmentation | 1 channel, 512×512, `[0, 1]`, sigmoid applied inside the network | `/api/analysis` with `analysis_type=single_channel` |
| `radiomics_cnn_model.pth` | Radiomics feature classifier | **21** radiomics features in the training column order | *not served* — see below |
| `segnet.pth` | Segmentation member 1 of the stacked ensemble | 4 channels (T1, T1ce, T2, FLAIR), 240×240, **per-channel** z-score, logits | the stacked ensemble only |
| `unet_best.pth` | Segmentation member 2 | same as member 1 | the stacked ensemble only |
| `raunet.pth` | Segmentation member 3 | same as member 1 | the stacked ensemble only |
| `radiomics_cnn.pth` + `radiomics_scaler.pkl` + `radiomics_feature_names.csv` | 46-feature radiomics branch | 46 features scaled by the saved scaler, column order from the saved schema | the radiomics branch of the stacked ensemble |
| `meta_scaler.pkl` + `meta_learner.pkl` | Stacked head | 10 meta features (3 members × {Dice, max p, mean p} + the radiomics label) | the stacked head |

`segnet_model.pth` and `UNet-[...].pt` have **different input specs**, so they are never averaged
into one ensemble: `/api/ensemble/compatibility` groups members by their exact input contract and
refuses to fuse across groups.

### `radiomics_cnn_model.pth` is not servable as shipped

The checkpoint's first layer is `Linear(21 -> 128)`, so it needs the 21-column feature table that
the training notebook built. The feature table in this repository (`features_df.csv`) has 11
columns and a different column order, no scaler was saved with the checkpoint, and the notebook's
training labels (`labels_subset`) are the BraTS **grade** (HGG / LGG) — not "tumour / no tumour" as
one comment in that notebook claims. Without the matching table, the column order and the label
mapping, any prediction would be meaningless, so the app reports this model as
*present but unusable* and names the mismatch instead of guessing. To make it servable, export the
notebook's training table (21 columns, same order, same names) together with the label map and
place it next to the checkpoint.

### The stacked ensemble needs all eight files, in `models/weights/`

`FINAL ENSEMBLE NB.ipynb` trains three four-channel segmentation models on 240×240 slices, turns
the ground-truth mask into 46 handcrafted features for a small CNN, and stacks the members'
statistics with that CNN's output in a `GradientBoostingClassifier`. The eight files above are the
frozen artefacts of that run. Their originals are in the extracted
`brain_mri_ensemble_artifacts` folder (on the author's machine,
`C:\Users\Tilak Studio\Downloads\brain_mri_ensemble_artifacts`).

Copy them with **all eight present** — the meta-learner was fitted on a fixed ten-column matrix, so
a missing member would mean imputing a statistic it was never trained on. The service refuses to
run and names the file instead:

```bash
# from the repository root (Linux/macOS)
cp "$HOME/Downloads/brain_mri_ensemble_artifacts"/{segnet.pth,unet_best.pth,raunet.pth,radiomics_cnn.pth,radiomics_scaler.pkl,radiomics_feature_names.csv,meta_scaler.pkl,meta_learner.pkl} models/weights/

# Windows PowerShell
$src = "$HOME\Downloads\brain_mri_ensemble_artifacts"
Copy-Item "$src\segnet.pth","$src\unet_best.pth","$src\raunet.pth","$src\radiomics_cnn.pth","$src\radiomics_scaler.pkl","$src\radiomics_feature_names.csv","$src\meta_scaler.pkl","$src\meta_learner.pkl" -Destination "models\weights\"

# Windows, the same thing with the plain command prompt
for %F in (segnet.pth unet_best.pth raunet.pth radiomics_cnn.pth radiomics_scaler.pkl radiomics_feature_names.csv meta_scaler.pkl meta_learner.pkl) do copy "%USERPROFILE%\Downloads\brain_mri_ensemble_artifacts\%F" "models\weights\"
```

`GET /api/ensemble/stacked` (and `python scripts/audit_integration.py`) then reports
`state: ready` and each file's size and digest. Until then it reports `state: artifacts_missing`
and lists exactly which of the eight are absent. `unet_best.pth` needs **MONAI**
(`pip install monai`) because the notebook built that member with `monai.nets.UNet`; without MONAI
the model is reported as `dependency_missing`, never as broken.

The three members are *not* interchangeable with `segnet_model.pth` or `UNet-[...].pt`: those two
were trained at 256×256/512×512 with a single global z-score, the members at 240×240 with a
per-channel z-score. The compatibility endpoint therefore never groups them together.

## Adding a checkpoint

1. Copy the file into `models/weights/` (or any of the other two locations).
2. Reload the interface, or call `/api/capabilities` - the resolver picks the file up on the next
   request; no restart and no retraining is needed.
3. Check the model card: it must report `state: ready`. If it reports `metadata_missing` or
   `incompatible`, the reason string says exactly which file or which shape disagrees.

## Files at the repository root that are kept on purpose

`final.py`, `model.py`, `bts/`, `features_df.csv`, `temp_image.png`, `install.bat`, `req.txt`,
`Copy_of_Ensemble_Framework_for_Brain_Abnormality(1).ipynb` and `firstnb.ipynb` are the supplied
source artefacts. They are referenced by the architecture modules, the audit documents and the
tests (for example `backend/app/ml/architectures/segnet.py` derives from `model.py`, and
`backend/app/ml/architectures/dynamic_unet.py` from `bts/model.py`), so none of them is dead code.
`Screen Recording 2025-01-31 012926.mp4` (23.8 MB) is referenced only by
`docs/PROJECT_AUDIT.md`; untrack it with `git rm --cached` if the repository should stay small -
nothing in the application reads it.
