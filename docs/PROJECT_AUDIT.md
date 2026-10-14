# PROJECT AUDIT

Repository: `Guri964/Brain-segmentation-and-classification`
Audit branch: `arena/3cb642d4-brain-segmentation-and-classif`
Base commit: `a9aa1ea` ("Initial commit - existing company project")
Audit date: 2026-10-13
Auditor role: ML / medical imaging engineering review

All statements below were verified by inspecting the files in the repository and by
executing the existing checkpoints with PyTorch 2.4.1 (CPU). Where a statement is an
inference rather than a measurement, it is marked as such.

---

## 1. Repository inventory (actual contents)

```
Copy_of_Ensemble_Framework_for_Brain_Abnormality(1).ipynb   1,520,156 B   50 cells
Screen Recording 2025-01-31 012926.mp4                     23,779,935 B   app demo capture
UNet-[16, 32, 64, 128, 256].pt                              7,771,377 B   checkpoint
bts/__pycache__/*.pyc                                                     build artifact
bts/classifier.py                                              12,870 B   training wrapper class
bts/loss.py                                                     2,843 B   Dice / BCE+Dice losses
bts/model.py                                                    6,758 B   DynamicUNet
features_df.csv                                                16,286 B   100 rows x 11 cols
final.py                                                        5,868 B   Streamlit application
install.bat                                                       850 B   Windows bootstrap
model.py                                                        1,546 B   SegNet
radiomics_cnn_model.pth                                        47,552 B   checkpoint
req.txt                                                           123 B   dependency list
segnet_model.pth                                              614,495 B   checkpoint
temp_image.png                                                116,828 B   589x590 8-bit grayscale MRI export
```

Not present in the repository, although referenced by the task description and/or the notebook:

| Expected artifact | Status |
| --- | --- |
| `Brain Abnormality Prediction Project Document.docx` | **absent** |
| `Project modification [Ruksana].docx` | **absent** |
| `Brain Tumor Detection and Analysis Using Deep Learning, Radiomics, and Explainable AI` | **absent** |
| `A Deep Ensemble-Driven Framework ... Localization and Classification` | **absent** |
| BraTS 2020 HDF5 slices / `meta_data.csv` / `name_mapping.csv` | **absent** |
| `X_train.npy`, `y_train_masks.npy`, `X_val.npy`, `X_test.npy` (Drive artifacts) | **absent** |
| MONAI U-Net checkpoint (`best_u-net_model*.pth`, `unet_final.pth`) | **absent** |
| README / HOW_TO_RUN / tests / backend / database | **absent** |

Consequence: the four DOCX/PDF source documents could not be read. The requirement set
implemented here was reconstructed from (a) the written specification supplied by the client,
(b) the notebook, and (c) the code. This is recorded as an open item in
`docs/IMPLEMENTATION_PLAN.md` (section "Open questions").

---

## 2. Dataset and label availability

The notebook downloads `awsaf49/brats2020-training-data` from Kaggle, reads HDF5 slices
containing an `image` array (BraTS 2020 four-channel slice) and a `mask` array, and derives
grade labels (`HGG` = 0, `LGG` = 1) by joining `meta_data.csv` with `name_mapping.csv` on
`BraTS20_Training_XXX`.

* **No part of that dataset is present in the repository.**
* **No labels, label vectors, split indices, or validation/test artifacts are present.**
* The notebook's sampling strategy would in any case be unusable as a research protocol:
  `os.listdir` order + `limit=2000`, no patient identifier in the split, and
  `train_test_split` over slices (see section 7).
* The radiomics/classification branch cannot be reproduced: `features_df.csv` has no label
  column, and the notebook variable `labels_subset` used for supervised training of
  `radiomics_cnn_model.pth` is **never defined anywhere in the notebook**. The checkpoint was
  therefore produced by an edited or incomplete execution. There is no recorded metric,
  split, seed, or feature-scaling artifact for it.

**Conclusion: supervised classification, any reporting of classification accuracy, and any
patient-level or grade-level claim are blocked by missing data. The classification serving
path must remain disabled until a labelled table and label mapping are supplied.**

---

## 3. Checkpoint audit (verified by loading and executing)

### 3.1 `UNet-[16, 32, 64, 128, 256].pt`

| Property | Value | Evidence |
| --- | --- | --- |
| Architecture | `DynamicUNet` from `bts/model.py` (modified 4-block U-Net, no dropout, no attention) | state-dict keys `conv1_1 ... conv10` match the class exactly; 46 tensors |
| Serialization | **legacy torch pickle** (zip-less, magic `0x1950a86a20f9469cfc6c`), tensors stored on `cuda:0` | `torch.load(..., map_location="cpu", weights_only=False)` required; `weights_only=True` cannot parse the legacy container |
| Input | 1 channel, 512 x 512 (`conv1_1.weight` = `(16, 1, 3, 3)`) | tensor shape |
| Output | 1 channel, 512 x 512, **sigmoid applied inside `forward`** (probability, not logit) | `bts/model.py: output = F.sigmoid(self.conv10(conv9))`; measured output range `[0.0, 0.99999]` |
| Threshold used by the original app | 0.5 | `final.py`, `bts/classifier.py: predict` |
| Preprocessing in the original app | grayscale, resize 512 x 512, `ToTensor()` -> `[0,1]`, **no z-score** | `final.py: preprocess_image` |
| Load result | **loads and runs**; mean wall time ~0.6 s per 512 x 512 image on 2 CPU threads | executed |
| Sanity check | on the repository's own `temp_image.png` the model produces a structured, brain-shaped mask covering 1.8 % of pixels (not empty, not saturated) | executed; the mask is not a ground-truth comparison and no accuracy claim is made |

Reusable: **yes**, for single-channel 512 x 512 grayscale MRI images.

### 3.2 `segnet_model.pth`

| Property | Value | Evidence |
| --- | --- | --- |
| Architecture | `SegNet` from `model.py` / notebook cell 22 (2-stage encoder, bilinear decoder, **no skip connections**) | 23 tensors: `encoder.{0,1,4,5}`, `decoder.{1,2,5}` |
| Serialization | zip container named `best_segnet_model`, tensors on `cuda:0` | archive member name matches `torch.save(model.state_dict(), f"best_{name.lower()}_model.pth")` from notebook cell 29 |
| Input | **4 channels**, spatial size divisible by 4 (trained at 256 x 256) | `encoder.0.weight` = `(64, 4, 3, 3)` |
| Output | 1 channel **logits** - the sigmoid in the model was commented out and is applied by the loss and by every inference path in the notebook | `model.py` (commented `nn.Sigmoid()`), notebook cells 29/33/37 apply `torch.sigmoid` explicitly; measured output range `[-350, -46]` for random input |
| Documented preprocessing | resize 256 x 256 (`INTER_LINEAR`, mask `INTER_NEAREST`), then **single global z-score over all four channels jointly** (mean and std of the whole `(H,W,4)` block, not per channel) | notebook cell 27 `normalize_image` |
| Threshold | 0.5 after sigmoid | notebook cell 37 |
| Load result | **loads and runs** (0.23 s per 256 x 256 x 4 tensor on CPU) | executed |
| Response check | with random input the sigmoid output is ~0 (expected); with z-scored real MRI data the model produces a non-degenerate map. **No labelled BraTS slice is available, so no Dice/IoU can be computed and none is claimed.** | executed |

Reusable: **yes**, but only for a genuine four-channel input. Feeding a one-channel image by
duplicating it four times is invalid and is rejected by the new preprocessing layer.

### 3.3 `radiomics_cnn_model.pth`

| Property | Value |
| --- | --- |
| Architecture | `TumorClassifier`: `Linear(21->128) - ReLU - Linear(128->64) - ReLU - Linear(64->2)` |
| Evidence | `fc.0.weight` = `(128, 21)`, `fc.2.weight` = `(64, 128)`, `fc.4.weight` = `(2, 64)` |
| Expected input | **21 radiomics features**, un-standardised, NaNs filled with 0 |
| Compatible feature table present? | **No.** `features_df.csv` has 11 columns |
| Feature-name ordering artifact? | **None present.** The column order of the training table is not recorded anywhere except the checkpoint's 21-wide weight matrix |
| Labels used | `labels_subset` - undefined in the notebook; the intended mapping (`Grade`: HGG=0, LGG=1) is *grade*, not "tumour / no tumour" as the comment in cell 40 claims |
| Metrics / calibration / scaling artifacts | **none** |

**Conclusion: the radiomics classifier is not loadable in a meaningful way against the
artifacts that exist. It must be treated as an unvalidated, incompatible artifact. The new
classification service refuses to serve predictions from it and reports the incompatibility
explicitly.**

### 3.4 `features_df.csv` (scientific validity check)

100 rows, 11 columns: 5 `diagnostics_*`, 6 `original_shape_*`. No first-order and no texture
features are present even though the notebook calls `extractor.enableAllFeatures()`; the
notebook filter `isinstance(v, (int, float))` drops NumPy scalar feature values, which
removed essentially the entire feature classes.

More seriously, the extraction geometry is invalid:

* The notebook stacks the four MRI modalities on the *z* axis: `X_train[i]` has shape
  `(4, 256, 256)` and is fed to SimpleITK as a 3-D "volume".
* The 2-D mask is transposed to `(1, 256, 256)` by the notebook's axis heuristic and then
  **edge-padded** from 1 to 4 slices by `match_slices`, so the segmentation is replicated
  across the "z" axis.
* Confirmed by the data: `diagnostics_Image-original_Mean` ~ 1e-9 (z-scored input),
  `original_shape_VoxelVolume` identical to `diagnostics_Mask-original_VoxelNum` (spacing 1),
  `diagnostics_Mask-original_VolumeNum` 1-14 (connected components of a replicated 2-D mask).

Therefore these values are *not* radiomics of an anatomical lesion: the z axis is the
modality axis, and the mask is duplicated. The table is retained as an archival artifact but
is not used to train or serve anything.

---

### 3.5 `brain_tumor_xception_final.keras` + `metadata.json` (four-class tumour type)

| Property | Value |
| --- | --- |
| Source | `firstnb.ipynb` (supplied notebook, committed at the repository root): Xception + global average pooling + 4-way softmax |
| Present in this workspace? | **No.** Neither file exists here, in `models/weights/`, or in `data/models/` |
| What is therefore unverifiable | The real architecture summary, the real parameter count, the real class order and any real prediction. Nothing is simulated in its place |
| What *is* verified | The serving contract: the endpoint, the metadata validation, the NumPy preprocessing (bit-identical to the Keras function, max abs difference `0.0`), and the end-to-end path driven with a locally built stand-in that has the notebook's exact input/output signature |
| Metadata contract enforced | `classes` (>= 2), `class_to_index` (exact permutation), `input_size` (3 channels), `preprocessing` (must name Xception) |
| Reported metrics | 0.9091 test accuracy / 0.9067 macro F1 / 0.9072 weighted F1 / 0.9392 best validation accuracy - **as recorded in the metadata of the training run**, re-measured nowhere |
| Served by | `backend/app/ml/image_classifier.py`, `backend/app/services/image_classification_service.py`, `POST /api/classification/image`, UI *Classification* page |

## 4. Existing functionality (works as-is)

| Component | Status | Notes |
| --- | --- | --- |
| `bts/model.py: DynamicUNet` | functional | exact architecture definition required to load `UNet-[...].pt` |
| `model.py: SegNet` | functional | exact architecture definition required to load `segnet_model.pth` |
| `bts/loss.py: DiceLoss`, `BCEDiceLoss` | functional, correct | usable as a baseline objective |
| `bts/classifier.py` | functional training wrapper | `train/predict/_dice_coefficient`; assumes 512 x 512, 1 channel, uses `tensorboardX` |
| Notebook: HDF5 loading, training loop, ensemble averaging, radiomics extraction, Grad-CAM | partially functional | see defects below |
| Streamlit `final.py` | runs, U-Net + (broken) Grad-CAM only | see defects below |

## 5. Defects and incompatibilities found

1. **`final.py` never uses SegNet.** `segnet` is constructed and `.eval()`-ed but never called;
   the UI exposes only the U-Net and a Grad-CAM tab. The advertised "U-Net & SegNet" pipeline
   is not implemented in the deployed application.
2. **No ensemble exists in the application**, although the notebook fuses the two models.
   With the available checkpoints an ensemble is *not possible on one input*: SegNet needs
   4 x 256 x 256, the shipped U-Net needs 1 x 512 x 512. They are not input-compatible.
3. **SegNet output semantics are not honoured by the app**: the checkpoint emits logits, so a
   caller that thresholds raw output at 0.5 would produce an all-foreground mask.
4. **Grad-CAM implementation is broken/inaccurate**: `register_backward_hook` is deprecated and
   unreliable; the target score is `output.sum()` (the activation mass of the whole map, not the
   foreground/class score); `.eval()` is not set; there is no division-by-zero guard; and the
   heatmap is attributed to whichever model was passed in, so a U-Net map can easily be
   presented as a SegNet explanation.
5. **Notebook ensemble is invalid as written**: cell 31 averages *logits of SegNet* with
   *probabilities of the MONAI U-Net* without applying sigmoid to SegNet; cell 37 fixes this.
   Both variants exist in the notebook, so the documented `P = (P_SegNet + P_UNet)/2` cannot be
   trusted unless the correct cell is used.
6. **Notebook train/val/test split is slice-level, not patient-level, and disables data leakage
   protection**: masks and images come from the same volumes in all three splits; moreover
   `train_test_split` is applied to `X_filtered` *after* filtering for tumour slices, and
   `y_train_masks = (y_train_masks > 0)` is applied only to the first 2000 loaded slices.
7. **One `best_val_loss` variable is shared by both models** in the notebook training loop, so
   the "best" U-Net checkpoint can be overwritten according to SegNet's validation loss; early
   stopping (`< 0.001`) applies to both models simultaneously.
8. **`bts/model.py` imports `torchsummary` at module scope** although `summary()` is optional and
   unused by `final.py`. This makes the whole application fail at import time if an unused
   dependency is missing (observed: `ModuleNotFoundError: torchsummary`).
9. **`req.txt` is broken**: `streamit` (typo for `streamlit`) makes installation fail; there is no
   Flask/JS/pytest/report/DB dependency list; `install.bat` pins Python 3.10 while nothing in the
   code requires that version.
10. **`install.bat` has no error handling** for the virtual-environment creation and no checksum
    on the downloaded Python installer.
11. **No tests, no CI, no `.gitignore`.** `bts/__pycache__/*.pyc` is committed.
12. **No persistence, no history, no reports, no backend API.** Everything happens in a Streamlit
    session; nothing survives a page reload except the last uploaded image (`temp_image.png` is
    the leftover artifact of such a session).
13. **No uncertainty, no calibration, no model registry, no version metadata** anywhere.
14. **No clinical-safety language**: the app calls the output "Brain Tumor Detection" while the
    only verified behaviour is a binary segmentation map produced by an unvalidated model.
15. **The radiomics path in the notebook produces a 2-D mask on a 4-slice "volume"** (section 3.4)
    and is therefore not reusable as the deployed feature extractor.
16. **`final.py` writes/reads nothing but the last upload**; `st.image(..., use_column_width=True)`
    is deprecated in current Streamlit; processing is unconditional (no validation of modality
    count, size, or dynamic range) and errors are not handled.
17. **`features_df.csv` and `radiomics_cnn_model.pth` mismatch**: 11 columns vs 21 expected inputs.

## 6. Reusable assets

| Asset | Reuse decision |
| --- | --- |
| `UNet-[16, 32, 64, 128, 256].pt` | **Reused** (1-ch, 512 x 512, probability output, threshold 0.5, `[0,1]` normalisation). |
| `segnet_model.pth` | **Reused** (4-ch, 256 x 256, logit output, z-score global normalisation, sigmoid then threshold 0.5). |
| `SegNet` / `DynamicUNet` source | **Reused verbatim** (re-implemented in `backend/app/ml/architectures/` with the same module paths so the checkpoints load unchanged, minus the optional `torchsummary` import). |
| `bts/loss.py` | **Reused** as `DiceLoss` / `BCEDiceLoss` baseline objectives. |
| Notebook radiomics extraction | **Rejected** (invalid geometry, dropped feature classes); replaced by a validated extractor. |
| `radiomics_cnn_model.pth`, `features_df.csv` | **Preserved but marked incompatible**; the classifier is disabled until labels and a matching feature schema exist. |
| `temp_image.png` | **Reused** as the bundled single-channel demo/test input. |
| `install.bat`, `req.txt` | Kept for reference; superseded by `backend/requirements.txt`, `install.bat` updated. |

## 7. Dataset requirements (what a full research run needs)

1. **BraTS 2020 training data** (369 patients, four co-registered modalities T1, T1ce, T2, FLAIR,
   240 x 240 x 155, 1 mm isotropic) or an equivalent four-modality cohort, converted to HDF5 or
   NPZ with the modality order recorded.
2. **Patient identifiers** (`BraTS20_Training_XXX`) in the metadata, to permit patient-level
   splitting. Without them the pipeline refuses to claim patient-level validation and logs the
   limitation.
3. **Reference masks** for every evaluation case (BraTS `mask` or manual delineation), plus the
   BraTS label convention (1 = NCR, 2 = ED, 4 = ET) if multi-class evaluation is desired.
4. **Grade labels** (`HGG`/`LGG` from `name_mapping.csv`) if any classification is to be trained;
   these are *grade* labels, not malignancy or tumour-presence labels.
5. **A defined external test cohort** that is never used for model selection, for the "reserved
   test set" requirement.
6. For the shipped single-channel U-Net, a matched single-channel 512 x 512 dataset is needed if
   the new 4-channel architectures are to be trained on the same distribution.

Memory/time estimate recorded for planning (not executed): the notebook loads arrays into RAM
(`limit=2000` slices of 4 x 256 x 256 float32 ~ 0.5 GB, plus masks), which is feasible on CPU
for research-scale runs; a 50-epoch SegNet/attention-U-Net run at 256 x 256 batch 8 on CPU is on
the order of hours per model, so it was **not** attempted here (2 CPU cores, 3 GB RAM, no GPU in
this environment). `scripts/*.py` therefore ship as reproducible entry points with
`--max-cases`, `--resume`, and `--dry-run` controls.

## 8. What must be retrained

| Component | Retrain? | Reason |
| --- | --- | --- |
| Single-channel U-Net | No | loads and runs; reusable baseline |
| SegNet (4-ch) | No (reuse) | loads and runs; the only four-channel model available |
| Four-channel partner for ensemble | **Yes, when data exists** | the shipped MONAI U-Net checkpoint is absent, so no input-compatible partner exists. The new `RAUNet` (residual attention U-Net, deep supervision, boundary-aware loss) is the intended partner; `scripts/train_unet.py` trains it |
| Ensemble weights (confidence-aware fusion) | **Yes, when validation data exists** | weights must be estimated from validation data only; until then the system runs equal weights and says so |
| Radiomics classifier | **Yes, when labels exist** | current checkpoint is incompatible (21 vs 11 features) and untraceable |
| Radiomics extractor | No (new implementation) | replaced by an IBSI-aligned extractor with PyRadiomics as an optional backend |

## 9. Risk register

| Risk | Impact | Mitigation implemented |
| --- | --- | --- |
| No labelled data | no accuracy claims possible | evaluation module returns `"Not evaluated"`; UI never fabricates metrics |
| Two models on incompatible inputs | ensemble could silently produce meaningless output | explicit input-spec compatibility check; ensemble endpoint returns `unavailable` with reasons |
| 1-channel image into 4-channel model | physically meaningless prediction | upload validation rejects it; modality-count is explicit in the UI |
| 2-D masks treated as volumes | wrong "volume" numbers | extractor records `dimensionality` and never prints `cm^3` for 2-D |
| Unvalidated classifier | false clinical signal | classifier gated behind a validated artifact + label mapping + metrics file |

---

## 10. Integration status of every artefact (audited by execution)

Produced by `python scripts/audit_integration.py`, which loads each checkpoint through the same
registry the API uses and runs one forward pass on the documented input shape. A successful
forward pass proves the artefact is loadable and contract-compatible; it proves nothing about
accuracy, and no accuracy number is produced by it.

| Artefact | State | Evidence |
| --- | --- | --- |
| `segnet_model.pth` | **loads and runs** | 23 tensors, 151,105 parameters, forward pass on `1 x 4 x 256 x 256` returns logits `[-89.71, -15.19]` -> probability ~`0.0` (synthetic input, sigmoid applied outside the network as the notebook does) |
| `UNet-[16, 32, 64, 128, 256].pt` | **loads and runs** | 46 tensors, 1,940,945 parameters, forward pass on `1 x 1 x 512 x 512` returns probabilities in `[0.0000, 0.0080]` (sigmoid inside the network) |
| `brain_tumor_xception_final.keras` + `metadata.json` | **absent** | reported as `checkpoint_missing` with the directory to install into; serving path verified separately with a stand-in of the same contract |
| `radiomics_cnn_model.pth` | **present, not usable** | first layer is `Linear(21 -> 128)`; the feature table in the repository has 11 columns, no scaler, no column order and no label map were saved with it. Reported as incompatible; never fed a padded vector |
| MONAI U-Net | **served as a member of the stacked ensemble** | registered as `unet_ensemble_240` (1,604,024 parameters) for `unet_best.pth`; needs MONAI installed and is reported as `dependency_missing` without it. The shipped `UNet-[...].pt` is a *different* model (1-channel Dynamic U-Net) and is unchanged |
| `raunet_brats_4ch.pt` | **absent** | produced by `scripts/train_unet.py`; this is the company RAUNet, unrelated to the notebook's `raunet.pth` member |
| Probability ensemble | **not formable in this deployment** | the two present company checkpoints take different tensors (`4x256x256` z-scored vs `1x512x512` in `[0, 1]`); `/api/ensemble/compatibility` explains this and refuses to fuse across contracts |
| Stacked ensemble (`FINAL ENSEMBLE NB.ipynb`) | **served; artefacts absent** | eight files, all in `models/weights/`; with them installed the audit executes the full pipeline on a synthetic slice (three members → consensus → 46 features → 10 meta features → label). Without them it reports `artifacts_missing` and names each file. See §10.1-10.2 |

Radiomics extraction was verified by executing it: the native backend produces **68 features**
(diagnostics 17, first-order 19, GLCM 10, GLRLM 9, shape 13) on a synthetic region.

### 10.1 The two notebooks, read end to end

Both supplied notebooks were read in full before anything was changed, because models from
different notebooks are not automatically compatible.

| | `firstnb.ipynb` | `FINAL ENSEMBLE NB.ipynb` |
| --- | --- | --- |
| Task | Tumour **type**, four classes (glioma / meningioma / pituitary / no tumour) | The notebook's own label `(mask pixel count > 100)` on BraTS 2020 slices |
| Input | RGB 299×299×3, `x / 127.5 - 1` | Four channels (T1, T1ce, T2, FLAIR), 240×240, per-channel z-score |
| Output | Softmax over four classes | `GradientBoostingClassifier` over ten meta features |
| Artefacts | `brain_tumor_xception_final.keras` + `metadata.json` | `segnet.pth`, `unet_best.pth`, `raunet.pth`, `radiomics_cnn.pth`, `radiomics_scaler.pkl`, `radiomics_feature_names.csv`, `meta_scaler.pkl`, `meta_learner.pkl` |
| In this deployment | Serving path complete; **no artefact installed**, so no class is predicted | Served end to end; **no artefact installed**, so it reports `artifacts_missing` |

Findings that shaped the integration:

1. **The two notebooks share nothing that could be fused.** Different input tensors, different
   label spaces, different purposes. They are served as separate capabilities and are never
   combined; the four Xception classes are never derived from a segmentation mask or a binary label.
2. **The ensemble's target is a mask-size proxy, not a grade.** `labels = (np.array([m.sum() for m
   in masks]) > 100).astype(int)` restricted to slices whose GT mask already has ≥ 50 pixels. Only
   the notebook's *report* prints those classes as `LGG`/`HGG`; the `Grade` column is never used for
   training. The deployment therefore serves `small_lesion_proxy` / `large_lesion_proxy`.
3. **The meta-learner consumes exactly ten columns** — three members × {Dice, max probability, mean
   probability} plus the radiomics CNN's label — in the order the notebook built them. That is now a
   frozen contract, checked at load time against `radiomics_feature_names.csv`, the radiomics scaler,
   the CNN's first layer, the meta-scaler and the meta-learner's `n_features_in_`.
4. **Two slots cannot be reproduced from one image.** The notebook filled each member's Dice from the
   ground-truth mask and extracted the 46 features from the ground-truth mask. At serving time there
   is no ground truth, so the Dice slot holds the member's Dice against the consensus mask and the
   features come from that same consensus mask. Both substitutions are printed in every payload
   (`dice_reference`, `feature_mask_source`) rather than being left implicit.
5. **The notebook's own numbers are reported, not re-measured.** `stacked_ensemble_test_accuracy
   0.9633` and friends are what its run printed, on a *slice-level* split of 2,000 slices, with the
   label derived from the same mask the meta-features summarise. They travel with a provenance
   sentence and a leakage note; no accuracy improvement is claimed for this deployment, because no
   held-out evaluation was possible without the data.
6. **The members are not the shipped checkpoints.** `segnet.pth` shares the architecture of
   `segnet_model.pth` but not its weights (the root checkpoint's key set differs by 20 missing / 23
   unexpected tensors), and it is a 4×240×240 model. It is therefore registered as a *separate*
   model (`segnet_ensemble_240`) and never substituted for the company checkpoint.
7. **The notebook's own pipeline has loose ends, recorded rather than patched over:** its
   `meta_X_*` cells recompute a different split and overwrite `y_val`; the report cell references
   `sn_val` / `un_val` / `ra_val`, which are never defined; the "saved names" list has no `raunet`
   entry. None of these change the served pipeline, and none is silently "fixed" in the notebook.

### 10.2 The notebook models, verified by execution

`tests/test_final_ensemble.py` (41 tests, torch-only) writes stand-ins with the notebook's real
architectures and exercises the whole path; `python scripts/audit_integration.py` performs the same
kind of execution check on the real artefacts once they are installed, including one full
end-to-end run of the pipeline on a synthetic four-channel slice.

| Check | Result |
| --- | --- |
| 46-feature extractor vs SciPy on 200 random regions | max abs difference `1.67e-16` (skew), `1.78e-15` (kurtosis), `0.0` (the rest) |
| Feature order | identical to the notebook's list, `len(feature_order()) == 46` |
| Empty mask | returns `None`, exactly like the notebook |
| Parameter counts | SegNet 151,105 · MONAI U-Net 1,604,024 · RAUNet 1,994,794 · RadiomicsCNN 54,082 |
| Registry contracts | three members at `1×4×240×240` `per_channel`; the CNN at 46 features; the company models unchanged at `1×4×256×256` / `1×1×512×512` |
| Refusals | missing artefact → `artifacts_missing`; missing member → `members_missing`; wrong tensor → `ValidationError`; empty consensus → `ModelUnavailableError`; schema/scaler/CNN of different widths → `incompatible` |
| HTTP path | `POST /api/analysis` with four modalities + `enable_stacked_ensemble=true` → 201, four models completed, `stacked_ensemble.available: true`, 10 meta features, 46 radiomics features; single-channel + the flag → 422 |
| Determinism | identical prediction and identical feature/scaled digests on repeated runs, and members are reused from the analysis instead of recomputed |

One genuine defect was found and fixed by this pass: the service's standalone `predict()` path (used
by the audit script and by any caller without precomputed member results) handed a NumPy array to
the segmentation service, which expects a tensor; `_TensorWrapper` now converts once, and
`test_a_numpy_slice_is_accepted_when_the_members_have_to_be_run_here` locks it in.
