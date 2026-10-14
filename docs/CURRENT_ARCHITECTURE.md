# CURRENT ARCHITECTURE (as found) AND TARGET ARCHITECTURE

## 1. As-found architecture

```
                    +---------------------------+
  user uploads  --> |  Streamlit  final.py      |  (single process, no API)
  PNG/JPG           |  - preprocess 512 grayscale
                    |  - U-Net inference only
                    |  - Grad-CAM (broken hooks)
                    +------------+--------------+
                                 |
                    +------------v--------------+
                    | bts/model.py DynamicUNet  |  <-- UNet-[16,32,64,128,256].pt (1ch, 512)
                    | model.py    SegNet        |  <-- segnet_model.pth (4ch, 256) UNUSED by the app
                    | bts/classifier.py         |  <-- training wrapper (not used at inference time)
                    | bts/loss.py               |  <-- Dice / BCE+Dice
                    +---------------------------+

  Notebook (offline, Colab):
     BraTS HDF5 -> z-score -> SegNet(4ch) + MONAI UNet(4ch) -> P = (P1+P2)/2
                 -> PyRadiomics (invalid 4-slice geometry) -> features_df.csv
                 -> 21-feature MLP -> radiomics_cnn_model.pth   (labels undefined)
                 -> Grad-CAM

  Persistence: none.  API: none.  Tests: none.  Reports: none.
```

Data flow of the as-found application: `bytes -> PIL RGB -> Grayscale -> Resize(512) -> ToTensor -> U-Net -> threshold(0.5) -> red overlay -> st.image`.

## 2. Findings that drive the redesign

* The two usable checkpoints **do not share an input specification** (1 x 512 x 512 probability vs 4 x 256 x 256 logits). Any architecture that pretends otherwise would silently produce meaningless comparisons.
* The single-channel U-Net is the only model that can legitimately serve a plain PNG/JPG upload.
* Four-channel inference is legitimate only with four genuinely different modality images.
* Grad-CAM, radiomics, uncertainty and evaluation are all absent or invalid, and none of them can be produced inside a Streamlit script without a real service layer.
* There is no persistence layer, so "analysis history", "reports" and "model evaluation" cannot exist in the current design.

## 3. Target architecture

```
frontend/  React 18 + TypeScript + Tailwind (Vite)
  pages: Overview, New MRI Analysis, Segmentation Results, Model Comparison,
         Quantitative Radiomics, Explainable AI, Clinical Decision Support,
         Analysis History, Reports, Model Evaluation, Settings
  services/api.ts  ->  HTTP (relative /api, proxied by Vite in dev)
        |
        v
backend/  Flask application factory
  routes/      health, analysis, models, radiomics, reports, history, evaluation, assistant
  services/    preprocessing, segmentation, ensemble, radiomics, classification,
               explainability, uncertainty, analysis (orchestrator), report, assistant
  ml/          architectures/{segnet,dynamic_unet,attention_unet}, checkpoint_registry,
               gradcam, losses, models (trainable novelty: RAUNet)
  database/    sqlite3 schema + migration runner + repository
  utils/       images, metrics, io, errors
        |
        +--> data/models/      checkpoints (read-only, never overwritten)
        +--> data/uploads/     per-analysis uploaded inputs
        +--> data/artifacts/   masks, overlays, heatmaps, feature tables (PNG/CSV/JSON)
        +--> data/reports/     generated PDFs
        +--> data/app.db       SQLite metadata (no image arrays, no weights)

scripts/   train_segnet.py, train_unet.py (4-channel RAUNet), evaluate_models.py,
           calibrate_ensemble.py, extract_radiomics_dataset.py, train_classifier.py,
           run_ablations.py, make_sample_assets.py
docs/      PROJECT_AUDIT.md, CURRENT_ARCHITECTURE.md, IMPLEMENTATION_PLAN.md,
           ALGORITHM_NOVELTY.md, API.md, LIMITATIONS.md, DATASET.md
```

### Input specifications enforced by the backend

| Model | Input spec | Normalisation | Output | Threshold |
| --- | --- | --- | --- | --- |
| `unet_dynamic_512` (reused) | 1 x 512 x 512 float32 | `[0,1]` (image/255) | probability (sigmoid inside the network) | 0.5 |
| `segnet_brats_4ch` (reused) | 4 x 256 x 256 float32 | single global z-score over the `(H,W,4)` block, then CHW | **logit**; sigmoid applied by the service | 0.5 |
| `raunet_brats_4ch` (new, to be trained) | 4 x 256 x 256 float32 | same as SegNet (identical pipeline required for ensembling) | logit + deep-supervision aux logits | 0.5 |
| `radiomics_mlp_v1` (incompatible) | 21 features | none recorded | 2 logits | n/a - **disabled** |

The registry marks each model with `input_spec` and `output_semantics`; the ensemble service
refuses to fuse models whose `input_spec` differs, which is exactly what happens today
(SegNet/RAUNet vs the single-channel U-Net).

### Ensemble policy

1. Both models must be loaded, validated against their checkpoints, and share an identical
   `input_spec`.
2. `P_ensemble = w_s * P_segnet + w_u * P_raunet` with `w` from
   `data/models/ensemble_calibration.json` if present (estimated on **validation** data by
   `scripts/calibrate_ensemble.py`), else `w = 0.5, 0.5` flagged as `"equal_weights_default"`.
3. Confidence-aware mode (`--strategy confidence`) rescales the weights per pixel by the
   predictive confidence of each model; the reliability terms still come from validation data.
4. Fusion never runs with a single available model: the endpoint returns `available: false`
   with a machine-readable reason list.

### Explainability policy

* Grad-CAM is computed from the actual model that produced the mask, using a registered
  target layer per architecture and the documented target score (mean foreground probability),
  with `torch.autograd.grad` (no deprecated hooks).
* The response always states `model_id`, `target_layer`, `target_score` and the method, so a
  U-Net map can never be labelled as a SegNet map.
* Feature attribution for the radiomics path uses permutation importance computed only when a
  validated classifier artifact exists; otherwise the endpoint reports `available: false`.

### Persistence model

`analyses` (1) - `segmentation_results` (n) - `radiomics_features` (n) - `classification_results` (0..1)
- `explainability_artifacts` (n) - `reports` (n) - `assistant_messages` (n), plus `model_versions`
and `schema_migrations`. Only file paths and scalar measurements are stored; images stay on disk.
