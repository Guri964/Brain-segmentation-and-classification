# How to run

Everything below is verified on CPU with Python 3.11 and Node 18+. Commands are given
for bash; on Windows PowerShell the same commands work, with `.venv\Scripts\Activate.ps1`
for the virtual environment.

---

## 1. Install

### 1.1 Python environment

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r backend/requirements.txt
```

Check what you ended up with:

```bash
python scripts/setup_env.py
```

That report tells you which optional pieces are present. Two of them matter:

* **SimpleITK** — required only if you want to feed NIfTI, NRRD or DICOM volumes.
  Without it the platform accepts PNG/JPEG and says so.
* **PyRadiomics** — has no wheel for Python 3.11+. The bundled native extractor is
  used instead and the API reports `selected_backend: native`; the feature values it
  produces are verified against analytic shapes in `tests/test_radiomics.py`.

### 1.2 Frontend

```bash
cd frontend
npm install
npm run build      # writes frontend/dist, which Flask serves
cd ..
```

### 1.3 GPU (optional)

The platform detects CUDA automatically and falls back to CPU. To install a CUDA
build of PyTorch:

```bash
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
```

Nothing else changes: `BRAIN_DEVICE=auto` (the default) picks the GPU when it is
usable, `BRAIN_DEVICE=cpu` pins it to CPU for reproducibility.

---

## 2. Run

### 2.1 Production-style (one process, one port)

```bash
python backend/run.py
```

Then open <http://localhost:5000>. Flask serves the built dashboard and the API from
the same origin, so no proxy or CORS configuration is involved.

### 2.2 Development (two processes, hot reload)

```bash
python backend/run.py            # terminal 1, API on :5000
cd frontend && npm run dev       # terminal 2, UI on :5173
```

Vite proxies `/api` to `http://127.0.0.1:5000`; change `VITE_API_TARGET` in
`frontend/.env` if the API listens elsewhere. The browser only ever talks to the Vite
origin.

### 2.3 Tests

```bash
python -m pytest tests -q              # 58 tests, ~20 s on 2 CPU cores
python -m pytest tests -q -k api       # only the HTTP-level tests
cd frontend && npm run typecheck       # TypeScript
```

The suite is CPU-only by design, uses deterministic synthetic phantoms, and writes
into a temporary directory. It never modifies the shipped checkpoints.

---

## 3. Configuration

All configuration is environment driven. Nothing is hard-coded, and no secret ever
has a default.

| Variable | Default | Purpose |
|---|---|---|
| `BRAIN_ENV` | `development` | `development`, `testing` or `production` |
| `BRAIN_DATA_DIR` | `./data` | Root for uploads, artefacts, reports, models and the database |
| `BRAIN_MAX_UPLOAD_BYTES` | `67108864` | Per-request upload limit |
| `BRAIN_MAX_FILES_PER_ANALYSIS` | `8` | Upload count limit |
| `BRAIN_DEVICE` | `auto` | `auto`, `cpu` or `cuda` |
| `BRAIN_TORCH_THREADS` | `2` | CPU thread cap — raise it on a bigger machine |
| `BRAIN_SEG_THRESHOLD` | `0.5` | Default mask threshold (also settable per analysis) |
| `BRAIN_RADIOMICS_BACKEND` | `auto` | `auto`, `pyradiomics` or `native` |
| `BRAIN_RADIOMICS_DISCRETIZATION` | `bin_count` | `bin_count` or `bin_width` |
| `BRAIN_RADIOMICS_BIN_COUNT` | `32` | Bin count when using `bin_count` |
| `BRAIN_ASSISTANT_LLM_ENABLED` | `false` | Opt in to a language-model backend |
| `BRAIN_ASSISTANT_LLM_API_KEY` | — | Read from the environment only, never stored or logged |
| `BRAIN_ASSISTANT_LLM_MODEL` | `gpt-4o-mini` | Model name for the optional backend |
| `BRAIN_CORS_ORIGINS` | dev origins | Comma-separated allow-list |
| `BRAIN_LOG_LEVEL` | `INFO` | Log verbosity |

Model paths are configurable as well (`BRAIN_MODEL_SEGNET`, `BRAIN_MODEL_UNET`,
`BRAIN_MODEL_XCEPTION`, `BRAIN_MODEL_RADIOMICS_MLP`, …), so checkpoints can live outside the
repository.

### Model weights

Nothing is downloaded and nothing is retrained by the application. Copy the artefacts into
`models/weights/` (the default of `BRAIN_WEIGHTS_DIR`); `models/README.md` lists the expected
file names and each one's input contract. `python scripts/audit_integration.py` then reports,
per artefact, whether it was found, whether it loads and whether a forward pass on the
documented input succeeds - and names exactly what is missing when it does not.

The four-class image classifier additionally needs TensorFlow in the API environment
(`pip install tensorflow-cpu`) plus `metadata.json` beside the checkpoint; without either, the
endpoint answers 409 with the missing file and nothing else on the platform is affected.

> **Do not run `torch.compile` in the same process as TensorFlow.** Importing TensorFlow and
> then `torch.compile` (which pulls in `torch._dynamo` → `triton`) crashes the interpreter on
> some environments, this one included. The API never calls `torch.compile`; the training
> notebook does, so train and serve in different processes. See `docs/LIMITATIONS.md` §17.

---

## 4. First run checklist

1. `python scripts/check_models.py` — confirms which checkpoints actually work. Expect
   `unet_dynamic_512` responsive and `segnet_brats_4ch` `no_foreground` until it is
   retrained.
2. Open the dashboard → **New MRI Analysis** → upload a PNG/JPEG export of a slice.
3. Choose the single-channel pipeline and run. The analysis takes a few seconds on CPU.
4. Inspect **Segmentation Results**, **Quantitative Radiomics**, **Explainable AI** and
   **Clinical Decision Support**. Every panel states what it could not compute.
5. **Reports** → download the PDF for that analysis.

## 5. Working with the real dataset

Training, calibration and evaluation need the original data, which is not part of this
repository.

```bash
# 1. describe the dataset (see docs/DATASET.md for the format)
cp data/manifest.example.json data/manifest.json
$EDITOR data/manifest.json

# 2. patient-level splits
python scripts/make_splits.py --data data/manifest.json

# 3. retrain what needs retraining
python scripts/train_segnet.py --data data/manifest.json --epochs 60
python scripts/train_unet.py   --data data/manifest.json --architecture raunet --epochs 60

# 4. measure on the reserved test split
python scripts/evaluate_models.py --data data/manifest.json --split test

# 5. estimate fusion weights on validation data only
python scripts/calibrate_ensemble.py --data data/manifest.json --metric dice

# 6. attribute each modification
python scripts/run_ablations.py --data data/manifest.json --models raunet_brats_4ch

# 7. enable the classification branch, if labels exist
python scripts/train_classifier.py --features radiomics.csv --label-column grade --patient-column patient_id
```

### 5.1 BraTS 2020 end to end

```bash
# 1. look before you train (read-only; writes nothing into the dataset)
python scripts/inspect_brats.py --root data/brats \
  --grade-table data/brats/name_mapping.csv --json data/artifacts/brats_inspection.json

# 2. build a patient-level manifest with the HGG/LGG labels and frozen splits
python scripts/prepare_brats.py --root data/brats \
  --out data/manifests/brats2020.json --grade-table data/brats/name_mapping.csv --splits

# 3. train segmentation (lazy loading reads slices from disk on demand)
python scripts/train_segnet.py --data data/manifests/brats2020.json \
  --size 256 --epochs 60 --batch-size 8 --lazy-loading --model-id segnet_brats_4ch

# 4. one labelled radiomics row per case, in the application's own schema
python scripts/extract_radiomics.py --data data/manifests/brats2020.json \
  --out data/manifests/brats_features.csv

# 5. train and compare the grade classifier, then serve it
python scripts/train_classification.py --features data/manifests/brats_features.csv \
  --split-column split --model-id radiomics_grade_classifier \
  --target-description "BraTS 2020 HGG vs LGG (WHO grade group)"
```

No BraTS data? `python scripts/make_brats_fixture.py --out /tmp/brats_fixture` writes a synthetic
tree with the same layout so the five steps above can be rehearsed. It prints a warning: the
labels are made up and any metric from it is meaningless. Details: `docs/BRAFTS2020.md`,
`docs/CLASSIFICATION.md`.

### 5.2 Train on a free Kaggle T4 (the route this machine cannot take)

BraTS 2020 is ~60 GB and needs a GPU, so the full training run does not fit here: this delivery
machine has no GPU, 3 GB of RAM and 11 GB of disk, and `www.kaggle.com` is unreachable from it.
Nothing about that is worked around with fake numbers -- instead the whole run is packaged as a
notebook that you execute once on Kaggle.

**Two datasets, two tasks.** The notebook trains segmentation on BraTS 2020 and a *separate*
multiclass tumour-type model on the Brain Tumor MRI Dataset
(`masoudnickparvar/brain-tumor-mri-dataset`, four classes: glioma / meningioma / pituitary /
no tumour). Attach both datasets; a missing one is reported and its section skips, while the
other still runs. The classes of the second dataset are read from its directories -- they are
never assumed to be HGG/LGG, and the two label spaces are never mixed (the stacked ensemble
rejects a member from the other dataset with the reason printed).

```bash
# 0. the notebook is generated from its sources; regenerate it only if you edited them
python notebooks/build_kaggle_notebook.py            # writes notebooks/brain_tumor_training_kaggle_t4.ipynb
python notebooks/build_kaggle_notebook.py --check     # verifies the committed notebook is current
python -m pytest tests/test_kaggle_notebook.py -q     # executes every cell against both synthetic fixtures
```

Then, on Kaggle:

1. Create a notebook and **File -> Import Notebook ->** `notebooks/brain_tumor_training_kaggle_t4.ipynb`.
2. **Settings -> Accelerator -> GPU T4 x2** (any CUDA GPU works; cell 1 prints the device, the
   memory and whether mixed precision is on before anything is trained).
3. **Add Data -> Datasets ->** `awsaf49/brats2020-training-data` **and**
   `masoudnickparvar/brain-tumor-mri-dataset`. Internet access is not needed: the notebook reads
   both datasets from `/kaggle/input` and embeds the project's own code.
4. Run all cells. Cell 2 (`RUN`) holds every size, epoch count and time budget; the training cells
   checkpoint every 500 steps / 20 minutes and resume from `checkpoints/<model>/last.pt`, so a
   session that ends mid-run continues in the next one instead of starting over.
5. Download `brain_mri_artifacts.zip` from the notebook output panel.

```bash
# back on this machine: install the artefacts and tell the API about them
unzip brain_mri_artifacts.zip
python scripts/install_trained_artifacts.py --source brain_mri_artifacts --data-dir data
# the script prints the environment variables (BRAIN_MODEL_SEGNET, BRAIN_MODEL_RAUNET,
# BRAIN_MODEL_UNET, BRAIN_CLASSIFIER_DIR); export them before starting the API
python scripts/check_models.py --force     # every checkpoint is loaded and exercised
python -m pytest tests -q
python backend/run.py
```

What the notebook writes (all inside `brain_mri_artifacts/`):

| File | Content |
| --- | --- |
| `checkpoints/segnet_brats_4ch_real_brats.pt` | retrained SegNet, 4x256x256, logits |
| `checkpoints/raunet_brats_4ch.pt` | RAUNet, 4x256x256, logits |
| `checkpoints/unet_dynamic_512_real_brats.pt` | retrained U-Net, 1x512x512, probability |
| `data/models/classification/<version>/` | classifier `.joblib` + `manifest.json` (ordered feature schema, scaler, label mapping, metrics, caveats) |
| `data/models/evaluation_results.json` | Dice/IoU/precision/recall + HD95 with bootstrap intervals on the held-out test split |
| `data/models/model_selection.json` | which checkpoint was chosen, on validation data only |
| `data/models/ensemble_calibration.json` | fusion weights estimated on the validation split |
| `data/models/artifacts_manifest.json` | provenance the API reads to label each model |
| `checkpoints/brain_tumor_mri_cnn.pt` | multiclass CNN trained from scratch on the Brain Tumor MRI Dataset |
| `data/models/classification_multiclass/<version>/` | stacking layer + both radiomics members + `manifest.json` (ordered schema, scaler, class mapping, metrics, caveats) |
| `data/models/models_index.json` | every artefact with its task, label space, dataset and whether the API serves it |
| `reports/` | EDA, mask and Grad-CAM figures, comparison tables, confusion matrix |

Three rules the notebook enforces and the code checks: the reserved test split is read exactly once
(after selection, which uses validation data only); nothing in the repository is overwritten --
`install_trained_artifacts.py` writes new files next to the shipped ones and records the hashes; and
the second dataset's splits are audited for near-duplicates of training images before any metric is
computed (the audit reports its candidates and its confirmations, and refuses to empty a split).

The classifier is trained on the feature schema the API computes at serving time
(`analysis_service._run_radiomics()` extracts from the first display plane, i.e. T1, with bare
feature names), so the exported artefact can be served without re-ordering or padding features.

A smoke run of any script is available with `--dry-run`, and `--max-cases 2` caps the
work for a quick check.

---

## 5.3 When training is slow, or the GPU sits at 0%

If the notebook reports `batch preparation` in seconds rather than milliseconds, the cache on
disk is in the compressed `.npz` layout. Every single slice read then inflates that patient's
whole archive (165 MiB for a real BraTS volume), and the slice index does the same once per
patient per epoch. Cell 9 measures both layouts on your own data and prints the difference; it
is not an estimate.

What cell 9 does about it:

| Step | What happens |
|---|---|
| Build | New caches are written as `cache/<volume>.cache/{images,masks,singles}.npy` (memory-mappable: reading slice *i* reads slice *i*) |
| Measure | Before converting, it times the archive reader and the store on the same four volumes and the same slices, and writes `artifacts/cache_io_benchmark.json` |
| Convert | `upgrade_patient_cache()` converts an existing `.npz` cache in place: write `.part`, read it back, compare every value, publish, then delete the archive. One patient at a time, resumable, and a volume that fails keeps its archive and is reported |
| Index | `cache/_cache_index.json` records slice counts so the per-epoch index costs header reads instead of payload loads |
| Verify | `validate_cache()` re-reads a slice from each sampled volume and checks the shape, dtype, finiteness and the mask's binary contract |

Knobs (cell 2, `RUN`):

| Key | Default | Meaning |
|---|---|---|
| `cache_format` | `"memmap"` | Layout for newly built cache entries; `"npz"` writes the old compressed archive |
| `upgrade_cache_to_memmap` | `True` | Convert existing archives during cell 9 (set `False` to keep them and stay slow) |
| `patient_cache_mb` | `1536` | Upper bound on materialised payloads held in RAM |
| `patient_cache_handles` | `32` | Upper bound on open memory maps |
| `prefetch_batches` | `False` | Prepare the next batch on a worker thread. Measured slower than direct reads here (12.2-13.1 ms vs 10.0-11.3 ms per batch of 8), so it is off |
| `benchmark_before_training` | `True` | Run the one-batch checks and the five-phase benchmark before the first epoch |
| `benchmark_steps` / `benchmark_warmup` | `6` / `2` | Steps measured (after the warm-up) in that benchmark |

Cell 18 prints the result: batch preparation, CPU->GPU transfer, forward, backward+optimiser
step and the whole step, separately, with `torch.cuda.synchronize()` around the GPU phases and
a real `nvidia-smi` utilisation sample. The five phases are also written to
`artifacts/segnet_benchmark.json`; the pre-training checks (shapes, dtypes, finite loss, real
gradients, checkpoint round-trip) go to `artifacts/segnet_self_test.json`.

Disk cost of the conversion: the memory-mappable layout is about 20% larger than the
compressed one (measured 165 MiB vs 138 MiB per BraTS volume). Cell 9 prints free space, the
measured size per volume and what the full 369-volume release would need before it starts, and
skips a volume it cannot fit (reported, never silently).

---

## 6. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `Checkpoint not found` for every model | Run from the repository root, or set `BRAIN_MODEL_SEGNET` / `BRAIN_MODEL_UNET` to absolute paths |
| `no_foreground` verdict for SegNet | Expected on this deployment. The checkpoint loads but predicts an empty mask; retrain with `scripts/train_segnet.py` |
| Ensemble says "different inputs" | Correct behaviour. 4×256×256 and 1×512×512 models cannot be fused; train a partner with a matching input spec |
| Classification disabled | No validated classifier artefact exists. `GET /api/classification` names the directory it searched and what an artefact must contain; `scripts/train_classification.py` produces one from a labelled radiomics table |
| "Every panel says Not available" | Open an analysis first (Overview → New MRI Analysis), or select one from History |
| Upload rejected with `unsupported_input` | Only PNG/JPEG are accepted per file. NIfTI/NRRD/DICOM need SimpleITK and the corresponding pipeline |
| Upload rejected as duplicate channels | Four-channel analyses need four *different* sequences; the API reports which ones matched |
| Blank page after `npm run build` | The build output must exist at `frontend/dist`; run the build from inside `frontend/` |
| Slow inference | Two CPU threads are the default. Raise `BRAIN_TORCH_THREADS`, or use a GPU |
| Notebook: GPU at 0%, `batch preparation` in seconds | The cache is in the compressed `.npz` layout. See §5.3: rebuild or let cell 9 convert it to the memory-mappable layout |
| `PyRadiomics unavailable` in the log | Expected on Python 3.11+. The native extractor is used and reported |
| PDF is missing for an old analysis | Regenerate it from the Reports page; the payload is still stored |

Logs are written to `data/logs/backend.log` and to standard output.
