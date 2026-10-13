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

Model paths are configurable as well (`BRAIN_MODEL_SEGNET`, `BRAIN_MODEL_UNET`, …), so
checkpoints can live outside the repository.

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

A smoke run of any script is available with `--dry-run`, and `--max-cases 2` caps the
work for a quick check.

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
| `PyRadiomics unavailable` in the log | Expected on Python 3.11+. The native extractor is used and reported |
| PDF is missing for an old analysis | Regenerate it from the Reports page; the payload is still stored |

Logs are written to `data/logs/backend.log` and to standard output.
