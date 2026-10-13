#!/usr/bin/env python3
"""Build notebooks/brain_tumor_training_kaggle_t4.ipynb from the sources in _notebook_source/.

Why a builder instead of editing the .ipynb by hand:
  * the notebook must embed the project's own architectures, losses, metrics and radiomics
    extractor verbatim, otherwise a checkpoint trained in the notebook could fail to load in
    the Flask application (or compute a different feature schema);
  * embedding the files programmatically keeps them identical to backend/app/** and lets the
    test suite verify the hashes.

Run:
    python notebooks/build_kaggle_notebook.py            # writes the notebook
    python notebooks/build_kaggle_notebook.py --check     # fails if the notebook is stale
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
SOURCE_DIR = HERE / "_notebook_source"
OUTPUT = HERE / "brain_tumor_training_kaggle_t4.ipynb"

# Files copied verbatim from the application into the notebook, under the same
# package layout, so `from app.ml.architectures.segnet import SegNet` works there
# exactly as it does in the API.
EMBEDDED_FILES = [
    "backend/app/ml/architectures/segnet.py",
    "backend/app/ml/architectures/dynamic_unet.py",
    "backend/app/ml/architectures/attention_unet.py",
    "backend/app/ml/losses.py",
    "backend/app/ml/gradcam.py",
    "backend/app/ml/radiomics_native.py",
    "backend/app/services/evaluation_service.py",
    "backend/app/services/radiomics_service.py",
    "backend/app/utils/metrics.py",
    "backend/app/utils/images.py",
    "backend/app/utils/io.py",
]
EMPTY_INIT_FILES = [
    "backend/app/__init__.py",
    "backend/app/ml/__init__.py",
    "backend/app/ml/architectures/__init__.py",
    "backend/app/services/__init__.py",
    "backend/app/utils/__init__.py",
]
# These must be empty in the notebook: the real files build the Flask application.
EMPTY_INIT_CONTENT = ""


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def embedded_cell_source() -> tuple[str, dict]:
    """Cell that writes the project modules to disk and imports them."""
    payload = {}
    for relative in EMBEDDED_FILES:
        source = (REPO / relative).read_text(encoding="utf-8")
        payload[relative] = {"sha256": file_sha256(REPO / relative), "source": source}
    for relative in EMPTY_INIT_FILES:
        payload[relative] = {"sha256": file_sha256_bytes(EMPTY_INIT_CONTENT), "source": EMPTY_INIT_CONTENT}

    lines = [
        "# =============================================================================",
        "# SECTION 1c - The project's own inference code, embedded verbatim",
        "# =============================================================================",
        "# Architectures, loss, metrics, Grad-CAM, the native radiomics extractor and the",
        "# evaluation service are copied here *from the repository*, byte for byte. That is what",
        "# makes the exports below loadable by the Flask application: the same builders build the",
        "# same graphs, and the same extractor produces the same feature names.",
        "#",
        "# Each entry records the sha256 of the file it came from. `pytest tests/test_kaggle_notebook.py`",
        "# recomputes those hashes against backend/app/**, so the notebook cannot silently drift",
        "# away from the application: regenerate it with notebooks/build_kaggle_notebook.py.",
        "# =============================================================================",
        "EMBEDDED_SOURCES = json.loads(r'''" + json.dumps(payload) + "''')",
        "",
        "EMBEDDED_ROOT = PATHS['work'] / '_embedded'",
        "for relative, entry in EMBEDDED_SOURCES.items():",
        "    target = EMBEDDED_ROOT / relative",
        "    target.parent.mkdir(parents=True, exist_ok=True)",
        "    current = target.read_text(encoding='utf-8') if target.exists() else None",
        "    if current != entry['source']:",
        "        target.write_text(entry['source'], encoding='utf-8')",
        "if str(EMBEDDED_ROOT / 'backend') not in sys.path:",
        "    sys.path.insert(0, str(EMBEDDED_ROOT / 'backend'))",
        "",
        "from app.ml.architectures.segnet import SegNet",
        "from app.ml.architectures.dynamic_unet import DEFAULT_FILTERS, DynamicUNet",
        "from app.ml.architectures.attention_unet import build_raunet",
        "from app.ml.losses import BCEDiceLoss, DiceLoss, build_loss",
        "from app.utils import metrics as project_metrics",
        "",
        "LOCK['embedded'] = {relative: entry['sha256'] for relative, entry in EMBEDDED_SOURCES.items()}",
        "print(f'embedded {len(EMBEDDED_SOURCES)} project files under {EMBEDDED_ROOT}')",
        "print('architectures:', SegNet.__module__, DynamicUNet.__module__, build_raunet.__module__)",
        "print('loss:', build_loss.__module__, '| metrics:', project_metrics.__name__)",
        "",
        "write_json(PATHS['artifacts'] / 'embedded_sources.json', LOCK['embedded'])",
    ]
    return "\n".join(lines), payload


def file_sha256_bytes(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_cells() -> list[tuple[str, str]]:
    cells: list[tuple[str, str]] = []
    for path in sorted(SOURCE_DIR.glob("nbs_*.py")):
        text = path.read_text(encoding="utf-8")
        chunks = text.split("# %% [cell] ")
        for index, chunk in enumerate(chunks):
            if index == 0:
                if chunk.strip():
                    raise SystemExit(f"{path}: content before the first '# %% [cell]' marker")
                continue
            title, _, body = chunk.partition("\n")
            cells.append((f"{path.name}:{title.strip()}", body.rstrip() + "\n"))
    return cells


INTRO_MARKDOWN = """# Brain tumour segmentation and multiclass classification on a Kaggle T4

This notebook trains and evaluates every model of the **Brain segmentation and classification**
project and exports artefacts the Flask application loads unchanged. It uses the two datasets the
project needs, for two different tasks:

| Task | Dataset | Kaggle |
| --- | --- | --- |
| Segmentation (plus the HGG/LGG grade experiment) | BraTS 2020 | `awsaf49/brats2020-training-data` |
| Multiclass tumour type (classes come from the dataset) | Brain Tumor MRI Dataset | `masoudnickparvar/brain-tumor-mri-dataset` |

The two datasets are never mixed: the label spaces are different, and the stacked ensemble accepts
a member only when its dataset and class list match the other members'.

Verified context for this notebook (see `docs/DELIVERY.md`):

* the shipped `segnet_model.pth` is degenerate -- its sigmoid output never crosses 0.5 on any
  available input, so SegNet is retrained here;
* `UNet-[16, 32, 64, 128, 256].pt` runs but was never evaluated against labelled masks, so the
  same architecture is retrained from scratch on real data;
* `raunet_brats_4ch.pt` does not exist yet;
* the shipped classifier expects 21 unnamed features and no scaler was saved, so neither
  classification artefact is trained to fit it: each one records its own ordered feature schema,
  scaler and label mapping, which is exactly what the API requires.

What each part produces is recorded next to it: every artefact carries its task, its dataset and
whether that dataset is real or a fixture, and `data/models/models_index.json` lists all of them.

## How to run it

1. **Settings -> Accelerator -> GPU T4 x2** (or any CUDA GPU). Cell 1 verifies the device and
   prints the memory before anything is trained.
2. **Add Data -> `awsaf49/brats2020-training-data`** and **Add Data ->
   `masoudnickparvar/brain-tumor-mri-dataset`.** Internet access is not required; the notebook
   reads both datasets from `/kaggle/input` and embeds all the code it needs. Each dataset is
   optional in the sense that a missing one is *reported* and its section skips -- with exact
   reasons and no invented results -- while the other still runs.
3. Run the cells in order. Cell 2 holds every size, epoch count and budget in one place.
4. A full run does **not** have to finish in one session. Checkpoints carry model, optimiser,
   scheduler, AMP scaler and RNG state; re-running the training cells continues where the last
   session stopped (see the runbook at the end).

## What it produces

Per-model checkpoints, training histories, held-out metrics in the application's own
`evaluation_results.json` / `model_selection.json` / `ensemble_calibration.json` formats, a
classifier artefact in the format `ClassificationService` requires, figures, and a
`brain_mri_artifacts.zip` containing everything plus a README and a provenance manifest.

## What it never does

* It never uses the test split for model selection, threshold tuning or ensemble weighting.
* It never fabricates a label, a metric or a model: a section that has no data says so and stops.
* It never overwrites an artefact that is already in the repository.
"""

RUNBOOK_MARKDOWN = """# Runbook, resume and export

## Session planning (Tesla T4, 15 GB)

The workload is dominated by three things: how many slices an epoch sees, the input size, and
the batch size. Measure it on your first run instead of trusting a guess -- the trainer prints
`seconds/epoch` and every checkpoint carries `elapsed_minutes`:

| Model | Input | Typical step time on a T4 (fp16) | Notes |
| --- | --- | --- | --- |
| SegNet | 4 x 256 x 256 | fast, small network | epoch time scales linearly with slices per epoch |
| RAUNet | 4 x 256 x 256 | several times SegNet per step | the deep-supervision heads add memory; reduce `batch_size` first |
| DynamicUNet | 1 x 512 x 512 | heaviest per step | keep `batch_size_single` at 2-4 on a T4 |

Set `RUN["max_steps_per_epoch"]` to time-box an epoch (the index is reshuffled every epoch, so a
capped epoch still sees different slices). `RUN["session_budget_minutes"]` stops training cleanly
before Kaggle ends the session, so the next session resumes instead of losing the work.

## Resume

* `RUN["smoke_test"] = True` runs the whole pipeline on a handful of patients: use it to verify
  the environment before spending GPU quota.
* Re-run the training cells. `checkpoints/<model_id>/last.pt` holds the model, optimiser,
  scheduler, AMP scaler, epoch, step, best metric and the RNG state; training continues from
  there. `best.pt` keeps the checkpoint with the best validation Dice.
* `history.json` and `training_summary.json` accumulate across sessions, so the final curves
  cover everything that was trained.
* If a session dies mid-epoch, only the work since the last checkpoint is lost
  (`checkpoint_every_steps`, `checkpoint_every_minutes`).

## Export and install

```bash
# 1. download brain_mri_artifacts.zip from the notebook output and unpack it in the project root
unzip brain_mri_artifacts.zip -d /path/to/Brain-segmentation-and-classification/

# 2. copy the artefacts into place (never overwriting the shipped checkpoints)
python scripts/install_trained_artifacts.py --source brain_mri_artifacts

# 3. what the application now sees
python scripts/check_models.py
python -m pytest tests -q
```

`install_trained_artifacts.py` also prints the `BRAIN_MODEL_*` environment variables that point
the API at the newly trained checkpoints, and writes `data/models/artifacts_manifest.json` so the
interface can label each model *trained on real BraTS data* / *trained on a synthetic fixture* /
*implemented but not trained*.

## Honest reporting

* Metrics come from the reserved patient-level test split, with the project's own metric code.
* The classification label is the dataset's HGG/LGG grade annotation. It is not a malignancy
  verdict, not a stage and not a prognosis.
* If a model is missing from `checkpoints/`, it was not trained in that session -- the export
  marks it `not_trained` and the API keeps reporting it as unavailable rather than pretending.
"""


def build_notebook() -> dict:
    cells = load_cells()
    embedded_source, embedded_payload = embedded_cell_source()

    notebook_cells = [
        {"cell_type": "markdown", "metadata": {}, "source": INTRO_MARKDOWN.splitlines(keepends=True)},
    ]
    inserted_embedded = False
    for title, body in cells:
        if not inserted_embedded and "Environment and GPU verification" in title:
            notebook_cells.append(
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": ["# Part 1. Environment, configuration and project code\n"],
                }
            )
        notebook_cells.append(
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [f"## {title.split(':', 1)[1].strip()}\n", f"`{title.split(':', 1)[0]}`\n"],
            }
        )
        notebook_cells.append(
            {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": body.splitlines(keepends=True)}
        )
        if not inserted_embedded and "Run configuration" in title:
            # The embedded cell writes the project modules under PATHS['work'] and imports them,
            # so it has to run *after* the configuration cell, which defines PATHS and write_json.
            notebook_cells.append(
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [
                        "## The project's inference code, embedded verbatim\n",
                        "`build_kaggle_notebook.embedded`\n",
                    ],
                }
            )
            notebook_cells.append(
                {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": embedded_source.splitlines(keepends=True)}
            )
            inserted_embedded = True
    if not inserted_embedded:  # pragma: no cover - only if the configuration cell is renamed
        raise SystemExit("no run-configuration cell found to attach the embedded sources to")
    notebook_cells.append({"cell_type": "markdown", "metadata": {}, "source": RUNBOOK_MARKDOWN.splitlines(keepends=True)})

    notebook = {
        "cells": notebook_cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
            "accelerator": "GPU",
            "kaggle": {
                "accelerator": "nvidiaTeslaT4",
                "dataSources": [{"databundleVersionId": 0, "sourceId": 0, "sourceType": "datasetVersion"}],
                "isGpuEnabled": True,
                "isInternetEnabled": False,
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
        "_embedded_hashes": {relative: entry["sha256"] for relative, entry in embedded_payload.items()},
    }
    return notebook


def stale_reasons(notebook: dict) -> list[str]:
    reasons = []
    recorded = notebook.get("_embedded_hashes", {})
    for relative in EMBEDDED_FILES:
        path = REPO / relative
        if not path.exists():
            reasons.append(f"{relative} is missing from the repository")
            continue
        if recorded.get(relative) != file_sha256(path):
            reasons.append(f"{relative} changed since the notebook was generated")
    for relative in EMPTY_INIT_FILES:
        if recorded.get(relative) != file_sha256_bytes(EMPTY_INIT_CONTENT):
            reasons.append(f"{relative} changed since the notebook was generated")
    return reasons


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify that the committed notebook is current")
    args = parser.parse_args()

    notebook = build_notebook()
    if args.check:
        if not OUTPUT.exists():
            print(f"{OUTPUT} does not exist")
            return 1
        current = json.loads(OUTPUT.read_text(encoding="utf-8"))
        reasons = stale_reasons(current)
        if reasons:
            print("The notebook is stale:")
            for reason in reasons:
                print(" -", reason)
            print("Run: python notebooks/build_kaggle_notebook.py")
            return 1
        if len(current["cells"]) != len(notebook["cells"]):
            print(f"the notebook has {len(current['cells'])} cells, expected {len(notebook['cells'])}")
            return 1
        print(f"{OUTPUT.name} is current ({len(current['cells'])} cells)")
        return 0

    OUTPUT.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
    code_cells = sum(1 for cell in notebook["cells"] if cell["cell_type"] == "code")
    markdown_cells = len(notebook["cells"]) - code_cells
    print(f"wrote {OUTPUT} : {len(notebook['cells'])} cells ({code_cells} code, {markdown_cells} markdown)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
