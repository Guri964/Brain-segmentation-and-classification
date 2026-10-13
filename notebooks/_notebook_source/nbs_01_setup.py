# %% [cell] 1. Environment and GPU verification
# =============================================================================
# SECTION 1 - Environment, GPU and dependency verification
# =============================================================================
# Everything the notebook needs is checked here and reported, never assumed.
# The notebook works on a Kaggle T4 (preferred), on any other CUDA GPU, and on
# CPU for a tiny smoke run -- but the full run is sized for a T4.
# =============================================================================
import importlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

LOCK = {}
STARTED_AT = time.time()


def _pip_install(packages, optional=True):
    """Install only what is missing; never fail the notebook for an optional dep."""
    missing = []
    for name, module in packages:
        try:
            importlib.import_module(module)
        except Exception:
            missing.append(name)
    if not missing:
        return {"installed": [], "already_present": [p for p, _ in packages]}
    print(f"Installing {missing} ...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", *missing],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        message = (result.stderr or result.stdout or "").strip()[-400:]
        if optional:
            print(f"WARNING: could not install {missing}; optional features will be skipped.\n{message}")
            return {"installed": [], "failed": missing, "error": message}
        raise RuntimeError(f"Required packages {missing} could not be installed: {message}")
    return {"installed": missing}


INSTALL_REPORT = _pip_install(
    [
        ("h5py", "h5py"),
        ("scikit-learn", "sklearn"),
        ("joblib", "joblib"),
        ("matplotlib", "matplotlib"),
        ("pandas", "pandas"),
        ("tqdm", "tqdm"),
        ("nibabel", "nibabel"),
        ("SimpleITK", "SimpleITK"),
    ]
)

import numpy as np
import torch
import torch.nn as nn

CUDA_AVAILABLE = torch.cuda.is_available()
DEVICE = torch.device("cuda:0" if CUDA_AVAILABLE else "cpu")
GPU_NAME = torch.cuda.get_device_name(0) if CUDA_AVAILABLE else "none (CPU)"
GPU_TOTAL_GB = (
    round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2) if CUDA_AVAILABLE else 0.0
)
COMPUTE_CAPABILITY = torch.cuda.get_device_capability(0) if CUDA_AVAILABLE else None

AMP_SUPPORTED = bool(CUDA_AVAILABLE and torch.cuda.is_bf16_supported() is not None)
# T4 (SM 7.5) supports fp16 tensor-core maths; AMP is used when a CUDA device exists.
AMP_ENABLED = bool(CUDA_AVAILABLE)

print("=" * 78)
print("ENVIRONMENT")
print("=" * 78)
print(f"python           : {sys.version.split()[0]}")
print(f"platform         : {platform.platform()}")
print(f"torch            : {torch.__version__} (cuda build {torch.version.cuda})")
print(f"cuda available   : {CUDA_AVAILABLE}")
print(f"device           : {DEVICE}  |  {GPU_NAME}")
if CUDA_AVAILABLE:
    print(f"gpu memory       : {GPU_TOTAL_GB} GB  | capability {COMPUTE_CAPABILITY}")
    print(f"cuda runtime     : {torch.version.cuda}")
    print(f"cudnn            : {torch.backends.cudnn.version()}")
print(f"amp enabled      : {AMP_ENABLED}")
print(f"cpu count        : {os.cpu_count()}")
print(f"numpy            : {np.__version__}")
print(f"install report   : {INSTALL_REPORT}")

if not CUDA_AVAILABLE:
    print(
        "\nWARNING: no CUDA device. The notebook will still run, but only a small smoke "
        "run is realistic on CPU. Set RUN['smoke_test'] = True, or enable the GPU "
        "accelerator in Kaggle (Settings -> Accelerator -> GPU T4 x2) and rerun."
    )
elif GPU_TOTAL_GB < 12:
    print(
        f"\nWARNING: {GPU_TOTAL_GB} GB of GPU memory is below the 15 GB a Kaggle T4 offers. "
        "Reduce RUN['batch_size'] and RUN['cache_size'] before the full run."
    )

LOCK["environment"] = {
    "torch": torch.__version__,
    "cuda_available": CUDA_AVAILABLE,
    "gpu": GPU_NAME,
    "gpu_memory_gb": GPU_TOTAL_GB,
    "amp": AMP_ENABLED,
    "python": sys.version.split()[0],
}

# %% [cell] 2. Run configuration and resumable state
# =============================================================================
# SECTION 1b - Configuration: paths, sizes, epochs, memory budget, reproducibility
# =============================================================================
# Every number that affects a result lives here. Nothing downstream hard-codes a
# duration, a batch size or a metric.
# =============================================================================
import random

RUN = {
    # --- dataset -----------------------------------------------------------
    "dataset_slug": "awsaf49/brats2020-training-data",
    "dataset_roots": [  # searched in order; first match wins
        "/kaggle/input/brats2020-training-data",
        "/kaggle/input/brats2020-training-data/BraTS2020_training_data",
        "/kaggle/input/brats2020_training_data",
        "/kaggle/input/brats-2020-training-data",
    ],
    # --- segmentation ------------------------------------------------------
    "image_size": 256,          # 4-channel models (SegNet, RAUNet): 256x256
    "single_image_size": 512,   # single-channel model (DynamicUNet): 512x512
    "single_channel_index": 3,  # FLAIR is the channel the shipped U-Net was fed
    "batch_size": 16,           # 4-channel, 256x256, fp16 on a T4
    "batch_size_single": 4,     # 1-channel, 512x512
    "epochs_segnet": 40,
    "epochs_unet": 30,
    "epochs_raunet": 40,
    "max_steps_per_epoch": None,   # set an int to time-box an epoch (indices reshuffled each epoch)
    "min_steps_per_epoch": 200,
    "learning_rate": 3e-4,
    "learning_rate_single": 1e-4,
    "weight_decay": 1e-5,
    "dice_weight": 0.6,          # project convention: 0.6 * Dice + 0.4 * BCE
    "early_stopping_patience": 8,
    "validate_every_epochs": 1,
    "validation_cases": 64,      # patient-level validation cases per validation pass
    "grad_clip": 1.0,
    "num_workers": 2,
    # --- time budget -------------------------------------------------------
    "session_budget_minutes": 480,   # stop cleanly before the Kaggle session ends
    "checkpoint_every_steps": 500,
    "checkpoint_every_minutes": 20,
    # --- classification ----------------------------------------------------
    "radiomics_slices_per_patient": 3,   # slices around the largest tumour slice
    "radiomics_max_patients": None,      # None = every labelled patient
    "classifier_improved": "hist_gradient_boosting",  # or "mlp"
    "image_classifier_epochs": 12,
    "image_classifier_enabled": True,
    "image_classifier_slices_per_patient": 6,
    # --- second dataset: multiclass image classification --------------------
    # Brain Tumor MRI Dataset (the dataset's own classes; never assumed to be HGG/LGG):
    #   https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset
    "classification_dataset_slug": "masoudnickparvar/brain-tumor-mri-dataset",
    "classification_dataset_roots": [
        "/kaggle/input/brain-tumor-mri-dataset",
        "/kaggle/input/brain-tumor-mri-dataset/Training",
        "/kaggle/input/brain-tumor-mri-classification-dataset",
        "/kaggle/input/brain-tumor-classification-mri",
        "data/brain_tumor_mri_dataset",
    ],
    # "auto" checks for a fixture manifest first and accepts a tree with >= 1000 images as the
    # real release; anything else is recorded as unverified rather than claimed as real data.
    "classification_dataset_provenance": "auto",
    "classification_task_name": "brain tumour type (multiclass, classes defined by the dataset)",
    "classification_image_size": 224,
    "classification_validation_fraction": 0.15,
    # Near-duplicate audit. The hash only proposes candidates; the pixel correlation confirms them,
    # because an average hash cannot tell two similar-looking brain slices apart on its own.
    "classification_duplicate_hash_size": 16,             # 16x16 = 256-bit average hash
    "classification_duplicate_pixel_size": 64,             # size at which the pixels are compared
    "classification_duplicate_hamming_threshold": 24,      # <= this many differing bits = candidate
    "classification_duplicate_correlation_threshold": 0.995,  # Pearson r >= this = confirmed duplicate
    "duplicate_exclusion_max_fraction": 0.05,              # above this share the audit is inconclusive
    "exclude_duplicate_eval_images": True,                 # drop confirmed duplicates from validation/test
    "radiomics_image_size": 256,                       # the size the application's radiomics runs at
    "multiclass_radiomics_max_images": None,           # None = every image; an int caps it per class/split
    "cnn_channels": [32, 64, 128],
    "cnn_dropout": 0.3,
    "cnn_epochs": 30,
    "cnn_batch_size": 32,
    "cnn_learning_rate": 3e-4,
    "cnn_weight_decay": 1e-4,
    "cnn_max_steps_per_epoch": None,
    "cnn_validation_every_epochs": 1,
    "cnn_early_stopping_patience": 8,
    "skip_multiclass_cnn": False,
    # --- ensemble ----------------------------------------------------------
    "ensemble_members": ["segnet_brats_4ch", "raunet_brats_4ch"],
    "ensemble_weight_grid": 0.05,
    # --- reproducibility ---------------------------------------------------
    "seed": 20240101,
    "smoke_test": False,   # True -> tiny subsets, 1 epoch, 1 case per split
    # "auto" detects the synthetic fixture from its HDF5 marker and accepts a real BraTS
    # tree of >= 100 volumes as real. Anything else is recorded as "unknown" rather than
    # being claimed as real data. Override with "real_brats2020" / "synthetic_fixture".
    "dataset_provenance": "auto",
}

WORK = Path("/kaggle/working")
if not WORK.exists():  # running outside Kaggle
    WORK = Path("./kaggle_working").resolve()
PATHS = {
    "work": WORK,
    "cache": WORK / "cache",             # resized volumes + masks (resumable)
    "checkpoints": WORK / "checkpoints", # per model: last.pt / best.pt / history.json
    "artifacts": WORK / "artifacts",     # metrics, manifests, calibration
    "reports": WORK / "reports",         # figures, tables
    "logs": WORK / "logs",
    "export": WORK / "brain_mri_artifacts",
}
for path in PATHS.values():
    path.mkdir(parents=True, exist_ok=True)

SMOKE_OVERRIDES = {
    "epochs_segnet": 1,
    # the multiclass pipeline is exercised end to end too, on a handful of small images
    "classification_image_size": 64,
    "classification_validation_fraction": 0.34,
    "multiclass_radiomics_max_images": 4,
    "cnn_epochs": 1,
    "cnn_batch_size": 8,
    "cnn_max_steps_per_epoch": 3,
    "cnn_early_stopping_patience": 1,
    "epochs_unet": 1,
    "epochs_raunet": 1,
    "max_steps_per_epoch": 4,
    "min_steps_per_epoch": 1,
    "validation_cases": 2,
    "radiomics_max_patients": 4,
    "image_classifier_epochs": 1,
    "image_classifier_slices_per_patient": 2,
    "radiomics_slices_per_patient": 1,
    "session_budget_minutes": 30,
}


def apply_smoke_overrides() -> None:
    """Shrink every workload to a few minutes.

    Called here when RUN['smoke_test'] is True; the test suite flips the flag after this
    cell and calls the same function, so the tested configuration is the notebook's own.
    Idempotent: calling it twice changes nothing.
    """
    RUN["smoke_test"] = True
    RUN.update(SMOKE_OVERRIDES)


if RUN["smoke_test"]:
    apply_smoke_overrides()

SEED = int(RUN["seed"])


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False  # speed over bitwise determinism
    torch.backends.cudnn.benchmark = True


def set_deterministic_namespace() -> None:
    """Nothing random is allowed to depend on hidden state across sessions."""
    os.environ.setdefault("PYTHONHASHSEED", str(SEED))


seed_everything()
set_deterministic_namespace()


def write_json(path, payload) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def read_json(path, default=None):
    path = Path(path)
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{path} is not valid JSON: {exc}") from exc


def utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


LOCK["run"] = RUN
LOCK["paths"] = {key: str(value) for key, value in PATHS.items()}
write_json(PATHS["artifacts"] / "run_config.json", {"run": RUN, "paths": LOCK["paths"], "generated_at": utc_now()})

print("Work directory :", PATHS["work"])
print("Smoke test     :", RUN["smoke_test"])
print("Image size     :", RUN["image_size"], "| single-channel:", RUN["single_image_size"])
print("Batch sizes    :", RUN["batch_size"], "|", RUN["batch_size_single"])
print("Epochs         : segnet", RUN["epochs_segnet"], "| unet", RUN["epochs_unet"], "| raunet", RUN["epochs_raunet"])
print("Session budget :", RUN["session_budget_minutes"], "minutes (checkpoints let the next session continue)")
print("Seed           :", SEED)
