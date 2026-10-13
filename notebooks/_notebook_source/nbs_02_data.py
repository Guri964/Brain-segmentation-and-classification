# %% [cell] 3. Dataset discovery
# =============================================================================
# SECTION 2 - Dataset discovery
# =============================================================================
# The BraTS 2020 Kaggle mirror is mounted read-only at
# /kaggle/input/brats2020-training-data. Its layout is
#   BraTS2020_training_data/content/data/volume_<n>/volume_<n>_slice_<m>.h5
# plus name_mapping.csv with Grade (HGG/LGG) and BrATS_2020_subject_ID.
# Nothing is downloaded here: internet access is not required.
# =============================================================================
def discover_dataset(roots=RUN["dataset_roots"]):
    """Return the directory that holds the slice files and any grade table found."""
    candidates = []
    for root in roots:
        base = Path(root)
        if not base.exists():
            continue
        candidates.append(base)
        for nested in base.rglob("data"):
            if nested.is_dir():
                candidates.append(nested)
    for base in candidates:
        slice_files = list(base.glob("volume_*/volume_*_slice_*.h5")) or list(
            base.glob("*/volume_*_slice_*.h5")
        )
        if slice_files:
            return {
                "found": True,
                "root": str(base),
                "layout": "hdf5_slices",
                "slice_files": len(slice_files),
                "searched": [str(item) for item in roots],
            }
    # NIfTI release (one directory per patient)
    for base in candidates:
        nifti = list(base.glob("**/*_seg.nii*")) or list(base.glob("**/*_flair.nii*"))
        if nifti:
            return {
                "found": True,
                "root": str(base),
                "layout": "nifti",
                "files": len(nifti),
                "searched": [str(item) for item in roots],
            }
    return {
        "found": False,
        "searched": [str(item) for item in roots],
        "hint": (
            "Add the dataset to this notebook (Add Data -> awsaf49/brats2020-training-data). "
            "If it is mounted elsewhere, put the path first in RUN['dataset_roots']."
        ),
    }


DATASET = discover_dataset()
LOCK["dataset_discovery"] = DATASET

if not DATASET["found"]:
    message = [
        "BraTS 2020 data was not found in this environment.",
        f"searched: {DATASET['searched']}",
        DATASET["hint"],
        "",
        "Nothing else in this notebook can run without the dataset: no metrics are produced,",
        "no model is trained, and no synthetic substitute is used. Kaggle dataset:",
        "https://www.kaggle.com/datasets/awsaf49/brats2020-training-data",
    ]
    print("\n".join(message))
    raise SystemExit("Dataset missing -- attach it and re-run this cell.")

DATA_ROOT = Path(DATASET["root"])
GRADE_TABLE_CANDIDATES = [
    DATA_ROOT / "name_mapping.csv",
    DATA_ROOT.parent / "name_mapping.csv",
    DATA_ROOT.parent.parent / "name_mapping.csv",
    Path(DATASET["searched"][0]) / "name_mapping.csv",
    Path(DATASET["searched"][0]) / "BraTS2020_training_data" / "name_mapping.csv",
]
GRADE_TABLE = next((path for path in GRADE_TABLE_CANDIDATES if path.exists()), None)
SLICE_FILES = sorted(DATA_ROOT.glob("volume_*/volume_*_slice_*.h5")) or sorted(
    DATA_ROOT.glob("*/volume_*_slice_*.h5")
)

print(f"Dataset root   : {DATA_ROOT}")
print(f"Slice files    : {len(SLICE_FILES)}")
print(f"Grade table    : {GRADE_TABLE if GRADE_TABLE else 'NOT FOUND (labels unavailable -> classification section will refuse to run)'}")
if not SLICE_FILES:
    raise SystemExit("No HDF5 slice files matched the expected pattern.")
# --- what is this dataset? -----------------------------------------------------------
# Every artefact this notebook exports says whether it was trained on the real dataset or
# on a synthetic fixture. That claim is not decoration: it is what the application shows
# next to every prediction, so it is derived from the data and never assumed.
VOLUME_DIRECTORIES = sorted({path.parent.name for path in SLICE_FILES})
FIXTURE_MARKER = None
try:
    import h5py  # also imported by the structure-inspection cell below

    with h5py.File(SLICE_FILES[0], "r") as handle:
        FIXTURE_MARKER = dict(handle.attrs).get("source")
except Exception as exc:  # unreadable file -> not a provenance claim
    print("Could not read the HDF5 attributes of the first slice file:", exc)


def resolve_dataset_provenance(declared, marker, volumes):
    """Return (is_synthetic, note); None means "cannot be established, do not claim real data"."""
    declared = str(declared or "auto").lower()
    if declared in {"synthetic", "synthetic_fixture", "fixture"}:
        return True, "declared in RUN['dataset_provenance']"
    if declared in {"real", "real_brats2020", "brats2020"}:
        return False, "declared in RUN['dataset_provenance']"
    if marker == "synthetic-fixture":
        return True, "HDF5 file attribute source='synthetic-fixture' (scripts/make_brats_fixture.py)"
    if volumes >= 100:
        return False, f"{volumes} patient volumes with a name_mapping.csv, matching the BraTS 2020 release"
    return None, (
        f"neither the fixture marker nor a full BraTS release ({volumes} volumes); set "
        "RUN['dataset_provenance'] explicitly if this really is the real dataset"
    )


DATASET_IS_SYNTHETIC, DATASET_PROVENANCE_NOTE = resolve_dataset_provenance(
    RUN.get("dataset_provenance", "auto"), FIXTURE_MARKER, len(VOLUME_DIRECTORIES)
)
DATASET_NAME = (
    "Synthetic BraTS-format fixture (scripts/make_brats_fixture.py; the labels and masks are "
    "generated, any metric computed from them is meaningless)"
    if DATASET_IS_SYNTHETIC
    else "BraTS 2020 training data (Kaggle mirror awsaf49/brats2020-training-data)"
    if DATASET_IS_SYNTHETIC is False
    else f"Unverified HDF5 tree at {DATA_ROOT} (provenance not established)"
)
print("Patient volumes:", len(VOLUME_DIRECTORIES))
print(
    "Dataset provenance:",
    {True: "SYNTHETIC FIXTURE", False: "real BraTS 2020", None: "unknown"}[DATASET_IS_SYNTHETIC],
    "--",
    DATASET_PROVENANCE_NOTE,
)
if DATASET_IS_SYNTHETIC:
    print(
        "\n" + "!" * 78 + "\n"
        "WARNING: this is the synthetic fixture. Every number this notebook prints describes\n"
        "made-up images, made-up masks and made-up HGG/LGG labels. Nothing in the exported\n"
        "artefacts may be quoted as a result, and the application will label them\n"
        "'trained on a synthetic fixture'.\n" + "!" * 78
    )
LOCK["dataset"] = {
    "root": str(DATA_ROOT),
    "slice_files": len(SLICE_FILES),
    "volumes": len(VOLUME_DIRECTORIES),
    "grade_table": str(GRADE_TABLE) if GRADE_TABLE else None,
    "synthetic": DATASET_IS_SYNTHETIC,
    "provenance_note": DATASET_PROVENANCE_NOTE,
}

# %% [cell] 4. Dataset structure inspection
# =============================================================================
# SECTION 3 - Structure inspection (keys, shapes, dtypes, channel order, masks)
# =============================================================================
import h5py

def inspect_slice_file(path):
    with h5py.File(path, "r") as handle:
        keys = list(handle.keys())
        info = {"path": str(path), "keys": keys, "file_attributes": dict(handle.attrs)}
        for key in keys:
            dataset = handle[key]
            info[key] = {
                "shape": list(dataset.shape),
                "dtype": str(dataset.dtype),
                "attributes": {name: str(value) for name, value in dataset.attrs.items()},
            }
    return info


SAMPLE_COUNT = min(8, len(SLICE_FILES))
STRUCTURE_SAMPLES = []
for path in SLICE_FILES[:SAMPLE_COUNT]:
    STRUCTURE_SAMPLES.append(inspect_slice_file(path))

first = STRUCTURE_SAMPLES[0]
shape_variants = {}
key_variants = {}
mask_shape_variants = {}
for sample in STRUCTURE_SAMPLES:
    key_variants[tuple(sorted(sample["keys"]))] = key_variants.get(tuple(sorted(sample["keys"])), 0) + 1
    if "image" in sample:
        shape_variants[tuple(sample["image"]["shape"])] = shape_variants.get(tuple(sample["image"]["shape"]), 0) + 1
    if "mask" in sample:
        mask_shape_variants[tuple(sample["mask"]["shape"])] = mask_shape_variants.get(tuple(sample["mask"]["shape"]), 0) + 1

with h5py.File(SLICE_FILES[0], "r") as handle:
    image0 = np.asarray(handle["image"][...])
    mask0 = np.asarray(handle["mask"][...])

STRUCTURE = {
    "sampled_files": SAMPLE_COUNT,
    "key_variants": {str(key): value for key, value in key_variants.items()},
    "image_shape_variants": {str(key): value for key, value in shape_variants.items()},
    "mask_shape_variants": {str(key): value for key, value in mask_shape_variants.items()},
    "image_dtype": str(image0.dtype),
    "mask_dtype": str(mask0.dtype),
    "mask_unique_values": [int(item) for item in np.unique(mask0)[:12]],
    "image_range": [float(image0.min()), float(image0.max())],
    "sample_attributes": STRUCTURE_SAMPLES[0],
}
write_json(PATHS["artifacts"] / "dataset_structure.json", STRUCTURE)

print("=" * 78)
print("STRUCTURE")
print("=" * 78)
print("HDF5 keys        :", STRUCTURE["key_variants"])
print("image shapes     :", STRUCTURE["image_shape_variants"])
print("mask shapes      :", STRUCTURE["mask_shape_variants"])
print("image dtype      :", STRUCTURE["image_dtype"], "| mask dtype:", STRUCTURE["mask_dtype"])
print("mask values      :", STRUCTURE["mask_unique_values"])
print("stored image range:", STRUCTURE["image_range"])

if tuple(sorted(STRUCTURE_SAMPLES[0]["keys"])) != ("image", "mask"):
    print(
        "\nWARNING: the HDF5 keys are not exactly ('image', 'mask'). The readers below look for "
        "these two names; adjust `IMAGE_KEY` / `MASK_KEY` if your copy differs."
    )
IMAGE_KEY, MASK_KEY = "image", "mask"

# The mirror stores one binary plane per tumour sub-region (NCR/NET, oedema, enhancing)
# or a single label map. Both are handled: the training target is always the union,
# because which plane means which sub-region is not declared inside the file.
MASK_KIND = "one_hot_channels" if mask0.ndim == 3 and mask0.shape[-1] == 3 else "single_or_label_map"
print("mask encoding    :", MASK_KIND, f"(mask shape {mask0.shape})")
print("channels         : T1, T1ce, T2, FLAIR in the stored order -- the notebook never reorders them")

# %% [cell] 5. Patient identifiers and label verification
# =============================================================================
# SECTION 4 - Patient identifiers and HGG/LGG label verification
# =============================================================================
import csv
import re

VOLUME_PATTERN = re.compile(r"(volume_\d+)")


def volume_key_of(path):
    for part in reversed(path.parts):
        match = VOLUME_PATTERN.search(part)
        if match:
            return match.group(1)
    match = VOLUME_PATTERN.search(path.stem)
    return match.group(1) if match else path.parent.name


def slice_index_of(path):
    match = re.search(r"slice_(\d+)", path.stem)
    return int(match.group(1)) if match else None


def load_grade_table(path):
    """name_mapping.csv -> {identifier: {'grade': 'HGG'|'LGG'|None, 'raw': ...}}"""
    if path is None:
        return {}
    rows = list(csv.DictReader(path.open(encoding="utf-8-sig")))
    if not rows:
        return {}
    columns = list(rows[0].keys())
    id_column = next((c for c in columns if "brats" in c.lower() and "id" in c.lower()), None)
    id_column = id_column or next((c for c in columns if "id" in c.lower() or "name" in c.lower()), columns[0])
    grade_column = next((c for c in columns if "grade" in c.lower()), None)
    if grade_column is None:
        return {}
    table = {}
    for row in rows:
        identifier = str(row.get(id_column, "")).strip()
        if not identifier:
            continue
        raw = str(row.get(grade_column, "")).strip()
        grade = raw.upper() if raw.upper() in {"HGG", "LGG"} else None
        table[identifier] = {"grade": grade, "raw": raw, "id_column": id_column, "grade_column": grade_column}
    return table


GRADES = load_grade_table(GRADE_TABLE) if GRADE_TABLE else {}
GRADE_BY_VOLUME = {}
for identifier, entry in GRADES.items():
    GRADE_BY_VOLUME[identifier.lower()] = entry
    case_insensitive = identifier.strip().lower()
    GRADE_BY_VOLUME[case_insensitive] = entry
    # BraTS_2020_subject_ID / BraTS20_Training_001 style identifiers also map by number
    number_match = re.search(r"(\d{3,4})$", identifier)
    if number_match:
        GRADE_BY_VOLUME[f"volume_{int(number_match.group(1))}"] = entry

VOLUMES = {}
for path in SLICE_FILES:
    key = volume_key_of(path)
    entry = VOLUMES.setdefault(key, {"volume": key, "slices": [], "grade": None, "grade_raw": None})
    entry["slices"].append(path)
for key, entry in VOLUMES.items():
    label_entry = GRADE_BY_VOLUME.get(key.lower()) or GRADE_BY_VOLUME.get(key)
    entry["grade"] = (label_entry or {}).get("grade")
    entry["grade_raw"] = (label_entry or {}).get("raw")
    entry["slices"].sort(key=lambda item: (slice_index_of(item) is None, slice_index_of(item)))

LABELLED = {key: entry for key, entry in VOLUMES.items() if entry["grade"] in {"HGG", "LGG"}}
UNLABELLED = sorted(key for key, entry in VOLUMES.items() if entry["grade"] not in {"HGG", "LGG"})
CLASS_COUNTS = {}
for entry in LABELLED.values():
    CLASS_COUNTS[entry["grade"]] = CLASS_COUNTS.get(entry["grade"], 0) + 1

LABEL_REPORT = {
    "volumes": len(VOLUMES),
    "slices": len(SLICE_FILES),
    "grade_table": str(GRADE_TABLE) if GRADE_TABLE else None,
    "grade_table_rows": len(GRADES),
    "labelled_patients": len(LABELLED),
    "unlabelled_patients": len(UNLABELLED),
    "class_counts": CLASS_COUNTS,
    "label_level": "patient (a slice inherits its volume's grade; a mask can never carry a grade)",
    "slices_per_volume": {
        "min": min(len(entry["slices"]) for entry in VOLUMES.values()),
        "max": max(len(entry["slices"]) for entry in VOLUMES.values()),
    },
}
write_json(PATHS["artifacts"] / "label_verification.json", LABEL_REPORT)

print("=" * 78)
print("IDENTIFIERS AND LABELS")
print("=" * 78)
print("Volumes (patients):", len(VOLUMES), "| slices:", len(SLICE_FILES))
print("Grade table rows  :", len(GRADES), "-> labelled volumes:", len(LABELLED))
print("Class counts      :", CLASS_COUNTS)
print("Unlabelled        :", len(UNLABELLED), UNLABELLED[:5])
print("Slices per volume :", LABEL_REPORT["slices_per_volume"])
print(
    "\nHGG/LGG is the WHO grade group recorded by the dataset for a patient. It is not a\n"
    "malignancy verdict, not a stage and not a prognosis, and it is never derived from a mask."
)

if not GRADE_TABLE:
    print(
        "\nWARNING: no grade table was found, so classification cannot be trained here. "
        "Segmentation sections still run. Set RUN/GRADE_TABLE once the CSV is available."
    )
elif len(LABELLED) < 20:
    print("\nWARNING: fewer than 20 labelled patients -- classification metrics would not mean much.")

CLASSIFICATION_POSSIBLE = bool(GRADE_TABLE) and len(LABELLED) >= 2

# %% [cell] 6. Exploratory data analysis
# =============================================================================
# SECTION 5 - Exploratory data analysis (per-patient slice counts, tumour slices,
# intensity distributions, class balance)
# =============================================================================
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def tumour_fraction(path):
    with h5py.File(path, "r") as handle:
        mask = np.asarray(handle[MASK_KEY][...])
    return float((mask > 0).any(axis=-1).mean()) if mask.ndim == 3 else float((mask > 0).mean())


EDA_SAMPLE = min(400, len(SLICE_FILES))
step = max(1, len(SLICE_FILES) // EDA_SAMPLE)
EDA_FILES = SLICE_FILES[::step][:EDA_SAMPLE]
eda_rows = []
for path in EDA_FILES:
    with h5py.File(path, "r") as handle:
        image = np.asarray(handle[IMAGE_KEY][...], dtype=np.float32)
        mask = np.asarray(handle[MASK_KEY][...])
    whole = (mask > 0).any(axis=-1) if mask.ndim == 3 else (mask > 0)
    eda_rows.append(
        {
            "volume": volume_key_of(path),
            "slice": slice_index_of(path),
            "tumour_fraction": float(whole.mean()),
            "has_tumour": bool(whole.any()),
            "mean_intensity": float(image.mean()),
            "channel_means": [float(image[..., c].mean()) for c in range(image.shape[-1])],
        }
    )

tumour_rows = [row for row in eda_rows if row["has_tumour"]]
fractions = [row["tumour_fraction"] for row in tumour_rows] or [0.0]
per_volume = {}
for row in eda_rows:
    per_volume.setdefault(row["volume"], 0)
    per_volume[row["volume"]] += 1

EDA = {
    "sampled_slices": len(eda_rows),
    "sampled_patients": len(per_volume),
    "slices_with_tumour": len(tumour_rows),
    "slices_without_tumour": len(eda_rows) - len(tumour_rows),
    "tumour_fraction_mean": float(np.mean(fractions)),
    "tumour_fraction_median": float(np.median(fractions)),
    "slices_per_patient_percentiles": [
        float(item) for item in np.percentile(list(per_volume.values()) or [0], [0, 10, 50, 90, 100])
    ],
    "channel_mean_intensity": [float(np.mean([row["channel_means"][c] for row in eda_rows])) for c in range(4)],
    "class_counts": CLASS_COUNTS,
}
write_json(PATHS["artifacts"] / "eda.json", EDA)

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes[0, 0].hist([row["tumour_fraction"] for row in eda_rows], bins=40, color="#0ea5e9")
axes[0, 0].set_title("Fraction of the slice covered by tumour")
axes[0, 0].set_xlabel("tumour pixels / total pixels")
axes[0, 1].hist(list(per_volume.values()), bins=30, color="#8b5cf6")
axes[0, 1].set_title("Slices per patient (sampled)")
axes[0, 1].set_xlabel("slices")
axes[1, 0].bar(list(CLASS_COUNTS.keys()) or ["no labels"], list(CLASS_COUNTS.values()) or [0], color="#14b8a6")
axes[1, 0].set_title("Patients per grade (HGG/LGG)")
channel_means = EDA["channel_mean_intensity"]
axes[1, 1].bar(["T1", "T1ce", "T2", "FLAIR"][: len(channel_means)], channel_means, color="#f59e0b")
axes[1, 1].set_title("Mean raw intensity per channel")
plt.tight_layout()
plt.savefig(PATHS["reports"] / "01_eda.png", dpi=130)
plt.close(fig)

print("=" * 78)
print("EXPLORATORY SUMMARY")
print("=" * 78)
for key in ("sampled_slices", "sampled_patients", "slices_with_tumour", "slices_without_tumour", "tumour_fraction_mean", "tumour_fraction_median"):
    print(f"{key:26s}: {EDA[key]}")
print("class counts              :", CLASS_COUNTS)
print(f"figure saved              : {PATHS['reports'] / '01_eda.png'}")

# %% [cell] 7. MRI visualisation
# =============================================================================
# SECTION 6 - MRI visualisation (the four modalities of a patient)
# =============================================================================
def read_slice(path):
    with h5py.File(path, "r") as handle:
        image = np.asarray(handle[IMAGE_KEY][...], dtype=np.float32)
        mask = np.asarray(handle[MASK_KEY][...])
    channels_last = image.ndim == 3 and image.shape[-1] <= 8
    image = image if channels_last or image.ndim == 2 else np.moveaxis(image, 0, -1)
    if image.ndim == 2:
        image = image[..., None]
    whole = ((mask > 0).any(axis=-1) if mask.ndim == 3 else (mask > 0))
    return image, whole


EXAMPLE_VOLUME = sorted(LABELLED or VOLUMES)[0]
EXAMPLE_ENTRY = VOLUMES[EXAMPLE_VOLUME]
print(f"Example patient: {EXAMPLE_VOLUME} (grade {EXAMPLE_ENTRY['grade']}) with {len(EXAMPLE_ENTRY['slices'])} slices")

tumour_slice_path = None
for path in EXAMPLE_ENTRY["slices"]:
    _, whole = read_slice(path)
    if whole.any():
        tumour_slice_path = path
        break
if tumour_slice_path is None:
    tumour_slice_path = EXAMPLE_ENTRY["slices"][len(EXAMPLE_ENTRY["slices"]) // 2]
EXAMPLE_IMAGE, EXAMPLE_MASK = read_slice(tumour_slice_path)
print("Showing", tumour_slice_path)

fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
for index, name in enumerate(["T1", "T1ce", "T2", "FLAIR"]):
    if index >= EXAMPLE_IMAGE.shape[-1]:
        axes[index].axis("off")
        continue
    axes[index].imshow(EXAMPLE_IMAGE[..., index], cmap="gray")
    axes[index].set_title(f"{name} (channel {index})")
    axes[index].axis("off")
plt.tight_layout()
plt.savefig(PATHS["reports"] / "02_modalities.png", dpi=130)
plt.close(fig)
print(f"figure saved: {PATHS['reports'] / '02_modalities.png'}")

# %% [cell] 8. Segmentation-mask visualisation
# =============================================================================
# SECTION 7 - Segmentation-mask visualisation (sub-region planes and the union)
# =============================================================================
with h5py.File(tumour_slice_path, "r") as handle:
    raw_mask = np.asarray(handle[MASK_KEY][...])

fig, axes = plt.subplots(1, 5, figsize=(20, 4.4))
axes[0].imshow(EXAMPLE_IMAGE[..., 0], cmap="gray")
axes[0].set_title("T1")
axes[1].imshow(EXAMPLE_IMAGE[..., 3], cmap="gray")
axes[1].set_title("FLAIR")
if raw_mask.ndim == 3:
    for plane in range(min(3, raw_mask.shape[-1])):
        axes[2 + plane].imshow(raw_mask[..., plane], cmap="magma")
        axes[2 + plane].set_title(f"stored mask plane {plane}")
else:
    axes[2].imshow(raw_mask, cmap="magma")
    axes[2].set_title("stored mask")
axes[4].imshow(EXAMPLE_IMAGE[..., 3], cmap="gray")
axes[4].imshow(np.ma.masked_where(~EXAMPLE_MASK, EXAMPLE_MASK), cmap="autumn", alpha=0.45)
axes[4].set_title("FLAIR + whole tumour (union)")
for axis in axes:
    axis.axis("off")
plt.tight_layout()
plt.savefig(PATHS["reports"] / "03_masks.png", dpi=130)
plt.close(fig)

sub_region_note = (
    "The stored planes are treated as sub-region planes: they are kept separate for reporting, "
    "and only their union is used as the training target, because the file does not declare "
    "which plane is which sub-region."
)
MASK_REPORT = {
    "example_slice": str(tumour_slice_path),
    "stored_shape": list(raw_mask.shape),
    "stored_unique_values": [int(item) for item in np.unique(raw_mask)[:12]],
    "whole_tumour_pixels": int(EXAMPLE_MASK.sum()),
    "note": sub_region_note,
}
write_json(PATHS["artifacts"] / "mask_visualisation.json", MASK_REPORT)
print(sub_region_note)
print("stored mask values:", MASK_REPORT["stored_unique_values"], "| whole-tumour pixels:", MASK_REPORT["whole_tumour_pixels"])
print(f"figure saved: {PATHS['reports'] / '03_masks.png'}")

# %% [cell] 9. Data validation and preprocessing (resumable cache)
# =============================================================================
# SECTION 8 - Validation and preprocessing
# =============================================================================
# For every patient volume:
#   * read each slice, build the whole-tumour mask (union of every non-zero label),
#   * resize to the size the model consumes (nearest for masks, bilinear for images),
#   * normalise exactly as the application does:
#       4-channel models : global z-score over the (H, W, C) block, then CHW
#       single-channel   : intensity / max-style scaling to [0, 1]  (the shipped
#                          U-Net contract is "image / 255")
#   * store float16 arrays so the whole cache fits comfortably on the Kaggle disk.
# The cache is written per patient and skipped when it already exists, so a
# resumed session does not repeat this work.
# =============================================================================
from PIL import Image as PILImage


def resize2d(array, size, nearest=False):
    if array.shape[0] == size and array.shape[1] == size:
        return array.astype(np.float32)
    resample = PILImage.NEAREST if nearest else PILImage.BILINEAR
    image = PILImage.fromarray(array.astype(np.float32), mode="F")
    resized = image.resize((size, size), resample=resample)
    return np.asarray(resized, dtype=np.float32)


def zscore_channels(block):
    """Global z-score over the (H, W, C) block, matching the application contract."""
    mean = float(block.mean())
    std = float(block.std()) or 1.0
    return (block - mean) / std


def build_patient_cache(volume_key, entry, size, single_size, single_channel, force=False):
    target = PATHS["cache"] / f"{volume_key}.npz"
    if target.exists() and not force:
        return {"volume": volume_key, "cached": True, "path": str(target)}
    channels, singles, masks = [], [], []
    for path in entry["slices"]:
        image, whole = read_slice(path)
        channels.append(zscore_channels(resize2d_channels(image, size)))
        masks.append(resize2d(whole.astype(np.float32), size, nearest=True) > 0.5)
        flair = image[..., single_channel] if image.shape[-1] > single_channel else image[..., 0]
        pairs = single_pair(flair, single_size)
        singles.append(pairs)
    np.savez_compressed(
        target,
        images=np.stack(channels).astype(np.float16),
        masks=np.stack(masks).astype(np.uint8),
        singles=np.stack(singles).astype(np.float16),
        grade=np.array([entry["grade"] or ""]),
        slices=np.array([str(path) for path in entry["slices"]]),
    )
    return {
        "volume": volume_key,
        "cached": False,
        "path": str(target),
        "slices": len(masks),
        "size_bytes": target.stat().st_size,
    }


def resize2d_channels(image, size):
    return np.stack([resize2d(image[..., c], size) for c in range(image.shape[-1])], axis=-1)


def single_pair(flair, size):
    """Return the 512x512 single-channel input and the [0, 1] scaling used for it."""
    resized = resize2d(flair, size)
    low, high = float(resized.min()), float(resized.max())
    if high <= low:
        scaled = np.zeros_like(resized)
    elif high <= 1.0 + 1e-6:
        scaled = resized  # already [0, 1]
    elif high <= 255.1:
        scaled = resized / 255.0  # the / 255 contract of the shipped U-Net
    else:
        scaled = (resized - low) / (high - low)
    return scaled.astype(np.float32)


CACHE_REPORT = []
CACHE_STARTED = time.time()
for index, (volume_key, entry) in enumerate(sorted(VOLUMES.items()), start=1):
    CACHE_REPORT.append(
        build_patient_cache(volume_key, entry, RUN["image_size"], RUN["single_image_size"], RUN["single_channel_index"])
    )
    if index % 25 == 0 or index == len(VOLUMES):
        elapsed = time.time() - CACHE_STARTED
        print(f"cached {index}/{len(VOLUMES)} volumes ({elapsed:.0f}s, {elapsed / index:.2f}s per volume)")
write_json(PATHS["artifacts"] / "cache_report.json", CACHE_REPORT)

CACHED = [item for item in CACHE_REPORT if Path(item["path"]).exists()]
total_bytes = sum(Path(item["path"]).stat().st_size for item in CACHED)
print("=" * 78)
print("PREPROCESSING")
print("=" * 78)
print(f"cached volumes   : {len(CACHED)}")
print(f"cache size       : {total_bytes / 1024 ** 2:.1f} MiB")
print(f"4-channel tensors: {len(CACHED)} x [N_slices, {RUN['image_size']}, {RUN['image_size']}, 4] (z-scored, CHW at load time)")
print(f"1-channel tensors: {len(CACHED)} x [N_slices, {RUN['single_image_size']}, {RUN['single_image_size']}] (scaled to [0, 1])")
print("A rerun skips existing files, so this cell is cheap the second time.")


class CacheError(RuntimeError):
    pass


def load_patient(volume_key, split_guard=True):
    path = PATHS["cache"] / f"{volume_key}.npz"
    if not path.exists():
        raise CacheError(f"{volume_key} is not cached; run the preprocessing cell first")
    with np.load(path, allow_pickle=False) as payload:
        return {
            "volume": volume_key,
            "images": payload["images"],       # (S, H, W, 4) float16
            "masks": payload["masks"],         # (S, H, W) uint8
            "singles": payload["singles"],     # (S, 512, 512) float16
            "grade": str(payload["grade"][0]) or None,
            "slices": [str(item) for item in payload["slices"]],
        }


def validate_cache(sample=5):
    """Fail early and loudly on a corrupt cache instead of training on garbage."""
    problems = []
    for item in CACHED[:sample]:
        payload = np.load(Path(item["path"]), allow_pickle=False)
        images, masks = payload["images"], payload["masks"]
        if images.ndim != 4 or images.shape[-1] != 4:
            problems.append(f"{item['volume']}: image tensor {images.shape} is not (S, H, W, 4)")
        if masks.ndim != 3:
            problems.append(f"{item['volume']}: mask tensor {masks.shape} is not (S, H, W)")
        if not np.isfinite(images.astype(np.float32)).all():
            problems.append(f"{item['volume']}: non-finite image values")
        if masks.astype(np.uint8).max(initial=0) > 1:
            problems.append(f"{item['volume']}: mask is not binary (values above 1)")
    if problems:
        raise CacheError("cache validation failed: " + "; ".join(problems[:5]))
    return {"checked": min(sample, len(CACHED)), "problems": problems}


CACHE_VALIDATION = validate_cache()
print("cache validation :", CACHE_VALIDATION)
