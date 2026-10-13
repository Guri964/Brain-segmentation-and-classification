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
#
# Two on-disk layouts, one reader. The difference is measured at the bottom of this cell,
# not assumed:
#
#   "npy" (default)   cache/<volume>.cache/{images,masks,singles}.npy + meta.json
#                     One memory-mappable file per member: reading slice i reads slice i.
#   "npz" (legacy)    cache/<volume>.npz, a savez_compressed archive. ANY access to a
#                     member inflates that whole member, so reading a single slice of a
#                     real BraTS volume costs a full decompression of images + masks +
#                     singles (165 MiB). A cache written by an earlier version of this
#                     notebook still reads, and `upgrade_patient_cache()` converts it in
#                     place -- verifying every array before it deletes anything.
#
# Both layouts are read through `load_patient()` and `PATIENT_STORE`, so every other
# section of the notebook is layout-agnostic and an un-migrated cache still trains.
# =============================================================================
from PIL import Image as PILImage
from collections import OrderedDict
import shutil
import zipfile


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


class CacheError(RuntimeError):
    pass


# =============================================================================
# Cache layout
# =============================================================================
CACHE_MEMBERS = ("images", "masks", "singles")
MEMORY_MAP_FORMAT = "npy"
ARCHIVE_FORMAT = "npz"
CACHE_INDEX_NAME = "_cache_index.json"   # the underscore keeps it out of the patient namespace
CACHE_DIRECTORY_SUFFIX = ".cache"


def cache_archive(volume_key):
    """The compressed single-file layout (what earlier versions of this notebook wrote)."""
    return PATHS["cache"] / f"{volume_key}.npz"


def cache_directory(volume_key):
    """The memory-mappable per-patient layout."""
    return PATHS["cache"] / f"{volume_key}{CACHE_DIRECTORY_SUFFIX}"


def cache_member_path(volume_key, member):
    return cache_directory(volume_key) / f"{member}.npy"


def cache_meta_path(volume_key):
    return cache_directory(volume_key) / "meta.json"


def cache_layout(volume_key):
    """`"npy"`, `"npz"` or None. The memory-mappable layout wins when both are present."""
    if cache_member_path(volume_key, "images").exists() and cache_member_path(volume_key, "masks").exists():
        return MEMORY_MAP_FORMAT
    if cache_archive(volume_key).exists():
        return ARCHIVE_FORMAT
    return None


def cache_entry_path(volume_key):
    """Where this patient lives now."""
    return cache_directory(volume_key) if cache_layout(volume_key) == MEMORY_MAP_FORMAT else cache_archive(volume_key)


def cached_volumes():
    """Every patient key present in the cache, whatever layout it is in."""
    keys = set()
    for path in PATHS["cache"].glob(f"*{CACHE_DIRECTORY_SUFFIX}"):
        if path.is_dir():
            keys.add(path.name[: -len(CACHE_DIRECTORY_SUFFIX)])
    for path in PATHS["cache"].glob(f"*.{ARCHIVE_FORMAT}"):
        keys.add(path.stem)
    return sorted(keys)


def round_mib(value):
    return round(value / 1024 ** 2, 1)


def layout_bytes(volume_key):
    """What the cached payload of one patient costs on disk."""
    layout = cache_layout(volume_key)
    if layout == MEMORY_MAP_FORMAT:
        return sum(path.stat().st_size for path in cache_directory(volume_key).glob("*.npy"))
    if layout == ARCHIVE_FORMAT:
        return cache_archive(volume_key).stat().st_size
    return 0


# =============================================================================
# Slice counts without reading the arrays
# =============================================================================
# The old code learned how many slices a patient has by loading the whole payload:
#
#     for volume_key in patient_ids:
#         payload = load_patient(volume_key)              # a whole archive inflation
#         index.extend(... for position in range(payload["masks"].shape[0]))
#
# For the 369-volume BraTS 2020 release that is 369 archive inflations per epoch, for
# training AND for every validation pass, to read one integer per patient. Both layouts
# expose the shape in a header, so it costs a header read instead.


def read_npy_header(path):
    """(shape, dtype) of a .npy file, without reading its data."""
    with open(path, "rb") as handle:
        version = np.lib.format.read_magic(handle)
        reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
        shape, fortran_order, dtype = reader(handle)
    return tuple(shape), np.dtype(dtype)


def read_archive_member_header(path, member):
    """(shape, dtype) of one member of an .npz archive, inflating only its header."""
    with zipfile.ZipFile(path) as archive:
        with archive.open(f"{member}.npy") as handle:
            version = np.lib.format.read_magic(handle)
            reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
            shape, fortran_order, dtype = reader(handle)
    return tuple(shape), np.dtype(dtype)


def probe_patient(volume_key):
    """Slice count, member shapes and dtypes of a cached patient. Touches headers only."""
    layout = cache_layout(volume_key)
    if layout is None:
        return None
    members = {}
    if layout == MEMORY_MAP_FORMAT:
        for member in CACHE_MEMBERS:
            path = cache_member_path(volume_key, member)
            if path.exists():
                members[member] = read_npy_header(path)
    else:
        archive = cache_archive(volume_key)
        with zipfile.ZipFile(archive) as handle:
            available = set(handle.namelist())
        for member in CACHE_MEMBERS:
            if f"{member}.npy" in available:
                members[member] = read_archive_member_header(archive, member)
    if not members:
        return None
    return {
        "volume": volume_key,
        "layout": layout,
        "slices": int(members.get("masks", members["images"])[0][0]),
        "members": {member: list(shape) for member, (shape, _) in members.items()},
        "dtypes": {member: str(dtype) for member, (_, dtype) in members.items()},
        "bytes": layout_bytes(volume_key),
    }


CACHE_INDEX: dict = {}
CACHE_INDEX_PATH = PATHS["cache"] / CACHE_INDEX_NAME
CACHE_INDEX_PROBES = {"header_reads": 0, "arrays_read": 0}


def load_cache_index(rebuild=False):
    """Per-patient slice counts: from the index file, else one header read per patient."""
    global CACHE_INDEX
    if CACHE_INDEX and not rebuild:
        return CACHE_INDEX
    stored = read_json(CACHE_INDEX_PATH, default=None) or {}
    entries = {}
    for volume_key in cached_volumes():
        entry = stored.get(volume_key)
        if entry and entry.get("layout") == cache_layout(volume_key):
            entries[volume_key] = entry
            continue
        probed = probe_patient(volume_key)
        CACHE_INDEX_PROBES["header_reads"] += 1
        if probed is not None:
            entries[volume_key] = probed
    CACHE_INDEX = entries
    return CACHE_INDEX


def write_cache_index():
    write_json(CACHE_INDEX_PATH, {key: value for key, value in sorted(CACHE_INDEX.items())})
    return CACHE_INDEX


def cache_index_entry(volume_key, refresh=False):
    if not CACHE_INDEX:
        load_cache_index()
    entry = CACHE_INDEX.get(volume_key)
    if entry and not refresh and entry.get("layout") == cache_layout(volume_key):
        return entry
    probed = probe_patient(volume_key)
    CACHE_INDEX_PROBES["header_reads"] += 1
    if probed is not None:
        CACHE_INDEX[volume_key] = probed
    return probed


def patient_slice_count(volume_key):
    entry = cache_index_entry(volume_key)
    if entry is None:
        raise CacheError(f"{volume_key} is not cached; run the preprocessing cell first")
    return int(entry["slices"])


# =============================================================================
# Writing the cache
# =============================================================================
def write_npy_part(path, array):
    """Write `<path>.part` and return it. Nothing at `path` changes until it is published."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    if part.exists():
        part.unlink()
    with open(part, "wb") as handle:
        np.save(handle, array, allow_pickle=False)
    return part


def parts_of(volume_key):
    return {member: cache_member_path(volume_key, member).with_name(f"{member}.npy.part") for member in CACHE_MEMBERS}


def clean_parts(volume_key):
    for part in parts_of(volume_key).values():
        if part.exists():
            part.unlink()


def write_cache_members(volume_key, arrays, *, verify=True):
    """Write every member of one patient, and only publish them once they are all complete.

    `cache_layout()` reports this patient as cached as soon as `images.npy` and `masks.npy`
    exist, so those two are published last: a session that dies (or a conversion that fails)
    can leave `.part` files or a partial set behind, but it can never leave a cache that
    looks complete while a member is missing. With `verify=True` every part file is read
    back and compared value by value before it is published.
    """
    parts = {}
    for member in ("singles", "masks", "images"):        # `images` is published last on purpose
        if member in arrays:
            parts[member] = write_npy_part(cache_member_path(volume_key, member), arrays[member])
    if verify:
        for member, part in parts.items():
            written = np.load(part, mmap_mode="r")
            same = (
                written.shape == np.shape(arrays[member])
                and written.dtype == np.asarray(arrays[member]).dtype
                and np.array_equal(np.asarray(written), arrays[member])
            )
            del written
            if not same:
                clean_parts(volume_key)
                raise CacheError(f"{volume_key}: the {member} member did not survive the write")
    for member in ("singles", "masks", "images"):
        if member in parts:
            parts[member].replace(cache_member_path(volume_key, member))
    return {member: cache_member_path(volume_key, member) for member in parts}


def build_patient_cache(volume_key, entry, size, single_size, single_channel, force=False):
    layout = cache_layout(volume_key)
    if layout is not None and not force:
        return {"volume": volume_key, "cached": True, "layout": layout, "path": str(cache_entry_path(volume_key))}
    channels, singles, masks = [], [], []
    for path in entry["slices"]:
        image, whole = read_slice(path)
        channels.append(zscore_channels(resize2d_channels(image, size)))
        masks.append(resize2d(whole.astype(np.float32), size, nearest=True) > 0.5)
        flair = image[..., single_channel] if image.shape[-1] > single_channel else image[..., 0]
        pairs = single_pair(flair, single_size)
        singles.append(pairs)
    arrays = {
        "images": np.stack(channels).astype(np.float16),
        "masks": np.stack(masks).astype(np.uint8),
        "singles": np.stack(singles).astype(np.float16),
    }
    slice_paths = [str(path) for path in entry["slices"]]
    grade = entry["grade"] or ""

    if RUN.get("cache_format", MEMORY_MAP_FORMAT) == ARCHIVE_FORMAT:
        target = cache_archive(volume_key)
        np.savez_compressed(target, grade=np.array([grade]), slices=np.array(slice_paths), **arrays)
        written, layout = target, ARCHIVE_FORMAT
    else:
        try:
            write_cache_members(volume_key, arrays)
        except CacheError as exc:
            return {"volume": volume_key, "cached": False, "layout": MEMORY_MAP_FORMAT, "error": str(exc)}
        write_json(cache_meta_path(volume_key), {"volume": volume_key, "grade": grade, "slices": slice_paths})
        written, layout = cache_directory(volume_key), MEMORY_MAP_FORMAT

    CACHE_INDEX[volume_key] = {
        "volume": volume_key,
        "layout": layout,
        "slices": len(masks),
        "members": {member: list(array.shape) for member, array in arrays.items()},
        "dtypes": {member: str(array.dtype) for member, array in arrays.items()},
        "bytes": layout_bytes(volume_key),
    }
    return {
        "volume": volume_key,
        "cached": False,
        "layout": layout,
        "path": str(written),
        "slices": len(masks),
        "size_bytes": layout_bytes(volume_key),
    }


def upgrade_patient_cache(volume_keys=None, *, headroom_bytes=512 * 1024 ** 2, verbose=True):
    """Convert legacy .npz archives to the memory-mappable layout, in place and verified.

    One-time, and it costs one archive inflation per patient -- the same read the old
    training loop paid for EVERY slice and for every patient just to count slices. Nothing
    is deleted before it has been verified:

      1. the arrays are written to `<member>.npy.part`;
      2. each written file is memory-mapped and compared with the archive member by shape,
         dtype and every value;
      3. only then is the part file renamed, i.e. the member becomes visible to the reader;
      4. only when every member is complete is the archive deleted.

    A patient whose conversion fails keeps its archive and is listed in `failed`; the reader
    keeps serving it from the archive. Free space is checked per patient first, so the peak
    cost is one patient's payload rather than the whole cache.
    """
    report = {"converted": [], "already": [], "failed": [], "bytes_before": 0, "bytes_after": 0, "seconds": 0.0}
    candidates = [
        volume_key
        for volume_key in (volume_keys if volume_keys is not None else cached_volumes())
        if cache_layout(volume_key) == ARCHIVE_FORMAT
    ]
    if not candidates:
        return report
    started = time.time()
    for index, volume_key in enumerate(candidates, start=1):
        archive = cache_archive(volume_key)
        try:
            size = archive.stat().st_size
            free = shutil.disk_usage(PATHS["cache"]).free
            if free < size + headroom_bytes:
                report["failed"].append(
                    {"volume": volume_key, "reason": f"only {free / 1024 ** 2:.0f} MiB free, the archive needs {size / 1024 ** 2:.0f} MiB"}
                )
                continue
            with np.load(archive, allow_pickle=False) as payload:
                sources = {member: np.asarray(payload[member]) for member in CACHE_MEMBERS if member in payload.files}
                meta = {
                    "volume": volume_key,
                    "grade": str(payload["grade"][0]) if "grade" in payload.files else "",
                    "slices": [str(item) for item in payload["slices"]] if "slices" in payload.files else [],
                }
            # Every member is written, verified against the archive, and only then published;
            # `images.npy` is published last, so the patient only becomes "memory-mappable"
            # once all of it is on disk.
            write_cache_members(volume_key, sources, verify=True)
            write_json(cache_meta_path(volume_key), meta)
            archive.unlink()
            report["converted"].append(volume_key)
            report["bytes_before"] += size
            report["bytes_after"] += layout_bytes(volume_key)
        except Exception as exc:  # noqa: BLE001 - a failed conversion must never lose the archive
            clean_parts(volume_key)
            report["failed"].append({"volume": volume_key, "reason": f"{type(exc).__name__}: {exc}"})
        if verbose and (index % 25 == 0 or index == len(candidates)):
            done = time.time() - started
            print(
                f"  cache upgrade {index}/{len(candidates)} ({done:.0f}s, {done / index:.2f}s per volume) -- "
                f"{len(report['converted'])} converted, {len(report['failed'])} failed"
            )
    report["seconds"] = round(time.time() - started, 2)
    report["bytes_before"] = int(report["bytes_before"])
    report["bytes_after"] = int(report["bytes_after"])
    if report["failed"]:
        print(
            f"WARNING: {len(report['failed'])} volumes kept their .npz archive "
            f"({report['failed'][0]['reason']}). They still load, just more slowly."
        )
    return report


# =============================================================================
# Reading the cache: one payload per patient, one member per request
# =============================================================================
class PatientPayloadStore:
    """Bounded reader for the cached patient payloads.

    What went wrong before this class existed: `iterate_batches()` called `load_patient()`
    once per patient (to count slices) and once per SLICE (to build the batch), and every
    call inflated that patient's whole archive -- images + masks + singles, 165 MiB for a
    real BraTS volume -- to look at one slice. A batch of 8 therefore cost 20 archive
    inflations. On the reported run that measured 109.7 s for the first batch; the profile
    put 100% of it inside `zipfile.read` (deflate), and the steady-state cost was 7.7 s per
    batch of 8 -- about 1.6 hours of CPU per epoch, with the GPU waiting the whole time.

    What this class does instead:

      * reads are counted per PATIENT, not per slice -- a batch of 8 slices from 8 patients
        costs at most 8 archive reads, and slices that share a patient share one read;
      * only the members the caller asks for are touched: a 4-channel batch never opens
        `singles` (half the payload), a single-channel batch never opens `images`;
      * the memory-mappable layout decodes nothing -- slice i is read directly from disk;
      * memory stays bounded: `patient_cache_mb` caps materialised arrays and
        `patient_cache_handles` caps open memory maps, so nothing loads the dataset into RAM.

    `PATIENT_STORE.stats_report()` is what the benchmark prints.
    """

    def __init__(self, *, budget_bytes=None, max_handles=None):
        self.budget_bytes = (
            int(budget_bytes) if budget_bytes is not None else int(RUN.get("patient_cache_mb", 1536)) * 1024 ** 2
        )
        self.max_handles = (
            int(max_handles) if max_handles is not None else int(RUN.get("patient_cache_handles", 32))
        )
        self._handles = OrderedDict()        # volume -> {member: memory map}
        self._entries = OrderedDict()        # volume -> {"layout", "members", "meta", "bytes", "slices"}
        self._materialised_bytes = 0
        self.stats = {
            "payload_reads": 0,
            "member_reads": 0,
            "member_hits": 0,
            "slice_requests": 0,
            "slices_returned": 0,
            "archive_reads": 0,
            "evictions": 0,
            "bytes_materialised": 0,
        }

    # -- bookkeeping ------------------------------------------------------
    def _entry(self, volume_key):
        entry = self._entries.get(volume_key)
        if entry is not None:
            self._entries.move_to_end(volume_key)
        return entry

    def _new_entry(self, volume_key, layout):
        entry = {"layout": layout, "members": {}, "meta": None, "bytes": 0, "slices": None}
        self._entries[volume_key] = entry
        self._trim_materialised()
        return entry

    def _trim_materialised(self):
        while self._materialised_bytes > self.budget_bytes and len(self._entries) > 1:
            _, entry = self._entries.popitem(last=False)
            self._materialised_bytes -= entry["bytes"]
            self.stats["evictions"] += 1

    def _trim_handles(self):
        while len(self._handles) > self.max_handles:
            self._handles.popitem(last=False)

    def _read_member(self, volume_key, member):
        """The member as it is stored: a memory map for the npy layout, an array for an archive.

        Memory maps live only in `_handles`, never in the entry: each open map holds a file
        descriptor, and with 369 patients x 3 members a dict that keeps them all would hit the
        process file-descriptor limit. `_handles` is capped (`patient_cache_handles`), so an
        evicted map is closed by refcounting as soon as its slices have been copied out.
        Materialised arrays (the archive layout, and `load_patient`) live in the entry instead,
        where they are counted against the byte budget.
        """
        layout = cache_layout(volume_key)
        if layout is None:
            raise CacheError(f"{volume_key} is not cached; run the preprocessing cell first")
        entry = self._entry(volume_key) or self._new_entry(volume_key, layout)
        current = entry["members"].get(member)
        if current is not None:
            self.stats["member_hits"] += 1
            return current
        handles = self._handles.get(volume_key)
        if handles and member in handles:
            self.stats["member_hits"] += 1
            self._handles.move_to_end(volume_key)
            return handles[member]
        self.stats["member_reads"] += 1
        if layout == MEMORY_MAP_FORMAT:
            path = cache_member_path(volume_key, member)
            if not path.exists():
                raise CacheError(f"{volume_key}: the cache has no {member} member")
            array = np.load(path, mmap_mode="r")
            self._handles.setdefault(volume_key, {})[member] = array
            self._handles.move_to_end(volume_key)
            self._trim_handles()
            return array
        with np.load(cache_archive(volume_key), allow_pickle=False) as payload:
            self.stats["archive_reads"] += 1
            if member not in payload.files:
                raise CacheError(f"{volume_key}: the archive has no {member} member")
            array = np.asarray(payload[member])
        entry["members"][member] = array
        entry["bytes"] += int(array.nbytes)
        self._materialised_bytes += int(array.nbytes)
        self.stats["bytes_materialised"] += int(array.nbytes)
        self._trim_materialised()
        return array

    def _materialised_member(self, volume_key, member):
        """The member as an in-memory array (what `load_patient()` promises callers).

        The array is returned directly, so it stays valid even if the eviction below or a
        later request pushes this patient out of the store.
        """
        array = self._read_member(volume_key, member)
        if isinstance(array, np.memmap):
            array = np.array(array, copy=True)          # detach from the file
            entry = self._entry(volume_key)
            if entry is not None:
                entry["members"][member] = array
                entry["bytes"] += int(array.nbytes)
                self._materialised_bytes += int(array.nbytes)
                self.stats["bytes_materialised"] += int(array.nbytes)
            handles = self._handles.get(volume_key)
            if handles is not None:                      # this member no longer needs its map
                handles.pop(member, None)
                if not handles:
                    self._handles.pop(volume_key, None)
            self._trim_materialised()
        return array

    def _meta(self, volume_key, layout):
        entry = self._entry(volume_key) or self._new_entry(volume_key, layout)
        if entry["meta"] is not None:
            return entry["meta"]
        if layout == MEMORY_MAP_FORMAT:
            stored = read_json(cache_meta_path(volume_key), default=None) or {}
            meta = {"grade": str(stored.get("grade") or "") or None,
                    "slices": [str(item) for item in stored.get("slices", [])]}
        else:
            with np.load(cache_archive(volume_key), allow_pickle=False) as payload:
                self.stats["archive_reads"] += 1
                grade = str(payload["grade"][0]) if "grade" in payload.files else ""
                slice_paths = [str(item) for item in payload["slices"]] if "slices" in payload.files else []
            meta = {"grade": grade or None, "slices": slice_paths}
        entry["meta"] = meta
        return meta

    # -- public API -------------------------------------------------------
    def release(self, volume_key=None):
        """Forget cached state (the benchmark uses this to measure a cold read)."""
        if volume_key is None:
            self._handles.clear()
            self._entries.clear()
            self._materialised_bytes = 0
            return
        entry = self._entries.pop(volume_key, None)
        if entry is not None:
            self._materialised_bytes -= entry["bytes"]
        self._handles.pop(volume_key, None)

    def slice_count(self, volume_key):
        entry = self._entry(volume_key)
        if entry is not None and entry["slices"] is not None:
            return entry["slices"]
        count = patient_slice_count(volume_key)
        entry = self._entry(volume_key) or self._new_entry(volume_key, cache_layout(volume_key) or MEMORY_MAP_FORMAT)
        entry["slices"] = count
        return count

    def slices(self, volume_key, positions, *, member):
        """Rows `positions` of one member, in the order requested. Nothing else is read."""
        array = self._read_member(volume_key, member)
        self.stats["slice_requests"] += 1
        if len(positions) == 0:
            return np.empty((0,) + tuple(array.shape[1:]), dtype=array.dtype)
        self.stats["slices_returned"] += len(positions)
        rows = array[np.asarray(positions, dtype=np.int64)]
        return np.array(rows, copy=True)      # always a fresh array, never a view into a map

    def payload(self, volume_key):
        """The full payload dict `load_patient()` has always returned.

        The arrays are detached copies: a caller may keep them after the store evicted the
        patient, without holding a file descriptor open.
        """
        self.stats["payload_reads"] += 1
        layout = cache_layout(volume_key)
        if layout is None:
            raise CacheError(f"{volume_key} is not cached; run the preprocessing cell first")
        images = self._materialised_member(volume_key, "images")
        masks = self._materialised_member(volume_key, "masks")
        singles = self._materialised_member(volume_key, "singles")
        meta = self._meta(volume_key, layout)
        return {
            "volume": volume_key,
            "images": images,     # (S, H, W, 4) float16, channels T1, T1ce, T2, FLAIR
            "masks": masks,       # (S, H, W) uint8, whole tumour
            "singles": singles,   # (S, 512, 512) float16, scaled to [0, 1]
            "grade": meta["grade"],
            "slices": meta["slices"],
            "layout": layout,
        }

    def stats_report(self):
        report = dict(self.stats)
        report["budget_mib"] = round_mib(self.budget_bytes)
        report["materialised_mib"] = round_mib(self._materialised_bytes)
        report["open_memory_maps"] = len(self._handles)
        report["max_handles"] = self.max_handles
        return report


PATIENT_STORE = None


def patient_store(*, budget_bytes=None, max_handles=None, reset=False):
    """The process-wide payload store (created on first use, recreatable for tests)."""
    global PATIENT_STORE
    if PATIENT_STORE is None or reset:
        PATIENT_STORE = PatientPayloadStore(budget_bytes=budget_bytes, max_handles=max_handles)
    return PATIENT_STORE


def load_patient(volume_key, split_guard=True, legacy_archive=False):
    """One patient's cached payload.

    `legacy_archive=True` reads the compressed archive directly, which is what this function
    did before the store existed: one full inflation per call. It is kept so the benchmark
    can measure the old behaviour on the same cache, and so an un-migrated cache stays
    readable even if its index cannot be built.
    """
    if legacy_archive:
        path = cache_archive(volume_key)
        if not path.exists():
            raise CacheError(f"{volume_key} has no .npz archive to read in legacy mode")
        with np.load(path, allow_pickle=False) as payload:
            return {
                "volume": volume_key,
                "images": np.asarray(payload["images"]),
                "masks": np.asarray(payload["masks"]),
                "singles": np.asarray(payload["singles"]),
                "grade": str(payload["grade"][0]) or None,
                "slices": [str(item) for item in payload["slices"]],
                "layout": ARCHIVE_FORMAT,
            }
    return patient_store().payload(volume_key)


def measure_cache_io(volume_keys, *, slices=4, repeats=3, legacy_repeats=1):
    """Cost of preparing a batch, of reading one slice, and of building the slice index.

    Every row is measured on the same volumes, the same slice positions and the same cache
    state, so the layouts and the two readers can be compared like for like:

      * `legacy_*`   what the old batch loop did -- `load_patient()` per slice, i.e. one
                     whole-archive inflation per slice. Reported only for volumes that
                     still have their `.npz` archive, and `None` otherwise, because it
                     cannot be measured honestly once the archive is gone.
      * `batch_*`    what `iterate_batches()` does now -- one member read per patient plus a
                     direct slice read (or one inflation per patient on an archive).
      * `index_*`    building the slice index from the in-memory index (per epoch).
      * `cold_index_*` the first-time cost: one header probe per patient instead of one
                     payload load per patient.
    """
    store = patient_store()
    keys = [key for key in volume_keys if cache_layout(key) is not None]
    if not keys:
        return {"volumes": 0, "note": "no cached volume to measure"}
    positions = list(range(min(slices, max(1, patient_slice_count(keys[0])))))

    def best(fn, repeats=repeats):
        values = []
        for _ in range(repeats):
            started = time.perf_counter()
            fn()
            values.append(time.perf_counter() - started)
        return min(values)

    def optimised_batch():
        for volume_key in keys:
            store.release(volume_key)                     # cold patient, as a shuffled batch sees it
            store.slices(volume_key, positions, member="images")
            store.slices(volume_key, positions, member="masks")

    def legacy_batch():
        # one archive read per slice: that is what the old loop did (`load_patient` per
        # slice, inflating images + masks + singles each time), so this row is the
        # like-for-like "before" number rather than a cheaper approximation of it.
        for volume_key in keys:
            for position in positions:
                with np.load(cache_archive(volume_key), allow_pickle=False) as payload:
                    _ = payload["images"][position], payload["masks"][position]

    def optimised_index():
        return sum(patient_slice_count(key) for key in keys)

    def cold_index():
        for key in keys:
            probe_patient(key)
        CACHE_INDEX_PROBES["header_reads"] += len(keys)
        return len(keys)

    def legacy_index():
        total = 0
        for key in keys:
            total += load_patient(key, legacy_archive=True)["masks"].shape[0]
        return total

    archives_only = all(cache_archive(key).exists() for key in keys)
    layout = cache_layout(keys[0])
    optimised = best(optimised_batch)
    # The legacy reader is the slow one by design: once is enough to measure it.
    legacy = best(legacy_batch, repeats=legacy_repeats) if archives_only else None
    store.release()
    optimised_index_seconds = best(optimised_index, repeats=1)
    cold_index_seconds = best(cold_index, repeats=1)
    legacy_index_seconds = best(legacy_index, repeats=legacy_repeats) if archives_only else None
    store.release()
    total_slices = len(keys) * len(positions)
    return {
        "volumes": len(keys),
        "layout": layout,
        "slices_per_volume": len(positions),
        "batch_seconds": round(optimised, 4),
        "batch_seconds_per_volume": round(optimised / len(keys), 5),
        "batch_seconds_per_slice": round(optimised / total_slices, 6),
        "legacy_batch_seconds": None if legacy is None else round(legacy, 4),
        "legacy_batch_seconds_per_slice": None if legacy is None else round(legacy / total_slices, 6),
        "index_seconds": round(optimised_index_seconds, 4),
        "index_seconds_per_volume": round(optimised_index_seconds / len(keys), 6),
        "cold_index_seconds_per_volume": round(cold_index_seconds / len(keys), 6),
        "legacy_index_seconds_per_volume": None if legacy_index_seconds is None
        else round(legacy_index_seconds / len(keys), 6),
        "legacy_measured": archives_only,
        "store": store.stats_report(),
    }


# =============================================================================
# Build the cache (skipped for every volume that is already cached)
# =============================================================================
CACHE_REPORT = []
CACHE_STARTED = time.time()
for index, (volume_key, entry) in enumerate(sorted(VOLUMES.items()), start=1):
    CACHE_REPORT.append(
        build_patient_cache(volume_key, entry, RUN["image_size"], RUN["single_image_size"], RUN["single_channel_index"])
    )
    if index % 25 == 0 or index == len(VOLUMES):
        elapsed = time.time() - CACHE_STARTED
        print(f"cached {index}/{len(VOLUMES)} volumes ({elapsed:.0f}s, {elapsed / index:.2f}s per volume)")

CACHED = [item for item in CACHE_REPORT if cache_layout(item["volume"]) is not None]
total_bytes = sum(layout_bytes(item["volume"]) for item in CACHED)
write_json(
    PATHS["artifacts"] / "cache_report.json",
    {
        "generated_at": utc_now(),
        "volumes": len(VOLUMES),
        "cached": len(CACHED),
        "preferred_layout": RUN.get("cache_format", MEMORY_MAP_FORMAT),
        "bytes": int(total_bytes),
        "per_volume": CACHE_REPORT,
    },
)

# --- the old behaviour, measured on this cache before it is migrated -------------------
# `load_patient()` inflated the whole compressed archive once per slice read, and once per
# patient just to count slices. Both are measured here on the same volumes, the same slices
# and the same cache state as the optimised reader below, so the before/after comparison is
# a measurement rather than an expectation.
LEGACY_VOLUMES = [item["volume"] for item in CACHED if cache_layout(item["volume"]) == ARCHIVE_FORMAT]
# Four volumes x four slices, read in both layouts. A batch of RUN["batch_size"] is that
# many reads; the per-slice column scales to it without re-measuring.
SAMPLE = LEGACY_VOLUMES[:4] or [item["volume"] for item in CACHED][:4]
CACHE_IO_BEFORE = measure_cache_io(SAMPLE) if LEGACY_VOLUMES else None

# --- one-time, verified conversion of a legacy cache ----------------------------------
CACHE_UPGRADE = {"converted": [], "already": [], "failed": [], "seconds": 0.0}
if RUN.get("upgrade_cache_to_memmap", True) and LEGACY_VOLUMES:
    print(
        f"{len(LEGACY_VOLUMES)} volumes are still in the compressed layout; converting them in "
        "place (verified array by array, one patient at a time)..."
    )
    CACHE_UPGRADE = upgrade_patient_cache()
load_cache_index(rebuild=True)
write_cache_index()

# Same patient IDs, same slices, same batch size: only the cache layout changed.
CACHE_IO_AFTER = measure_cache_io(SAMPLE)
write_json(
    PATHS["artifacts"] / "cache_io_benchmark.json",
    {"sample": SAMPLE, "before": CACHE_IO_BEFORE, "after": CACHE_IO_AFTER, "upgrade": CACHE_UPGRADE, "generated_at": utc_now()},
)

# =============================================================================
# Summary
# =============================================================================
# Sizes are re-read after the conversion, so the numbers describe the cache as it is now.
DISK = shutil.disk_usage(PATHS["cache"])
total_bytes = sum(layout_bytes(item["volume"]) for item in CACHED)
per_volume_bytes = total_bytes / max(1, len(CACHED))
print("=" * 78)
print("PREPROCESSING")
print("=" * 78)
print(f"cached volumes   : {len(CACHED)}")
print(f"cache size       : {total_bytes / 1024 ** 2:.1f} MiB ({per_volume_bytes / 1024 ** 2:.0f} MiB per volume)")
print(f"cache layout     : {sum(1 for item in CACHED if cache_layout(item['volume']) == MEMORY_MAP_FORMAT)} memory-mappable, "
      f"{sum(1 for item in CACHED if cache_layout(item['volume']) == ARCHIVE_FORMAT)} compressed")
if per_volume_bytes:
    print(f"disk space       : {DISK.free / 1024 ** 3:.1f} GiB free of {DISK.total / 1024 ** 3:.1f} GiB; at the measured "
          f"{per_volume_bytes / 1024 ** 3:.3f} GiB per volume the full 369-volume release needs about "
          f"{369 * per_volume_bytes / 1024 ** 3:.0f} GiB")
print(f"4-channel tensors: {len(CACHED)} x [N_slices, {RUN['image_size']}, {RUN['image_size']}, 4] (z-scored, CHW at load time)")
print(f"1-channel tensors: {len(CACHED)} x [N_slices, {RUN['single_image_size']}, {RUN['single_image_size']}] (scaled to [0, 1])")
print("A rerun skips existing cache entries, so this cell is cheap the second time.")
print()
print("CACHE READ BENCHMARK -- same volumes, same slices, same cache state")
print("-" * 78)
if CACHE_IO_BEFORE:
    print(f"  before  archive   : {CACHE_IO_BEFORE['legacy_batch_seconds_per_slice'] * 1000:9.3f} ms per slice "
          f"({CACHE_IO_BEFORE['legacy_batch_seconds']:.3f} s to read "
          f"{CACHE_IO_BEFORE['slices_per_volume']} slices x {CACHE_IO_BEFORE['volumes']} volumes; "
          f"{CACHE_IO_BEFORE['legacy_batch_seconds_per_slice'] * RUN['batch_size'] * 1000:.0f} ms for a batch of {RUN['batch_size']})")
    print(f"                      index build: {CACHE_IO_BEFORE['legacy_index_seconds_per_volume'] * 1000:.1f} ms per volume")
else:
    print("  before  archive   : not measured -- this cache was already converted (or built new)")
print(f"  after   memory map: {CACHE_IO_AFTER['batch_seconds_per_slice'] * 1000:9.4f} ms per slice "
      f"({CACHE_IO_AFTER['batch_seconds']:.4f} s for {CACHE_IO_AFTER['slices_per_volume']} slices x "
      f"{CACHE_IO_AFTER['volumes']} volumes, cold patient each time; "
      f"{CACHE_IO_AFTER['batch_seconds_per_slice'] * RUN['batch_size'] * 1000:.2f} ms for a batch of {RUN['batch_size']})")
print(f"                      index build: {CACHE_IO_AFTER['cold_index_seconds_per_volume'] * 1000:.3f} ms per volume "
      f"for the first probe, then {CACHE_IO_AFTER['index_seconds_per_volume'] * 1e6:.1f} us per volume from the index")
if CACHE_IO_BEFORE and CACHE_IO_BEFORE["legacy_batch_seconds_per_slice"] and CACHE_IO_AFTER["batch_seconds_per_slice"]:
    print(f"  batch preparation is {CACHE_IO_BEFORE['legacy_batch_seconds_per_slice'] / CACHE_IO_AFTER['batch_seconds_per_slice']:,.0f}x "
          f"cheaper per slice; the index no longer reads any array data")
print(f"  in a batch of {RUN['batch_size']}: about "
      f"{CACHE_IO_AFTER['batch_seconds_per_slice'] * RUN['batch_size'] * 1000:.1f} ms of CPU preparation, "
      "one page-cache read per patient, no decompression")
print("=" * 78)


def validate_cache(sample=5):
    """Fail early and loudly on a corrupt cache instead of training on garbage.

    Works for both layouts and reads only a couple of slices per patient, plus the index
    header, so validating the cache does not itself cost a payload load.
    """
    problems = []
    store = patient_store()
    checked = 0
    for item in CACHED[:sample]:
        volume_key = item["volume"]
        entry = cache_index_entry(volume_key, refresh=True)
        if entry is None:
            problems.append(f"{volume_key}: no cache entry")
            continue
        shapes = entry["members"]
        if len(shapes.get("images", [])) != 4 or shapes["images"][-1] != 4:
            problems.append(f"{volume_key}: image tensor {shapes.get('images')} is not (S, H, W, 4)")
        if len(shapes.get("masks", [])) != 3:
            problems.append(f"{volume_key}: mask tensor {shapes.get('masks')} is not (S, H, W)")
        if shapes.get("images", [None])[:2] != shapes.get("masks", [None])[:2]:
            problems.append(
                f"{volume_key}: mask resolution {shapes.get('masks', [])[:2]} does not match images {shapes.get('images', [])[:2]}"
            )
        positions = sorted({0, max(0, int(entry["slices"]) - 1)})
        images = np.asarray(store.slices(volume_key, positions, member="images"), dtype=np.float32)
        masks = np.asarray(store.slices(volume_key, positions, member="masks"))
        if not np.isfinite(images).all():
            problems.append(f"{volume_key}: non-finite image values")
        if masks.max(initial=0) > 1:
            problems.append(f"{volume_key}: mask is not binary (values above 1)")
        if masks.shape[-2:] != images.shape[-3:-1]:
            problems.append(f"{volume_key}: mask and image resolutions disagree ({masks.shape} vs {images.shape})")
        checked += 1
    store.release()
    if problems:
        raise CacheError("cache validation failed: " + "; ".join(problems[:5]))
    return {"checked": checked, "problems": problems, "index_probes": CACHE_INDEX_PROBES["header_reads"]}


CACHE_VALIDATION = validate_cache()
print("cache validation :", CACHE_VALIDATION)
print("payload store    :", patient_store().stats_report())
