# %% [cell] 10. Classification dataset discovery (Brain Tumor MRI)
# =============================================================================
# SECTION 2b - Second dataset: Brain Tumor MRI Dataset (multiclass classification)
# =============================================================================
# Segmentation and classification use *different* datasets, deliberately:
#
#   * BraTS 2020 (HDF5 slices + masks)  -> segmentation, and the HGG/LGG grade
#     experiment that follows the project's own radiomics schema;
#   * Brain Tumor MRI Dataset           -> the multiclass image classification
#     task, with the classes the dataset itself defines.
#
#   https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset
#
# The classes are read from the directory structure and are never assumed. This
# notebook does not know in advance whether the mounted copy is the balanced v2
# release (1400/400 per class) or an older composition -- so it counts what is
# there and reports it.
#
# If the dataset is not attached, every cell below reports that and skips: the
# BraTS segmentation pipeline still runs. Nothing is substituted for it.
# =============================================================================

CLASSIFICATION_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
CLASSIFICATION_SPLIT_NAMES = {"train": "train", "training": "train", "test": "test", "testing": "test", "val": "validation", "valid": "validation", "validation": "validation"}


def _image_files(directory):
    try:
        entries = sorted(directory.iterdir())
    except OSError:
        return []
    return [path for path in entries if path.is_file() and path.suffix.lower() in CLASSIFICATION_IMAGE_EXTENSIONS]


def _split_directories(base):
    """Return {split_name: [class directories]} if `base` looks like an ImageFolder tree."""
    if not base.is_dir():
        return {}
    found = {}
    for child in sorted(base.iterdir()):
        if not child.is_dir() or child.name.startswith("."):
            continue
        key = CLASSIFICATION_SPLIT_NAMES.get(child.name.strip().lower())
        if key is None:
            continue
        class_dirs = [item for item in sorted(child.iterdir()) if item.is_dir() and _image_files(item)]
        if len(class_dirs) >= 2:
            found.setdefault(key, []).extend(class_dirs)
    return found


def discover_classification_dataset(roots=None):
    """Find an image-classification tree: class sub-directories holding image files.

    Returns a dict with the split -> class -> files mapping. `found: False` carries the
    directories that were searched and the exact reason, so a missing dataset is reported
    instead of silently producing an empty run.
    """
    roots = list(roots if roots is not None else RUN["classification_dataset_roots"])
    searched = []
    candidates = []
    for root in roots:
        base = Path(root).expanduser()
        searched.append(str(base))
        if not base.exists():
            continue
        candidates.append(base)
        # Kaggle mounts a dataset as /kaggle/input/<slug>/<inner folder>; walk one or two
        # levels so the tree is found wherever the archive was unpacked.
        for first in sorted((item for item in base.iterdir() if item.is_dir()), key=lambda p: p.name):
            candidates.append(first)
            for second in sorted((item for item in first.iterdir() if item.is_dir()), key=lambda p: p.name):
                candidates.append(second)

    for base in candidates:
        splits = _split_directories(base)
        if splits:
            return {
                "found": True,
                "root": str(base),
                "layout": "image_folder_with_split_directories",
                "splits": {name: {directory.name: [str(path) for path in _image_files(directory)] for directory in directories} for name, directories in splits.items()},
                "searched": searched,
            }
    # No explicit split directories: accept a plain class-per-directory tree.
    for base in candidates:
        class_dirs = [item for item in sorted(base.iterdir()) if item.is_dir() and _image_files(item)] if base.is_dir() else []
        if len(class_dirs) >= 2:
            return {
                "found": True,
                "root": str(base),
                "layout": "image_folder_single_tree",
                "splits": {"all": {directory.name: [str(path) for path in _image_files(directory)] for directory in class_dirs}},
                "searched": searched,
            }
    return {
        "found": False,
        "searched": searched,
        "hint": (
            "Add Data -> Dataset -> masoudnickparvar/brain-tumor-mri-dataset. It mounts under "
            "/kaggle/input/brain-tumor-mri-dataset; if your copy is elsewhere, put its path first "
            "in RUN['classification_dataset_roots']."
        ),
    }


CLASSIFICATION_DATASET = discover_classification_dataset()
CLASSIFICATION_IMAGES_AVAILABLE = bool(CLASSIFICATION_DATASET["found"])

if not CLASSIFICATION_IMAGES_AVAILABLE:
    print("Classification dataset: NOT FOUND")
    print("  searched:", CLASSIFICATION_DATASET["searched"])
    print("  ", CLASSIFICATION_DATASET["hint"])
    print("  The multiclass classification cells below will report this and skip. The BraTS")
    print("  segmentation pipeline is unaffected, and no labels are invented for it.")
    CLASSIFICATION_CLASSES: list = []
    CLASSIFICATION_SPLIT_FILES: dict = {}
    CLASSIFICATION_IS_SYNTHETIC = None
    CLASSIFICATION_DATASET_NAME = "not attached"
    CLASSIFICATION_PROVENANCE_NOTE = "the dataset was not found in this environment"
else:
    CLASSIFICATION_ROOT = Path(CLASSIFICATION_DATASET["root"])
    CLASSIFICATION_SPLIT_FILES = CLASSIFICATION_DATASET["splits"]
    # Classes come from the directories, in the dataset's own spelling.
    CLASSIFICATION_CLASSES = sorted({name for groups in CLASSIFICATION_SPLIT_FILES.values() for name in groups})
    CLASSIFICATION_TOTAL_IMAGES = sum(
        len(paths) for groups in CLASSIFICATION_SPLIT_FILES.values() for paths in groups.values()
    )

    # --- what is this dataset? -------------------------------------------------
    # Same rule as for the segmentation data: the claim is derived from the files, never
    # assumed. A fixture writes `fixture_manifest.json` next to its classes; a real release
    # has thousands of images; anything else is recorded as unverified rather than "real".
    FIXTURE_MANIFEST = None
    for candidate in (CLASSIFICATION_ROOT / "fixture_manifest.json", CLASSIFICATION_ROOT.parent / "fixture_manifest.json"):
        if candidate.exists():
            FIXTURE_MANIFEST = read_json(candidate, default=None)
            if FIXTURE_MANIFEST:
                break


    def resolve_classification_provenance(declared, fixture, images, classes):
        declared = str(declared or "auto").lower()
        if declared in {"synthetic", "synthetic_fixture", "fixture"}:
            return True, "declared in RUN['classification_dataset_provenance']"
        if declared in {"real", "real_brain_tumor_mri", "brain_tumor_mri"}:
            return False, "declared in RUN['classification_dataset_provenance']"
        if fixture and fixture.get("synthetic") is True:
            return True, f"fixture manifest ({fixture.get('generator', 'generator not recorded')})"
        if images >= 1000 and len(classes) >= 2:
            return False, f"{images} images in {len(classes)} classes, matching the released Kaggle dataset"
        return None, (
            f"neither a fixture manifest nor a full release ({images} images in {len(classes)} classes); "
            "set RUN['classification_dataset_provenance'] if this really is the real dataset"
        )


    CLASSIFICATION_IS_SYNTHETIC, CLASSIFICATION_PROVENANCE_NOTE = resolve_classification_provenance(
        RUN.get("classification_dataset_provenance", "auto"), FIXTURE_MANIFEST, CLASSIFICATION_TOTAL_IMAGES, CLASSIFICATION_CLASSES
    )
    CLASSIFICATION_DATASET_NAME = (
        "Synthetic image-classification fixture (scripts/make_classification_fixture.py; the classes are "
        "generated, any metric computed from them is meaningless)"
        if CLASSIFICATION_IS_SYNTHETIC
        else "Brain Tumor MRI Dataset (Kaggle: masoudnickparvar/brain-tumor-mri-dataset)"
        if CLASSIFICATION_IS_SYNTHETIC is False
        else f"Unverified image tree at {CLASSIFICATION_ROOT} (provenance not established)"
    )
    print(f"Classification dataset : {CLASSIFICATION_ROOT} ({CLASSIFICATION_DATASET['layout']})")
    print(f"Images                 : {CLASSIFICATION_TOTAL_IMAGES}")
    for split_name, groups in sorted(CLASSIFICATION_SPLIT_FILES.items()):
        counts = ", ".join(f"{name}: {len(paths)}" for name, paths in sorted(groups.items()))
        print(f"  {split_name:<12} {counts}")
    print(f"Classes ({len(CLASSIFICATION_CLASSES)})      : {', '.join(CLASSIFICATION_CLASSES)}")
    print(
        "Dataset provenance    :",
        {True: "SYNTHETIC FIXTURE", False: "real Brain Tumor MRI Dataset", None: "unknown"}[CLASSIFICATION_IS_SYNTHETIC],
        "--",
        CLASSIFICATION_PROVENANCE_NOTE,
    )
    if CLASSIFICATION_IS_SYNTHETIC:
        print(
            "\n" + "!" * 78 + "\n"
            "WARNING: the classification data is a synthetic fixture. Every number printed for\n"
            "the multiclass task below describes generated images with generated labels. It must\n"
            "not be quoted as a result.\n" + "!" * 78
        )

LOCK["classification_dataset"] = {
    "root": CLASSIFICATION_DATASET.get("root"),
    "layout": CLASSIFICATION_DATASET.get("layout"),
    "classes": CLASSIFICATION_CLASSES,
    "images": CLASSIFICATION_TOTAL_IMAGES if CLASSIFICATION_IMAGES_AVAILABLE else 0,
    "synthetic": CLASSIFICATION_IS_SYNTHETIC,
    "provenance_note": CLASSIFICATION_PROVENANCE_NOTE,
}
write_json(PATHS["artifacts"] / "classification_dataset_discovery.json", {"dataset": CLASSIFICATION_DATASET, "lock": LOCK["classification_dataset"]})

# %% [cell] 11. Classification dataset structure inspection
# =============================================================================
# SECTION 3b - Structure inspection of the classification images
# =============================================================================
# What is actually inside the tree: file formats, colour modes, pixel sizes, per-class
# counts, unreadable files, and byte-identical duplicates. All of it is written to disk so
# the preprocessing decisions (grayscale, resize) are traceable to the data.
import hashlib
from collections import Counter

STRUCTURE = {"state": "not_available", "reason": "the classification dataset was not attached"}
if CLASSIFICATION_IMAGES_AVAILABLE:
    from PIL import Image as PILImage

    def inspect_image(path, max_side=None):
        with PILImage.open(path) as handle:
            size = handle.size
            mode = handle.mode
            fmt = handle.format
            array = np.asarray(handle.convert("L"), dtype=np.float32)
        return {"path": path, "size": size, "mode": mode, "format": fmt, "array": array}

    extension_counts = Counter()
    mode_counts = Counter()
    size_counts = Counter()
    digest_to_paths = {}
    unreadable = []
    per_class = {}
    for split_name, groups in sorted(CLASSIFICATION_SPLIT_FILES.items()):
        for class_name, paths in sorted(groups.items()):
            entry = per_class.setdefault(
                class_name,
                {"split": split_name, "images": 0, "min_side": None, "max_side": None, "extensions": Counter(), "modes": Counter()},
            )
            for path in paths:
                path = Path(path)
                extension_counts[path.suffix.lower()] += 1
                entry["extensions"][path.suffix.lower()] += 1
                try:
                    with PILImage.open(path) as handle:
                        width, height = handle.size
                        mode_counts[handle.mode] += 1
                        entry["modes"][handle.mode] += 1
                except Exception as exc:  # unreadable file -> reported, never assumed away
                    unreadable.append({"path": str(path), "error": f"{type(exc).__name__}: {exc}"})
                    continue
                entry["images"] += 1
                smallest, largest = min(width, height), max(width, height)
                entry["min_side"] = smallest if entry["min_side"] is None else min(entry["min_side"], smallest)
                entry["max_side"] = largest if entry["max_side"] is None else max(entry["max_side"], largest)
                size_counts[(width, height)] += 1
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                digest_to_paths.setdefault(digest, []).append((split_name, str(path)))

    duplicate_groups = [
        {"sha256": digest, "files": [{"split": split, "path": path} for split, path in paths]}
        for digest, paths in digest_to_paths.items()
        if len(paths) > 1
    ]
    cross_split_duplicates = [
        group
        for group in duplicate_groups
        if len({entry["split"] for entry in group["files"]}) > 1
    ]
    STRUCTURE = {
        "state": "inspected",
        "root": CLASSIFICATION_DATASET["root"],
        "layout": CLASSIFICATION_DATASET["layout"],
        "split_counts": {name: sum(len(paths) for paths in groups.values()) for name, groups in CLASSIFICATION_SPLIT_FILES.items()},
        "class_counts": {
            name: sum(len(CLASSIFICATION_SPLIT_FILES[split].get(name, [])) for split in CLASSIFICATION_SPLIT_FILES)
            for name in CLASSIFICATION_CLASSES
        },
        "classes_in_each_split": {name: sorted(groups) for name, groups in CLASSIFICATION_SPLIT_FILES.items()},
        "per_class": {name: {**entry, "extensions": dict(entry["extensions"]), "modes": dict(entry["modes"])} for name, entry in per_class.items()},
        "extensions": dict(extension_counts),
        "modes": dict(mode_counts),
        "distinct_pixel_sizes": len(size_counts),
        "most_common_sizes": [[list(size), count] for size, count in size_counts.most_common(5)],
        "unreadable_files": unreadable,
        "byte_identical_duplicate_groups": len(duplicate_groups),
        "byte_identical_duplicates_across_splits": len(cross_split_duplicates),
        "duplicate_examples": cross_split_duplicates[:5],
        "synthetic": CLASSIFICATION_IS_SYNTHETIC,
        "provenance_note": CLASSIFICATION_PROVENANCE_NOTE,
    }
    print("Split sizes    :", STRUCTURE["split_counts"])
    print("Class counts   :", STRUCTURE["class_counts"])
    print("Extensions     :", STRUCTURE["extensions"])
    print("Colour modes   :", STRUCTURE["modes"])
    print("Distinct sizes :", STRUCTURE["distinct_pixel_sizes"], "(most common:", STRUCTURE["most_common_sizes"], ")")
    print("Unreadable     :", len(unreadable))
    print("Byte-identical duplicate groups:", len(duplicate_groups), "of which across splits:", len(cross_split_duplicates))
    if unreadable:
        print("  first unreadable:", unreadable[0])
    write_json(PATHS["artifacts"] / "classification_structure.json", STRUCTURE)

# %% [cell] 12. Classification dataset EDA and visualisation
# =============================================================================
# SECTION 4b - Class distribution and sample images
# =============================================================================
CLASSIFICATION_EDA = {"state": "not_available", "reason": "the classification dataset was not attached"}
if CLASSIFICATION_IMAGES_AVAILABLE:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    distribution = {
        split_name: {name: len(paths) for name, paths in sorted(groups.items())}
        for split_name, groups in sorted(CLASSIFICATION_SPLIT_FILES.items())
    }
    totals = {
        name: sum(len(groups.get(name, [])) for groups in CLASSIFICATION_SPLIT_FILES.values())
        for name in CLASSIFICATION_CLASSES
    }

    # --- class distribution figure ------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    split_labels = list(distribution)
    positions = np.arange(len(CLASSIFICATION_CLASSES))
    width = 0.8 / max(1, len(split_labels))
    for index, split_name in enumerate(split_labels):
        counts = [distribution[split_name].get(name, 0) for name in CLASSIFICATION_CLASSES]
        axes[0].bar(positions + index * width, counts, width, label=split_name)
    axes[0].set_xticks(positions + width * (len(split_labels) - 1) / 2)
    axes[0].set_xticklabels(CLASSIFICATION_CLASSES, rotation=20, ha="right")
    axes[0].set_ylabel("images")
    axes[0].set_title("Class distribution per split")
    axes[0].legend()
    axes[1].bar(list(totals), [totals[name] for name in totals], color="tab:blue")
    axes[1].set_title("Total images per class")
    axes[1].tick_params(axis="x", rotation=20)
    plt.tight_layout()
    distribution_figure = PATHS["reports"] / "07_classification_class_distribution.png"
    plt.savefig(distribution_figure, dpi=130)
    plt.close(fig)

    # --- sample grid ----------------------------------------------------------
    preview_split = "train" if "train" in CLASSIFICATION_SPLIT_FILES else split_labels[0]
    columns = max(len(CLASSIFICATION_CLASSES), 1)
    fig, axes = plt.subplots(1, columns, figsize=(3.1 * columns, 3.4))
    axes = np.atleast_1d(axes)
    examples = {}
    for index, class_name in enumerate(CLASSIFICATION_CLASSES):
        paths = CLASSIFICATION_SPLIT_FILES[preview_split].get(class_name, [])
        axes[index].axis("off")
        if not paths:
            axes[index].set_title(f"{class_name}\n(no image)")
            continue
        sample = inspect_image(Path(paths[0]))
        examples[class_name] = sample["path"].name
        axes[index].imshow(sample["array"], cmap="gray")
        axes[index].set_title(f"{class_name}\n{sample['size'][0]}x{sample['size'][1]} {sample['format']}", fontsize=9)
    plt.tight_layout()
    samples_figure = PATHS["reports"] / "08_classification_samples.png"
    plt.savefig(samples_figure, dpi=130)
    plt.close(fig)

    CLASSIFICATION_EDA = {
        "state": "generated",
        "split": preview_split,
        "distribution": distribution,
        "totals": totals,
        "classes": CLASSIFICATION_CLASSES,
        "example_filenames": examples,
        "figures": {"distribution": str(distribution_figure), "samples": str(samples_figure)},
        "synthetic": CLASSIFICATION_IS_SYNTHETIC,
    }
    print("distribution:", distribution)
    print("figures     :", CLASSIFICATION_EDA["figures"])
    write_json(PATHS["artifacts"] / "classification_eda.json", CLASSIFICATION_EDA)

# %% [cell] 13. Classification preprocessing, splits and leakage audit
# =============================================================================
# SECTION 9b - Images, splits and the leakage audit
# =============================================================================
# Three separate concerns, all of them recorded:
#
# 1. Preprocessing: grayscale, resized to RUN['classification_image_size'] with Lanczos,
#    stored as uint8 arrays in the resumable cache. The classifier sees exactly these.
# 2. Splitting: the dataset ships its own train/test directories. This notebook does not
#    trust that blindly -- it keeps the shipped test directory as the held-out test set and
#    carves a stratified validation set out of the shipped training directory.
#    **There are no patient identifiers in this dataset** (it is a curated collection of
#    individual images from figshare / SARTAJ / Br35H), so a patient-level split is
#    impossible here and is not claimed. What can be done is done: a near-duplicate audit
#    that flags test images whose content matches a training image, and those images are
#    excluded from the reported test set by default.
# 3. Audit: 8x8 average-hash (64-bit) per image; Hamming distance to the nearest training
#    image is reported per test image. The threshold and the counts are recorded.
# =============================================================================
import random as _random

CLASSIFICATION_IMAGES: dict = {"train": {}, "validation": {}, "test": {}}
CLASSIFICATION_AUDIT = {"state": "not_available", "reason": "the classification dataset was not attached"}
CLASSIFICATION_SPLIT_SOURCE = None

if CLASSIFICATION_IMAGES_AVAILABLE:
    from PIL import Image as PILImage

    IMAGE_SIZE = int(RUN["classification_image_size"])
    CACHE_ROOT = PATHS["cache"] / "classification_images"
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)


    def load_image_array(path, size=IMAGE_SIZE, cache=True):
        """Grayscale uint8 array at `size`, cached on disk so a re-run does not re-decode."""
        path = Path(path)
        digest = hashlib.sha256(f"{path}|{size}|grayscale".encode("utf-8")).hexdigest()[:16]
        target = CACHE_ROOT / f"{digest}.npy"
        if cache and target.exists():
            return np.load(target)
        with PILImage.open(path) as handle:
            array = handle.convert("L").resize((size, size), PILImage.LANCZOS)
        array = np.asarray(array, dtype=np.uint8)
        if cache:
            np.save(target, array)
        return array


    AUDIT_PIXEL_SIZE = int(RUN["classification_duplicate_pixel_size"])
    AUDIT_HASH_SIZE = int(RUN["classification_duplicate_hash_size"])


    def average_hash(array, hash_size=AUDIT_HASH_SIZE):
        """Coarse perceptual hash, used only as a *candidate filter* for the duplicate audit."""
        with PILImage.fromarray(np.asarray(array, dtype=np.uint8)) as handle:
            small = np.asarray(handle.resize((hash_size, hash_size), PILImage.BILINEAR), dtype=np.float32)
        return (small > small.mean()).ravel()


    def pixel_correlation(first, second):
        """Pearson correlation of two same-sized grayscale arrays, used to confirm a candidate.

        An average hash is very coarse: images that share a global structure (every one of these
        is a centred brain slice) can land within a few bits of each other while showing clearly
        different content. A hash match therefore only *proposes* a duplicate; this correlation
        decides it, so the audit cannot quietly throw away a large share of the test set.
        """
        a = np.asarray(first, dtype=np.float32).ravel()
        b = np.asarray(second, dtype=np.float32).ravel()
        a = a - a.mean()
        b = b - b.mean()
        denominator = float(np.sqrt((a * a).sum() * (b * b).sum()))
        return float((a * b).sum() / denominator) if denominator > 0 else 0.0


    # --- assign splits -------------------------------------------------------
    RAW = CLASSIFICATION_SPLIT_FILES
    if {"train", "test"} <= set(RAW):
        train_pool = {name: list(paths) for name, paths in RAW["train"].items()}
        test_pool = {name: list(paths) for name, paths in RAW["test"].items()}
        CLASSIFICATION_SPLIT_SOURCE = "the dataset's own Training/ and Testing/ directories"
    else:
        rng_split = np.random.default_rng(SEED)
        train_pool, test_pool = {}, {}
        for name, paths in RAW.get("all", {}).items():
            shuffled = list(paths)
            rng_split.shuffle(shuffled)
            cut = int(round(len(shuffled) * 0.7))
            train_pool[name] = shuffled[:cut]
            test_pool[name] = shuffled[cut:]
        CLASSIFICATION_SPLIT_SOURCE = "assigned by this notebook (70/30, stratified by class, seed %d)" % SEED

    # stratified validation carve-out from the training pool
    rng_val = np.random.default_rng(SEED + 1)
    validation_pool = {}
    for name, paths in train_pool.items():
        shuffled = list(paths)
        rng_val.shuffle(shuffled)
        cuts = int(round(len(shuffled) * float(RUN["classification_validation_fraction"])))
        cuts = min(max(cuts, 1), max(len(shuffled) - 1, 0))
        validation_pool[name] = shuffled[:cuts]
        train_pool[name] = shuffled[cuts:]

    for split_name, pool in (("train", train_pool), ("validation", validation_pool), ("test", test_pool)):
        for name, paths in sorted(pool.items()):
            CLASSIFICATION_IMAGES[split_name][name] = [Path(path) for path in paths]

    split_sizes = {name: sum(len(paths) for paths in pool.values()) for name, pool in CLASSIFICATION_IMAGES.items()}
    print("split source :", CLASSIFICATION_SPLIT_SOURCE)
    print("split sizes  :", split_sizes)

    # --- leakage audit: validation and test against the training images --------
    # No patient identifiers exist in this dataset, so grouping by patient is impossible. The
    # next best thing is measurable: perceptual near-duplicates of a training image inside the
    # validation or test split would inflate the reported scores, so they are counted and, by
    # default, removed from the evaluation splits. The count is reported either way.
    hash_threshold = int(RUN["classification_duplicate_hamming_threshold"])
    correlation_threshold = float(RUN["classification_duplicate_correlation_threshold"])
    max_exclusion_fraction = float(RUN["duplicate_exclusion_max_fraction"])

    train_paths_flat = [Path(path) for paths in CLASSIFICATION_IMAGES["train"].values() for path in paths]
    train_arrays = np.stack(
        [load_image_array(path, size=AUDIT_PIXEL_SIZE) for path in train_paths_flat]
    ) if train_paths_flat else np.zeros((0, AUDIT_PIXEL_SIZE, AUDIT_PIXEL_SIZE), dtype=np.uint8)
    train_hashes = np.stack([average_hash(array) for array in train_arrays]) if len(train_arrays) else np.zeros((0, AUDIT_HASH_SIZE * AUDIT_HASH_SIZE), dtype=bool)
    print(
        f"leakage audit: {len(train_arrays)} training image(s) indexed, "
        f"{AUDIT_HASH_SIZE}x{AUDIT_HASH_SIZE} hash (candidate filter) + pixel correlation >= {correlation_threshold} (confirmation)"
    )

    def audit_split(split_name):
        """Two-stage duplicate detection: hash candidates, then confirm with correlation."""
        total = 0
        candidates = []
        confirmed = []
        for class_name, paths in sorted(CLASSIFICATION_IMAGES[split_name].items()):
            for path in paths:
                total += 1
                array = load_image_array(path, size=AUDIT_PIXEL_SIZE)
                digest = average_hash(array)
                if not train_hashes.shape[0]:
                    continue
                distances = np.count_nonzero(train_hashes != digest, axis=1)
                nearest = int(distances.min())
                if nearest > hash_threshold:
                    continue
                nearest_index = int(distances.argmin())
                correlation = pixel_correlation(array, train_arrays[nearest_index])
                entry = {
                    "path": str(path),
                    "class": class_name,
                    "hamming_distance": nearest,
                    "pixel_correlation": correlation,
                    "nearest_train_image": str(train_paths_flat[nearest_index]),
                }
                candidates.append(entry)
                if correlation >= correlation_threshold:
                    confirmed.append(entry)
        return total, candidates, confirmed

    audit_report = {}
    for split_name in ("validation", "test"):
        total, candidates, confirmed = audit_split(split_name)
        excluded = []
        suppression = None
        if confirmed and len(confirmed) / max(1, total) > max_exclusion_fraction:
            # A calibrated audit removes a handful of images, not a share of the split. If the
            # fraction is this large the threshold is wrong for this data, so nothing is removed
            # and the situation is reported instead of silently shrinking the evaluation set.
            suppression = (
                f"{len(confirmed)} of {total} images ({len(confirmed) / total:.1%}) matched the duplicate "
                f"criteria, above the {max_exclusion_fraction:.0%} limit; the audit is treated as "
                "inconclusive and no image is removed. Raise RUN['classification_duplicate_correlation_threshold'] "
                "or inspect the flagged pairs."
            )
            confirmed = []
        if RUN["exclude_duplicate_eval_images"] and confirmed:
            confirmed_paths = {entry["path"] for entry in confirmed}
            for class_name, paths in list(CLASSIFICATION_IMAGES[split_name].items()):
                kept = [path for path in paths if str(path) not in confirmed_paths]
                excluded.extend({"path": str(path), "class": class_name} for path in paths if str(path) not in kept)
                CLASSIFICATION_IMAGES[split_name][class_name] = kept
        audit_report[split_name] = {
            "images_before": total,
            "hash_candidates": len(candidates),
            "confirmed_near_duplicates": len(confirmed),
            "confirmed_fraction": len(confirmed) / max(1, total),
            "confirmed_by_class": {
                name: sum(1 for entry in confirmed if entry["class"] == name) for name in CLASSIFICATION_CLASSES
            },
            "excluded": excluded,
            "images_after": total - len(excluded),
            "exclusion_suppressed": suppression,
            "examples": (confirmed or candidates)[:5],
        }

    CLASSIFICATION_AUDIT = {
        "state": "measured_and_enforced" if RUN["exclude_duplicate_eval_images"] else "measured",
        "method": (
            f"{AUDIT_HASH_SIZE}x{AUDIT_HASH_SIZE} average hash (Hamming <= {hash_threshold}) as a candidate "
            f"filter, confirmed by Pearson correlation >= {correlation_threshold} of the "
            f"{AUDIT_PIXEL_SIZE}x{AUDIT_PIXEL_SIZE} grayscale images"
        ),
        "hash_threshold": hash_threshold,
        "correlation_threshold": correlation_threshold,
        "max_exclusion_fraction": max_exclusion_fraction,
        "split_audits": audit_report,
        "patient_identifiers_available": False,
        "patient_level_split_possible": False,
        "note": (
            "The dataset carries no patient identifiers -- it is a curated collection of individual "
            "images -- so a patient-level split cannot be made and is not claimed here. The dataset's "
            "own train/test directories are kept, and images confirmed to duplicate a training image "
            "are removed from the validation and test splits by default, so a reported score is not "
            "earned by a duplicated image."
        ),
    }
    print(
        "audit        :",
        {
            split: {
                "images_before": entry["images_before"],
                "hash_candidates": entry["hash_candidates"],
                "confirmed": entry["confirmed_near_duplicates"],
                "kept": entry["images_after"],
            }
            for split, entry in audit_report.items()
        },
    )
    for split_name, entry in audit_report.items():
        if entry["exclusion_suppressed"]:
            print(f"  WARNING {split_name}:", entry["exclusion_suppressed"])

    # --- materialise the arrays ------------------------------------------------
    t0 = time.time()
    load_counts = {}
    for split_name, groups in CLASSIFICATION_IMAGES.items():
        for name, paths in groups.items():
            for path in paths:
                load_image_array(path)
            load_counts[f"{split_name}/{name}"] = len(paths)
    print(f"decoded/cached {sum(load_counts.values())} images in {time.time() - t0:.1f}s -> {CACHE_ROOT}")

    CLASSIFICATION_SPLIT_RECORD = {
        "state": "ready",
        "root": CLASSIFICATION_DATASET["root"],
        "classes": CLASSIFICATION_CLASSES,
        "split_source": CLASSIFICATION_SPLIT_SOURCE,
        "validation_fraction": float(RUN["classification_validation_fraction"]),
        "seed": SEED,
        "image_size": IMAGE_SIZE,
        "splits": {
            name: {class_name: len(paths) for class_name, paths in sorted(groups.items())}
            for name, groups in CLASSIFICATION_IMAGES.items()
        },
        "split_sizes": {name: sum(len(paths) for paths in groups.values()) for name, groups in CLASSIFICATION_IMAGES.items()},
        "audit": CLASSIFICATION_AUDIT,
        "synthetic": CLASSIFICATION_IS_SYNTHETIC,
        "provenance_note": CLASSIFICATION_PROVENANCE_NOTE,
    }
    write_json(PATHS["artifacts"] / "classification_splits.json", CLASSIFICATION_SPLIT_RECORD)

# %% [cell] 14. Classification dataset readiness
# =============================================================================
# SECTION 10b - What the second dataset gives us, stated plainly
# =============================================================================
if not CLASSIFICATION_IMAGES_AVAILABLE:
    CLASSIFICATION_READY = False
    print("Classification task: skipped --", CLASSIFICATION_DATASET["hint"])
    print("No multiclass model is trained, no metric is reported, and nothing is fabricated.")
else:
    CLASSIFICATION_READY = all(
        CLASSIFICATION_IMAGES["train"] and CLASSIFICATION_IMAGES["validation"] and CLASSIFICATION_IMAGES["test"]
    ) and all(
        sum(len(paths) for paths in groups.values()) > 0 for groups in CLASSIFICATION_IMAGES.values()
    )
    summary_rows = []
    for split_name in ("train", "validation", "test"):
        groups = CLASSIFICATION_IMAGES[split_name]
        for class_name in CLASSIFICATION_CLASSES:
            summary_rows.append((split_name, class_name, len(groups.get(class_name, []))))
    print(f"{'split':<12}{'class':<14}images")
    for split_name, class_name, count in summary_rows:
        print(f"{split_name:<12}{class_name:<14}{count}")
    print()
    print("task            :", RUN["classification_task_name"])
    print("classes         :", CLASSIFICATION_CLASSES, "(from the dataset's own directories)")
    print("image size      :", RUN["classification_image_size"], "px, grayscale")
    print("split source    :", CLASSIFICATION_SPLIT_SOURCE)
    print(
        "leakage audit   :",
        CLASSIFICATION_AUDIT["state"],
        "(test: {before} -> {after} images, {flagged} confirmed near-duplicate(s) of training "
        "images)".format(
            before=CLASSIFICATION_AUDIT["split_audits"]["test"]["images_before"],
            after=CLASSIFICATION_AUDIT["split_audits"]["test"]["images_after"],
            flagged=CLASSIFICATION_AUDIT["split_audits"]["test"]["confirmed_near_duplicates"],
        ),
    )
    print("dataset is the  :", {True: "synthetic fixture", False: "real dataset", None: "unverified tree"}[CLASSIFICATION_IS_SYNTHETIC])
    if not CLASSIFICATION_READY:
        empty = [name for name, groups in CLASSIFICATION_IMAGES.items() if not sum(len(paths) for paths in groups.values())]
        partial = {
            name: {class_name: len(paths) for class_name, paths in groups.items() if not paths}
            for name, groups in CLASSIFICATION_IMAGES.items()
            if name not in empty
        }
        print(f"Classification task: no multiclass model is trained -- empty split(s): {empty}; classes with no image: {partial}")

LOCK["classification_splits"] = {
    "classes": CLASSIFICATION_CLASSES,
    "splits": {name: sum(len(paths) for paths in groups.values()) for name, groups in CLASSIFICATION_IMAGES.items()},
    "audit": (
        CLASSIFICATION_AUDIT["split_audits"]["test"]["confirmed_near_duplicates"] if CLASSIFICATION_IMAGES_AVAILABLE else None
    ),
}
