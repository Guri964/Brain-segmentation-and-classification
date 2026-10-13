from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

IMAGE_KEY_CANDIDATES = ("image", "images", "img", "mri", "data", "volume", "t1")
MASK_KEY_CANDIDATES = ("mask", "masks", "seg", "segmentation", "label", "labels", "gt", "y")
BRATS_MODALITY_SUFFIXES = ("t1ce", "t1", "t2", "flair", "seg")
BRATS_LABEL_NAMES = {
    0: "background",
    1: "necrotic_and_non_enhancing_tumour_core",
    2: "peritumoural_edema",
    4: "enhancing_tumour",
}
WHOLE_TUMOUR_LABELS = (1, 2, 4)
SLICE_PATTERN = re.compile(r"(?P<volume>.+?)[._-]slice[._-](?P<index>\d+)", re.IGNORECASE)
VOLUME_PATTERNS = (
    re.compile(r"volume[._-]?(?P<index>\d+)", re.IGNORECASE),
    re.compile(r"BraTS[-_]?(?P<year>\d{2,4})?[-_]?(?:Training|Validation)?[-_]?(?P<index>\d{2,5})", re.IGNORECASE),
)


@dataclass
class BratsSlice:
    image: np.ndarray
    mask_channels: np.ndarray
    mask_semantics: dict[str, Any]
    image_key: str
    mask_key: str
    source: str
    attributes: dict[str, Any] = field(default_factory=dict)

    @property
    def channels(self) -> int:
        return int(self.image.shape[0])

    def whole_tumour(self) -> np.ndarray:
        if self.mask_channels.ndim == 2:
            return (self.mask_channels > 0).astype(np.uint8)
        return (self.mask_channels > 0).any(axis=0).astype(np.uint8)

    def label_map(self) -> np.ndarray | None:
        if self.mask_semantics.get("kind") in {"brats_label_map", "binary", "multilabel"}:
            return self.mask_channels.astype(np.int32)
        return None

    def summary(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "image_key": self.image_key,
            "mask_key": self.mask_key,
            "image_shape": list(self.image.shape),
            "channels": self.channels,
            "image_dtype": str(self.image.dtype),
            "image_range": [float(self.image.min()), float(self.image.max())],
            "mask_shape": list(self.mask_channels.shape),
            "mask_semantics": self.mask_semantics,
            "foreground_fraction": float(self.whole_tumour().mean()),
            "attributes": self.attributes,
        }


def hdf5_entries(path: Path) -> list[dict[str, Any]]:
    import h5py

    entries: list[dict[str, Any]] = []
    with h5py.File(path, "r") as handle:
        def visit(name: str, node: Any) -> None:
            if isinstance(node, h5py.Dataset):
                entry: dict[str, Any] = {
                    "name": name,
                    "shape": list(node.shape),
                    "dtype": str(node.dtype),
                }
                if node.size and node.size <= 8_000_000 and node.dtype.kind in "iuifb":
                    block = np.asarray(node[...])
                    entry["min"] = float(np.min(block))
                    entry["max"] = float(np.max(block))
                    if node.dtype.kind in "iub" and float(np.min(block)) >= 0 and float(np.max(block)) <= 8:
                        entry["unique_values"] = [int(item) for item in np.unique(block)]
                    entry["mean"] = float(np.mean(block))
                entries.append(entry)
            if isinstance(node, h5py.Group):
                attributes = {key: _jsonify(value) for key, value in node.attrs.items()}
                if attributes:
                    entries.append({"name": name, "group_attributes": attributes})

        handle.visititems(visit)
        root_attributes = {key: _jsonify(value) for key, value in handle.attrs.items()}
    if root_attributes:
        entries.append({"name": "/", "attributes": root_attributes})
    return entries


def _jsonify(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def detect_keys(names: Iterable[str]) -> tuple[str, str | None]:
    keys = [name for name in names]
    lowered = {name.lower().split("/")[-1]: name for name in keys}
    image_key = None
    for candidate in IMAGE_KEY_CANDIDATES:
        for lower, original in lowered.items():
            if lower == candidate:
                image_key = original
                break
        if image_key:
            break
    if image_key is None:
        for name in keys:
            if name.lower().split("/")[-1].startswith("image") or name.lower().split("/")[-1].startswith("img"):
                image_key = name
                break
    mask_key = None
    for candidate in MASK_KEY_CANDIDATES:
        for lower, original in lowered.items():
            if lower == candidate:
                mask_key = original
                break
        if mask_key:
            break
    if mask_key is None:
        for name in keys:
            base = name.lower().split("/")[-1]
            if base.startswith("mask") or base.startswith("seg") or base.startswith("label"):
                mask_key = name
                break
    if image_key is None:
        raise ValueError(
            "No image dataset was found in this file. Datasets present: "
            + ", ".join(keys[:10])
        )
    return image_key, mask_key


def _channels_first(block: np.ndarray) -> tuple[np.ndarray, bool]:
    if block.ndim == 2:
        return block[None, ...], False
    if block.ndim != 3:
        raise ValueError(f"unsupported array shape {block.shape}")
    first, last = block.shape[0], block.shape[-1]
    channel_candidates = (1, 3, 4)
    if last in channel_candidates and first not in channel_candidates:
        return np.moveaxis(block, -1, 0), True
    if first in channel_candidates and last not in channel_candidates:
        return block, False
    if first == last:
        return block, False
    if last in channel_candidates:
        return np.moveaxis(block, -1, 0), True
    return block, False


def classify_mask(mask: np.ndarray, declared_channel_labels: Sequence[int] | None = None) -> dict[str, Any]:
    block = np.asarray(mask)
    semantics: dict[str, Any] = {}
    if block.ndim == 3:
        block, channels_last = _channels_first(block)
        semantics["channel_axis"] = "last" if channels_last else "first"
        semantics["channels"] = int(block.shape[0])
        is_binary = bool(np.isin(np.unique(block), (0, 1)).all())
        if block.shape[0] == 1:
            block = block[0]
            semantics["kind"] = "binary" if is_binary else "multilabel"
        elif is_binary:
            semantics["kind"] = "one_hot_channels"
            if declared_channel_labels:
                semantics["channel_labels"] = [int(item) for item in declared_channel_labels]
            else:
                semantics["channel_labels_declared"] = False
                semantics["note"] = (
                    "The mask stores one binary plane per tumour sub-region. The sub-region each "
                    "plane refers to is not declared by the file, so the planes are kept separate "
                    "and only their union is used. Pass --mask-channel-labels to declare the order."
                )
        else:
            semantics["kind"] = "multi_channel_labels"
    else:
        block = np.asarray(block)
        unique = np.unique(block)
        if np.isin(unique, (0, 1)).all():
            semantics["kind"] = "binary"
        elif set(int(item) for item in unique if item) <= {1, 2, 3}:
            semantics["kind"] = "brats_label_map"
            semantics["note"] = (
                "Non-zero values are BraTS tumour sub-region labels; the union of all non-zero "
                "values is the whole tumour."
            )
        elif set(int(item) for item in unique if item) <= {1, 2, 4}:
            semantics["kind"] = "brats_label_map"
            semantics["labels_present"] = sorted(int(item) for item in unique if item)
        else:
            semantics["kind"] = "multilabel"
    semantics["observed_values"] = [int(item) for item in np.unique(block)[:12]]
    semantics["dimension"] = "2d" if np.asarray(block).ndim == 2 else "3d"
    return {"array": np.asarray(block), "semantics": semantics}


def read_slice_file(
    path: Path,
    *,
    image_key: str | None = None,
    mask_key: str | None = None,
    mask_channel_labels: Sequence[int] | None = None,
) -> BratsSlice:
    import h5py

    with h5py.File(path, "r") as handle:
        names = _dataset_names(handle)
        if image_key is None or mask_key is None:
            detected_image, detected_mask = detect_keys(names)
            image_key = image_key or detected_image
            mask_key = mask_key or detected_mask
        if image_key not in names:
            raise ValueError(f"{path}: dataset '{image_key}' not present ({names[:8]})")
        attributes = {
            "root": {key: _jsonify(value) for key, value in handle.attrs.items()},
            "image": {key: _jsonify(value) for key, value in handle[image_key].attrs.items()},
        }
        image_block = np.asarray(handle[image_key][...], dtype=np.float32)
        mask_raw = None
        if mask_key and mask_key in names:
            mask_raw = np.asarray(handle[mask_key][...])
            attributes["mask"] = {
                key: _jsonify(value) for key, value in handle[mask_key].attrs.items()
            }
        name_attributes = parse_name_attributes(path)
    image, channels_last = _channels_first(image_block)
    image_metadata = {
        "stored_shape": list(image_block.shape),
        "channel_axis": "last" if channels_last else "first",
        "channels": int(image.shape[0]),
    }
    if mask_raw is None:
        mask_channels = np.zeros(image.shape[1:], dtype=np.int32)
        mask_semantics: dict[str, Any] = {"kind": "absent"}
    else:
        classified = classify_mask(mask_raw, mask_channel_labels)
        mask_channels = classified["array"]
        mask_semantics = classified["semantics"]
    attributes.update({key: value for key, value in name_attributes.items()})
    return BratsSlice(
        image=image,
        mask_channels=mask_channels,
        mask_semantics={"image": image_metadata, **mask_semantics},
        image_key=image_key,
        mask_key=mask_key or "",
        source=str(path),
        attributes=attributes,
    )


def _dataset_names(handle: Any) -> list[str]:
    names: list[str] = []

    def visit(name: str, node: Any) -> None:
        import h5py

        if isinstance(node, h5py.Dataset):
            names.append(name)

    handle.visititems(visit)
    return names


def parse_name_attributes(path: Path) -> dict[str, Any]:
    stem = path.name
    for suffix in (".h5", ".hdf5", ".npy", ".npz"):
        if stem.lower().endswith(suffix):
            stem = stem[: -len(suffix)]
    attributes: dict[str, Any] = {"file_stem": stem}
    match = SLICE_PATTERN.search(stem)
    if match:
        attributes["volume_key"] = match.group("volume")
        attributes["slice_index"] = int(match.group("index"))
    else:
        attributes["volume_key"] = stem
    volume_match = None
    for pattern in VOLUME_PATTERNS:
        volume_match = pattern.search(stem)
        if volume_match:
            attributes["volume_number"] = int(volume_match.group("index"))
            break
    return attributes


def discover_slice_files(root: Path, extensions: Sequence[str] = (".h5", ".hdf5")) -> list[Path]:
    files: list[Path] = []
    for extension in extensions:
        files.extend(root.rglob(f"*{extension}"))
    return sorted({path for path in files})


def group_by_volume(paths: Sequence[Path], *, level: str = "volume") -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = defaultdict(list)
    for path in paths:
        attributes = parse_name_attributes(path)
        if level == "patient":
            key = str(attributes.get("volume_number") or attributes.get("volume_key") or path.stem)
        else:
            key = str(attributes.get("volume_key") or path.stem)
        groups[key].append(path)
    for key in groups:
        groups[key].sort(key=lambda item: parse_name_attributes(item).get("slice_index", 0))
    return dict(groups)


def read_nifti(path: Path) -> tuple[np.ndarray, tuple[float, ...] | None]:
    import SimpleITK as sitk

    image = sitk.ReadImage(str(path))
    array = np.asarray(sitk.GetArrayFromImage(image), dtype=np.float32)
    spacing = tuple(float(item) for item in image.GetSpacing())
    return array, spacing


def read_volume_directory(directory: Path) -> dict[str, Any]:
    modalities: dict[str, np.ndarray] = {}
    segmentations: dict[str, np.ndarray] = {}
    spacing = None
    for path in sorted(directory.iterdir()):
        name = path.name
        if name.endswith(".nii.gz"):
            base = name[: -len(".nii.gz")]
        elif path.suffix in {".nii", ".mha", ".mhd", ".nrrd"}:
            base = path.stem
        else:
            continue
        token = base.lower().split("_")[-1]
        array, current_spacing = read_nifti(path)
        spacing = current_spacing or spacing
        if token == "seg":
            segmentations["seg"] = array
        elif token in BRATS_MODALITY_SUFFIXES:
            modalities[token] = array
    return {
        "modalities": modalities,
        "segmentation": segmentations.get("seg"),
        "spacing": spacing,
        "source": str(directory),
    }


def find_volume_directories(root: Path) -> list[Path]:
    directories: list[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_dir():
            continue
        children = [item.name.lower() for item in path.iterdir() if item.is_file()]
        if any(name.endswith(("_t1.nii", "_t1.nii.gz", "_t1ce.nii", "_t1ce.nii.gz")) for name in children):
            directories.append(path)
    return directories


def read_grade_table(path: Path, column: str | None = None) -> dict[str, dict[str, Any]]:
    rows = _read_csv_rows(path)
    if not rows:
        raise ValueError(f"{path} contains no rows")
    columns = list(rows[0].keys())
    id_column = _pick_column(columns, ("brats", "subject", "patient", "id", "name", "case"))
    if column is not None:
        match = next(
            (item for item in columns if item.strip().lower().replace(" ", "_") == column.strip().lower().replace(" ", "_")),
            None,
        )
        if match is None:
            raise ValueError(
                f"--grade-column '{column}' does not exist in {path}. Columns: {', '.join(columns)}"
            )
        grade_column = match
    else:
        grade_column = _pick_column(columns, ("grade", "who", "class", "label"))
    if id_column is None:
        raise ValueError(
            f"No identifier column was found in {path}. Columns: {', '.join(columns)}"
        )
    if grade_column is None:
        raise ValueError(
            f"No grade column was found in {path}. Columns: {', '.join(columns)}. "
            "BraTS 2020 ships the grade in name_mapping.csv; if your copy stores it under a "
            "different name, pass --grade-column explicitly."
        )
    table: dict[str, dict[str, Any]] = {}
    for row in rows:
        raw_id = str(row.get(id_column, "")).strip()
        if not raw_id:
            continue
        keys = subject_keys(raw_id)
        payload = {
            "grade": normalise_grade(str(row.get(grade_column, "")).strip()),
            "raw_grade": str(row.get(grade_column, "")).strip(),
            "source_column": grade_column,
            "source_file": str(path),
            "row": {key: value for key, value in row.items() if key not in {id_column, grade_column}},
        }
        for key in keys:
            table[key] = payload
    return table


def _pick_column(columns: Sequence[str], hints: Sequence[str]) -> str | None:
    for hint in hints:
        for column in columns:
            if column.strip().lower().replace(" ", "_") == hint:
                return column
    for hint in hints:
        for column in columns:
            if hint in column.strip().lower():
                return column
    return None


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        sample = handle.read(4096)
        handle.seek(0)
        delimiter = ","
        if sample.count("\t") > sample.count(","):
            delimiter = "\t"
        return [dict(row) for row in csv.DictReader(handle, delimiter=delimiter)]


def subject_keys(identifier: str) -> set[str]:
    keys = {identifier.strip().lower()}
    tokens = re.split(r"[^A-Za-z0-9]+", identifier.strip())
    digits = [token for token in tokens if token.isdigit()]
    for token in digits:
        keys.add(str(int(token)))
        keys.add(token.zfill(3))
        keys.add(f"brats20_training_{token.zfill(3)}".lower())
        keys.add(f"brats20_training_{int(token):03d}".lower())
    joined = "".join(token for token in tokens if token).lower()
    if joined:
        keys.add(joined)
    return {key for key in keys if key}


def normalise_grade(value: str, *, high_values: Sequence[str] = (), low_values: Sequence[str] = ()) -> str | None:
    if not value:
        return None
    text = value.strip().lower()
    if text in {"hgg", "high", "high_grade", "high-grade", "highgrade", "g3", "g4", "grade3", "grade4", "iii", "iv", "3", "4"}:
        if text in {"3", "4"} and text not in high_values:
            return None
        return "HGG"
    if text in {"lgg", "low", "low_grade", "low-grade", "lowgrade", "g2", "grade2", "ii", "2"}:
        if text == "2" and text not in low_values:
            return None
        return "LGG"
    for item in high_values:
        if text == item.strip().lower():
            return "HGG"
    for item in low_values:
        if text == item.strip().lower():
            return "LGG"
    return None


def volume_identifier(path: Path) -> str:
    attributes = parse_name_attributes(path)
    raw = str(attributes.get("volume_number") or attributes.get("volume_key") or path.stem)
    return raw


def inspect_tree(
    root: Path,
    *,
    limit: int = 5,
    extensions: Sequence[str] = (".h5", ".hdf5"),
) -> dict[str, Any]:
    paths = discover_slice_files(root, extensions)
    report: dict[str, Any] = {
        "root": str(root),
        "files": len(paths),
        "examples": [],
        "extensions": {},
        "volumes": 0,
        "slices_per_volume": {},
    }
    for path in paths:
        report["extensions"][path.suffix.lower()] = report["extensions"].get(path.suffix.lower(), 0) + 1
    if not paths:
        directories = sorted({item.parent for item in root.rglob("*") if item.is_dir()})[:limit]
        report["note"] = (
            "No HDF5 slice files were found. Directory levels inspected: "
            + ", ".join(str(item) for item in directories)
        )
        return report
    groups = group_by_volume(paths)
    report["volumes"] = len(groups)
    report["slices_per_volume"] = {
        "min": min(len(items) for items in groups.values()),
        "max": max(len(items) for items in groups.values()),
        "mean": round(sum(len(items) for items in groups.values()) / len(groups), 2),
    }
    image_keys: dict[str, int] = defaultdict(int)
    mask_keys: dict[str, int] = defaultdict(int)
    shapes: dict[str, int] = defaultdict(int)
    mask_kinds: dict[str, int] = defaultdict(int)
    foreground: list[float] = []
    for path in paths[: max(1, limit)]:
        slice_obj = read_slice_file(path)
        report["examples"].append({"file": str(path), **slice_obj.summary()})
        image_keys[slice_obj.image_key] += 1
        mask_keys[slice_obj.mask_key] += 1
        shapes[json.dumps(list(slice_obj.image.shape))] += 1
        mask_kinds[str(slice_obj.mask_semantics.get("kind"))] += 1
        foreground.append(float(slice_obj.whole_tumour().mean()))
    report["image_keys"] = dict(image_keys)
    report["mask_keys"] = dict(mask_keys)
    report["image_shapes"] = dict(shapes)
    report["mask_kinds"] = dict(mask_kinds)
    if foreground:
        report["foreground_fraction_examples"] = [round(value, 6) for value in foreground]
    return report
