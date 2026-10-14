from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import BACKEND_ROOT, configure_logging, utc_now
from brats import (
    discover_slice_files,
    find_volume_directories,
    group_by_volume,
    inspect_tree,
    read_grade_table,
    read_slice_file,
    read_volume_directory,
)

sys.path.insert(0, str(BACKEND_ROOT))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect a BraTS-style dataset before anything is trained: file layout, HDF5 keys, "
            "channel order, mask encoding, patient identifiers and grade labels."
        )
    )
    parser.add_argument("--root", type=Path, required=True, help="Directory with the dataset")
    parser.add_argument("--grade-table", type=Path, default=None, help="name_mapping.csv or survival_info.csv")
    parser.add_argument("--grade-column", default=None)
    parser.add_argument("--examples", type=int, default=3, help="How many files to read in full")
    parser.add_argument("--json", type=Path, default=None, help="Write the report here")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    logger = configure_logging(args.verbose)
    root = args.root.expanduser().resolve()
    if not root.exists():
        raise SystemExit(f"{root} does not exist")

    report: dict = {
        "generated_at": utc_now(),
        "root": str(root),
        "layout": None,
    }
    slice_files = discover_slice_files(root)
    volume_directories = find_volume_directories(root)

    if slice_files:
        report["layout"] = "hdf5_slices"
        tree = inspect_tree(root, limit=args.examples)
        report["tree"] = tree
        logger.info("HDF5 slice files: %d across %d volume(s)", tree["files"], tree["volumes"])
        logger.info(
            "Slices per volume: min %s · max %s · mean %s",
            tree["slices_per_volume"].get("min"),
            tree["slices_per_volume"].get("max"),
            tree["slices_per_volume"].get("mean"),
        )
        logger.info("Image dataset keys: %s", tree["image_keys"])
        logger.info("Mask dataset keys: %s", tree["mask_keys"])
        logger.info("Image shapes: %s", tree["image_shapes"])
        logger.info("Mask encodings: %s", tree["mask_kinds"])
        for example in tree["examples"]:
            logger.info(
                "  %s | image %s | channels %s | mask %s | foreground %.4f",
                Path(example["source"]).name,
                example["image_shape"],
                example["channels"],
                example["mask_semantics"].get("kind"),
                example["foreground_fraction"],
            )
        keys = group_by_volume(slice_files)
        example_key = sorted(keys)[0]
        report["example_volume"] = {
            "volume_key": example_key,
            "slices": len(keys[example_key]),
            "first_slices": [str(item) for item in keys[example_key][:3]],
        }
    elif volume_directories:
        report["layout"] = "nifti_volumes"
        logger.info("NIfTI volume directories: %d", len(volume_directories))
        report["volumes"] = []
        for directory in volume_directories[: max(1, args.examples)]:
            volume = read_volume_directory(directory)
            entry = {
                "directory": str(directory),
                "modalities": {name: list(array.shape) for name, array in volume["modalities"].items()},
                "ordered_modalities": sorted(volume["modalities"]),
                "segmentation_shape": list(volume["segmentation"].shape) if volume["segmentation"] is not None else None,
                "spacing": volume["spacing"],
            }
            if volume["segmentation"] is not None:
                import numpy as np

                values = [int(item) for item in np.unique(volume["segmentation"])[:12]]
                entry["segmentation_values"] = values
                entry["segmentation_is_multi_label"] = any(item > 1 for item in values)
            report["volumes"].append(entry)
            logger.info("  %s -> %s", directory.name, entry["modalities"])
    else:
        report["layout"] = "unknown"
        logger.error(
            "No HDF5 slice files (*.h5) and no NIfTI volume directories were found under %s. "
            "Pass the directory that actually contains them.",
            root,
        )
        if args.json:
            args.json.parent.mkdir(parents=True, exist_ok=True)
            args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return 2

    if args.grade_table:
        table = read_grade_table(args.grade_table, column=args.grade_column)
        counts: dict[str, int] = {}
        seen_rows: set[tuple] = set()
        for entry in table.values():
            key = (entry.get("source_file"), entry.get("raw_grade"), tuple(sorted(entry.get("row", {}).items())))
            if key in seen_rows:
                continue
            seen_rows.add(key)
            counts[str(entry.get("grade"))] = counts.get(str(entry.get("grade")), 0) + 1
        report["grade_table"] = {
            "path": str(args.grade_table),
            "rows": len(seen_rows),
            "class_counts": counts,
        }
        logger.info("Grade table %s: %s", args.grade_table.name, counts)
        if counts.get("None"):
            logger.warning(
                "%s identifier(s) did not map to HGG or LGG. Those cases stay unlabelled rather "
                "than being guessed.",
                counts["None"],
            )

    contracts = _model_contracts()
    report["model_contracts"] = contracts
    logger.info("Registered segmentation input contracts:")
    for entry in contracts:
        logger.info(
            "  %-20s %d x %d x %d · %s",
            entry["model_id"],
            entry["channels"],
            entry["height"],
            entry["width"],
            entry["normalisation"],
        )
    logger.info(
        "Interpretation: four-channel slices with %s are the match for a four-channel model; a "
        "single-channel model needs one modality resized to its declared size. Channel order is "
        "whatever the file stores — the inspector above reports it, and nothing reorders it "
        "silently.",
        "240x240",
    )

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        logger.info("Report written to %s", args.json)
    return 0


def _model_contracts() -> list[dict]:
    try:
        from app.config import DevelopmentConfig
        from app.ml.registry import ModelRegistry

        registry = ModelRegistry(DevelopmentConfig)
        return [
            {
                "model_id": descriptor.model_id,
                "channels": descriptor.input_spec.channels,
                "height": descriptor.input_spec.height,
                "width": descriptor.input_spec.width,
                "normalisation": descriptor.input_spec.normalisation,
                "output_semantics": descriptor.output_semantics,
            }
            for descriptor in registry.descriptors.values()
            if descriptor.task == "segmentation"
        ]
    except Exception as exc:
        return [{"error": f"{type(exc).__name__}: {exc}"}]


if __name__ == "__main__":
    raise SystemExit(main())
