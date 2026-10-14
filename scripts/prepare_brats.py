from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import configure_logging, utc_now
from _splits import assign_splits, describe_splits
from brats import (
    discover_slice_files,
    find_volume_directories,
    group_by_volume,
    parse_name_attributes,
    read_grade_table,
    read_slice_file,
    read_volume_directory,
    subject_keys,
    volume_identifier,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a BraTS 2020 dataset into a patient-level manifest that the training, "
            "evaluation and radiomics scripts consume."
        )
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="Manifest to write")
    parser.add_argument("--name", default="brats2020")
    parser.add_argument("--version", default="brats2020-training")
    parser.add_argument("--grade-table", type=Path, default=None)
    parser.add_argument("--grade-column", default=None)
    parser.add_argument(
        "--mask-channel-labels",
        default=None,
        help=(
            "Comma-separated sub-region labels for a one-hot mask, in stored order, for example "
            "'1,2,4'. Left empty, the planes stay separate and only their union is used."
        ),
    )
    parser.add_argument(
        "--slice-filter",
        choices=["tumour", "all"],
        default="tumour",
        help="'tumour' keeps slices whose mask has foreground (recommended for training).",
    )
    parser.add_argument("--min-foreground-pixels", type=int, default=1)
    parser.add_argument("--max-slices-per-volume", type=int, default=None)
    parser.add_argument("--scan-limit", type=int, default=None, help="Only inspect this many files")
    parser.add_argument("--channel-limit", type=int, default=4, help="Maximum channels kept per slice")
    parser.add_argument("--splits", action="store_true", help="Assign patient-level splits now")
    parser.add_argument("--train", type=float, default=0.70)
    parser.add_argument("--val", type=float, default=0.15)
    parser.add_argument("--test", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=20240101)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    logger = configure_logging(args.verbose)
    root = args.root.expanduser().resolve()
    if not root.exists():
        raise SystemExit(f"{root} does not exist")

    channel_labels = None
    if args.mask_channel_labels:
        channel_labels = [int(item) for item in args.mask_channel_labels.split(",") if item.strip()]

    grades: dict[str, dict] = {}
    if args.grade_table:
        grades = read_grade_table(args.grade_table, column=args.grade_column)
        logger.info(
            "Grade table: %d identifier(s) mapped from column '%s'",
            len(grades),
            next((entry.get("source_column") for entry in grades.values()), "?"),
        )

    slice_files = discover_slice_files(root)
    volume_directories = find_volume_directories(root)
    cases: list[dict] = []
    statistics = {"read": 0, "kept": 0, "empty_masks": 0, "unreadable": 0, "missing_grade": 0}
    modalities_seen: set[str] = set()

    if slice_files:
        logger.info("HDF5 layout: %d slice file(s)", len(slice_files))
        grouped = group_by_volume(slice_files)
        if args.scan_limit:
            scan: list[Path] = []
            for key in sorted(grouped):
                scan.extend(grouped[key])
                if len(scan) >= args.scan_limit:
                    break
            grouped = group_by_volume(scan)
            logger.info("Scan limited to %d file(s)", len(scan))
        for volume_key in sorted(grouped):
            kept_in_volume = 0
            patient_id = _patient_identifier(volume_key, grades)
            for path in grouped[volume_key]:
                if args.max_slices_per_volume and kept_in_volume >= args.max_slices_per_volume:
                    break
                statistics["read"] += 1
                try:
                    slice_obj = read_slice_file(
                        path,
                        mask_channel_labels=channel_labels,
                    )
                except Exception as exc:
                    statistics["unreadable"] += 1
                    logger.warning("Could not read %s: %s", path.name, exc)
                    continue
                foreground = int(slice_obj.whole_tumour().sum())
                slice_attributes = parse_name_attributes(path)
                if args.slice_filter == "tumour" and foreground < args.min_foreground_pixels:
                    statistics["empty_masks"] += 1
                    continue
                channels = min(slice_obj.channels, args.channel_limit)
                if channels < slice_obj.channels:
                    logger.warning(
                        "%s has %d channels; keeping the first %d in stored order unmodified.",
                        path.name,
                        slice_obj.channels,
                        channels,
                    )
                label_entry = grades.get(patient_id) or _grade_for(volume_key, grades)
                label = label_entry.get("grade") if label_entry else None
                if label is None:
                    statistics["missing_grade"] += 1
                images = [str(path)] * channels
                cases.append(
                    {
                        "case_id": f"{volume_key}_{slice_attributes.get('slice_index', statistics['read'])}",
                        "patient_id": str(patient_id),
                        "images": images,
                        "mask": str(path),
                        "label": label,
                        "options": {
                            "kind": "brats_hdf5_slice",
                            "channels": channels,
                            "channel_order": [f"channel_{index}" for index in range(channels)],
                            "mask_source": "same_file",
                            "mask_semantics": slice_obj.mask_semantics.get("kind"),
                            "mask_channel_labels": slice_obj.mask_semantics.get("channel_labels"),
                            "foreground_pixels": foreground,
                            "slice_index": slice_attributes.get("slice_index"),
                            "spacing": None,
                        },
                    }
                )
                kept_in_volume += 1
                statistics["kept"] += 1
        modalities_seen.add("hdf5_slice_channels")
    elif volume_directories:
        logger.info("NIfTI layout: %d volume director(ies)", len(volume_directories))
        for directory in volume_directories:
            statistics["read"] += 1
            volume = read_volume_directory(directory)
            modalities = volume["modalities"]
            segmentation = volume["segmentation"]
            if not modalities or segmentation is None:
                statistics["unreadable"] += 1
                logger.warning("%s has no usable modalities or no segmentation", directory.name)
                continue
            order = [name for name in ("t1", "t1ce", "t2", "flair") if name in modalities]
            order += [name for name in sorted(modalities) if name not in order]
            order = order[: args.channel_limit]
            modalities_seen.update(order)
            patient_id = subject_keys(directory.name)
            label_entry = None
            for key in patient_id:
                if key in grades:
                    label_entry = grades[key]
                    break
            label = label_entry.get("grade") if label_entry else None
            if label is None:
                statistics["missing_grade"] += 1
            spacing = volume["spacing"]
            cases.append(
                {
                    "case_id": directory.name,
                    "patient_id": directory.name,
                    "images": [str(directory / f"{directory.name}_{name}.nii.gz") for name in order],
                    "mask": str(directory / f"{directory.name}_seg.nii.gz"),
                    "label": label,
                    "options": {
                        "kind": "brats_nifti_volume",
                        "channels": len(order),
                        "channel_order": order,
                        "mask_source": "same_directory",
                        "mask_semantics": "brats_label_map",
                        "spacing": list(spacing) if spacing else None,
                    },
                }
            )
            statistics["kept"] += 1
    else:
        logger.error(
            "Neither HDF5 slice files nor NIfTI volumes were found under %s. Run "
            "scripts/inspect_brats.py first to see what the directory actually contains.",
            root,
        )
        return 2

    if not cases:
        logger.error(
            "No case survived the filters (read %d, empty masks %d, unreadable %d). Relax "
            "--slice-filter or --min-foreground-pixels.",
            statistics["read"],
            statistics["empty_masks"],
            statistics["unreadable"],
        )
        return 2

    payload: dict = {
        "name": args.name,
        "version": args.version,
        "root": str(root),
        "created_at": utc_now(),
        "source": {
            "layout": "hdf5_slices" if slice_files else "nifti_volumes",
            "root": str(root),
            "grade_table": str(args.grade_table) if args.grade_table else None,
            "slice_filter": args.slice_filter,
            "min_foreground_pixels": args.min_foreground_pixels,
            "max_slices_per_volume": args.max_slices_per_volume,
            "channel_limit": args.channel_limit,
            "channel_order": sorted(modalities_seen),
            "statistics": statistics,
            "channel_note": (
                "Channels are kept in the order the file stores them. No channel is renamed, "
                "reordered or averaged."
            ),
        },
        "cases": cases,
    }

    if args.splits:
        splits, metadata = assign_splits(
            cases,
            train=args.train,
            val=args.val,
            test=args.test,
            seed=args.seed,
            logger=logger,
        )
        payload["splits"] = splits
        payload["split_metadata"] = metadata
        for name in ("train", "validation", "test"):
            entry = metadata[name]
            logger.info(
                "%-10s %6d case(s) · %4d patient(s) · classes %s",
                name,
                entry["cases"],
                entry["patients"],
                entry["class_distribution"],
            )
        if not metadata.get("leakage_free", True):
            logger.error(
                "Patients appear in more than one split: %s. This manifest must not be used for "
                "training.",
                metadata.get("patient_overlap"),
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info(
        "Manifest written to %s · %d case(s) · %d patient(s) · labelled %d · unlabelled %d",
        args.out,
        len(cases),
        len({case["patient_id"] for case in cases}),
        sum(1 for case in cases if case["label"]), 
        statistics["missing_grade"],
    )
    if statistics["missing_grade"]:
        logger.warning(
            "%d case(s) have no grade label. They can train segmentation but not the classifier; "
            "supply --grade-table (BraTS 2020 name_mapping.csv) to label them.",
            statistics["missing_grade"],
        )
    return 0


def _patient_identifier(volume_key: str, grades: dict[str, dict]) -> str:
    for key in subject_keys(volume_key):
        if key in grades:
            return volume_key
    return volume_key


def _grade_for(volume_key: str, grades: dict[str, dict]) -> dict | None:
    for key in subject_keys(volume_key):
        if key in grades:
            return grades[key]
    return None


if __name__ == "__main__":
    raise SystemExit(main())
