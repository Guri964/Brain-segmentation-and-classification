#!/usr/bin/env python3
"""Write a synthetic *image-classification* fixture, for testing the pipeline only.

The real classification dataset is the Brain Tumor MRI Dataset
(https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset), which cannot be
downloaded on this machine. This script writes a directory tree with the same *shape* --
`Training/<class>/*.png` and `Testing/<class>/*.png` -- so the reader, the split logic, the
leakage audit, the CNN and the stacked ensemble can be exercised end to end.

**Nothing produced from it is a result.** The images are generated textures and the class
labels are made up; a `fixture_manifest.json` marks the tree as synthetic, which is how the
notebook detects it and records `synthetic: true` next to every artefact it exports.

Usage:
    python scripts/make_classification_fixture.py --out /tmp/classification_fixture \
        --classes glioma,meningioma,pituitary,notumor --train-per-class 6 --test-per-class 4
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

NOTICE = (
    "SYNTHETIC CLASSIFICATION FIXTURE - these images are generated textures with invented class "
    "labels. They exist so the pipeline can be tested without the real dataset. Any accuracy, F1 "
    "or AUC computed from them is meaningless by construction and must never be reported as a "
    "result."
)

DEFAULT_CLASSES = ("glioma", "meningioma", "pituitary", "notumor")


def _texture(kind: str, size: int, rng: np.random.Generator) -> np.ndarray:
    """A distinct, label-informative pattern per class so a tiny CNN can separate them.

    Every image gets its own geometry (lesion position and size, head tilt, noise) so the images
    are genuinely different from each other -- otherwise the fixture would look like a pile of
    near-duplicates and the notebook's leakage audit would (correctly) flag them.
    """
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    centre = size / 2.0
    skull = (((yy - centre) / (size * rng.uniform(0.40, 0.47))) ** 2 + ((xx - centre) / (size * rng.uniform(0.32, 0.40))) ** 2) <= 1.0
    lesion_y = centre * rng.uniform(0.62, 1.38)
    lesion_x = centre * rng.uniform(0.62, 1.38)
    base = np.zeros((size, size), dtype=np.float32)
    if kind == "glioma":
        blob = np.exp(
            -(((yy - lesion_y) ** 2 + (xx - lesion_x) ** 2) / (2 * (size * rng.uniform(0.09, 0.16)) ** 2))
        )
        base = 0.55 * skull + 0.5 * blob
    elif kind == "meningioma":
        blob = np.exp(
            -(((yy - lesion_y) ** 2 + (xx - lesion_x) ** 2) / (2 * (size * rng.uniform(0.07, 0.12)) ** 2))
        )
        ridge = np.exp(-(((yy - centre) ** 2) / (2 * (size * rng.uniform(0.04, 0.07)) ** 2)))
        base = 0.5 * skull + 0.45 * blob * ridge
    elif kind == "pituitary":
        blob = np.exp(
            -(((yy - lesion_y) ** 2 + (xx - lesion_x) ** 2) / (2 * (size * rng.uniform(0.04, 0.07)) ** 2))
        )
        base = 0.45 * skull + 0.7 * blob
    else:  # notumor / anything else: no focal lesion, only anatomy
        base = 0.5 * skull + 0.08 * np.sin(xx / rng.uniform(6.0, 14.0)) * np.cos(yy / rng.uniform(7.0, 15.0))
    image = base + 0.05 * np.sin(xx / 5.0 + rng.uniform(0, 6.28)) + rng.normal(0, rng.uniform(0.02, 0.05), base.shape)
    return np.clip(image, 0.0, 1.0)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=NOTICE, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True, help="Directory to write the tree into")
    parser.add_argument("--classes", default=",".join(DEFAULT_CLASSES), help="Comma-separated class names")
    parser.add_argument("--train-per-class", type=int, default=6)
    parser.add_argument("--test-per-class", type=int, default=4)
    parser.add_argument("--size", type=int, default=96, help="Image side in pixels")
    parser.add_argument("--seed", type=int, default=20240101)
    parser.add_argument("--jpeg", action="store_true", help="Write JPEGs instead of PNGs (format handling)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    try:
        from PIL import Image
    except ImportError:  # pragma: no cover
        print("Pillow is required: pip install Pillow", file=sys.stderr)
        return 2

    classes = [name.strip() for name in args.classes.split(",") if name.strip()]
    if len(classes) < 2:
        print("at least two classes are required", file=sys.stderr)
        return 2

    rng = np.random.default_rng(args.seed)
    root = args.out
    extension = ".jpg" if args.jpeg else ".png"
    counts = {"Training": {}, "Testing": {}}
    for split_name, per_class in (("Training", args.train_per_class), ("Testing", args.test_per_class)):
        for class_name in classes:
            directory = root / split_name / class_name
            directory.mkdir(parents=True, exist_ok=True)
            for index in range(per_class):
                array = _texture(class_name, args.size, rng)
                path = directory / f"{class_name}_{index:03d}{extension}"
                Image.fromarray((array * 255).astype(np.uint8)).save(path)
            counts[split_name][class_name] = per_class

    manifest = {
        "synthetic": True,
        "generator": "scripts/make_classification_fixture.py",
        "notice": NOTICE,
        "classes": classes,
        "image_size": args.size,
        "counts": counts,
        "seed": args.seed,
        "layout": "Training/<class>/*" + extension + " and Testing/<class>/*" + extension,
    }
    (root / "fixture_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # A small CSV mirroring the real dataset's companion metadata, so the label reader is exercised.
    with open(root / "fixture_labels.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["split", "class", "images"])
        for split_name, per_class_counts in counts.items():
            for class_name, value in per_class_counts.items():
                writer.writerow([split_name, class_name, value])

    print(NOTICE)
    print(f"wrote {root}: {counts}")
    print(f"marker: {root / 'fixture_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
