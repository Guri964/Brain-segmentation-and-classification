from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import configure_logging
from brats import hdf5_entries

NOTICE = (
    "SYNTHETIC STRUCTURE FIXTURE - this generator writes random arrays that follow the BraTS "
    "HDF5 slice layout so that the reader, the manifest builder, the splitters and the training "
    "loop can be verified without the real dataset. Nothing produced from it is a result: any "
    "metric computed on these files is meaningless by construction."
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=NOTICE)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--volumes", type=int, default=6)
    parser.add_argument("--slices-per-volume", type=int, default=4)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--width", type=int, default=240)
    parser.add_argument("--channels", type=int, default=4)
    parser.add_argument("--empty-every", type=int, default=3, help="Every Nth slice gets an empty mask")
    parser.add_argument("--seed", type=int, default=20240101)
    parser.add_argument("--grades", default="HGG,LGG", help="Grades assigned alternately to volumes")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    import h5py

    logger = configure_logging(args.verbose)
    logger.warning(NOTICE)
    rng = np.random.default_rng(args.seed)
    out = args.out.expanduser().resolve()
    data_dir = out / "BraTS2020_training_data" / "content" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    grades = [item.strip() for item in args.grades.split(",") if item.strip()]
    rows = [["BraTS_2020_subject_id", "Grade", "Age"]]
    written = 0
    for volume_index in range(1, args.volumes + 1):
        grade = grades[(volume_index - 1) % len(grades)]
        rows.append([f"BraTS20_Training_{volume_index:03d}", grade, str(40 + volume_index)])
        volume_dir = data_dir / f"volume_{volume_index}"
        volume_dir.mkdir(parents=True, exist_ok=True)
        for slice_index in range(args.slices_per_volume):
            image = rng.normal(0.15, 0.12, size=(args.height, args.width, args.channels)).astype(np.float32)
            image = np.clip(image, 0.0, 1.0)
            mask = np.zeros((args.height, args.width, 3), dtype=np.uint8)
            if args.empty_every and slice_index % args.empty_every == 0:
                pass
            else:
                row = rng.integers(args.height // 4, args.height // 2)
                column = rng.integers(args.width // 4, args.width // 2)
                radius = int(rng.integers(12, 32))
                rows_index, columns_index = np.ogrid[: args.height, : args.width]
                disk = (rows_index - row) ** 2 + (columns_index - column) ** 2 <= radius**2
                mask[disk, 0] = 1
                mask[disk & (np.abs(rows_index - row) < radius // 2), 1] = 1
                mask[disk & (np.abs(columns_index - column) < radius // 3), 2] = 1
            image[..., 1] = np.where(mask[..., 0] > 0, image[..., 1] + 0.2, image[..., 1])
            path = volume_dir / f"volume_{volume_index}_slice_{slice_index}.h5"
            with h5py.File(path, "w") as handle:
                handle.create_dataset("image", data=image)
                handle.create_dataset("mask", data=mask)
                handle.attrs["source"] = "synthetic-fixture"
            written += 1
    mapping_path = out / "name_mapping.csv"
    with open(mapping_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerows(rows)
    logger.info("Wrote %d slice file(s) to %s", written, data_dir)
    logger.info("Wrote grade table %s with %d row(s)", mapping_path, len(rows) - 1)
    example = next(iter(sorted(data_dir.rglob("*.h5"))))
    for entry in hdf5_entries(example):
        logger.info("  %s", entry)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
