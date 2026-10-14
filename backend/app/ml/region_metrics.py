"""Measurements of the segmented abnormal region.

These are the quantitative region measurements the project brief asks for
("abnormality measurements": pixel area, bounding box, width, height, image
coverage, centroid, perimeter and shape descriptors).  Everything is derived
from the binary mask that the analysis already produced, and nothing is
inferred about anatomy, malignancy or the patient.

Definitions
-----------
``pixel_area``             number of foreground pixels in the mask.
``image_coverage``         ``pixel_area / mask.size`` - the fraction of the
                           image occupied by the region.
``bounding_box``           tightest axis-aligned box around the region.  ``x``
                           and ``y`` are the top-left column and row, ``width``
                           and ``height`` are its size in pixels.
``centroid``               mean column (``x``) and mean row (``y``) of the
                           foreground pixels, in the same pixel coordinate
                           system as ``bounding_box``.
``perimeter_px``           boundary length of the region.  The estimator is the
                           one the native radiomics shape block already uses
                           (:func:`app.ml.radiomics_native._perimeter_2d`), so a
                           measurement quoted here can never disagree with the
                           ``shape_perimeter_px`` radiomics feature.
``equivalent_diameter_px`` diameter of the disc with the same area.

When no foreground pixel exists the function reports ``detected = False`` and
every geometric value as ``None``; it never substitutes ``0`` for "unknown",
because a zero-width box and an absent region are different statements.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .radiomics_native import _perimeter_2d

REGION_NOTE = (
    "Measurements describe the region the segmentation model produced.  They are "
    "image measurements in pixel units, not anatomical or clinical measurements, "
    "and they inherit any error in the mask."
)


def region_geometry(mask: np.ndarray) -> dict[str, Any]:
    """Axis-aware geometry of a mask of any dimensionality >= 2.

    Returned keys use the same axis order as the mask itself: ``bounding_box``
    and ``centroid`` are lists with one entry per axis (mirroring
    ``radiomics_service``'s ``roi_bounding_box``), and ``extent`` lists the
    inclusive size of each axis.
    """

    array = np.asarray(mask)
    if array.ndim < 2:
        raise ValueError(f"region_geometry expects a mask of at least 2 dimensions, got {array.ndim}")
    binary = array.astype(bool, copy=False)
    voxel_count = int(binary.sum())
    if voxel_count == 0:
        return {
            "detected": False,
            "voxel_count": 0,
            "bounding_box": None,
            "centroid": None,
            "extent": None,
            "touches_border": None,
        }
    coordinates = np.argwhere(binary)
    bounding_box = [
        [int(coordinates[:, axis].min()), int(coordinates[:, axis].max())]
        for axis in range(binary.ndim)
    ]
    centroid = [float(coordinates[:, axis].mean()) for axis in range(binary.ndim)]
    extent = [high - low + 1 for low, high in bounding_box]
    touches_border = any(
        low == 0 or high == size - 1
        for (low, high), size in zip(bounding_box, binary.shape)
    )
    return {
        "detected": True,
        "voxel_count": voxel_count,
        "bounding_box": bounding_box,
        "centroid": centroid,
        "extent": extent,
        "touches_border": bool(touches_border),
    }


def measure_region(mask: np.ndarray | None, *, source: str | None = None) -> dict[str, Any]:
    """Measurements of a 2-D binary region, ready to serialise into a payload."""

    if mask is None:
        return _absent(source, "No segmentation mask was available, so no region was measured.")
    array = np.asarray(mask)
    if array.ndim != 2:
        raise ValueError(f"measure_region expects a 2-D mask, got shape {array.shape}")
    geometry = region_geometry(array)
    if not geometry["detected"]:
        return _absent(
            source,
            "No pixel crossed the decision threshold, so no region was detected and no "
            "measurement is reported.",
        )
    rows, columns = array.shape
    total = int(array.size)
    voxel_count = int(geometry["voxel_count"])
    bbox = geometry["bounding_box"]
    centroid = geometry["centroid"]
    perimeter = _perimeter_2d(array.astype(bool, copy=False), (1.0, 1.0))
    height = int(bbox[0][1] - bbox[0][0] + 1)
    width = int(bbox[1][1] - bbox[1][0] + 1)
    return {
        "detected": True,
        "pixel_area": voxel_count,
        "image_coverage": round(voxel_count / total, 6),
        "image_coverage_percent": round(100.0 * voxel_count / total, 4),
        "bounding_box": {
            "x": int(bbox[1][0]),
            "y": int(bbox[0][0]),
            "width": width,
            "height": height,
        },
        "centroid": {"x": round(float(centroid[1]), 2), "y": round(float(centroid[0]), 2)},
        "perimeter_px": round(float(perimeter), 4),
        "equivalent_diameter_px": round(2.0 * math.sqrt(voxel_count / math.pi), 4),
        "units": "pixels",
        "image_size": {"width": int(columns), "height": int(rows)},
        "mask_source": source,
        "note": REGION_NOTE,
    }


def _absent(source: str | None, reason: str) -> dict[str, Any]:
    return {
        "detected": False,
        "pixel_area": 0,
        "image_coverage": 0.0,
        "image_coverage_percent": 0.0,
        "bounding_box": None,
        "centroid": None,
        "perimeter_px": None,
        "equivalent_diameter_px": None,
        "units": "pixels",
        "image_size": None,
        "mask_source": source,
        "note": reason,
    }
