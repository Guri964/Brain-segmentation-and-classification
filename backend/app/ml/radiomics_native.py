from __future__ import annotations
import logging
import math
from dataclasses import dataclass
from typing import Iterable
import numpy as np
logger = logging.getLogger(__name__)
EPS = np.finfo(np.float32).eps
FEATURE_CATEGORIES = {
    "shape": "Shape and size descriptors of the segmented region",
    "firstorder": "First-order intensity statistics inside the segmented region",
    "glcm": "Grey-level co-occurrence matrix texture features",
    "glrlm": "Grey-level run-length matrix texture features",
    "diagnostics": "Acquisition and segmentation diagnostics and global histogram descriptors",
}
GLCM_DIRECTIONS_2D = ((0, 1), (1, 0), (1, 1), (-1, 1))
GLRLM_DIRECTIONS_2D = ((0, 1), (1, 0), (1, 1), (-1, 1))
GLCM_DIRECTIONS_3D = (
    (0, 1, 0),
    (1, 0, 0),
    (0, 0, 1),
    (1, 1, 0),
    (1, -1, 0),
    (1, 0, 1),
    (1, 0, -1),
    (0, 1, 1),
    (0, 1, -1),
    (1, 1, 1),
    (1, 1, -1),
    (1, -1, 1),
    (1, -1, -1),
)
GLRLM_DIRECTIONS_3D = (
    (0, 1, 0),
    (1, 0, 0),
    (0, 0, 1),
    (1, 1, 0),
    (1, -1, 0),
    (1, 0, 1),
    (1, 0, -1),
    (0, 1, 1),
    (0, 1, -1),
    (1, 1, 1),
    (1, 1, -1),
    (1, -1, 1),
    (1, -1, -1),
)
@dataclass
class DiscretisationSettings:
    method: str = "bin_count"
    bin_count: int = 32
    bin_width: float = 25.0
    def as_dict(self) -> dict:
        return {
            "method": self.method,
            "bin_count": self.bin_count,
            "bin_width": self.bin_width,
            "note": (
                "Fixed bin count discretisation rescales every case to the same number of "
                "bins, which makes texture values comparable across cases but dependent on "
                "the intensity range. Fixed bin width is reported in the same intensity unit "
                "as the image."
            ),
        }
def discretise(
    intensities: np.ndarray, settings: DiscretisationSettings
) -> tuple[np.ndarray, dict]:
    values = np.asarray(intensities, dtype=np.float64).ravel()
    if values.size == 0:
        return np.zeros(0, dtype=np.int32), {"levels": 0}
    if settings.method == "bin_width" and settings.bin_width > 0:
        low = values.min()
        levels = np.floor((values - low) / settings.bin_width).astype(np.int32) + 1
        levels = np.clip(levels, 1, 256)
        info = {
            "levels": int(levels.max()),
            "method": "bin_width",
            "bin_width": settings.bin_width,
            "range": [float(low), float(values.max())],
        }
        return levels, info
    n_bins = max(2, int(settings.bin_count))
    low, high = float(values.min()), float(values.max())
    if high - low < 1e-12:
        return np.ones_like(values, dtype=np.int32), {
            "levels": 1,
            "method": "bin_count",
            "bin_count": n_bins,
            "range": [low, high],
        }
    edges = np.linspace(low, high + 1e-9, n_bins + 1)
    levels = np.digitize(values, edges, right=False).astype(np.int32)
    levels = np.clip(levels, 1, n_bins)
    return levels, {
        "levels": int(levels.max()),
        "method": "bin_count",
        "bin_count": n_bins,
        "range": [low, high],
    }
def _perimeter_2d(mask: np.ndarray, spacing: tuple[float, float] = (1.0, 1.0)) -> float:
    mask = np.asarray(mask).astype(bool)
    if mask.ndim != 2:
        raise ValueError("_perimeter_2d expects a 2-D mask")
    dy, dx = float(spacing[0]), float(spacing[1])
    rows, cols = mask.shape
    samples = np.pad(mask, 1)
    east_west = samples[:, :-1] != samples[:, 1:]
    north_south = samples[:-1, :] != samples[1:, :]
    top = east_west[:-1, : cols + 1]
    bottom = east_west[1:, : cols + 1]
    left = north_south[:, : cols + 1]
    right = north_south[:, 1 : cols + 2]
    count = top.astype(np.int8) + bottom + left + right
    vertical_segment = top & bottom
    horizontal_segment = left & right
    diagonal_segment = (count == 2) & ~(vertical_segment | horizontal_segment)
    four_crossings = (count == 4) | (count % 2 == 1)
    diagonal_length = math.hypot(dx / 2.0, dy / 2.0)
    return float(
        vertical_segment.sum() * dy
        + horizontal_segment.sum() * dx
        + diagonal_segment.sum() * diagonal_length
        + four_crossings.sum() * 2.0 * diagonal_length
    )
def shape_features_2d(mask: np.ndarray, spacing: tuple[float, float] | None = None) -> dict:
    mask = np.asarray(mask).astype(bool)
    area_px = float(mask.sum())
    features: dict[str, float | None] = {}
    if spacing is None:
        features["pixel_area"] = area_px
        features["perimeter_px"] = _perimeter_2d(mask, (1.0, 1.0))
        features["equivalent_diameter_px"] = float(2 * math.sqrt(area_px / math.pi)) if area_px else 0.0
        features["circularity"] = (
            float(4 * math.pi * area_px / (features["perimeter_px"] ** 2))
            if features["perimeter_px"] > 1e-9
            else None
        )
        features["area_mm2"] = None
        features["perimeter_mm"] = None
        features["equivalent_diameter_mm"] = None
        features["units"] = "pixels"
    else:
        area_mm2 = area_px * spacing[0] * spacing[1]
        perimeter = _perimeter_2d(mask, spacing)
        features["pixel_area"] = area_px
        features["perimeter_px"] = None
        features["equivalent_diameter_px"] = None
        features["area_mm2"] = area_mm2
        features["perimeter_mm"] = perimeter
        features["equivalent_diameter_mm"] = float(2 * math.sqrt(area_mm2 / math.pi)) if area_mm2 else 0.0
        features["circularity"] = (
            float(4 * math.pi * area_mm2 / (perimeter**2)) if perimeter > 1e-9 else None
        )
        features["units"] = "mm2"
    features["elongation"] = _elongation(mask)
    features["flatness"] = None
    extent = _bbox_extent(mask)
    features["bbox_height_px"] = extent[0] if extent else None
    features["bbox_width_px"] = extent[1] if extent else None
    features["sphericity"] = None
    return features
def _bbox_extent(mask: np.ndarray) -> list[int] | None:
    coordinates = np.argwhere(mask)
    if coordinates.size == 0:
        return None
    return [int(coordinates[:, axis].max() - coordinates[:, axis].min() + 1) for axis in range(coordinates.shape[1])]
def shape_features_3d(mask: np.ndarray, spacing: tuple[float, float, float] | None = None) -> dict:
    mask = np.asarray(mask).astype(bool)
    voxels = float(mask.sum())
    spacing = spacing or (1.0, 1.0, 1.0)
    volume = voxels * float(np.prod(spacing))
    axes = _principal_axes(mask, spacing)
    surface = _surface_area(mask, spacing)
    features = {
        "voxel_count": voxels,
        "volume_mm3": volume,
        "surface_area_mm2": surface,
        "major_axis_mm": axes[0] if axes else None,
        "minor_axis_mm": axes[1] if len(axes) > 1 else None,
        "least_axis_mm": axes[2] if len(axes) > 2 else None,
        "elongation": (axes[1] / axes[0]) if axes and axes[0] > 0 else None,
        "flatness": (axes[2] / axes[0]) if axes and len(axes) > 2 and axes[0] > 0 else None,
        "sphericity": (
            float((36 * math.pi * max(volume, 0.0) ** 2) ** (1 / 3) / surface)
            if surface > 1e-9 and volume > 0
            else None
        ),
        "units": "mm3",
    }
    features["equivalent_diameter_mm"] = (
        float((6 * volume / math.pi) ** (1 / 3)) if volume > 0 else None
    )
    return features
def _principal_axes(mask: np.ndarray, spacing: tuple[float, ...]) -> list[float]:
    coordinates = np.argwhere(mask).astype(np.float64)
    if coordinates.shape[0] < 3:
        return []
    coordinates *= np.asarray(spacing, dtype=np.float64)
    centred = coordinates - coordinates.mean(axis=0)
    covariance = np.cov(centred, rowvar=False)
    try:
        eigenvalues = np.linalg.eigvalsh(covariance)
    except np.linalg.LinAlgError:
        return []
    eigenvalues = np.clip(np.sort(eigenvalues)[::-1], 0, None)
    return [float(4 * math.sqrt(value)) for value in eigenvalues]
def _surface_area(mask: np.ndarray, spacing: tuple[float, ...]) -> float:
    padded = np.pad(mask.astype(np.uint8), 1)
    centre = padded[1:-1, 1:-1, 1:-1].astype(bool)
    area = 0.0
    for axis, step in enumerate(spacing):
        for shift in (-1, 1):
            shifted = np.roll(padded, shift, axis=axis)
            if shift == 1:
                sl = [slice(None)] * 3
                sl[axis] = 0
                shifted[tuple(sl)] = 0
            else:
                sl = [slice(None)] * 3
                sl[axis] = -1
                shifted[tuple(sl)] = 0
            neighbour = shifted[1:-1, 1:-1, 1:-1].astype(bool)
            faces = centre & ~neighbour
            other = [spacing[i] for i in range(3) if i != axis]
            area += float(faces.sum()) * float(np.prod(other))
    return area
def _elongation(mask: np.ndarray) -> float | None:
    coordinates = np.argwhere(mask).astype(np.float64)
    if coordinates.shape[0] < 3:
        return None
    centred = coordinates - coordinates.mean(axis=0)
    covariance = np.cov(centred, rowvar=False)
    try:
        eigenvalues = np.sort(np.clip(np.linalg.eigvalsh(covariance), 0, None))[::-1]
    except np.linalg.LinAlgError:
        return None
    if eigenvalues[0] <= 0:
        return None
    return float(math.sqrt(eigenvalues[1] / eigenvalues[0]))
def first_order_features(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64).ravel()
    if values.size == 0:
        return {}
    mean = float(values.mean())
    variance = float(values.var())
    std = math.sqrt(variance)
    sorted_values = np.sort(values)
    percentiles = np.percentile(sorted_values, [10, 25, 50, 75, 90])
    centred = values - mean
    skewness = float((centred**3).mean() / (std**3)) if std > 1e-12 else 0.0
    kurtosis = float((centred**4).mean() / (std**4) - 3.0) if std > 1e-12 else 0.0
    return {
        "mean": mean,
        "median": float(np.median(sorted_values)),
        "minimum": float(sorted_values[0]),
        "maximum": float(sorted_values[-1]),
        "range": float(sorted_values[-1] - sorted_values[0]),
        "variance": variance,
        "standard_deviation": std,
        "skewness": skewness,
        "kurtosis": kurtosis,
        "p10": float(percentiles[0]),
        "p25": float(percentiles[1]),
        "p50": float(percentiles[2]),
        "p75": float(percentiles[3]),
        "p90": float(percentiles[4]),
        "interquartile_range": float(percentiles[3] - percentiles[1]),
        "mean_absolute_deviation": float(np.abs(centred).mean()),
        "root_mean_squared": float(np.sqrt((values**2).mean())),
        "energy": float((values**2).sum()),
        "pixel_count": int(values.size),
    }
def histogram_descriptors(probabilities: np.ndarray) -> dict:
    values = np.asarray(probabilities, dtype=np.float64).ravel()
    values = values[values > 0]
    if values.size == 0:
        return {"entropy": None, "uniformity": None, "energy": None, "maximum_probability": None}
    return {
        "entropy": float(-(values * np.log2(values)).sum()),
        "uniformity": float((values**2).sum()),
        "energy": float(np.sqrt((values**2).sum())),
        "maximum_probability": float(values.max()),
    }
def _offsets_inside(shape: tuple[int, ...], direction: tuple[int, ...]) -> tuple[slice, slice]:
    src = []
    dst = []
    for size, step in zip(shape, direction):
        if step >= 0:
            src.append(slice(0, size - step))
            dst.append(slice(step, size))
        else:
            src.append(slice(-step, size))
            dst.append(slice(0, size + step))
    return tuple(src), tuple(dst)
def glcm_matrix(levels: np.ndarray, n_levels: int, direction: tuple[int, ...]) -> np.ndarray:
    src, dst = _offsets_inside(levels.shape, direction)
    a = levels[src].astype(np.int32)
    b = levels[dst].astype(np.int32)
    valid = (a > 0) & (b > 0)
    if not valid.any():
        return np.zeros((n_levels, n_levels), dtype=np.float64)
    index = (a[valid] - 1) * n_levels + (b[valid] - 1)
    counts = np.bincount(index, minlength=n_levels * n_levels).astype(np.float64)
    matrix = counts.reshape(n_levels, n_levels)
    matrix = matrix + matrix.T
    total = matrix.sum()
    return matrix / total if total > 0 else matrix
def glcm_features(levels: np.ndarray, n_levels: int, directions: Iterable[tuple[int, ...]], spacing=None) -> dict:
    matrices = [glcm_matrix(levels, n_levels, direction) for direction in directions]
    matrices = [m for m in matrices if m.sum() > 0]
    if not matrices:
        return {}
    i = np.arange(1, n_levels + 1, dtype=np.float64)[:, None]
    j = np.arange(1, n_levels + 1, dtype=np.float64)[None, :]
    contrast = []
    dissimilarity = []
    homogeneity = []
    asm = []
    energy = []
    correlation = []
    entropy = []
    mean_i = []
    variance_i = []
    variance_j = []
    for matrix in matrices:
        p = matrix
        diff = (i - j) ** 2
        contrast.append(float((p * diff).sum()))
        dissimilarity.append(float((p * np.abs(i - j)).sum()))
        homogeneity.append(float((p / (1 + diff)).sum()))
        asm.append(float((p**2).sum()))
        energy.append(float(np.sqrt((p**2).sum())))
        mu_i = float((p * i).sum())
        mu_j = float((p * j).sum())
        var_i = float((p * (i - mu_i) ** 2).sum())
        var_j = float((p * (j - mu_j) ** 2).sum())
        denom = math.sqrt(var_i * var_j)
        correlation.append(float(((p * (i - mu_i) * (j - mu_j)).sum() / denom)) if denom > 1e-12 else 0.0)
        entropy.append(float(-(p[p > 0] * np.log2(p[p > 0])).sum()))
        mean_i.append(mu_i)
        variance_i.append(var_i)
        variance_j.append(var_j)
    average = lambda values: float(np.mean(values)) if values else None
    return {
        "contrast": average(contrast),
        "dissimilarity": average(dissimilarity),
        "homogeneity": average(homogeneity),
        "asm": average(asm),
        "energy": average(energy),
        "correlation": average(correlation),
        "entropy": average(entropy),
        "mean_intensity": average(mean_i),
        "variance": average(variance_i),
        "n_directions": len(matrices),
    }
def _run_length_matrix(
    levels: np.ndarray, n_levels: int, direction: tuple[int, ...]
) -> np.ndarray:
    shape = levels.shape
    max_run = max(shape)
    matrix = np.zeros((n_levels, max_run), dtype=np.float64)
    steps = np.asarray(direction, dtype=np.int64)
    for start in np.argwhere(levels > 0):
        level = int(levels[tuple(start)])
        predecessor = start - steps
        if all(0 <= predecessor[k] < shape[k] for k in range(levels.ndim)):
            if int(levels[tuple(predecessor)]) == level:
                continue
        length = 0
        current = start.copy()
        while all(0 <= current[k] < shape[k] for k in range(levels.ndim)):
            if int(levels[tuple(current)]) != level:
                break
            length += 1
            current = current + steps
        if length:
            matrix[level - 1, min(length, max_run) - 1] += 1
    return matrix
def glrlm_features(levels: np.ndarray, n_levels: int, directions: Iterable[tuple[int, ...]]) -> dict:
    max_run = max(levels.shape)
    matrices = []
    for direction in directions:
        matrix = _run_length_matrix(levels, n_levels, direction)
        total = matrix.sum()
        if total > 0:
            matrices.append(matrix / total)
    if not matrices:
        return {}
    p = np.mean(matrices, axis=0)
    p = p / p.sum()
    j = np.arange(1, max_run + 1, dtype=np.float64)[None, :]
    i = np.arange(1, n_levels + 1, dtype=np.float64)[:, None]
    n_runs = p.sum()
    sre = float((p / (j**2)).sum())
    lre = float((p * (j**2)).sum())
    gln = float((p.sum(axis=1) ** 2).sum())
    rln = float((p.sum(axis=0) ** 2).sum())
    rp = float(n_runs / p.size)
    mean_level = float((p * i).sum())
    return {
        "short_run_emphasis": sre,
        "long_run_emphasis": lre,
        "grey_level_non_uniformity": gln,
        "run_length_non_uniformity": rln,
        "run_percentage": rp,
        "grey_level_variance": float((p * (i - mean_level) ** 2).sum()),
        "low_grey_level_run_emphasis": float((p / (i**2)).sum()),
        "high_grey_level_run_emphasis": float((p * (i**2)).sum()),
        "n_directions": len(matrices),
    }
def extract_features(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    spacing: tuple[float, ...] | None = None,
    settings: DiscretisationSettings | None = None,
    include_texture: bool = True,
) -> dict:
    image = np.asarray(image, dtype=np.float32)
    mask = np.asarray(mask).astype(bool)
    if image.shape != mask.shape:
        raise ValueError(f"image shape {image.shape} != mask shape {mask.shape}")
    if not mask.any():
        raise ValueError("The region of interest is empty; no features can be extracted.")
    settings = settings or DiscretisationSettings()
    dimensionality = "3D" if image.ndim == 3 else "2D"
    roi_values = image[mask]
    n_voxels = int(mask.sum())
    levels, discretisation_info = discretise(roi_values, settings)
    n_levels = max(1, int(discretisation_info.get("levels", 1)))
    result: dict = {}
    result["diagnostics_dimensionality"] = dimensionality
    result["diagnostics_roi_voxel_count"] = n_voxels
    result["diagnostics_image_mean"] = float(image.mean())
    result["diagnostics_image_min"] = float(image.min())
    result["diagnostics_image_max"] = float(image.max())
    result["diagnostics_roi_mean"] = float(roi_values.mean())
    result["diagnostics_roi_min"] = float(roi_values.min())
    result["diagnostics_roi_max"] = float(roi_values.max())
    result["diagnostics_intensity_range_in_roi"] = float(roi_values.max() - roi_values.min())
    result["diagnostics_discretisation_levels"] = n_levels
    result["diagnostics_discretisation_method"] = discretisation_info.get("method")
    result["diagnostics_spacing"] = list(spacing) if spacing else None
    result["diagnostics_spacing_available"] = bool(spacing)
    shape = shape_features_3d(mask, spacing) if image.ndim == 3 else shape_features_2d(mask, spacing)
    for key, value in shape.items():
        result[f"shape_{key}"] = value
    for key, value in first_order_features(roi_values).items():
        result[f"firstorder_{key}"] = value
    counts, _ = np.histogram(roi_values, bins=max(2, n_levels))
    probabilities = counts.astype(np.float64) / max(counts.sum(), 1)
    for key, value in histogram_descriptors(probabilities).items():
        result[f"diagnostics_{key}"] = value
    if include_texture:
        if n_voxels < 4 * n_levels:
            result["diagnostics_texture_warning"] = (
                f"The ROI contains {n_voxels} pixels for {n_levels} grey levels "
                f"({n_voxels / n_levels:.2f} pixels per level). Texture matrices with fewer than "
                "four pixels per grey level are not reliable; consider a fixed bin width or a "
                "coarser bin count."
            )
        levels_map = np.zeros(mask.shape, dtype=np.int32)
        levels_map[mask] = levels
        directions = GLCM_DIRECTIONS_2D if image.ndim == 2 else GLCM_DIRECTIONS_3D
        for key, value in glcm_features(levels_map, n_levels, directions).items():
            result[f"glcm_{key}"] = value
        run_directions = GLRLM_DIRECTIONS_2D if image.ndim == 2 else GLRLM_DIRECTIONS_3D
        for key, value in glrlm_features(levels_map, n_levels, run_directions).items():
            result[f"glrlm_{key}"] = value
    numeric = {
        key: (None if value is None or isinstance(value, str) or isinstance(value, bool) else float(value))
        for key, value in result.items()
        if not key.startswith("diagnostics_") or key in {
            "diagnostics_roi_voxel_count",
            "diagnostics_image_mean",
            "diagnostics_image_min",
            "diagnostics_image_max",
            "diagnostics_roi_mean",
            "diagnostics_roi_min",
            "diagnostics_roi_max",
            "diagnostics_intensity_range_in_roi",
            "diagnostics_discretisation_levels",
            "diagnostics_entropy",
            "diagnostics_uniformity",
            "diagnostics_energy",
            "diagnostics_maximum_probability",
        }
    }
    for key, value in numeric.items():
        if value is not None and not np.isfinite(value):
            result[key] = None
    return result
def categorise(feature_name: str) -> str:
    if feature_name.startswith("shape_"):
        return "shape"
    if feature_name.startswith("firstorder_"):
        return "firstorder"
    if feature_name.startswith("glcm_"):
        return "glcm"
    if feature_name.startswith("glrlm_"):
        return "glrlm"
    return "diagnostics"
