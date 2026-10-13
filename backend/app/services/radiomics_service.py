from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Any
import numpy as np
from ..ml.radiomics_native import (
    FEATURE_CATEGORIES,
    DiscretisationSettings,
    categorise,
    extract_features,
)
from ..utils.io import to_jsonable
logger = logging.getLogger(__name__)
FAMILY_ORDER = ("shape", "firstorder", "glcm", "glrlm", "diagnostics")
FEATURE_DESCRIPTIONS = {
    "shape_pixel_area": "Number of pixels the predicted region covers in the 2-D image.",
    "shape_area_mm2": "Physical area of the region, computed only when pixel spacing is known.",
    "shape_perimeter_px": "Length of the region boundary measured in pixels.",
    "shape_perimeter_mm": "Length of the region boundary in millimetres (requires spacing).",
    "shape_compactness": "How close the region is to a disc (1 = perfect disc). Sensitive to boundary detail.",
    "shape_sphericity": "Square root of compactness; 1 means the region is as round as possible.",
    "shape_elongation": "Ratio of the second to the first principal axis length of the region.",
    "shape_equivalent_diameter_px": "Diameter of the disc with the same area as the region, in pixels.",
    "shape_voxel_count": "Number of voxels in the region of interest.",
    "shape_volume_mm3": "Physical volume of the region, computed only when voxel spacing is known.",
    "shape_surface_area_mm2": "Estimated surface area of the segmented region.",
    "firstorder_mean": "Average intensity inside the predicted region.",
    "firstorder_median": "Median intensity inside the predicted region; less sensitive to outliers than the mean.",
    "firstorder_minimum": "Lowest intensity inside the region.",
    "firstorder_maximum": "Highest intensity inside the region.",
    "firstorder_variance": "Spread of intensities around the mean inside the region.",
    "firstorder_standard_deviation": "Square root of the intensity variance.",
    "firstorder_skewness": "Asymmetry of the intensity distribution inside the region.",
    "firstorder_kurtosis": "Tail weight of the intensity distribution (excess kurtosis, 0 = Gaussian).",
    "firstorder_energy": "Sum of squared intensities inside the region.",
    "firstorder_root_mean_squared": "Root mean square intensity inside the region.",
    "firstorder_interquartile_range": "Difference between the 75th and 25th intensity percentiles.",
    "firstorder_pixel_count": "Number of pixels contributing to the intensity statistics.",
    "glcm_contrast": "Grey-level co-occurrence contrast: how quickly intensities change between neighbouring pixels.",
    "glcm_dissimilarity": "Average absolute intensity difference between neighbouring pixels.",
    "glcm_homogeneity": "How uniform the local intensity pattern is (higher = more homogeneous texture).",
    "glcm_asm": "Angular second moment of the co-occurrence matrix; a measure of order in the texture.",
    "glcm_energy": "Square root of the angular second moment.",
    "glcm_correlation": "Linear dependency between neighbouring pixel intensities.",
    "glcm_entropy": "Information content of the co-occurrence matrix; higher means more heterogeneous texture.",
    "glcm_mean_intensity": "Mean grey level of the co-occurrence matrix.",
    "glrlm_short_run_emphasis": "Emphasis on short runs of identical intensity; higher in fine, granular textures.",
    "glrlm_long_run_emphasis": "Emphasis on long runs of identical intensity; higher in smooth textures.",
    "glrlm_run_percentage": "Fraction of the image covered by runs of each grey level.",
    "diagnostics_entropy": "Entropy of the intensity histogram inside the region.",
    "diagnostics_uniformity": "Uniformity of the intensity histogram; 1 means a single intensity value.",
    "diagnostics_roi_voxel_count": "Size of the region of interest, in pixels (2-D) or voxels (3-D).",
    "diagnostics_discretisation_levels": "Number of discrete grey levels used for the texture matrices.",
}
@dataclass
class RadiomicsResult:
    available: bool
    backend: str
    dimensionality: str
    feature_count: int
    features: dict[str, Any]
    categories: dict[str, list[str]]
    spacing_available: bool
    warnings: list[str] = field(default_factory=list)
    quality: dict = field(default_factory=dict)
    unavailable_reason: str | None = None
    backend_note: str = ""
    def to_dict(self) -> dict:
        return to_jsonable(
            {
                "available": self.available,
                "backend": self.backend,
                "backend_note": self.backend_note,
                "dimensionality": self.dimensionality,
                "feature_count": self.feature_count,
                "features": self.features,
                "categories": self.categories,
                "spacing_available": self.spacing_available,
                "warnings": self.warnings,
                "quality": self.quality,
                "unavailable_reason": self.unavailable_reason,
            }
        )
class RadiomicsService:
    def __init__(self, config):
        self.config = config
        self._pyradiomics = None
        self._pyradiomics_checked = False
    def _try_import_pyradiomics(self):
        if self._pyradiomics_checked:
            return self._pyradiomics
        self._pyradiomics_checked = True
        try:
            from radiomics import featureextractor
            self._pyradiomics = featureextractor
            logger.info("PyRadiomics is available and will be used as the primary backend")
        except Exception as exc:
            self._pyradiomics = None
            logger.info("PyRadiomics unavailable (%s); using the native extractor", type(exc).__name__)
        return self._pyradiomics
    def backend_status(self) -> dict:
        preference = (getattr(self.config, "RADIOMICS_BACKEND", "auto") or "auto").lower()
        available = self._try_import_pyradiomics() is not None
        selected = "native"
        if preference == "pyradiomics" and available:
            selected = "pyradiomics"
        elif preference == "auto" and available:
            selected = "pyradiomics"
        return {
            "preference": preference,
            "pyradiomics_available": available,
            "selected_backend": selected,
            "note": (
                "The native extractor is a NumPy implementation of shape, first-order, GLCM and "
                "GLRLM features. PyRadiomics, when installed, additionally provides GLDM, NGTDM "
                "and a full IBSI implementation. Both report the same geometry diagnostics so "
                "that results remain comparable."
            ),
        }
    def extract(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        *,
        spacing: tuple[float, ...] | None = None,
        mask_source: str = "predicted",
    ) -> RadiomicsResult:
        image = np.asarray(image, dtype=np.float32)
        mask = np.asarray(mask).astype(bool)
        dimensionality = "3D" if image.ndim == 3 else "2D"
        if image.shape != mask.shape:
            return RadiomicsResult(
                available=False,
                backend="none",
                dimensionality=dimensionality,
                feature_count=0,
                features={},
                categories={},
                spacing_available=bool(spacing),
                unavailable_reason=(
                    f"Image shape {list(image.shape)} and mask shape {list(mask.shape)} differ; "
                    "radiomics requires the mask to be defined on the image grid."
                ),
            )
        if not mask.any():
            return RadiomicsResult(
                available=False,
                backend="none",
                dimensionality=dimensionality,
                feature_count=0,
                features={},
                categories={},
                spacing_available=bool(spacing),
                unavailable_reason=(
                    "The predicted region is empty, so no region-based features can be computed. "
                    "An empty region is different from a measurement of zero."
                ),
            )
        settings = DiscretisationSettings(
            method=getattr(self.config, "RADIOMICS_DISCRETIZATION", "bin_count"),
            bin_count=int(getattr(self.config, "RADIOMICS_BIN_COUNT", 32)),
            bin_width=float(getattr(self.config, "RADIOMICS_BIN_WIDTH", 0.0) or 25.0),
        )
        backend = self.backend_status()["selected_backend"]
        warnings: list[str] = []
        if backend == "pyradiomics":
            try:
                return self._extract_pyradiomics(
                    image, mask, spacing=spacing, settings=settings, dimensionality=dimensionality,
                    mask_source=mask_source, warnings=warnings,
                )
            except Exception as exc:
                logger.warning("PyRadiomics extraction failed (%s); using the native extractor", exc)
                warnings.append(
                    f"PyRadiomics failed for this case ({type(exc).__name__}); the native extractor "
                    "was used instead."
                )
        return self._extract_native(
            image,
            mask,
            spacing=spacing,
            settings=settings,
            dimensionality=dimensionality,
            mask_source=mask_source,
            warnings=warnings,
        )
    def _extract_native(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        *,
        spacing,
        settings: DiscretisationSettings,
        dimensionality: str,
        mask_source: str,
        warnings: list[str],
    ) -> RadiomicsResult:
        features = extract_features(
            image, mask, spacing=spacing, settings=settings, include_texture=True
        )
        if "diagnostics_texture_warning" in features:
            warnings.append(features.pop("diagnostics_texture_warning"))
        if dimensionality == "2D":
            warnings.append(
                "This is a 2-D slice analysis. Areas are reported in pixels and no physical "
                "volume is computed, because a single slice has no voxel depth. Physical units "
                "require a 3-D volume together with its spacing."
            )
        elif not spacing:
            warnings.append(
                "No voxel spacing was supplied, so physical volume and surface area are not "
                "reported; measurements are given in voxel units."
            )
        ordered = self._order_features(features)
        categories = self._group_features(ordered)
        quality = self._quality(features, mask, dimensionality, mask_source)
        return RadiomicsResult(
            available=True,
            backend="native",
            dimensionality=dimensionality,
            feature_count=len(ordered),
            features=ordered,
            categories=categories,
            spacing_available=bool(spacing),
            warnings=warnings,
            quality=quality,
            backend_note=(
                "Native NumPy implementation: shape, first-order, GLCM and GLRLM features with "
                "IBSI-style naming."
            ),
        )
    def _extract_pyradiomics(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        *,
        spacing,
        settings: DiscretisationSettings,
        dimensionality: str,
        mask_source: str,
        warnings: list[str],
    ) -> RadiomicsResult:
        import SimpleITK as sitk
        from radiomics import featureextractor
        params = {
            "binWidth": settings.bin_width,
            "resampledPixelSpacing": None,
            "interpolator": "sitkBSpline",
            "normalize": False,
            "geometryTolerance": 1e-3,
        }
        extractor = featureextractor.RadiomicsFeatureExtractor(**params)
        extractor.enableAllFeatures()
        if spacing:
            kwargs = {"spacing": tuple(spacing)}
            image_sitk = sitk.GetImageFromArray(image.astype(np.float32))
            image_sitk.SetSpacing(tuple(reversed(spacing))) if image.ndim == 3 else image_sitk.SetSpacing(tuple(reversed(spacing)))
            mask_sitk = sitk.GetImageFromArray(mask.astype(np.uint8))
            mask_sitk.CopyInformation(image_sitk)
        else:
            kwargs = {}
            array = image[None, :, :] if image.ndim == 2 else image
            mask_array = mask[None, :, :] if mask.ndim == 2 else mask
            image_sitk = sitk.GetImageFromArray(array.astype(np.float32))
            mask_sitk = sitk.GetImageFromArray(mask_array.astype(np.uint8))
            mask_sitk.CopyInformation(image_sitk)
            warnings.append(
                "No spacing was available, so a nominal 1 x 1 voxel spacing was used for the "
                "PyRadiomics call; physical-size features are therefore expressed in pixels."
            )
        raw = extractor.execute(image_sitk, mask_sitk, **kwargs)
        features: dict[str, Any] = {}
        diagnostics: dict[str, Any] = {}
        for key, value in raw.items():
            if key.startswith("diagnostics_"):
                diagnostics[key] = value
                continue
            try:
                features[key] = float(value)
            except (TypeError, ValueError):
                continue
        features["diagnostics_dimensionality"] = dimensionality
        features["diagnostics_roi_voxel_count"] = int(mask.sum())
        features["diagnostics_spacing"] = list(spacing) if spacing else None
        features["diagnostics_spacing_available"] = bool(spacing)
        features["diagnostics_discretisation_method"] = "pyradiomics_bin_width"
        features["diagnostics_discretisation_levels"] = None
        for key, value in diagnostics.items():
            if isinstance(value, (int, float, str)) and not key.endswith("_Hash"):
                features.setdefault(f"pyradiomics_{key.replace('diagnostics_', '')}", value)
        ordered = self._order_features(features)
        return RadiomicsResult(
            available=True,
            backend="pyradiomics",
            dimensionality=dimensionality,
            feature_count=len(ordered),
            features=ordered,
            categories=self._group_features(ordered),
            spacing_available=bool(spacing),
            warnings=warnings,
            quality=self._quality(features, mask, dimensionality, mask_source),
            backend_note=f"PyRadiomics {getattr(featureextractor, '__version__', '')} with all feature classes enabled.",
        )
    @staticmethod
    def _order_features(features: dict[str, Any]) -> dict[str, Any]:
        buckets: dict[str, list[str]] = {family: [] for family in FAMILY_ORDER}
        for name in features:
            buckets.setdefault(categorise(name), []).append(name)
        ordered: dict[str, Any] = {}
        for family in FAMILY_ORDER:
            for name in sorted(buckets.get(family, [])):
                ordered[name] = features[name]
        for family, names in buckets.items():
            if family in FAMILY_ORDER:
                continue
            for name in sorted(names):
                ordered[name] = features[name]
        return to_jsonable(ordered)
    @staticmethod
    def _group_features(features: dict[str, Any]) -> dict[str, list[str]]:
        groups: dict[str, list[str]] = {}
        for name in features:
            groups.setdefault(categorise(name), []).append(name)
        return groups
    @staticmethod
    def _quality(features: dict, mask: np.ndarray, dimensionality: str, mask_source: str) -> dict:
        voxel_count = int(mask.sum())
        bbox = _bounding_box(mask)
        extent = None
        if bbox:
            extents = [high - low + 1 for low, high in bbox]
            extent = extents
        touching_border = any(low == 0 or high == size - 1 for (low, high), size in zip(bbox, mask.shape)) if bbox else False
        return {
            "mask_source": mask_source,
            "roi_voxel_count": voxel_count,
            "roi_bounding_box": bbox,
            "roi_extent": extent,
            "roi_touches_fov_border": bool(touching_border),
            "relative_roi_size": float(voxel_count / mask.size),
            "note": (
                "Shape and texture features describe the region produced by the segmentation "
                "model. If that region is inaccurate, every downstream value inherits the error, "
                "which is why segmentation quality should be reviewed before the numbers are read."
            ),
        }
def _bounding_box(mask: np.ndarray) -> list[list[int]] | None:
    coordinates = np.argwhere(mask)
    if coordinates.size == 0:
        return None
    return [
        [int(coordinates[:, axis].min()), int(coordinates[:, axis].max())]
        for axis in range(coordinates.shape[1])
    ]
def feature_tooltip(name: str) -> str:
    if name in FEATURE_DESCRIPTIONS:
        return FEATURE_DESCRIPTIONS[name]
    family = categorise(name)
    return FEATURE_CATEGORIES.get(family, "Radiomics feature.") + f" (feature: {name})"
