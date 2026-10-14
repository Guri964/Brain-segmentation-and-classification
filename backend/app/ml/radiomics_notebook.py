"""The 46 handcrafted radiomics features of ``FINAL ENSEMBLE NB.ipynb``.

``extract_features_fast`` (notebook cell 26) is transcribed here so the served features are the
same numbers the notebook trained on.  The notebook used SciPy's ``skew``/``kurtosis`` and
``scipy.stats.entropy``; the arithmetic is re-implemented in NumPy so the serving path has no
extra dependency, and ``tests/test_stacked_ensemble.py`` checks every implementation against
SciPy's on the same inputs (they agree to floating-point precision).

Feature order
-------------
The notebook built a plain ``dict`` and called ``pd.DataFrame(feat_list)``, so the column order
is the insertion order of ``extract_features_fast``: for each of the four channels
``mean, std, skew, kurt, energy, entropy, p10, p25, p75, p90, range`` (11 x 4 = 44), followed by
``tumor_size`` and ``tumor_ratio`` (46 total).  ``feature_order()`` reproduces exactly that, and
``artifacts/radiomics_feature_names.csv`` from the notebook is the authority at serving time:
the service refuses to run when the two disagree.

Target semantics (read this before quoting any number)
------------------------------------------------------
The notebook's label vector is::

    labels = (np.array([m.sum() for m in masks]) > 100).astype(int)   # cell 7

that is, *an image-level tumour-size proxy* (does the mask have more than 100 positive pixels),
restricted to slices whose mask has at least 50 positive pixels.  Its classification report
prints those two classes under the names ``LGG``/``HGG`` (cell 31), but the radiologist grade
column that the notebook computed in cell 6 is never used for training.  The stacked ensemble
therefore does **not** predict glioma grade, malignancy or tumour type, and this module never
labels its output as such.  :data:`TARGET_SEMANTICS` is the sentence every surface must carry.
"""

from __future__ import annotations

import numpy as np

#: Names in the exact order the notebook's dict produced them.
_CHANNEL_STATS = (
    "mean",
    "std",
    "skew",
    "kurt",
    "energy",
    "entropy",
    "p10",
    "p25",
    "p75",
    "p90",
    "range",
)

CHANNELS = 4
FEATURE_COUNT = CHANNELS * len(_CHANNEL_STATS) + 2  # 44 + tumor_size + tumor_ratio

TARGET_SEMANTICS = (
    "The notebook's label vector is (mask pixel count > 100), an image-level tumour-size proxy, "
    "not the WHO grade. The names 'LGG'/'HGG' printed in its classification report are attached "
    "to that proxy, so this output must not be read as a grade, a malignancy verdict or a tumour "
    "type."
)

LEAKAGE_NOTE = (
    "Target leakage and split leakage, both properties of the notebook rather than of this "
    "service, apply to every number it reported. (1) Target leakage: the label is derived from "
    "the ground-truth mask while the meta-features include each segmentation member's maximum "
    "and mean predicted probability - statistics of the very same mask - so the target is "
    "partly reconstructible from the inputs. (2) Split leakage: the train/validation/test split "
    "is over slices, not patients, so near-duplicate slices of one case appear on both sides of "
    "it. The reported accuracy is therefore an upper bound produced under those conditions and "
    "is never restated here as the performance of this deployment."
)

FEATURE_GROUPS = {
    name: tuple(f"ch{channel}_{name}" for channel in range(CHANNELS))
    for name in _CHANNEL_STATS
}
FEATURE_GROUPS["tumor"] = ("tumor_size", "tumor_ratio")


def feature_order() -> list[str]:
    """The 46 feature names, in the notebook's column order."""

    names = [f"ch{channel}_{stat}" for channel in range(CHANNELS) for stat in _CHANNEL_STATS]
    names.extend(["tumor_size", "tumor_ratio"])
    return names


# ------------------------------------------------------------------ statistics (SciPy parity)


def _skew(values: np.ndarray) -> float:
    """``scipy.stats.skew(values)`` - biased g1 estimator, the SciPy default."""

    if values.size < 1:
        return float("nan")
    mean = values.mean()
    deviations = values - mean
    variance = float((deviations**2).mean())
    if variance <= 0:
        return 0.0
    return float((deviations**3).mean() / variance**1.5)


def _kurtosis(values: np.ndarray) -> float:
    """``scipy.stats.kurtosis(values)`` - Fisher (normal -> 0), biased, the SciPy defaults."""

    if values.size < 1:
        return float("nan")
    mean = values.mean()
    deviations = values - mean
    variance = float((deviations**2).mean())
    if variance <= 0:
        return 0.0
    return float((deviations**4).mean() / variance**2 - 3.0)


def _entropy(probabilities: np.ndarray) -> float:
    """``scipy.stats.entropy`` - normalise, then ``-sum(p log p)`` with ``0 log 0 = 0``."""

    values = np.asarray(probabilities, dtype=np.float64)
    total = values.sum()
    if total <= 0:
        return 0.0
    values = values / total
    nonzero = values > 0
    return float(-(values[nonzero] * np.log(values[nonzero])).sum())


# ------------------------------------------------------------------------------ extraction


def extract_features(image_hwc: np.ndarray, mask: np.ndarray) -> dict[str, float] | None:
    """The notebook's ``extract_features_fast``.

    ``image_hwc`` is ``(H, W, 4)`` float; ``mask`` is ``(H, W)`` or ``(H, W, 1)``.  Returns
    ``None`` when the mask is empty, exactly as the notebook does, so callers can never score a
    region that does not exist.
    """

    array = np.asarray(image_hwc, dtype=np.float32)
    if array.ndim != 3 or array.shape[-1] != CHANNELS:
        raise ValueError(
            f"expected a (H, W, {CHANNELS}) image, received {tuple(array.shape)}"
        )
    mask_array = np.asarray(mask)
    if mask_array.size == 0 or np.max(mask_array) == 0:
        return None
    if mask_array.ndim == 3:
        mask_array = mask_array[..., 0]
    region = mask_array > 0
    if not region.any():
        return None
    if region.shape != array.shape[:2]:
        raise ValueError(
            f"mask shape {region.shape} does not match the image {array.shape[:2]}"
        )

    features: dict[str, float] = {}
    for channel in range(array.shape[-1]):
        roi = array[..., channel][region].astype(np.float32)
        histogram, _ = np.histogram(roi, bins=32, density=True)
        histogram = histogram + 1e-9
        features[f"ch{channel}_mean"] = float(roi.mean())
        features[f"ch{channel}_std"] = float(roi.std())
        features[f"ch{channel}_skew"] = _skew(roi.astype(np.float64))
        features[f"ch{channel}_kurt"] = _kurtosis(roi.astype(np.float64))
        features[f"ch{channel}_energy"] = float((roi**2).mean())
        features[f"ch{channel}_entropy"] = _entropy(histogram.astype(np.float64))
        features[f"ch{channel}_p10"] = float(np.percentile(roi, 10))
        features[f"ch{channel}_p25"] = float(np.percentile(roi, 25))
        features[f"ch{channel}_p75"] = float(np.percentile(roi, 75))
        features[f"ch{channel}_p90"] = float(np.percentile(roi, 90))
        features[f"ch{channel}_range"] = float(roi.max() - roi.min())

    features["tumor_size"] = float(region.sum())
    features["tumor_ratio"] = float(region.mean())
    return features


def vector(features: dict[str, float], order: list[str]) -> np.ndarray:
    """Order a feature dict into the vector the scaler/CNN expect.

    Missing keys are a hard error rather than a silent zero: the notebook's own
    ``reindex(columns=...)`` would have filled them with 0, but at serving time a missing feature
    means the extraction changed and the model must not be fed a made-up value.
    """

    missing = [name for name in order if name not in features]
    if missing:
        raise ValueError(
            "feature extraction did not produce "
            + ", ".join(missing[:8])
            + (" ..." if len(missing) > 8 else "")
        )
    return np.asarray([features[name] for name in order], dtype=np.float32)


__all__ = [
    "CHANNELS",
    "FEATURE_COUNT",
    "FEATURE_GROUPS",
    "LEAKAGE_NOTE",
    "TARGET_SEMANTICS",
    "extract_features",
    "feature_order",
    "vector",
]
