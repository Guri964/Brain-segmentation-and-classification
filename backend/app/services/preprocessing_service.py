from __future__ import annotations
import io
import logging
from dataclasses import dataclass, field
from typing import Sequence
import numpy as np
import torch
from ..errors import UnsupportedInputError, ValidationError
from ..utils.images import decode_image_bytes, resize_array
logger = logging.getLogger(__name__)
MODALITY_ORDER_BRATS = ("t1", "t1ce", "t2", "flair")
@dataclass
class PreprocessedInput:
    tensor: torch.Tensor
    pipeline: str
    channels: int
    height: int
    width: int
    normalisation: str
    display_stack: list[tuple[str, np.ndarray]]
    metadata: dict = field(default_factory=dict)
    def summary(self) -> dict:
        return {
            "pipeline": self.pipeline,
            "tensor_shape": list(self.tensor.shape),
            "channels": self.channels,
            "spatial_size": [self.height, self.width],
            "normalisation": self.normalisation,
            "metadata": self.metadata,
        }
# decoding
def _decode_single(raw: bytes, filename: str) -> tuple[np.ndarray, dict]:
    try:
        decoded = decode_image_bytes(raw)
    except Exception as exc:
        raise UnsupportedInputError(
            f"Could not decode '{filename}' as an image: {exc}",
            details={"filename": filename},
        ) from exc
    if decoded.array.ndim != 2:
        raise UnsupportedInputError(f"'{filename}' did not decode to a single channel.")
    stats = {
        "filename": filename,
        "source_format": decoded.source_format,
        "original_mode": decoded.original_mode,
        "original_size": list(decoded.original_size),
        "intensity_min": float(decoded.array.min()),
        "intensity_max": float(decoded.array.max()),
        "intensity_mean": float(decoded.array.mean()),
    }
    return decoded.array, stats
def assess_single_channel_acceptability(array: np.ndarray, filename: str) -> list[str]:
    warnings: list[str] = []
    if array.size == 0:
        warnings.append("Image is empty.")
    lo, hi = float(array.min()), float(array.max())
    if hi - lo < 1e-6:
        warnings.append("Image has no intensity variation (constant value).")
    unique = np.unique((array * 255).astype(np.uint8))
    if unique.size <= 4:
        warnings.append("Image appears to be near-binary or a synthetic mask rather than an MRI.")
    if min(array.shape) < 32:
        warnings.append("Image is smaller than 32 px on one axis; results will be unreliable.")
    coverage = float((array > 0.02).mean())
    if coverage > 0.98:
        warnings.append(
            "Nearly the entire image is bright; the input does not look like a cropped brain MRI."
        )
    if coverage < 0.01:
        warnings.append("Very little of the image carries signal; the input may be blank.")
    return warnings
def preprocess_single_channel(raw: bytes, filename: str, *, size: int = 512) -> PreprocessedInput:
    array, stats = _decode_single(raw, filename)
    warnings = assess_single_channel_acceptability(array, filename)
    resized = resize_array(array, (size, size), nearest=False)
    tensor = torch.from_numpy(resized.astype(np.float32))[None, None]
    stats["warnings"] = warnings
    return PreprocessedInput(
        tensor=tensor,
        pipeline="single_channel_512",
        channels=1,
        height=size,
        width=size,
        normalisation="[0,1] intensity scaling (image / 255), bilinear resize",
        display_stack=[("Grayscale MRI", resized.astype(np.float32))],
        metadata=stats,
    )
def _load_volume_from_npz(raw: bytes, filename: str) -> dict[str, np.ndarray]:
    try:
        with np.load(io.BytesIO(raw), allow_pickle=False) as payload:
            keys = list(payload.keys())
            arrays: dict[str, np.ndarray] = {}
            for key in keys:
                value = payload[key]
                if value.ndim == 3 and value.shape[-1] == 4:
                    for index, name in enumerate(MODALITY_ORDER_BRATS):
                        arrays[name] = value[..., index].astype(np.float32)
                    return arrays
                if value.ndim == 3 and value.shape[0] == 4:
                    for index, name in enumerate(MODALITY_ORDER_BRATS):
                        arrays[name] = value[index].astype(np.float32)
                    return arrays
            for key in keys:
                value = payload[key]
                if value.ndim != 2:
                    continue
                lowered = key.lower()
                name = next(
                    (m for m in MODALITY_ORDER_BRATS if m in lowered or lowered.startswith(m)), None
                )
                if name is None:
                    name = f"channel{len(arrays) + 1}"
                arrays[name] = value.astype(np.float32)
    except Exception as exc:
        raise UnsupportedInputError(f"Could not read '{filename}' as an .npz volume: {exc}") from exc
    return arrays
def _channel_signature(array: np.ndarray, bins: int = 16) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float32)
    height, width = arr.shape
    rows = np.linspace(0, height, bins + 1).astype(int)
    cols = np.linspace(0, width, bins + 1).astype(int)
    pooled = np.zeros((bins, bins), dtype=np.float32)
    for i in range(bins):
        for j in range(bins):
            block = arr[rows[i] : max(rows[i + 1], rows[i] + 1), cols[j] : max(cols[j + 1], cols[j] + 1)]
            pooled[i, j] = float(block.mean()) if block.size else 0.0
    return np.round(pooled, 4)
def _find_duplicate_channels(
    named_arrays: Sequence[tuple[str, np.ndarray]], *, tolerance: float = 1e-3
) -> list[tuple[str, str]]:
    signatures = [(name, _channel_signature(np.asarray(array))) for name, array in named_arrays]
    duplicates: list[tuple[str, str]] = []
    for i in range(len(signatures)):
        for j in range(i + 1, len(signatures)):
            name_a, sig_a = signatures[i]
            name_b, sig_b = signatures[j]
            if np.max(np.abs(sig_a - sig_b)) <= tolerance:
                duplicates.append((name_a, name_b))
    return duplicates
def normalise_modalities(arrays: Sequence[np.ndarray]) -> np.ndarray:
    if any(array is None for array in arrays):
        raise UnsupportedInputError("One or more modality channels could not be read.")
    stacked = np.stack([np.asarray(a, dtype=np.float32) for a in arrays], axis=-1)
    mean = float(stacked.mean())
    std = float(stacked.std())
    if std < 1e-5:
        raise ValidationError(
            "The four modality channels have (almost) no intensity variation; "
            "normalisation would divide by approximately zero."
        )
    return (stacked - mean) / std
def preprocess_four_channel(
    payloads: Sequence[tuple[str, bytes]],
    *,
    size: int = 256,
    modality_names: Sequence[str] | None = None,
) -> PreprocessedInput:
    if payloads is None or len(payloads) == 0:
        raise ValidationError("No four-channel input was provided.")
    warnings: list[str] = []
    if len(payloads) == 1 and payloads[0][0].lower().endswith(".npz"):
        filename, raw = payloads[0]
        arrays = _load_volume_from_npz(raw, filename)
        if len(arrays) != 4:
            raise UnsupportedInputError(
                f"'{filename}' contains {len(arrays)} usable 2-D channels; exactly 4 are required.",
                details={"found_channels": list(arrays.keys())},
            )
        names = modality_names or list(arrays.keys())
        ordered = [arrays[name] if name in arrays else list(arrays.values())[i] for i, name in enumerate(names)]
        source_stats = {
            name: {
                "original_shape": list(np.asarray(arrays.get(name, ordered[i])).shape),
                "min": float(np.asarray(ordered[i]).min()),
                "max": float(np.asarray(ordered[i]).max()),
            }
            for i, name in enumerate(names)
        }
    else:
        if len(payloads) != 4:
            raise UnsupportedInputError(
                f"The four-channel pipeline requires exactly 4 modality images; received {len(payloads)}.",
                details={"received": len(payloads), "expected_modalities": list(MODALITY_ORDER_BRATS)},
            )
        names = list(modality_names) if modality_names else list(MODALITY_ORDER_BRATS)
        ordered = []
        source_stats = {}
        for (filename, raw), name in zip(payloads, names):
            array, stats = _decode_single(raw, filename)
            ordered.append(array)
            source_stats[name] = stats
        duplicates = _find_duplicate_channels(list(zip(names, ordered)))
        if duplicates:
            raise UnsupportedInputError(
                "The four supplied channels are not all different: "
                + ", ".join(f"{a} == {b}" for a, b in duplicates)
                + ". A four-channel model must not be fed the same image four times, because the "
                "network would then be reading modalities that were never acquired.",
                details={
                    "duplicate_pairs": [list(pair) for pair in duplicates],
                    "modalities": [str(n).upper() for n in names],
                },
            )
    resized_stack = []
    display_stack: list[tuple[str, np.ndarray]] = []
    for name, array in zip(names, ordered):
        resized = resize_array(np.asarray(array, dtype=np.float32), (size, size), nearest=False)
        resized_stack.append(resized)
        display_stack.append((name.upper(), resized))
    normalised = normalise_modalities(resized_stack)
    chw = np.transpose(normalised, (2, 0, 1)).astype(np.float32)
    if chw.shape[0] != 4:
        raise ValidationError(f"Expected 4 channels after preprocessing, obtained {chw.shape[0]}.")
    tensor = torch.from_numpy(chw)[None]
    for name in names:
        stats = source_stats.get(name, {})
        lo = stats.get("intensity_min")
        hi = stats.get("intensity_max")
        if lo is not None and hi is not None and (hi - lo) < 1e-6:
            warnings.append(f"Channel '{name}' has no intensity variation.")
    metadata = {
        "modalities": [str(n).upper() for n in names],
        "channel_statistics": source_stats,
        "global_mean": float(normalised.mean()),
        "global_std": float(normalised.std()),
        "warnings": warnings,
    }
    return PreprocessedInput(
        tensor=tensor,
        pipeline="four_channel_256",
        channels=4,
        height=size,
        width=size,
        normalisation="global z-score over the (H,W,4) block (notebook cell 27), bilinear resize",
        display_stack=display_stack,
        metadata=metadata,
    )
def build_input_for_model(model_descriptor, payloads, *, modality_names=None) -> PreprocessedInput:
    spec = model_descriptor.input_spec
    if spec.channels == 1:
        if len(payloads) != 1:
            raise ValidationError(
                f"{model_descriptor.model_id} expects a single grayscale image, received "
                f"{len(payloads)} files."
            )
        filename, raw = payloads[0]
        return preprocess_single_channel(raw, filename, size=spec.height)
    if spec.channels == 4:
        return preprocess_four_channel(payloads, size=spec.height, modality_names=modality_names)
    raise UnsupportedInputError(
        f"Unsupported channel count {spec.channels} for {model_descriptor.model_id}."
    )
