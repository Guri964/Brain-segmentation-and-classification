from __future__ import annotations
import base64
import io
from dataclasses import dataclass
from typing import Iterable, Sequence
import numpy as np
from PIL import Image, ImageDraw, ImageFont
PNG_MIME = "image/png"
@dataclass(frozen=True)
class DecodedImage:
    array: np.ndarray
    original_mode: str
    original_size: tuple[int, int]
    source_format: str
def decode_image_bytes(raw: bytes) -> DecodedImage:
    with Image.open(io.BytesIO(raw)) as img:
        source_format = (img.format or "unknown").upper()
        original_mode = img.mode
        original_size = img.size
        if original_mode in ("I;16", "I;16B", "I", "F"):
            arr = np.asarray(img).astype(np.float32)
            finite = np.isfinite(arr)
            if finite.any():
                lo, hi = float(arr[finite].min()), float(arr[finite].max())
                arr = (arr - lo) / (hi - lo) if hi > lo else np.zeros_like(arr)
            else:
                arr = np.zeros_like(arr)
            arr = np.nan_to_num(arr, nan=0.0)
        else:
            arr = np.asarray(img.convert("L")).astype(np.float32) / 255.0
    return DecodedImage(
        array=arr.astype(np.float32),
        original_mode=original_mode,
        original_size=original_size,
        source_format=source_format,
    )
def decode_gray(raw: bytes) -> np.ndarray:
    return decode_image_bytes(raw).array
def resize_array(array: np.ndarray, size: tuple[int, int], *, nearest: bool = False) -> np.ndarray:
    height, width = size
    mode = Image.Resampling.NEAREST if nearest else Image.Resampling.BILINEAR
    if array.dtype == np.bool_:
        array = array.astype(np.uint8) * 255
    if np.issubdtype(array.dtype, np.floating):
        img = Image.fromarray(np.clip(array * 255.0, 0, 255).astype(np.uint8), mode="L")
        out = np.asarray(img.resize((width, height), mode), dtype=np.float32) / 255.0
        return out
    img = Image.fromarray(array.astype(np.uint8), mode="L")
    return np.asarray(img.resize((width, height), mode))
def to_uint8(array: np.ndarray, *, percentile_clip: tuple[float, float] | None = None) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float32)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros(arr.shape, dtype=np.uint8)
    arr = np.where(finite, arr, np.nan)
    if percentile_clip is not None:
        lo, hi = np.nanpercentile(arr, percentile_clip)
    else:
        lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr))
    if hi - lo < 1e-12:
        return np.zeros(arr.shape, dtype=np.uint8)
    scaled = (np.nan_to_num(arr, nan=lo) - lo) / (hi - lo)
    return (np.clip(scaled, 0.0, 1.0) * 255.0).astype(np.uint8)
def apply_colormap(probability: np.ndarray, name: str = "jet") -> np.ndarray:
    p = np.clip(np.asarray(probability, dtype=np.float32), 0.0, 1.0)
    if name == "viridis":
        stops = np.array(
            [
                [68, 1, 84],
                [65, 68, 135],
                [42, 120, 142],
                [34, 168, 132],
                [122, 209, 81],
                [253, 231, 37],
            ],
            dtype=np.float32,
        )
    else:
        stops = np.array(
            [
                [0, 0, 131],
                [0, 60, 170],
                [5, 255, 255],
                [255, 255, 0],
                [250, 0, 0],
                [128, 0, 0],
            ],
            dtype=np.float32,
        )
    idx = p * (len(stops) - 1)
    lo = np.floor(idx).astype(int)
    hi = np.clip(lo + 1, 0, len(stops) - 1)
    frac = (idx - lo)[..., None]
    rgb = stops[lo] * (1 - frac) + stops[hi] * frac
    return rgb.astype(np.uint8)
def mask_to_rgb(mask: np.ndarray, colour: Sequence[int] = (255, 64, 64)) -> np.ndarray:
    rgb = np.zeros(mask.shape + (3,), dtype=np.uint8)
    rgb[mask.astype(bool)] = np.array(colour, dtype=np.uint8)
    return rgb
def make_overlay(
    background: np.ndarray,
    mask: np.ndarray,
    *,
    colour: Sequence[int] = (255, 64, 64),
    alpha: float = 0.45,
) -> np.ndarray:
    gray = to_uint8(background)
    if gray.shape != mask.shape:
        mask = resize_array(mask.astype(np.float32), gray.shape, nearest=True) > 0.5
    base = np.repeat(gray[..., None], 3, axis=2).astype(np.float32)
    overlay = base.copy()
    colour_arr = np.array(colour, dtype=np.float32)
    hit = mask.astype(bool)
    overlay[hit] = base[hit] * (1 - alpha) + colour_arr * alpha
    return overlay.astype(np.uint8)
def make_heatmap_overlay(
    background: np.ndarray,
    heatmap: np.ndarray,
    *,
    alpha: float = 0.5,
    colormap: str = "jet",
) -> np.ndarray:
    gray = to_uint8(background)
    hm = np.asarray(heatmap, dtype=np.float32)
    if hm.shape != gray.shape:
        hm = resize_array(np.clip(hm, 0, 1), gray.shape)
    rgb = apply_colormap(np.clip(hm, 0, 1), colormap).astype(np.float32)
    base = np.repeat(gray[..., None], 3, axis=2).astype(np.float32)
    blended = base * (1 - alpha) + rgb * alpha
    return np.clip(blended, 0, 255).astype(np.uint8)
def encode_png(array: np.ndarray) -> bytes:
    arr = np.asarray(array)
    if arr.dtype != np.uint8:
        arr = to_uint8(arr)
    if arr.ndim == 2:
        img = Image.fromarray(arr, mode="L")
    else:
        img = Image.fromarray(arr.astype(np.uint8), mode="RGB")
    buffer = io.BytesIO()
    img.save(buffer, format="PNG", optimize=False)
    return buffer.getvalue()
def encode_data_url(array: np.ndarray) -> str:
    payload = base64.b64encode(encode_png(array)).decode("ascii")
    return f"data:{PNG_MIME};base64,{payload}"
def build_montage(
    panels: Sequence[tuple[str, np.ndarray]],
    *,
    panel_size: tuple[int, int] = (256, 256),
    padding: int = 8,
    label_height: int = 20,
    background: int = 18,
) -> np.ndarray:
    if not panels:
        raise ValueError("build_montage requires at least one panel")
    height, width = panel_size
    cols = len(panels)
    canvas_h = height + label_height + 2 * padding
    canvas_w = cols * width + (cols + 1) * padding
    canvas = np.full((canvas_h, canvas_w, 3), background, dtype=np.uint8)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    for index, (label, panel) in enumerate(panels):
        arr = np.asarray(panel)
        if arr.ndim == 2:
            arr = np.repeat(arr[..., None], 3, axis=2)
        if arr.shape[0] != height or arr.shape[1] != width:
            img = Image.fromarray(arr.astype(np.uint8))
            arr = np.asarray(img.resize((width, height), Image.Resampling.BILINEAR))
            if arr.ndim == 2:
                arr = np.repeat(arr[..., None], 3, axis=2)
        x0 = padding + index * (width + padding)
        y0 = padding
        canvas[y0 : y0 + height, x0 : x0 + width] = arr.astype(np.uint8)
        if font is not None:
            pil = Image.fromarray(canvas)
            draw = ImageDraw.Draw(pil)
            draw.text((x0 + 4, y0 + height + 4), label, fill=(230, 230, 230), font=font)
            canvas = np.asarray(pil)
    return canvas
def tile_pair(left: np.ndarray, right: np.ndarray, *, gap: int = 8) -> np.ndarray:
    a, b = np.asarray(left), np.asarray(right)
    if a.ndim == 2:
        a = np.repeat(a[..., None], 3, axis=2)
    if b.ndim == 2:
        b = np.repeat(b[..., None], 3, axis=2)
    if a.shape[0] != b.shape[0]:
        img = Image.fromarray(b.astype(np.uint8))
        b = np.asarray(img.resize((a.shape[1], a.shape[0]), Image.Resampling.BILINEAR))
        if b.ndim == 2:
            b = np.repeat(b[..., None], 3, axis=2)
    canvas = np.full((a.shape[0], a.shape[1] + b.shape[1] + gap, 3), 18, dtype=np.uint8)
    canvas[:, : a.shape[1]] = a.astype(np.uint8)
    canvas[:, a.shape[1] + gap :] = b.astype(np.uint8)
    return canvas
def normalise_blockwise(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float32)
    mean = float(arr.mean())
    std = float(arr.std())
    if std < 1e-5:
        return np.zeros_like(arr)
    return (arr - mean) / std
__all__ = [
    "DecodedImage",
    "apply_colormap",
    "build_montage",
    "decode_gray",
    "decode_image_bytes",
    "encode_data_url",
    "encode_png",
    "make_heatmap_overlay",
    "make_overlay",
    "mask_to_rgb",
    "normalise_blockwise",
    "resize_array",
    "tile_pair",
    "to_uint8",
]
