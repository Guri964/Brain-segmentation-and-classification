from __future__ import annotations
import json
import os
import re
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import numpy as np
_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")
ALLOWED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
ALLOWED_VOLUME_SUFFIXES = {".npz", ".npy", ".h5", ".hdf5"}
def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
def new_analysis_id() -> str:
    return f"an-{datetime.now(timezone.utc).strftime('%Y%m%d')}-{uuid.uuid4().hex[:10]}"
def safe_filename(name: str, *, default: str = "upload") -> str:
    base = os.path.basename(name or default).strip() or default
    cleaned = _SAFE_NAME.sub("_", base)
    if len(cleaned) > 120:
        stem, dot, suffix = cleaned.rpartition(".")
        cleaned = f"{stem[:100]}{dot}{suffix[:16]}" if dot else cleaned[:120]
    return cleaned or default
def ensure_within(base: Path, candidate: Path) -> Path:
    base = Path(base).resolve()
    resolved = (base / candidate).resolve()
    if base not in resolved.parents and resolved != base:
        raise ValueError(f"path escapes storage root: {candidate}")
    return resolved
def write_bytes_atomic(path: Path, payload: bytes) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=str(path.parent), delete=False) as handle:
        handle.write(payload)
        temp_name = handle.name
    os.replace(temp_name, path)
    return path
def write_json(path: Path, payload: Any) -> Path:
    return write_bytes_atomic(Path(path), json.dumps(to_jsonable(payload), indent=2).encode("utf-8"))
def read_json(path: Path, default: Any = None) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return default
def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return None
    if hasattr(value, "item") and callable(getattr(value, "item")):
        try:
            return to_jsonable(value.item())
        except Exception:
            return str(value)
    return value
def human_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024
    return f"{size:.1f} GB"
def relative_to_root(path: Path, root: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(Path(root).resolve()))
    except ValueError:
        return str(path)
