from __future__ import annotations
import argparse
import hashlib
import json
import logging
import os
import random
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence
import numpy as np
REPO_ROOT = Path(__file__).resolve().parent.parent
BACKEND_ROOT = REPO_ROOT / "backend"
DEFAULT_DATA_DIR = REPO_ROOT / "data"
DEFAULT_MODELS_DIR = DEFAULT_DATA_DIR / "models"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))
def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
def configure_logging(verbose: bool = False, log_file: Path | None = None) -> logging.Logger:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )
    return logging.getLogger("scripts")
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
def code_revision() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"
def file_digest(path: Path, limit: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        digest.update(handle.read(limit))
    return digest.hexdigest()[:16]
@dataclass
class Case:
    case_id: str
    patient_id: str
    images: list[Path]
    mask: Path | None = None
    label: int | None = None
    options: dict[str, Any] = field(default_factory=dict)
    @property
    def channels(self) -> int:
        return len(self.images)
    @property
    def source_kind(self) -> str:
        return str(self.options.get("kind") or "image_files")
    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "patient_id": self.patient_id,
            "images": [str(path) for path in self.images],
            "mask": str(self.mask) if self.mask else None,
            "label": self.label,
            "options": self.options,
        }
@dataclass
class Dataset:
    name: str
    version: str
    root: Path
    cases: list[Case]
    splits: dict[str, list[str]]
    warnings: list[str] = field(default_factory=list)
    def cases_for(self, split: str) -> list[Case]:
        wanted = set(self.splits.get(split, []))
        return [case for case in self.cases if case.case_id in wanted]
    def summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "root": str(self.root),
            "cases": len(self.cases),
            "patients": len({case.patient_id for case in self.cases}),
            "channels": sorted({case.channels for case in self.cases}),
            "with_masks": sum(1 for case in self.cases if case.mask is not None),
            "with_labels": sum(1 for case in self.cases if case.label is not None),
            "split_sizes": {name: len(ids) for name, ids in self.splits.items()},
            "split_strategy": "patient_level",
            "source_kinds": sorted({case.source_kind for case in self.cases}),
            "channel_orders": sorted(
                {",".join(case.options.get("channel_order") or []) for case in self.cases if case.options.get("channel_order")}
            ),
            "labelled": sum(1 for case in self.cases if case.label is not None),
            "warnings": self.warnings,
        }
def load_manifest(path: Path, *, require_patient_ids: bool = True) -> Dataset:
    if not path.exists():
        raise SystemExit(
            f"Dataset manifest not found: {path}\n"
            "Create one first - see docs/DATASET.md for the exact format and for how to "
            "generate it from a BraTS-style directory tree."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    root = Path(payload.get("root") or path.parent).expanduser().resolve()
    cases: list[Case] = []
    missing_patient_ids: list[str] = []
    for entry in payload.get("cases", []):
        case_id = str(entry.get("case_id") or entry.get("id") or "").strip()
        if not case_id:
            raise SystemExit(f"A case entry in {path} has no case_id: {entry}")
        patient_id = str(entry.get("patient_id") or entry.get("patient") or "").strip()
        if not patient_id:
            missing_patient_ids.append(case_id)
            patient_id = case_id
        images = [Path(item) for item in (entry.get("images") or [entry.get("image")]) if item]
        if not images:
            raise SystemExit(f"Case {case_id} lists no images.")
        images = [item if item.is_absolute() else (root / item) for item in images]
        mask_raw = entry.get("mask") or entry.get("segmentation")
        mask = Path(mask_raw) if mask_raw else None
        if mask is not None and not mask.is_absolute():
            mask = root / mask
        label = _coerce_label(entry.get("label"))
        options = dict(entry.get("options") or {})
        for key in ("mask_semantics", "mask_channel_labels", "channel_order", "spacing", "kind"):
            if key in entry and key not in options:
                options[key] = entry[key]
        cases.append(
            Case(
                case_id=case_id,
                patient_id=patient_id,
                images=images,
                mask=mask,
                label=label,
                options=options,
            )
        )
    if not cases:
        raise SystemExit(f"The manifest {path} contains no cases.")
    if missing_patient_ids and require_patient_ids:
        raise SystemExit(
            "These cases have no patient_id: "
            + ", ".join(missing_patient_ids[:5])
            + (" ..." if len(missing_patient_ids) > 5 else "")
            + "\nPatient-level splitting is mandatory: slices from one patient must never appear "
            "in two splits. Add patient_id to every case, or pass --allow-slice-splits to record "
            "the shortcut explicitly as a limitation (not recommended)."
        )
    warnings: list[str] = []
    missing = [
        str(case.images[0])
        for case in cases
        if not all(item.exists() for item in case.images)
    ]
    if missing:
        warnings.append(
            f"{len(missing)} case(s) reference files that do not exist yet, for example {missing[0]}. "
            "Only existing cases are used."
        )
        cases = [case for case in cases if all(item.exists() for item in case.images)]
    channels = {case.channels for case in cases}
    if len(channels) > 1:
        warnings.append(
            "The manifest mixes channel counts "
            f"({sorted(channels)}); train one architecture per channel count."
        )
    splits = _assign_splits(cases, payload.get("splits"))
    return Dataset(
        name=str(payload.get("name") or path.stem),
        version=str(payload.get("version") or "unversioned"),
        root=root,
        cases=cases,
        splits=splits,
        warnings=warnings,
    )
def _coerce_label(value: Any) -> Any:
    if value is None or str(value).strip() == "":
        return None
    text = str(value).strip()
    try:
        return int(float(text))
    except ValueError:
        return text
def _assign_splits(cases: Sequence[Case], declared: dict[str, Iterable[str]] | None) -> dict[str, list[str]]:
    if declared:
        assigned: dict[str, list[str]] = {name: [str(item) for item in ids] for name, ids in declared.items()}
        known = {case.case_id for case in cases}
        for name, ids in assigned.items():
            unknown = [item for item in ids if item not in known]
            if unknown:
                raise SystemExit(f"Split '{name}' lists unknown case ids: {unknown[:3]}")
        return assigned
    raise SystemExit(
        "The manifest does not declare splits. Run scripts/make_splits.py to create a "
        "patient-level train/validation/test split (recommended), or add a 'splits' object "
        "to the manifest yourself."
    )
def require_splits(dataset: Dataset) -> None:
    for name in ("train",):
        if not dataset.splits.get(name):
            raise SystemExit(f"The manifest declares no '{name}' split; nothing to train on.")
def save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    import torch
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
def write_run_record(directory: Path, record: dict[str, Any]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    name = record.get("run_id") or record.get("model_id") or "run"
    path = directory / f"{name}-{stamp}.json"
    path.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    return path
def resolve_data_dir(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()
    env = os.environ.get("BRAIN_DATA_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return DEFAULT_DATA_DIR
def torch_device(preference: str = "auto"):
    import torch
    if preference == "cpu":
        return torch.device("cpu")
    if preference == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if preference == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cpu")
def describe_environment() -> dict[str, Any]:
    import platform
    payload: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "revision": code_revision(),
    }
    try:
        import torch
        payload.update(
            {
                "torch": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "torch_threads": torch.get_num_threads(),
            }
        )
    except Exception as exc:
        payload["torch"] = f"unavailable ({exc})"
    try:
        import psutil
        payload["memory_gb"] = round(psutil.virtual_memory().total / (1024**3), 2)
    except Exception:
        payload["memory_gb"] = None
    return payload
def base_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data", type=Path, required=True, help="Path to the dataset manifest (JSON)")
    parser.add_argument("--data-dir", default=None, help="Where artefacts are written (defaults to BRAIN_DATA_DIR or ./data)")
    parser.add_argument("--seed", type=int, default=20240101, help="Random seed recorded with the run")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=12, help="Early-stopping patience in epochs")
    parser.add_argument("--scheduler", choices=["none", "cosine", "plateau"], default="plateau")
    parser.add_argument("--resume", type=Path, default=None, help="Checkpoint to resume from")
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--max-cases", type=int, default=None, help="Cap the number of cases (smoke runs)")
    parser.add_argument(
        "--lazy-loading",
        action="store_true",
        help=(
            "Read each case from disk per batch instead of holding the whole split in memory. "
            "Use this for BraTS-scale datasets; it also caps the validation pass with --val-cases."
        ),
    )
    parser.add_argument("--val-cases", type=int, default=64, help="Cases used per validation pass in lazy mode")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true", help="Validate data and configuration, then stop")
    parser.add_argument("--verbose", action="store_true")
    return parser
def report_dataset(logger: logging.Logger, dataset: Dataset) -> None:
    summary = dataset.summary()
    logger.info(
        "Dataset %s (%s): %s cases, %s patients, splits %s",
        summary["name"],
        summary["version"],
        summary["cases"],
        summary["patients"],
        summary["split_sizes"],
    )
    for warning in summary["warnings"]:
        logger.warning(warning)
def iter_batches(items: Sequence[Any], size: int) -> Iterator[Sequence[Any]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]
