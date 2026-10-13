#!/usr/bin/env python3
"""Install the artefacts produced by the Kaggle notebook into this project.

The notebook writes a self-contained directory (`brain_mri_artifacts/`, shipped as
`brain_mri_artifacts.zip`) and never touches the repository. This script is the step that puts
those files where the Flask application looks for them, and it is deliberately conservative:

* the checkpoints that ship with the repository (`segnet_model.pth`,
  `UNet-[16, 32, 64, 128, 256].pt`, `radiomics_cnn_model.pth`) are **never** overwritten -- the
  retrained files have their own names and are installed next to them;
* a destination that already exists with different content is only replaced with `--force`, and
  the replaced file is kept as `<name>.<timestamp>.bak`;
* everything installed is hashed and recorded in `<data-dir>/models/install_record.json`, and
  the notebook's provenance manifest is merged into `<data-dir>/models/artifacts_manifest.json`
  so the API and the UI can tell *which* model is being served and what it was trained on.

Usage
-----
    python scripts/install_trained_artifacts.py --source brain_mri_artifacts --data-dir data
    python scripts/install_trained_artifacts.py --source brain_mri_artifacts.zip --dry-run

The script prints the environment variables that make the API load the new artefacts, and it
finishes with a list of what was found, installed, skipped and missing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import configure_logging, utc_now  # noqa: E402

# checkpoint file in the export -> (destination name, environment variable that points at it)
CHECKPOINTS = {
    "segnet_brats_4ch_real_brats.pt": ("segnet_brats_4ch_real_brats.pt", "BRAIN_MODEL_SEGNET"),
    "raunet_brats_4ch.pt": ("raunet_brats_4ch.pt", "BRAIN_MODEL_RAUNET"),
    "unet_dynamic_512_real_brats.pt": ("unet_dynamic_512_real_brats.pt", "BRAIN_MODEL_UNET"),
}
# Checkpoints trained on the *second* dataset (multiclass tumour type). They are installed next to
# the segmentation ones, but nothing points the application at them: the API serves the radiomics
# classifier that matches its own input contract, and `models_index.json` records why.
EXTRA_CHECKPOINTS = {
    "brain_tumor_mri_cnn.pt": "brain_tumor_mri_cnn.pt",
}
# metric/provenance files in the export -> destination name under <data-dir>/models
MODEL_FILES = {
    "evaluation_results.json": "evaluation_results.json",
    "model_selection.json": "model_selection.json",
    "ensemble_calibration.json": "ensemble_calibration.json",
    "artifacts_manifest.json": "artifacts_manifest.json",
    "models_index.json": "models_index.json",
}
# artefact directories under <export>/data/models -> destination under <data-dir>/models
ARTEFACT_DIRECTORIES = {
    "classification": "classification",                      # BraTS HGG/LGG radiomics classifier (served)
    "classification_multiclass": "classification_multiclass",  # Brain Tumor MRI stacking ensemble (identified, not served)
}
REQUIRED_LAYOUT = ("checkpoints", "data/models")


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_source(source: Path, logger) -> tuple[Path, tempfile.TemporaryDirectory | None]:
    """Accept the extracted directory or the zip the notebook produces."""
    if source.is_dir():
        return source, None
    if source.is_file() and zipfile.is_zipfile(source):
        temporary = tempfile.TemporaryDirectory(prefix="brain-artifacts-")
        with zipfile.ZipFile(source) as archive:
            archive.extractall(temporary.name)
        inner = Path(temporary.name)
        # the archive may or may not contain a single top-level directory
        entries = [item for item in inner.iterdir() if item.is_dir()]
        if len(entries) == 1 and not (inner / "checkpoints").exists():
            inner = entries[0]
        logger.info("Extracted %s to %s", source, inner)
        return inner, temporary
    raise SystemExit(f"{source} is neither a directory nor a zip archive")


def install_file(source: Path, destination: Path, *, force: bool, dry_run: bool, logger) -> dict[str, Any]:
    """Copy one file, refusing to silently replace different content."""
    record: dict[str, Any]
    if not source.exists():
        return {"file": source.name, "state": "missing"}
    digest = sha256_of(source)
    if destination.exists():
        existing = sha256_of(destination)
        if existing == digest:
            return {
                "file": destination.name,
                "state": "already_installed",
                "destination": str(destination),
                "sha256": digest,
            }
        if not force:
            logger.warning(
                "%s exists with different content (%s); not replacing it. Re-run with --force to "
                "keep a backup and install this one.",
                destination,
                existing[:12],
            )
            return {
                "file": destination.name,
                "state": "kept_existing",
                "destination": str(destination),
                "existing_sha256": existing,
                "source_sha256": digest,
            }
        backup = destination.with_suffix(destination.suffix + f".{utc_now().replace(':', '')}.bak")
        if not dry_run:
            shutil.copy2(destination, backup)
        record = {"backup": str(backup)}
    else:
        record = {}
    if not dry_run:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    return {
        "file": destination.name,
        "state": "installed",
        "destination": str(destination),
        "source": str(source),
        "sha256": digest,
        "bytes": source.stat().st_size,
        **record,
    }


def install_tree(source: Path, destination: Path, *, force: bool, dry_run: bool, logger) -> list[dict]:
    entries = []
    for path in sorted(source.rglob("*")):
        if not path.is_file():
            continue
        target = destination / path.relative_to(source)
        entries.append(install_file(path, target, force=force, dry_run=dry_run, logger=logger))
    return entries


def build_report(source: Path, data_dir: Path, *, force: bool, dry_run: bool, logger) -> dict:
    models_dir = data_dir / "models"
    report: dict[str, Any] = {
        "generated_at": utc_now(),
        "source": str(source.resolve()),
        "data_dir": str(data_dir.resolve()),
        "dry_run": dry_run,
        "installed": [],
        "missing": [],
        "environment": {},
    }

    for name, (target_name, env_var) in CHECKPOINTS.items():
        origin = source / "checkpoints" / name
        if not origin.exists():
            report["missing"].append(f"checkpoints/{name}")
            continue
        entry = install_file(origin, models_dir / target_name, force=force, dry_run=dry_run, logger=logger)
        entry["model_id"] = target_name.replace(".pt", "")
        entry["environment_variable"] = env_var
        report["installed"].append(entry)
        report["environment"][env_var] = str((models_dir / target_name).resolve())

    for name, target_name in EXTRA_CHECKPOINTS.items():
        origin = source / "checkpoints" / name
        if not origin.exists():
            continue
        entry = install_file(origin, models_dir / target_name, force=force, dry_run=dry_run, logger=logger)
        entry["model_id"] = target_name.replace(".pt", "")
        entry["note"] = "trained on the second dataset (multiclass tumour type); the API does not serve it"
        report["installed"].append(entry)

    for name, target_name in MODEL_FILES.items():
        origin = source / "data" / "models" / name
        if not origin.exists():
            report["missing"].append(f"data/models/{name}")
            continue
        report["installed"].append(
            install_file(origin, models_dir / target_name, force=force, dry_run=dry_run, logger=logger)
        )

    for source_name, target_name in ARTEFACT_DIRECTORIES.items():
        artefact_source = source / "data" / "models" / source_name
        if not (artefact_source.is_dir() and any(artefact_source.iterdir())):
            report["missing"].append(f"data/models/{source_name}")
            continue
        entries = install_tree(
            artefact_source,
            models_dir / target_name,
            force=force,
            dry_run=dry_run,
            logger=logger,
        )
        report["installed"].extend(entries)
        root = models_dir / target_name
        versions = set()
        for item in entries:
            destination = item.get("destination")
            if not destination:
                continue
            try:
                versions.add(Path(destination).relative_to(root).parts[0])
            except (ValueError, IndexError):
                continue
        report[f"{target_name}_versions"] = sorted(versions)
        if target_name == "classification":
            # The application's default artefact directory is <data-dir>/models/classification, so
            # no environment variable is needed; recording it makes the default explicit.
            report["environment"]["BRAIN_CLASSIFIER_DIR"] = str(root.resolve())

    reports_source = source / "reports"
    if reports_source.is_dir() and any(reports_source.iterdir()):
        report["installed"].extend(
            install_tree(reports_source, models_dir / "reports", force=force, dry_run=dry_run, logger=logger)
        )

    provenance_path = source / "data" / "models" / "artifacts_manifest.json"
    if provenance_path.exists():
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        classification_dataset = provenance.get("classification_dataset") or {}
        report["provenance"] = {
            "produced_by": provenance.get("produced_by"),
            "generated_at": provenance.get("generated_at"),
            "synthetic": (provenance.get("dataset") or {}).get("synthetic"),
            "dataset": {
                "name": (provenance.get("dataset") or {}).get("name"),
                "synthetic": (provenance.get("dataset") or {}).get("synthetic"),
                "split_method": (provenance.get("dataset") or {}).get("split_method"),
            },
            "classification_dataset": {
                "name": classification_dataset.get("name"),
                "synthetic": classification_dataset.get("synthetic"),
                "classes": classification_dataset.get("classes"),
                "split_source": classification_dataset.get("split_source"),
            },
            "models": {
                model_id: {
                    "state": entry.get("state"),
                    "task": entry.get("task", "segmentation"),
                    "trained_on": entry.get("trained_on"),
                    "evaluated_on": entry.get("evaluated_on"),
                    "classes": entry.get("classes"),
                }
                for model_id, entry in (provenance.get("models") or {}).items()
            },
        }

    index_path = source / "data" / "models" / "models_index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        report["artefacts"] = [
            {
                "name": item.get("name"),
                "task": item.get("task"),
                "label_space": item.get("label_space"),
                "dataset": item.get("dataset"),
                "dataset_synthetic": item.get("dataset_synthetic"),
                "state": item.get("state"),
                "served_by_application": item.get("served_by_application"),
            }
            for item in (index.get("artefacts") or [])
        ]
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, required=True, help="brain_mri_artifacts directory or .zip")
    parser.add_argument("--data-dir", type=Path, default=None, help="project data directory (default: ./data)")
    parser.add_argument("--dry-run", action="store_true", help="report what would be installed, without writing")
    parser.add_argument("--force", action="store_true", help="replace different existing files, keeping a backup")
    parser.add_argument("--json", type=Path, default=None, help="also write the report to this JSON file")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    logger = configure_logging(args.verbose)
    if not args.source.exists():
        raise SystemExit(f"{args.source} does not exist")

    source, temporary = resolve_source(args.source, logger)
    data_dir = args.data_dir or Path("data")

    missing_layout = [item for item in REQUIRED_LAYOUT if not (source / item).exists()]
    if missing_layout:
        raise SystemExit(
            f"{source} does not look like the notebook's export: {', '.join(missing_layout)} missing. "
            "Extract brain_mri_artifacts.zip and pass its directory."
        )

    report = build_report(source, data_dir, force=args.force, dry_run=args.dry_run, logger=logger)
    if not args.dry_run:
        record_path = data_dir / "models" / "install_record.json"
        record_path.parent.mkdir(parents=True, exist_ok=True)
        record_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        report["record"] = str(record_path)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if report.get("artefacts"):
        logger.info("")
        logger.info("%-28s | %-26s | %-38s | %s", "artefact", "task", "label space", "served by the API")
        for item in report["artefacts"]:
            logger.info(
                "%-28s | %-26s | %-38s | %s",
                item["name"],
                item["task"],
                item["label_space"],
                "yes" if item["served_by_application"] else "no",
            )

    logger.info("")
    logger.info("%-42s | %-17s | %s", "file", "state", "destination")
    for entry in report["installed"]:
        logger.info(
            "%-42s | %-17s | %s",
            entry["file"],
            entry["state"],
            entry.get("destination") or entry.get("backup") or "",
        )
    for name in report["missing"]:
        logger.info("%-42s | %-17s | %s", name, "not in export", "rerun the notebook section that writes it")
    if report.get("provenance"):
        provenance = report["provenance"]
        logger.info("")
        logger.info("provenance: produced by %s at %s (synthetic=%s)", provenance["produced_by"], provenance["generated_at"], provenance["synthetic"])
        for model_id, entry in provenance["models"].items():
            logger.info(
                "  %-32s %-22s trained_on=%s evaluated_on=%s",
                model_id,
                entry["state"],
                entry["trained_on"],
                entry["evaluated_on"],
            )
    logger.info("")
    if report["environment"]:
        logger.info("Environment for the Flask application (export these before starting it):")
        for name, value in sorted(report["environment"].items()):
            logger.info("  %s=%s", name, value)
    else:
        logger.info("Nothing was installed, so no environment variable was produced.")
    if args.dry_run:
        logger.info("Dry run: nothing was written.")
    else:
        logger.info("Install record written to %s", report.get("record"))
    logger.info(
        "Next: python scripts/check_models.py --force   (loads every registered checkpoint and reports "
        "which ones are ready)"
    )
    if temporary is not None:
        temporary.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
