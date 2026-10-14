from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import (
    BACKEND_ROOT,
    configure_logging,
    describe_environment,
    load_manifest,
    resolve_data_dir,
    seed_everything,
    torch_device,
    utc_now,
    write_run_record,
)
from _training import TrainingConfig, load_case_arrays

sys.path.insert(0, str(BACKEND_ROOT))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Extract radiomics features for every case of a manifest, writing one labelled row "
            "per case. Rows use the same feature schema as the running application, so a model "
            "trained on this table can be served without re-ordering features."
        )
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="Feature table to write (CSV)")
    parser.add_argument("--split", default=None, choices=["train", "validation", "test"], help="Only this split")
    parser.add_argument("--model-id", default=None, help="Use this checkpoint's predicted mask instead of the reference mask")
    parser.add_argument("--image-channel", type=int, default=0, help="Channel used as the radiomics image")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--min-foreground-pixels", type=int, default=16)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--mask-source", default="reference", choices=["reference", "predicted"])
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--seed", type=int, default=20240101)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    logger = configure_logging(args.verbose)
    seed_everything(args.seed)
    dataset = load_manifest(args.data)
    cases = dataset.cases_for(args.split) if args.split else list(dataset.cases)
    if args.max_cases:
        cases = cases[: args.max_cases]
    if not cases:
        raise SystemExit("No case selected; check the manifest and --split.")

    from app.config import DevelopmentConfig
    from app.services.radiomics_service import RadiomicsService

    config = DevelopmentConfig
    radiomics = RadiomicsService(config)
    status = radiomics.backend_status()
    logger.info("Radiomics backend: %s (pyradiomics available: %s)", status["selected_backend"], status["pyradiomics_available"])
    if args.mask_source == "predicted" and not args.model_id:
        raise SystemExit("--mask-source predicted requires --model-id.")
    predictor = None
    if args.mask_source == "predicted":
        from app.ml.registry import ModelRegistry

        registry = ModelRegistry(config, device=torch_device(args.device))
        descriptor = registry.descriptors.get(args.model_id)
        if descriptor is None:
            raise SystemExit(
                f"'{args.model_id}' is not registered. Registered: {', '.join(registry.descriptors)}"
            )
        model = registry.load(args.model_id)
        predictor = (registry, descriptor, model)
        logger.info("Predicting masks with %s", args.model_id)

    import torch

    rows: list[dict] = []
    feature_names: list[str] = []
    skipped: list[dict] = []
    for index, case in enumerate(cases, start=1):
        try:
            image, reference_mask = load_case_arrays(case, args.image_size, max(1, args.image_channel + 1), log=None)
        except Exception as exc:
            skipped.append({"case_id": case.case_id, "reason": f"{type(exc).__name__}: {exc}"})
            continue
        if reference_mask is None and args.mask_source == "reference":
            skipped.append({"case_id": case.case_id, "reason": "no reference mask"})
            continue
        plane = image[min(args.image_channel, image.shape[0] - 1)]
        mask = reference_mask
        mask_source = "reference"
        if predictor is not None:
            registry, descriptor, model = predictor
            spec = descriptor.input_spec
            tensor_planes = []
            source_planes = image
            if spec.channels == 1:
                resized = _resize(plane, spec.height, spec.width)
                source_planes = [resized]
            else:
                source_planes = [plane] if image.shape[0] == spec.channels else image[: spec.channels]
                source_planes = [_resize(item, spec.height, spec.width) for item in source_planes]
                while len(source_planes) < spec.channels:
                    source_planes.append(np.zeros_like(source_planes[0]))
            stacked = np.stack(source_planes, axis=0).astype(np.float32)
            if spec.channels > 1:
                stacked = (stacked - float(stacked.mean())) / (float(stacked.std()) or 1.0)
            tensor_planes.append(stacked)
            tensor = torch.from_numpy(np.stack(tensor_planes).astype(np.float32))
            with torch.no_grad():
                output = model(tensor)
            if isinstance(output, (list, tuple)):
                output = output[0]
            probability = output
            probability = torch.sigmoid(probability) if descriptor.output_semantics == "logit" else probability
            predicted = (probability.numpy()[0, 0] > descriptor.fixed_threshold).astype(np.float32)
            mask = _resize(predicted, plane.shape[0], plane.shape[1])
            mask_source = f"predicted:{args.model_id}"
        foreground = int(mask.sum())
        if foreground < args.min_foreground_pixels:
            skipped.append(
                {
                    "case_id": case.case_id,
                    "reason": f"foreground {foreground} below --min-foreground-pixels",
                }
            )
            continue
        spacing = case.options.get("spacing")
        try:
            result = radiomics.extract(
                plane,
                mask,
                spacing=tuple(spacing) if spacing else None,
                mask_source=mask_source,
            )
        except Exception as exc:
            skipped.append({"case_id": case.case_id, "reason": f"{type(exc).__name__}: {exc}"})
            continue
        payload = result.to_dict()
        if not payload.get("available"):
            skipped.append({"case_id": case.case_id, "reason": payload.get("unavailable_reason") or "unavailable"})
            continue
        features = payload["features"]
        if not feature_names:
            feature_names = list(features)
        row = {
            "case_id": case.case_id,
            "patient_id": case.patient_id,
            "label": "" if case.label is None else case.label,
            "split": _split_of(dataset, case.case_id),
            "mask_source": mask_source,
            "foreground_pixels": foreground,
            "radiomics_backend": payload.get("backend"),
            "image_channel": args.image_channel,
        }
        for name in feature_names:
            value = features.get(name)
            row[name] = "" if value is None else value
        rows.append(row)
        if index % 25 == 0 or index == len(cases):
            logger.info("  %d/%d case(s) processed · %d usable", index, len(cases), len(rows))

    if not rows:
        raise SystemExit(
            "No usable case: every mask was empty, too small, or the extraction failed. "
            f"First reasons: {skipped[:3]}"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    data_dir = resolve_data_dir(args.data_dir)
    record = {
        "stage": "extract_radiomics",
        "status": "completed",
        "manifest": str(args.data),
        "split": args.split or "all",
        "mask_source": args.mask_source,
        "model_id": args.model_id,
        "image_channel": args.image_channel,
        "image_size": args.image_size,
        "cases_requested": len(cases),
        "cases_used": len(rows),
        "cases_skipped": len(skipped),
        "skipped_examples": skipped[:10],
        "feature_count": len(feature_names),
        "feature_schema": feature_names,
        "radiomics_backend": status["selected_backend"],
        "run_id": f"radiomics_{Path(args.out).stem}",
        "model_id": args.model_id or Path(args.out).stem,
        "output": str(args.out),
        "label_distribution": _label_distribution(rows),
        "environment": describe_environment(),
        "finished_at": utc_now(),
    }
    record_path = write_run_record(data_dir / "models" / "radiomics_runs", record)
    logger.info("Feature table written to %s (%d rows, %d features)", args.out, len(rows), len(feature_names))
    logger.info("Label distribution: %s", record["label_distribution"])
    logger.info("Run record: %s", record_path)
    if len(skipped):
        logger.warning("%d case(s) were skipped; the first reasons are in the run record.", len(skipped))
    return 0


def _split_of(dataset, case_id: str) -> str:
    for name, ids in dataset.splits.items():
        if case_id in ids:
            return name
    return "unassigned"


def _resize(array: np.ndarray, height: int, width: int) -> np.ndarray:
    from app.utils.images import resize_array

    return resize_array(np.asarray(array, dtype=np.float32), (height, width))


def _label_distribution(rows: list[dict]) -> dict:
    counts: dict[str, int] = {}
    for row in rows:
        key = str(row.get("label", ""))
        counts[key] = counts.get(key, 0) + 1
    return counts


if __name__ == "__main__":
    raise SystemExit(main())
