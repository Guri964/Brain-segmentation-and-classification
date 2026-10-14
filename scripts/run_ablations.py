from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (
    BACKEND_ROOT,
    code_revision,
    configure_logging,
    load_manifest,
    report_dataset,
    resolve_data_dir,
    seed_everything,
    utc_now,
)
from _training import TrainingConfig, prepare_split
sys.path.insert(0, str(BACKEND_ROOT))
MODALITY_LABELS = ["T1", "T1ce", "T2", "FLAIR"]
def main() -> int:
    parser = argparse.ArgumentParser(description="Ablate one component at a time on the validation split")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--models", required=True, help="Comma-separated model ids that share one input specification")
    parser.add_argument("--thresholds", default="0.3,0.5,0.7")
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20240101)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    logger = configure_logging(args.verbose)
    seed_everything(args.seed)
    dataset = load_manifest(args.data)
    report_dataset(logger, dataset)
    from app import create_app
    from app.services.preprocessing_service import preprocess_four_channel, preprocess_single_channel
    from app.utils.images import encode_png, to_uint8
    from app.utils.metrics import segmentation_metrics
    app = create_app(env="testing")
    services = app.extensions["brain_services"]
    registry = services["registry"]
    segmentation = services["segmentation"]
    members = [item.strip() for item in args.models.split(",")]
    for member in members:
        loaded = registry.load_model(member)
        if not loaded.is_ready:
            raise SystemExit(f"{member} is not usable: {loaded.message}")
        health = services["health"].check(member)
        if not health.responsive:
            raise SystemExit(f"{member} failed the reference check ({health.verdict}); ablations on it would be noise.")
    descriptor = registry.load_model(members[0]).descriptor
    config = TrainingConfig(
        model_id=members[0],
        architecture=descriptor.architecture,
        in_channels=descriptor.input_spec.channels,
        out_channels=1,
        spatial_size=descriptor.input_spec.height,
        normalisation=descriptor.input_spec.normalisation,
        seed=args.seed,
    )
    images, masks = prepare_split(dataset, "validation", config, logger, args.max_cases)
    logger.info("Ablating over %d validation case(s) with %s", len(images), ", ".join(members))
    if args.dry_run:
        logger.info("Dry run: %s case(s) would be evaluated.", len(images))
        return 0
    prepared = []
    for index in range(len(images)):
        spec = descriptor.input_spec
        if spec.channels == 1:
            prepared.append(preprocess_single_channel(encode_png(to_uint8(images[index][0])), f"case-{index}.png", size=spec.height))
        else:
            payloads = [
                (f"case-{index}_{label}.png", encode_png(to_uint8(images[index][channel])))
                for channel, label in enumerate(MODALITY_LABELS[: spec.channels])
            ]
            prepared.append(preprocess_four_channel(payloads, size=spec.height))
    thresholds = [float(item) for item in args.thresholds.split(",")]
    rows: list[dict] = []
    baseline = [segmentation.predict(members[0], item, threshold=0.5) for item in prepared]
    rows.append(_row("baseline", "single model, threshold 0.50, no TTA", masks, baseline, 0.5, logger, segmentation_metrics))
    rows.extend(
        _threshold_rows(masks, prepared, members[0], thresholds, segmentation, segmentation_metrics)
    )
    rows.append(_tta_row(masks, prepared, members[0], segmentation, segmentation_metrics, logger))
    if len(members) > 1:
        rows.append(_ensemble_row("ensemble_equal", "equal weights", members, prepared, masks, segmentation, segmentation_metrics, logger, weights=None))
        calibration = _load_calibration(resolve_data_dir(args.data_dir), members)
        if calibration is not None:
            rows.append(
                _ensemble_row(
                    "ensemble_calibrated",
                    f"validation-calibrated weights {calibration['weights']}",
                    members,
                    prepared,
                    masks,
                    segmentation,
                    segmentation_metrics,
                    logger,
                    weights=calibration["weights"],
                )
            )
        else:
            logger.warning("No calibration file for these members; the calibrated row is omitted rather than approximated.")
    output = {
        "state": "evaluated",
        "generated_at": utc_now(),
        "split": "validation",
        "members": members,
        "dataset": {"name": dataset.name, "version": dataset.version},
        "threshold": 0.5,
        "n_cases": len(masks),
        "ablations": rows,
        "seed": args.seed,
        "code_revision": code_revision(),
        "note": (
            "Each row differs from the baseline by one component. Differences between rows are the "
            "measured effect of that component on the validation split; they are not test-set "
            "results."
        ),
    }
    data_dir = resolve_data_dir(args.data_dir)
    target = data_dir / "models" / "ablation_results.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")
    logger.info("")
    for row in rows:
        logger.info(
            "%-22s dice %.4f | iou %.4f | precision %.4f | recall %.4f | specificity %.4f",
            row["configuration"],
            row["dice"],
            row["iou"],
            row["precision"],
            row["recall"],
            row["specificity"],
        )
    logger.info("Ablations written to %s", target)
    return 0
def _aggregate(masks, results, threshold, metrics_module) -> dict:
    scores = {"dice": [], "iou": [], "precision": [], "recall": [], "sensitivity": [], "specificity": []}
    for index, result in enumerate(results):
        single = metrics_module(result.probability, masks[index, 0], threshold)
        for key in scores:
            value = single.get(key)
            if value is not None:
                scores[key].append(float(value))
    return {key: (round(float(np.mean(values)), 6) if values else None) for key, values in scores.items()}
def _row(name, change, masks, results, threshold, logger, metrics_module) -> dict:
    aggregated = _aggregate(masks, results, threshold, metrics_module)
    logger.info("%s: dice %.4f over %d case(s)", name, aggregated["dice"] or float("nan"), len(masks))
    return {"configuration": name, "change": change, "n_cases": len(masks), **aggregated}
def _threshold_rows(masks, prepared, member, thresholds, segmentation, metrics_module) -> list[dict]:
    rows = []
    for threshold in thresholds:
        results = [segmentation.predict(member, item, threshold=threshold) for item in prepared]
        aggregated = _aggregate(masks, results, threshold, metrics_module)
        rows.append(
            {
                "configuration": f"threshold_{threshold:.2f}",
                "change": f"decision threshold {threshold:.2f} instead of 0.50",
                "n_cases": len(masks),
                **aggregated,
            }
        )
    return rows
def _tta_row(masks, prepared, member, segmentation, metrics_module, logger) -> dict:
    results = [segmentation.predict(member, item, threshold=0.5, tta=True) for item in prepared]
    aggregated = _aggregate(masks, results, 0.5, metrics_module)
    logger.info("tta_on: dice %.4f", aggregated["dice"] or float("nan"))
    return {
        "configuration": "tta_on",
        "change": "test-time augmentation over geometric transforms",
        "n_cases": len(masks),
        **aggregated,
    }
def _ensemble_row(name, change, members, prepared, masks, segmentation, metrics_module, logger, weights) -> dict:
    results = []
    for index in range(len(prepared)):
        probabilities = [segmentation.predict(member, prepared[index], threshold=0.5).probability for member in members]
        if weights is None:
            fused = np.mean(probabilities, axis=0)
        else:
            total = sum(float(weights.get(member, 0.0)) for member in members) or 1.0
            fused = sum(float(weights.get(member, 0.0)) * probability for member, probability in zip(members, probabilities, strict=False)) / total
        results.append(type("Fused", (), {"probability": fused, "mask": fused > 0.5})())
    aggregated = _aggregate(masks, results, 0.5, metrics_module)
    logger.info("%s: dice %.4f", name, aggregated["dice"] or float("nan"))
    return {"configuration": name, "change": change, "n_cases": len(masks), **aggregated}
def _load_calibration(data_dir: Path, members: list[str]):
    path = data_dir / "models" / "ensemble_calibration.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if set(payload.get("members", [])) != set(members):
        return None
    if payload.get("split") != "validation":
        return None
    return payload
if __name__ == "__main__":
    raise SystemExit(main())
