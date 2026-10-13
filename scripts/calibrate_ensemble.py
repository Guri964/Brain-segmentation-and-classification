from __future__ import annotations
import argparse
import itertools
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
    torch_device,
    utc_now,
)
from _training import prepare_split
sys.path.insert(0, str(BACKEND_ROOT))
MODALITY_LABELS = ["T1", "T1ce", "T2", "FLAIR"]
def main() -> int:
    parser = argparse.ArgumentParser(description="Estimate ensemble weights on the validation split")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--models", default=None, help="Comma-separated members (default: every model in the largest compatible group)")
    parser.add_argument("--metric", default="dice", choices=["dice", "iou"])
    parser.add_argument("--step", type=float, default=0.05, help="Weight grid resolution")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20240101)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
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
    from app.utils.metrics import dice as dice_metric
    from app.utils.metrics import iou as iou_metric
    app = create_app(env="testing")
    services = app.extensions["brain_services"]
    registry = services["registry"]
    segmentation = services["segmentation"]
    compatible = []
    for group in services["ensemble"].compatibility(registry.segmentation_model_ids())["groups"]:
        if group["ensemblable"]:
            compatible.append(group)
    if not compatible:
        raise SystemExit(
            "No group of models shares an identical input specification, so probability fusion "
            "cannot be calibrated. Fusing a 4x256x256 model with a 1x512x512 model would not "
            "produce a meaningful probability map."
        )
    group = max(compatible, key=lambda item: len(item["members"]))
    members = [item for item in (args.models.split(",") if args.models else group["members"])]
    members = [member for member in members if member in group["members"]]
    if len(members) < 2:
        raise SystemExit(f"At least two compatible members are required; this group has {group['members']}.")
    for member in members:
        health = services["health"].check(member)
        if not health.responsive:
            raise SystemExit(
                f"{member} failed the reference check ({health.verdict}). Calibrating weights for a "
                "model that produces no mask would produce weights that describe nothing."
            )
    device = torch_device(args.device)
    descriptor = registry.load_model(members[0]).descriptor
    from _training import TrainingConfig
    config = TrainingConfig(
        model_id=members[0],
        architecture=descriptor.architecture,
        in_channels=descriptor.input_spec.channels,
        out_channels=1,
        spatial_size=descriptor.input_spec.height,
        normalisation=descriptor.input_spec.normalisation,
        threshold=args.threshold,
        seed=args.seed,
    )
    images, masks = prepare_split(dataset, "validation", config, logger, args.max_cases)
    logger.info("Calibrating on %d validation case(s)", len(images))
    if args.dry_run:
        logger.info("Dry run: %s member(s) verified and compatible.", len(members))
        return 0
    probabilities: dict[str, list[np.ndarray]] = {}
    for member in members:
        prepared_list = []
        for index in range(len(images)):
            spec = descriptor.input_spec
            if spec.channels == 1:
                prepared = preprocess_single_channel(encode_png(to_uint8(images[index][0])), f"case-{index}.png", size=spec.height)
            else:
                payloads = [
                    (f"case-{index}_{label}.png", encode_png(to_uint8(images[index][channel])))
                    for channel, label in enumerate(MODALITY_LABELS[: spec.channels])
                ]
                prepared = preprocess_four_channel(payloads, size=spec.height)
            prepared_list.append(prepared)
        predictions = [segmentation.predict(member, prepared, threshold=args.threshold).probability for prepared in prepared_list]
        probabilities[member] = predictions
        logger.info("%s: %d probability map(s) computed on %s", member, len(predictions), device)
    grid = np.arange(0.0, 1.0 + 1e-9, args.step)
    best = None
    for candidate in itertools.product(grid, repeat=len(members) - 1):
        remainder = 1.0 - sum(candidate)
        if remainder < -1e-9:
            continue
        weights = np.asarray([*candidate, max(0.0, remainder)], dtype=np.float64)
        if weights.sum() <= 0:
            continue
        weights = weights / weights.sum()
        scores = []
        for index in range(len(masks)):
            fused = sum(weight * probabilities[member][index] for weight, member in zip(weights, members, strict=False))
            metric = dice_metric(fused, masks[index, 0], args.threshold) if args.metric == "dice" else iou_metric(fused, masks[index, 0], args.threshold)
            scores.append(metric)
        mean_score = float(np.mean(scores)) if scores else float("nan")
        if best is None or mean_score > best["score"]:
            best = {"score": mean_score, "weights": weights.tolist()}
    assert best is not None
    data_dir = resolve_data_dir(args.data_dir)
    target = data_dir / "models" / "ensemble_calibration.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state": "calibrated",
        "created_at": utc_now(),
        "split": "validation",
        "metric": args.metric,
        "threshold": args.threshold,
        "members": members,
        "weights": {member: round(float(weight), 6) for member, weight in zip(members, best["weights"], strict=False)},
        "score": round(float(best["score"]), 6),
        "n_cases": len(masks),
        "dataset": {"name": dataset.name, "version": dataset.version, "split_sizes": {name: len(ids) for name, ids in dataset.splits.items()}},
        "seed": args.seed,
        "test_split_used": False,
        "code_revision": code_revision(),
        "note": (
            "Weights were estimated on validation data only. Applying them is a fixed fusion rule; "
            "it does not make the ensemble more accurate than its members on data it has never "
            "seen."
        ),
    }
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Weights: %s", payload["weights"])
    logger.info("Validation %s: %.4f over %d case(s)", args.metric, best["score"], len(masks))
    logger.info("Written to %s", target)
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
