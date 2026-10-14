from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (
    BACKEND_ROOT,
    code_revision,
    configure_logging,
    describe_environment,
    file_digest,
    load_manifest,
    report_dataset,
    resolve_data_dir,
    seed_everything,
    torch_device,
    utc_now,
    write_run_record,
)
from _training import prepare_split
sys.path.insert(0, str(BACKEND_ROOT))
METRIC_KEYS = [
    "dice",
    "iou",
    "precision",
    "recall",
    "sensitivity",
    "specificity",
    "hausdorff95",
    "average_surface_distance",
]
def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate checkpoints on the reserved test split")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--split", default="test", choices=["test", "validation"])
    parser.add_argument("--models", default=None, help="Comma-separated model ids (default: every loaded segmentation model)")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20240101)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--select", action="store_true", help="Also write the model-selection record (validation split only)")
    parser.add_argument("--metric", default="dice", help="Selection metric when --select is used")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    logger = configure_logging(args.verbose)
    seed_everything(args.seed)
    dataset = load_manifest(args.data)
    report_dataset(logger, dataset)
    if args.split == "test" and not dataset.splits.get("test"):
        raise SystemExit(
            "The manifest declares no test split. Run scripts/make_splits.py first. Reporting "
            "metrics computed on the training or validation split as test performance is exactly "
            "the mistake this pipeline exists to prevent."
        )
    from app import create_app
    from app.services.evaluation_service import EvaluationService
    app = create_app(env="testing")
    services = app.extensions["brain_services"]
    registry = services["registry"]
    segmentation = services["segmentation"]
    requested = [item.strip() for item in args.models.split(",")] if args.models else registry.segmentation_model_ids()
    device = torch_device(args.device)
    plan = []
    for model_id in requested:
        loaded = registry.load_model(model_id)
        if not loaded.is_ready:
            logger.warning("Skipping %s: %s", model_id, loaded.message)
            continue
        health = services["health"].check(model_id)
        if not health.responsive:
            logger.warning("Skipping %s: it failed the reference check (%s).", model_id, health.verdict)
            continue
        plan.append((model_id, loaded.descriptor))
    if not plan:
        raise SystemExit(
            "No checkpoint passed the reference check, so there is nothing to evaluate. Metrics "
            "from a model that cannot produce a mask would be meaningless."
        )
    logger.info("Evaluating on the '%s' split with %s", args.split, ", ".join(model_id for model_id, _ in plan))
    if args.dry_run:
        logger.info("Dry run: %d case(s) would be read.", len(dataset.cases_for(args.split)))
        return 0
    from app.services.preprocessing_service import preprocess_four_channel, preprocess_single_channel
    from app.utils.images import encode_png, to_uint8
    rubric: dict[str, list[np.ndarray]] = {}
    references: list[np.ndarray] = []
    case_ids: list[str] = []
    runtimes: list[dict] = []
    for position, (model_id, descriptor) in enumerate(plan):
        config = _config_for(descriptor, args, dataset)
        images, masks = prepare_split(dataset, args.split, config, logger, args.max_cases)
        cases = [case.case_id for case in dataset.cases_for(args.split)][: len(images)]
        logger.info("%s: %s case(s), input %s", model_id, len(images), images.shape)
        probabilities: list[np.ndarray] = []
        started = time.perf_counter()
        for index in range(len(images)):
            prepared = _prepare_input(
                images[index], descriptor, model_id, cases[index], preprocess_single_channel, preprocess_four_channel, encode_png, to_uint8
            )
            result = segmentation.predict(model_id, prepared, threshold=args.threshold)
            probabilities.append(result.probability)
        elapsed = time.perf_counter() - started
        rubric[model_id] = probabilities
        if position == 0:
            references = [mask[0] for mask in masks]
            case_ids = cases
        runtimes.append(
            {
                "model_id": model_id,
                "state": "measured",
                "device": str(device),
                "input_shape": list(images.shape),
                "parameter_count": getattr(descriptor, "parameter_count", None),
                "seconds_per_image": round(elapsed / max(1, len(images)), 4),
                "images_per_second": round(len(images) / elapsed, 3) if elapsed else None,
                "note": "Measured while evaluating the reserved split on this machine.",
            }
        )
    report = EvaluationService.evaluate_segmentation_predictions(
        rubric,
        references,
        threshold=args.threshold,
        case_ids=case_ids,
    )
    payload = {
        "state": "evaluated",
        "evaluated_at": utc_now(),
        "split": args.split,
        "dataset": dataset.summary(),
        "segmentation": report,
        "runtime": {"state": "measured", "generated_at": utc_now(), "models": runtimes},
        "protocol": {
            "seed": args.seed,
            "threshold": args.threshold,
            "bootstrap_samples": args.bootstrap,
            "selection_split": "validation",
            "test_split_used_for_tuning": False,
            "code_revision": code_revision(),
        },
        "environment": describe_environment(),
    }
    classification_table = _classification_table_from(data_dir=resolve_data_dir(args.data_dir))
    if classification_table is not None:
        payload["classification"] = classification_table
        logger.info("Classification evaluation attached from %s", classification_table.get("source"))
    data_dir = resolve_data_dir(args.data_dir)
    results_path = data_dir / "models" / "evaluation_results.json"
    if results_path.exists():
        backup = results_path.with_suffix(".previous.json")
        backup.write_text(results_path.read_text(encoding="utf-8"), encoding="utf-8")
        logger.info("Previous evaluation preserved at %s", backup)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    logger.info("Evaluation written to %s", results_path)
    if args.select:
        if args.split != "validation":
            raise SystemExit(
                "--select must run on the validation split: selecting a model using test metrics "
                "makes those same metrics optimistic. Re-run with --split validation."
            )
        best = _best_model(report, args.metric)
        selection = {
            "state": "selected",
            "selected_at": utc_now(),
            "metric": args.metric,
            "split": "validation",
            "model_id": best["model_id"] if best else None,
            "candidates": [
                {"model_id": model["model_id"], "metrics": model.get("aggregate", {})} for model in report.get("models", [])
            ],
            "rationale": (
                "Selection used the validation split only. The reserved test split is evaluated "
                "afterwards by scripts/evaluate_models.py --split test and is not used to choose "
                "anything."
            ),
            "seed": args.seed,
            "dataset_version": dataset.version,
            "code_revision": code_revision(),
        }
        selection_path = data_dir / "models" / "model_selection.json"
        selection_path.write_text(json.dumps(selection, indent=2, default=str), encoding="utf-8")
        logger.info("Model selection written to %s (%s = %s)", selection_path, args.metric, selection["model_id"])
    write_run_record(
        data_dir / "models" / "evaluation_runs",
        {
            "model_id": "evaluation",
            "status": "completed",
            "split": args.split,
            "models": [model_id for model_id, _ in plan],
            "dataset": dataset.summary(),
            "code_revision": code_revision(),
            "checkpoint_digests": {
                model_id: file_digest(Path(descriptor.path)) if descriptor.path and Path(descriptor.path).exists() else None
                for model_id, descriptor in plan
            },
        },
    )
    for model_id, measures in rubric.items():
        logger.info("%s: %d case(s) evaluated", model_id, len(measures))
    logger.info("Aggregate metrics are in %s - quote them with the dataset version and split.", results_path)
    return 0
MODALITY_LABELS = ["T1", "T1ce", "T2", "FLAIR"]
def _prepare_input(array, descriptor, model_id, case_id, single, four, encode_png, to_uint8):
    spec = descriptor.input_spec
    if spec.channels == 1:
        raw = encode_png(to_uint8(array[0]))
        return single(raw, f"{case_id}.png", size=spec.height)
    payloads = [
        (f"{case_id}_{label}.png", encode_png(to_uint8(array[index])))
        for index, label in enumerate(MODALITY_LABELS[: spec.channels])
    ]
    return four(payloads, size=spec.height)
def _config_for(descriptor, args, dataset):
    from _training import TrainingConfig
    spec = descriptor.input_spec
    channels = getattr(spec, "channels", None) or getattr(spec, "in_channels", 1)
    height = getattr(spec, "height", None) or getattr(spec, "size", 256)
    return TrainingConfig(
        model_id=descriptor.model_id,
        architecture=descriptor.architecture,
        in_channels=channels,
        out_channels=1,
        spatial_size=height,
        normalisation=getattr(spec, "normalisation", "scaling") or "scaling",
        threshold=args.threshold,
        seed=args.seed,
    )
def _classification_table_from(*, data_dir: Path):
    path = data_dir / "models" / "classification" / "evaluation.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.setdefault("source", str(path))
    return payload
def _best_model(report: dict, metric: str):
    best = None
    for model in report.get("models", []):
        value = (model.get("aggregate") or {}).get(metric)
        if value is None:
            continue
        if best is None or value > best["value"]:
            best = {"model_id": model.get("model_id"), "value": value}
    return best
if __name__ == "__main__":
    raise SystemExit(main())
