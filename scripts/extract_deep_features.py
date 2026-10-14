from __future__ import annotations
import argparse
import csv
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (
    BACKEND_ROOT,
    configure_logging,
    load_manifest,
    report_dataset,
    resolve_data_dir,
    seed_everything,
    torch_device,
    utc_now,
)
from _training import TrainingConfig, prepare_split
sys.path.insert(0, str(BACKEND_ROOT))
MODALITY_LABELS = ["T1", "T1ce", "T2", "FLAIR"]
def main() -> int:
    parser = argparse.ArgumentParser(description="Extract pooled encoder features for the fusion experiment")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--model", default="unet_dynamic_512")
    parser.add_argument("--splits", default="train,validation,test")
    parser.add_argument("--layer", default=None, help="Named submodule to pool (default: the last encoder stage)")
    parser.add_argument("--output", type=Path, default=None)
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
    splits = [item.strip() for item in args.splits.split(",")]
    from app import create_app
    from app.services.preprocessing_service import preprocess_four_channel, preprocess_single_channel
    from app.utils.images import encode_png, to_uint8
    app = create_app(env="testing")
    services = app.extensions["brain_services"]
    registry = services["registry"]
    loaded = registry.load_model(args.model)
    if not loaded.is_ready:
        raise SystemExit(f"{args.model} is not usable: {loaded.message}")
    descriptor = loaded.descriptor
    module = loaded.module
    device = torch_device(args.device)
    module.to(device).eval()
    layer = _resolve_layer(module, args.layer)
    logger.info("Pooling '%s' on %s", layer, device)
    import torch
    rows: list[dict] = []
    for split in splits:
        config = TrainingConfig(
            model_id=args.model,
            architecture=descriptor.architecture,
            in_channels=descriptor.input_spec.channels,
            out_channels=1,
            spatial_size=descriptor.input_spec.height,
            normalisation=descriptor.input_spec.normalisation,
            seed=args.seed,
        )
        try:
            images, _ = prepare_split(dataset, split, config, logger, args.max_cases)
        except SystemExit as exc:
            logger.warning("Skipping split '%s': %s", split, exc)
            continue
        if args.dry_run:
            logger.info("Dry run: %d case(s) in '%s'", len(images), split)
            continue
        cases = [case for case in dataset.cases_for(split)][: len(images)]
        logger.info("Split '%s': %d case(s)", split, len(images))
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
            activation = _capture(module, layer, prepared.tensor.to(device))
            pooled = activation.mean(dim=(2, 3)).squeeze(0).detach().cpu().numpy().astype(np.float32)
            record = {"case_id": cases[index].case_id, "patient_id": cases[index].patient_id, "split": split}
            if cases[index].label is not None:
                record["label"] = cases[index].label
            record.update({f"deep_{position:03d}": float(value) for position, value in enumerate(pooled)})
            rows.append(record)
    if args.dry_run:
        logger.info("Dry run complete; nothing written.")
        return 0
    if not rows:
        raise SystemExit("No embedding was extracted; check the split names and the dataset.")
    data_dir = resolve_data_dir(args.data_dir)
    output = args.output or (data_dir / "models" / "deep_features.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    columns = sorted({key for row in rows for key in row}, key=lambda name: (name not in {"case_id", "patient_id", "split", "label"}, name))
    with open(output, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    logger.info("%d row(s) × %d column(s) written to %s", len(rows), len(columns), output)
    logger.info(
        "Merge this table with the radiomics table on case_id to obtain the fusion representation, "
        "then train with scripts/train_classifier.py --architecture feature_fusion. Extracted at %s.",
        utc_now(),
    )
    return 0
def _resolve_layer(module, name: str | None):
    import torch.nn as nn
    if name:
        target = module
        for part in name.split("."):
            target = getattr(target, part)
        return target
    candidates = [child for child in module.modules() if isinstance(child, nn.Conv2d)]
    if not candidates:
        raise SystemExit("No convolutional layer was found to pool.")
    return candidates[-1]
def _capture(module, layer, tensor):
    import torch
    captured: dict[str, object] = {}
    def hook(_module, _inputs, output):
        captured["activation"] = output
    handle = layer.register_forward_hook(hook)
    try:
        with torch.no_grad():
            module(tensor)
    finally:
        handle.remove()
    activation = captured.get("activation")
    if activation is None:
        raise SystemExit("The selected layer was never reached by a forward pass.")
    return activation
if __name__ == "__main__":
    raise SystemExit(main())
