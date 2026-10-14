from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import base_parser, configure_logging, load_manifest, report_dataset, resolve_data_dir
from _training import TrainingConfig, assert_no_test_leakage, resolve_config_defaults, train
def main() -> int:
    parser = base_parser("Train a U-Net (single-channel baseline or four-channel attention variant)")
    parser.add_argument("--architecture", default="dynamic_unet", choices=["dynamic_unet", "raunet"])
    parser.add_argument("--size", type=int, default=None, help="Input resolution (default: 512 for dynamic_unet, 256 for raunet)")
    parser.add_argument("--channels", type=int, default=None, help="Channel count (default: 1 for dynamic_unet, 4 for raunet)")
    parser.add_argument("--loss", default="bce_dice", choices=["bce", "dice", "bce_dice", "focal_tversky", "boundary"])
    parser.add_argument("--filters", default="16,32,64,128,256")
    parser.add_argument("--model-id", default=None)
    args = parser.parse_args()
    logger = configure_logging(args.verbose)
    dataset = load_manifest(args.data)
    report_dataset(logger, dataset)
    assert_no_test_leakage(dataset)
    is_attention = args.architecture == "raunet"
    size = args.size or (256 if is_attention else 512)
    channels = args.channels or (4 if is_attention else 1)
    if size % 32:
        raise SystemExit("--size must be a multiple of 32.")
    filters = [int(item) for item in args.filters.split(",")]
    if len(filters) != 5:
        raise SystemExit("--filters must contain exactly five values, for example 16,32,64,128,256")
    model_id = args.model_id or ("raunet_brats_4ch" if is_attention else "unet_dynamic_512")
    config = resolve_config_defaults(
        TrainingConfig(
            model_id=model_id,
            architecture=args.architecture,
            in_channels=channels,
            out_channels=1,
            spatial_size=size,
            normalisation="zscore" if channels > 1 else "scaling",
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            patience=args.patience,
            scheduler=args.scheduler,
            loss=args.loss,
            seed=args.seed,
            architecture_kwargs={"filters": filters},
        ),
        dataset,
    )
    if args.dry_run:
        logger.info("Dry run: configuration validated, no training performed.")
        logger.info("Model: %s", config.descriptor())
        return 0
    data_dir = resolve_data_dir(args.data_dir)
    checkpoint_dir = args.checkpoint_dir or (data_dir / "models" / "checkpoints")
    run_dir = data_dir / "models" / "training"
    record = train(dataset, config, args, logger, checkpoint_dir, run_dir)
    logger.info(
        "Report the numbers from this run as validation performance only: the reserved test "
        "split has not been touched. Run scripts/evaluate_models.py for test metrics "
        "(best validation Dice %.4f).",
        record["best_validation_dice"],
    )
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
