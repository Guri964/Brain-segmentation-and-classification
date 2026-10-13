from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import base_parser, configure_logging, load_manifest, report_dataset, resolve_data_dir
from _training import TrainingConfig, assert_no_test_leakage, resolve_config_defaults, train
def main() -> int:
    parser = base_parser("Train the four-channel SegNet on a labelled dataset")
    parser.add_argument("--size", type=int, default=256, help="Input resolution (must be a multiple of 32)")
    parser.add_argument("--loss", default="bce_dice", choices=["bce", "dice", "bce_dice", "focal_tversky", "boundary"])
    parser.add_argument("--model-id", default="segnet_brats_4ch")
    args = parser.parse_args()
    logger = configure_logging(args.verbose)
    dataset = load_manifest(args.data)
    report_dataset(logger, dataset)
    assert_no_test_leakage(dataset)
    config = resolve_config_defaults(
        TrainingConfig(
            model_id=args.model_id,
            architecture="segnet",
            in_channels=4,
            out_channels=1,
            spatial_size=args.size,
            normalisation="zscore",
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            patience=args.patience,
            scheduler=args.scheduler,
            loss=args.loss,
            seed=args.seed,
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
    train(dataset, config, args, logger, checkpoint_dir, run_dir)
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
