# %% [cell] 18. SegNet training / retraining (4-channel)
# =============================================================================
# SECTION 11b - SegNet
# =============================================================================
# This is the model that ships as `segnet_model.pth` and that was found to be
# degenerate (its sigmoid output never crossed 0.5 on any available input).
# This section retrains it on real BraTS data with the architecture the
# application loads: SegNet(in_channels=4, out_channels=1), 256x256, z-scored
# channels, logits out. A checkpoint trained here is written to the notebook's
# own directory; the repository's original file is never overwritten.
# =============================================================================
from app.ml.architectures.segnet import SegNet

SEGNET_CONFIG = {
    "model_id": "segnet_brats_4ch",
    "architecture": "SegNet(in_channels=4, out_channels=1) -- application-compatible state dict",
    "input_size": RUN["image_size"],
    "channels": 4,
    "normalisation": "global z-score over (H,W,C), then CHW",
    "output_semantics": "logit (BCE with logits; sigmoid applied for Dice and at inference)",
    "loss": f"{RUN['dice_weight']}*Dice + {1 - RUN['dice_weight']:.1f}*BCE",
    "optimiser": f"AdamW lr={RUN['learning_rate']} weight_decay={RUN['weight_decay']}",
    "scheduler": "cosine annealing",
    "batch_size": RUN["batch_size"],
    "epochs": RUN["epochs_segnet"],
    "seed": SEED,
    "dataset": LOCK.get("dataset", {}),
    "splits": {"train": len(TRAIN_PATIENTS), "validation": len(VAL_PATIENTS), "test": len(TEST_PATIENTS)},
    "amp": AMP_ENABLED,
}

segnet = SegNet(in_channels=4, out_channels=1)
SEGNET_PARAMETERS = sum(parameter.numel() for parameter in segnet.parameters())
segnet_trainer = SegmentationTrainer(
    "segnet_brats_4ch",
    segnet,
    loss_weights=(RUN["dice_weight"], 1 - RUN["dice_weight"]),
    learning_rate=RUN["learning_rate"],
    weight_decay=RUN["weight_decay"],
    from_logits=SegNet.output_semantics.fget(segnet) == "logit",
)
print(f"SegNet parameters: {SEGNET_PARAMETERS:,} (the application's builder produces the same graph)")

if not RUN.get("skip_segnet_training", False):
    SEGNET_BUDGET = TimeBudget(RUN["session_budget_minutes"])
    SEGNET_SUMMARY = segnet_trainer.train(
        train_patients=TRAIN_PATIENTS,
        val_patients=VAL_PATIENTS,
        epochs=RUN["epochs_segnet"],
        batch_size=RUN["batch_size"],
        size=RUN["image_size"],
        single_channel=False,
        max_steps_per_epoch=RUN["max_steps_per_epoch"],
        validation_cases=RUN["validation_cases"],
        early_stopping_patience=RUN["early_stopping_patience"],
        budget=SEGNET_BUDGET,
        config=SEGNET_CONFIG,
    )
    SEGNET_SUMMARY["parameters"] = SEGNET_PARAMETERS
    write_json(PATHS["artifacts"] / "segnet_training_summary.json", SEGNET_SUMMARY)
    print("=" * 78)
    print("SEGNET TRAINING SUMMARY")
    print("=" * 78)
    for key in ("epochs_completed", "epochs_planned", "best_dice", "best_epoch", "stop_reason", "amp", "device"):
        print(f"{key:18s}: {SEGNET_SUMMARY[key]}")
    if SEGNET_SUMMARY["stop_reason"]:
        print(
            "\nThe run did not reach its planned epoch count. Re-run this cell in a new session "
            "with RESUME=True (default): training continues from checkpoints/segnet_brats_4ch/last.pt."
        )
else:
    print("skip_segnet_training = True -> not training SegNet in this session")

# %% [cell] 19. U-Net baseline (single-channel 512x512)
# =============================================================================
# SECTION 12 - U-Net baseline on real data
# =============================================================================
# The shipped `UNet-[16, 32, 64, 128, 256].pt` is the only artefact that already
# worked, but it was never evaluated on labelled data. This section trains the
# same architecture from scratch on real BraTS FLAIR slices resized to 512x512
# with the input contract the application uses (1 channel, [0, 1] scaling), so
# the reported Dice belongs to a checkpoint whose provenance is known.
# =============================================================================
from app.ml.architectures.dynamic_unet import DEFAULT_FILTERS, DynamicUNet

UNET_CONFIG = {
    "model_id": "unet_dynamic_512",
    "architecture": f"DynamicUNet filters {DEFAULT_FILTERS}, in_channels=1, out_channels=1",
    "input_size": RUN["single_image_size"],
    "channels": 1,
    "channel_used": {0: "T1", 1: "T1ce", 2: "T2", 3: "FLAIR"}[RUN["single_channel_index"]],
    "normalisation": "[0, 1] intensity scaling (the application's single-channel contract)",
    "output_semantics": "probability (sigmoid inside the network; loss uses the probability branch)",
    "loss": f"{RUN['dice_weight']}*Dice + {1 - RUN['dice_weight']:.1f}*BCE",
    "optimiser": f"AdamW lr={RUN['learning_rate_single']} weight_decay={RUN['weight_decay']}",
    "batch_size": RUN["batch_size_single"],
    "epochs": RUN["epochs_unet"],
    "seed": SEED,
    "splits": {"train": len(TRAIN_PATIENTS), "validation": len(VAL_PATIENTS), "test": len(TEST_PATIENTS)},
    "amp": AMP_ENABLED,
}

unet = DynamicUNet(DEFAULT_FILTERS, input_channels=1, output_channels=1)
UNET_PARAMETERS = sum(parameter.numel() for parameter in unet.parameters())
unet_trainer = SegmentationTrainer(
    "unet_dynamic_512_real_brats",
    unet,
    loss_weights=(RUN["dice_weight"], 1 - RUN["dice_weight"]),
    learning_rate=RUN["learning_rate_single"],
    weight_decay=RUN["weight_decay"],
    from_logits=DynamicUNet.output_semantics.fget(unet) == "logit",
)
print(f"DynamicUNet parameters: {UNET_PARAMETERS:,}")

if not RUN.get("skip_unet_training", False):
    UNET_BUDGET = TimeBudget(RUN["session_budget_minutes"])
    UNET_SUMMARY = unet_trainer.train(
        train_patients=TRAIN_PATIENTS,
        val_patients=VAL_PATIENTS,
        epochs=RUN["epochs_unet"],
        batch_size=RUN["batch_size_single"],
        size=RUN["single_image_size"],
        single_channel=True,
        max_steps_per_epoch=RUN["max_steps_per_epoch"],
        validation_cases=max(4, RUN["validation_cases"] // 4),
        early_stopping_patience=RUN["early_stopping_patience"],
        budget=UNET_BUDGET,
        config=UNET_CONFIG,
    )
    UNET_SUMMARY["parameters"] = UNET_PARAMETERS
    write_json(PATHS["artifacts"] / "unet_training_summary.json", UNET_SUMMARY)
    print("=" * 78)
    print("U-NET TRAINING SUMMARY")
    print("=" * 78)
    for key in ("epochs_completed", "epochs_planned", "best_dice", "best_epoch", "stop_reason", "amp", "device"):
        print(f"{key:18s}: {UNET_SUMMARY[key]}")
else:
    print("skip_unet_training = True -> not training the U-Net in this session")

# %% [cell] 20. Residual Attention U-Net (RAUNet, 4-channel)
# =============================================================================
# SECTION 13 - RAUNet (residual attention U-Net with deep supervision)
# =============================================================================
# Same input contract as SegNet (4 x 256 x 256, z-score), so the two can be
# fused later. It is the heaviest model here: if a session ends mid-run, rerun
# this cell and it continues from checkpoints/raunet_brats_4ch/last.pt.
# =============================================================================
from app.ml.architectures.attention_unet import build_raunet

RAUNET_CONFIG = {
    "model_id": "raunet_brats_4ch",
    "architecture": "RAUNet, base_filters=16, depth=4, attention gates, deep supervision",
    "input_size": RUN["image_size"],
    "channels": 4,
    "normalisation": "global z-score over (H,W,C), then CHW (identical to SegNet)",
    "output_semantics": "logit",
    "loss": f"{RUN['dice_weight']}*Dice + {1 - RUN['dice_weight']:.1f}*BCE",
    "optimiser": f"AdamW lr={RUN['learning_rate']} weight_decay={RUN['weight_decay']}",
    "batch_size": RUN["batch_size"],
    "epochs": RUN["epochs_raunet"],
    "seed": SEED,
    "splits": {"train": len(TRAIN_PATIENTS), "validation": len(VAL_PATIENTS), "test": len(TEST_PATIENTS)},
    "amp": AMP_ENABLED,
}

raunet = build_raunet(in_channels=4, out_channels=1, base_filters=16, depth=4)
RAUNET_PARAMETERS = sum(parameter.numel() for parameter in raunet.parameters())
raunet_trainer = SegmentationTrainer(
    "raunet_brats_4ch",
    raunet,
    loss_weights=(RUN["dice_weight"], 1 - RUN["dice_weight"]),
    learning_rate=RUN["learning_rate"],
    weight_decay=RUN["weight_decay"],
    from_logits=type(raunet).output_semantics.fget(raunet) == "logit",
)
print(f"RAUNet parameters: {RAUNET_PARAMETERS:,}")

if not RUN.get("skip_raunet_training", False):
    RAUNET_BUDGET = TimeBudget(RUN["session_budget_minutes"])
    RAUNET_SUMMARY = raunet_trainer.train(
        train_patients=TRAIN_PATIENTS,
        val_patients=VAL_PATIENTS,
        epochs=RUN["epochs_raunet"],
        batch_size=RUN["batch_size"],
        size=RUN["image_size"],
        single_channel=False,
        max_steps_per_epoch=RUN["max_steps_per_epoch"],
        validation_cases=RUN["validation_cases"],
        early_stopping_patience=RUN["early_stopping_patience"],
        budget=RAUNET_BUDGET,
        config=RAUNET_CONFIG,
    )
    RAUNET_SUMMARY["parameters"] = RAUNET_PARAMETERS
    write_json(PATHS["artifacts"] / "raunet_training_summary.json", RAUNET_SUMMARY)
    print("=" * 78)
    print("RAUNET TRAINING SUMMARY")
    print("=" * 78)
    for key in ("epochs_completed", "epochs_planned", "best_dice", "best_epoch", "stop_reason", "amp", "device"):
        print(f"{key:18s}: {RAUNET_SUMMARY[key]}")
    if RAUNET_SUMMARY["stop_reason"]:
        print("\nResume in the next session by re-running this cell: the checkpoint carries optimiser, scheduler and RNG state.")
else:
    print("skip_raunet_training = True -> not training RAUNet in this session")

# %% [cell] 21. Training curves
# =============================================================================
# Loss and validation curves for every model trained above
# =============================================================================
def plot_history(model_id, axis_pair):
    history = read_json(PATHS["checkpoints"] / model_id / "history.json", default=[]) or []
    if not history:
        return False
    epochs = [entry["epoch"] for entry in history]
    losses = [entry.get("train_loss") for entry in history]
    dice = [entry.get("val_dice") for entry in history]
    axis_pair[0].plot(epochs, losses, marker="o", label=f"{model_id} train loss")
    axis_pair[0].set_ylabel("loss")
    axis_pair[1].plot(epochs, dice, marker="o", label=f"{model_id} val Dice")
    axis_pair[1].set_ylabel("Dice")
    return True


fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
plotted = False
for model_id in ("segnet_brats_4ch", "unet_dynamic_512_real_brats", "raunet_brats_4ch"):
    plotted = plot_history(model_id, axes) or plotted
axes[0].set_title("Training loss"); axes[1].set_title("Validation Dice")
axes[0].set_xlabel("epoch"); axes[1].set_xlabel("epoch")
if plotted:
    axes[0].legend(fontsize=8); axes[1].legend(fontsize=8)
plt.tight_layout()
plt.savefig(PATHS["reports"] / "04_training_curves.png", dpi=130)
plt.close(fig)
print("training curves ->", PATHS["reports"] / "04_training_curves.png", "| histories found:", plotted)
