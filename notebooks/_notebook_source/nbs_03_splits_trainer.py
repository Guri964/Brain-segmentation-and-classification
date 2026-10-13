# %% [cell] 15. Patient-level train / validation / test splitting
# =============================================================================
# SECTION 9 - Patient-level splits (HGG/LGG stratified, leakage checked, frozen)
# =============================================================================
# BraTS slices from one patient are near-duplicates of each other. Splitting by
# slice would leak the same tumour into train and test, so the split is made on
# the patient (volume) and every slice follows its patient. The split is written
# to disk and reused by every later section; nothing reselects it.
# =============================================================================
SPLIT_PATH = PATHS["artifacts"] / "patient_splits.json"


def stratified_patient_splits(volumes, *, train=0.70, val=0.15, test=0.15, seed=SEED):
    """Stratify on the grade when grades exist, otherwise fall back to plain random."""
    rng = random.Random(seed)
    groups = {}
    for volume_key, entry in volumes.items():
        key = entry["grade"] if entry["grade"] in {"HGG", "LGG"} else "unlabelled"
        groups.setdefault(key, []).append(volume_key)
    splits = {"train": [], "validation": [], "test": []}
    warnings = []
    for key, patient_ids in sorted(groups.items()):
        ordered = sorted(patient_ids)
        rng.shuffle(ordered)
        count = len(ordered)
        if count < 3:
            splits["train"].extend(ordered)
            warnings.append(f"stratum '{key}' has {count} patient(s); kept in training (no metric can come from it)")
            continue
        n_val = max(1, int(round(count * val)))
        n_test = max(1, int(round(count * test)))
        n_train = count - n_val - n_test
        if n_train < 1:
            n_train, n_val, n_test = 1, max(1, (count - 1) // 2), max(0, count - 1 - max(1, (count - 1) // 2))
        splits["train"].extend(ordered[:n_train])
        splits["validation"].extend(ordered[n_train : n_train + n_val])
        splits["test"].extend(ordered[n_train + n_val : n_train + n_val + n_test])
    return splits, warnings


def split_summary(splits, volumes):
    placement = {}
    summary = {}
    for name in ("train", "validation", "test"):
        grades = {}
        slices = 0
        for volume_key in splits[name]:
            grades[volumes[volume_key]["grade"] or "unlabelled"] = grades.get(volumes[volume_key]["grade"] or "unlabelled", 0) + 1
            slices += len(volumes[volume_key]["slices"])
            placement.setdefault(volume_key, []).append(name)
        summary[name] = {"patients": len(splits[name]), "slices": slices, "grade_counts": grades}
    overlap = sorted(key for key, names in placement.items() if len(names) > 1)
    summary["patient_overlap"] = overlap
    summary["leakage_free"] = not overlap
    summary["patients_assigned"] = len(placement)
    summary["patients_total"] = len(volumes)
    return summary


if SPLIT_PATH.exists() and not RUN.get("refreeze_splits"):
    SPLIT_RECORD = read_json(SPLIT_PATH)
    SPLITS = SPLIT_RECORD["splits"]
    print(f"Reusing the frozen split from {SPLIT_PATH}")
else:
    SPLITS, SPLIT_WARNINGS = stratified_patient_splits(VOLUMES, seed=SEED)
    SPLIT_RECORD = {
        "created_at": utc_now(),
        "method": "patient_level_stratified_by_grade",
        "seed": SEED,
        "ratios": {"train": 0.70, "validation": 0.15, "test": 0.15},
        "splits": SPLITS,
        "warnings": SPLIT_WARNINGS,
        "summary": split_summary(SPLITS, VOLUMES),
    }
    write_json(SPLIT_PATH, SPLIT_RECORD)

SPLIT_SUMMARY = split_summary(SPLITS, VOLUMES)
assert SPLIT_SUMMARY["leakage_free"], f"patients appear in more than one split: {SPLIT_SUMMARY['patient_overlap']}"

print("=" * 78)
print("SPLITS (patient level, leakage checked)")
print("=" * 78)
for name in ("train", "validation", "test"):
    entry = SPLIT_SUMMARY[name]
    print(f"{name:11s}: {entry['patients']:4d} patients | {entry['slices']:6d} slices | grades {entry['grade_counts']}")
print("leakage-free      :", SPLIT_SUMMARY["leakage_free"])
print("frozen at         :", SPLIT_PATH)
print(
    "\nThe test split is written once here and is not touched until Section 19. Model\n"
    "selection, threshold tuning and ensemble weights use the validation split only."
)

TRAIN_PATIENTS = list(SPLITS["train"])
VAL_PATIENTS = list(SPLITS["validation"])
TEST_PATIENTS = list(SPLITS["test"])
if RUN["smoke_test"]:
    TRAIN_PATIENTS, VAL_PATIENTS, TEST_PATIENTS = TRAIN_PATIENTS[:2], VAL_PATIENTS[:1], TEST_PATIENTS[:1]
    print("smoke test: reduced to", len(TRAIN_PATIENTS), len(VAL_PATIENTS), len(TEST_PATIENTS), "patients")

# %% [cell] 16. Reproducible baselines
# =============================================================================
# SECTION 10 - Reproducible baselines
# =============================================================================
# Two reference points so every later number has something to be compared with:
#   * an all-foreground predictor (what Dice a trivial mask gets),
#   * the training-split prevalence of tumour pixels.
# They are computed on the validation split, never on the test split.
# =============================================================================
def validation_prevalence():
    total, foreground, slices = 0, 0, 0
    for volume_key in VAL_PATIENTS:
        payload = load_patient(volume_key)
        masks = payload["masks"].astype(np.float32)
        foreground += float(masks.sum())
        total += float(masks.size)
        slices += masks.shape[0]
    return {"slices": slices, "pixels": total, "foreground_pixels": foreground, "prevalence": foreground / max(1.0, total)}


VAL_PREVALENCE = validation_prevalence()
# Dice of an all-foreground prediction against the reference masks.
dice_all_foreground = 2 * VAL_PREVALENCE["foreground_pixels"] / max(
    1.0, VAL_PREVALENCE["pixels"] + VAL_PREVALENCE["foreground_pixels"]
)
BASELINES = {
    "all_foreground_dice": dice_all_foreground,
    "validation_prevalence": VAL_PREVALENCE["prevalence"],
    "validation_slices": VAL_PREVALENCE["slices"],
    "note": (
        "Any trained model has to beat the all-foreground Dice on the same split to be worth "
        "anything; both numbers are reported next to the model results."
    ),
}
write_json(PATHS["artifacts"] / "baselines.json", BASELINES)
print("=" * 78)
print("REFERENCE POINTS (validation split)")
print("=" * 78)
print(f"validation slices        : {VAL_PREVALENCE['slices']}")
print(f"tumour pixel prevalence  : {VAL_PREVALENCE['prevalence']:.4f}")
print(f"all-foreground Dice      : {dice_all_foreground:.4f}")

# %% [cell] 17. Trainer core (AMP, resume, progress, early stopping, scheduling)
# =============================================================================
# SECTION 11 - Trainer
# =============================================================================
# One trainer is used by every segmentation model, so the comparison is fair:
# same loss (0.6 * Dice + 0.4 * BCE), same optimiser, same schedule, same
# validation protocol, same checkpoint format.
#
# Checkpoints contain everything needed to continue in a later Kaggle session:
# model, optimiser, scheduler, AMP scaler, epoch, step, best metric, RNG state.
# =============================================================================
from tqdm.auto import tqdm

from app.ml.losses import BCEDiceLoss  # embedded copy of the project's own loss class


class TimeBudgetExceeded(RuntimeError):
    pass


class TimeBudget:
    """Stops training cleanly before the Kaggle session dies, so the run can resume."""

    def __init__(self, minutes):
        self.deadline = time.time() + minutes * 60

    @property
    def remaining_minutes(self):
        return max(0.0, (self.deadline - time.time()) / 60)

    @property
    def expired(self):
        return self.remaining_minutes <= 0

    def require(self, minutes_needed=5.0):
        if self.remaining_minutes < minutes_needed:
            raise TimeBudgetExceeded(
                f"only {self.remaining_minutes:.1f} min left of the session budget; "
                "checkpoints are saved and the next session continues from them"
            )


def dice_score(prediction, target, eps=1e-7):
    prediction = (np.asarray(prediction) > 0.5).astype(np.float64)
    target = (np.asarray(target) > 0.5).astype(np.float64)
    intersection = float((prediction * target).sum())
    return float((2 * intersection + eps) / (prediction.sum() + target.sum() + eps))


def upsample_mask_nearest(mask, size):
    """Nearest-neighbour resize of a binary mask.

    The cache stores masks at the 4-channel size (256); the single-channel U-Net consumes
    512x512 inputs, so its masks are resized the same way the images are -- nearest, so a
    pixel is either tumour or not, and never a half-labelled boundary invented by bilinear
    interpolation.
    """
    tensor = torch.from_numpy(np.asarray(mask, dtype=np.float32))
    if tensor.dim() == 3:
        tensor = tensor[:, None]
    resized = torch.nn.functional.interpolate(tensor, size=(size, size), mode="nearest")
    return (resized[:, 0].numpy() > 0.5).astype(np.float32)


def iterate_batches(patient_ids, *, batch_size, size, single_channel=False, shuffle=True, rng=None, max_steps=None):
    """Yield (images, masks) batches read from the cache on the fly."""
    rng = rng or np.random.default_rng(SEED)
    index = []
    for volume_key in patient_ids:
        payload = load_patient(volume_key)
        count = payload["masks"].shape[0]
        index.extend((volume_key, position) for position in range(count))
    if shuffle:
        rng.shuffle(index)
    steps = 0
    for start in range(0, len(index), batch_size):
        chunk = index[start : start + batch_size]
        images, masks = [], []
        for volume_key, position in chunk:
            payload = load_patient(volume_key)
            mask = payload["masks"][position][None, :, :].astype(np.float32)
            if single_channel:
                images.append(payload["singles"][position].astype(np.float32)[None, :, :])
                if mask.shape[-1] != size:
                    mask = upsample_mask_nearest(mask, size)
            else:
                images.append(np.moveaxis(payload["images"][position].astype(np.float32), -1, 0))
            masks.append(mask)
        yield np.stack(images), np.stack(masks)
        steps += 1
        if max_steps is not None and steps >= max_steps:
            return


def augment_batch(images, masks):
    """Left/right flips only: safe for anatomy, cheap, and never invents lesion tissue.

    Works on torch tensors (the training loop moves batches to the device first) and on numpy
    arrays. Torch rejects negative strides, so the flip goes through `torch.flip`.
    """
    for index in range(images.shape[0]):
        if random.random() >= 0.5:
            continue
        if isinstance(images, torch.Tensor):
            images[index] = torch.flip(images[index], dims=[2])
            masks[index] = torch.flip(masks[index], dims=[2])
        else:
            images[index] = np.ascontiguousarray(images[index][:, :, ::-1])
            masks[index] = np.ascontiguousarray(masks[index][:, :, ::-1])
    return images, masks


class SegmentationTrainer:
    """AMP training loop with resumable checkpoints and validation-based model selection.

    `from_logits` mirrors the architecture's own `output_semantics`: SegNet and RAUNet return
    logits, DynamicUNet returns probabilities (it applies the sigmoid inside `forward`). The
    loss and the Dice computation follow that flag instead of assuming one convention.
    """

    def __init__(
        self,
        model_id,
        model,
        *,
        loss_weights,
        learning_rate,
        weight_decay,
        from_logits=True,
        device=DEVICE,
        amp=AMP_ENABLED,
    ):
        self.model_id = model_id
        self.model = model.to(device)
        self.device = device
        self.from_logits = bool(from_logits)
        self.criterion = BCEDiceLoss(
            from_logits=self.from_logits,
            bce_weight=loss_weights[1],
            dice_weight=loss_weights[0],
        ).to(device)
        self.amp = bool(amp and device.type == "cuda")
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.optimizer_zero_grad = getattr(self.optimizer, "zero_grad", None)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=50, eta_min=learning_rate / 20)
        try:
            self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp)
            self.autocast = lambda: torch.amp.autocast("cuda", enabled=self.amp)
            self.amp_api = "torch.amp"
        except (AttributeError, TypeError):
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp)
            self.autocast = lambda: torch.cuda.amp.autocast(enabled=self.amp)
            self.amp_api = "torch.cuda.amp"
        self.loss_weights = loss_weights
        self.directory = PATHS["checkpoints"] / model_id
        self.directory.mkdir(parents=True, exist_ok=True)
        self.history_path = self.directory / "history.json"
        self.state = {
            "epoch": 0,
            "step": 0,
            "best_dice": -1.0,
            "best_epoch": None,
            "history": read_json(self.history_path, default=[]) or [],
        }

    # -- checkpointing ----------------------------------------------------
    def checkpoint_payload(self, *, metrics=None, config=None, epochs_planned=None):
        return {
            "state_dict": {key: value.detach().cpu() for key, value in self.model.state_dict().items()},
            "model_id": self.model_id,
            "epoch": self.state["epoch"],
            "step": self.state["step"],
            "best_dice": self.state["best_dice"],
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict() if self.amp else None,
            "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(),
            "torch_rng": torch.get_rng_state(),
            "config": config or {},
            "epochs_planned": epochs_planned,
            "metrics": metrics or {},
            "saved_at": utc_now(),
            "amp_api": self.amp_api,
        }

    def save(self, *, metrics=None, config=None, epochs_planned=None, best=False):
        payload = self.checkpoint_payload(metrics=metrics, config=config, epochs_planned=epochs_planned)
        torch.save(payload, self.directory / "last.pt")
        if best:
            torch.save(payload, self.directory / "best.pt")
        write_json(self.history_path, self.state["history"])
        return self.directory / ("best.pt" if best else "last.pt")

    def resume(self, *, config=None, allow=False):
        path = self.directory / "last.pt"
        if not allow or not path.exists():
            return {"resumed": False, "path": str(path) if path.exists() else None}
        payload = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(payload["state_dict"])
        try:
            self.optimizer.load_state_dict(payload["optimizer"])
            self.scheduler.load_state_dict(payload["scheduler"])
            if payload.get("scaler") and self.amp:
                self.scaler.load_state_dict(payload["scaler"])
        except (KeyError, ValueError) as exc:
            print(f"WARNING: optimiser state from {path.name} could not be restored ({exc}); continuing with fresh optimiser state")
        self.state["epoch"] = int(payload.get("epoch", 0))
        self.state["step"] = int(payload.get("step", 0))
        self.state["best_dice"] = float(payload.get("best_dice", -1.0))
        if isinstance(payload.get("history"), list):
            self.state["history"] = payload["history"]
        elif self.history_path.exists():
            self.state["history"] = read_json(self.history_path, default=self.state["history"])
        print(
            f"resumed {self.model_id} from {path} at epoch {self.state['epoch']} "
            f"(best validation Dice {self.state['best_dice']:.4f})"
        )
        return {"resumed": True, "epoch": self.state["epoch"], "best_dice": self.state["best_dice"], "path": str(path)}

    # -- training ---------------------------------------------------------
    def train(
        self,
        *,
        train_patients,
        val_patients,
        epochs,
        batch_size,
        size,
        single_channel=False,
        max_steps_per_epoch=None,
        validation_cases=None,
        early_stopping_patience=8,
        validate_every=1,
        budget=None,
        resume=True,
        config=None,
        max_steps_per_batch=None,
    ):
        self.resume(allow=resume, config=config)
        rng = np.random.default_rng(SEED + self.state["epoch"])
        checkpoint_every_steps = int(RUN["checkpoint_every_steps"])
        last_save = time.time()
        started = time.time()
        stop_reason = None
        for epoch in range(self.state["epoch"] + 1, epochs + 1):
            self.state["epoch"] = epoch
            self.model.train()
            epoch_losses = []
            batch_iterator = iterate_batches(
                train_patients,
                batch_size=batch_size,
                size=size,
                single_channel=single_channel,
                shuffle=True,
                rng=rng,
                max_steps=max_steps_per_epoch,
            )
            progress = tqdm(batch_iterator, desc=f"{self.model_id} epoch {epoch}/{epochs}", leave=False)
            epoch_started = time.time()
            for step_in_epoch, (images, masks) in enumerate(progress, start=1):
                images_t = torch.from_numpy(images).to(self.device, non_blocking=True)
                masks_t = torch.from_numpy(masks).to(self.device, non_blocking=True)
                if not single_channel:
                    images_t, masks_t = augment_batch(images_t, masks_t)
                self.optimizer.zero_grad(set_to_none=True)
                with self.autocast():
                    outputs = self.model(images_t)
                    loss = self.criterion(outputs, masks_t)
                self.scaler.scale(loss).backward()
                if RUN["grad_clip"]:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), RUN["grad_clip"])
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.state["step"] += 1
                epoch_losses.append(float(loss.detach()))
                progress.set_postfix(loss=f"{np.mean(epoch_losses[-20:]):.4f}")
                if budget is not None and budget.expired:
                    stop_reason = "session time budget reached"
                    break
                if checkpoint_every_steps and self.state["step"] % checkpoint_every_steps == 0:
                    self.save(config=config, epochs_planned=epochs)
                    last_save = time.time()
                    write_json(
                        PATHS["artifacts"] / f"{self.model_id}_progress.json",
                        {"epoch": epoch, "step": self.state["step"], "elapsed_minutes": round((time.time() - started) / 60, 2), "stop_reason": None},
                    )
            self.scheduler.step()
            epoch_seconds = time.time() - epoch_started
            entry = {
                "epoch": epoch,
                "train_loss": float(np.mean(epoch_losses)) if epoch_losses else None,
                "lr": float(self.optimizer.param_groups[0]["lr"]),
                "epoch_seconds": round(epoch_seconds, 2),
                "elapsed_minutes": round((time.time() - started) / 60, 2),
            }
            if validate_every and epoch % validate_every == 0 and val_patients:
                val_dice, val_loss = self.validate(
                    val_patients,
                    batch_size=batch_size,
                    size=size,
                    single_channel=single_channel,
                    max_cases=validation_cases,
                    budget=budget,
                )
                entry["val_dice"] = val_dice
                entry["val_loss"] = val_loss
                improved = val_dice > self.state["best_dice"]
                if improved:
                    self.state["best_dice"] = val_dice
                    self.state["best_epoch"] = epoch
                    self.state["epochs_without_improvement"] = 0
                else:
                    self.state["epochs_without_improvement"] = int(self.state.get("epochs_without_improvement", 0)) + 1
                self.state["history"].append(entry)
                self.save(metrics=entry, config=config, epochs_planned=epochs, best=improved)
                last_save = time.time()
                remaining = len(train_patients) * 155 // max(1, batch_size)  # rough remaining work
                print(
                    f"epoch {epoch:3d}/{epochs}  train_loss {entry['train_loss']:.4f}  "
                    f"val_dice {val_dice:.4f}  val_loss {val_loss:.4f}  "
                    f"best {self.state['best_dice']:.4f} @ {self.state['best_epoch']}  "
                    f"{epoch_seconds:.0f}s/epoch"
                )
                if self.state["epochs_without_improvement"] >= early_stopping_patience:
                    stop_reason = f"early stopping after {early_stopping_patience} epochs without validation improvement"
                    break
            else:
                self.state["history"].append(entry)
                self.save(metrics=entry, config=config, epochs_planned=epochs)
                print(f"epoch {epoch:3d}/{epochs}  train_loss {entry['train_loss']}  (no validation this epoch)")
            if budget is not None and budget.expired:
                stop_reason = "session time budget reached"
            if time.time() - last_save > RUN["checkpoint_every_minutes"] * 60:
                self.save(config=config, epochs_planned=epochs)
                last_save = time.time()
            if stop_reason:
                break
        self.save(config=config, epochs_planned=epochs)
        summary = {
            "model_id": self.model_id,
            "epochs_completed": self.state["epoch"],
            "epochs_planned": epochs,
            "best_dice": self.state["best_dice"],
            "best_epoch": self.state["best_epoch"],
            "stop_reason": stop_reason,
            "history": self.state["history"],
            "amp": self.amp,
            "device": str(self.device),
            "resumable": (self.directory / "last.pt").exists(),
        }
        write_json(self.directory / "training_summary.json", summary)
        return summary

    # -- validation -------------------------------------------------------
    def probabilities(self, outputs):
        """Detach a network output and turn it into probabilities, whatever it returned."""
        tensor = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
        tensor = tensor.float()
        if self.from_logits:
            tensor = torch.sigmoid(tensor)
        return tensor.cpu().numpy()

    def validate(self, patient_ids, *, batch_size, size, single_channel=False, max_cases=None, budget=None):
        self.model.eval()
        scores, losses = [], []
        cases = 0
        with torch.no_grad():
            for images, masks in iterate_batches(
                patient_ids,
                batch_size=batch_size,
                size=size,
                single_channel=single_channel,
                shuffle=False,
            ):
                images_t = torch.from_numpy(images).to(self.device)
                masks_t = torch.from_numpy(masks).to(self.device)
                with self.autocast():
                    outputs = self.model(images_t)
                    loss = self.criterion(outputs, masks_t)
                probabilities = self.probabilities(outputs)
                targets = masks_t.cpu().numpy()
                for index in range(probabilities.shape[0]):
                    scores.append(dice_score(probabilities[index, 0] > 0.5, targets[index, 0] > 0.5))
                    losses.append(float(loss))
                    cases += 1
                    if max_cases and cases >= max_cases:
                        break
                if max_cases and cases >= max_cases:
                    break
                if budget is not None and budget.expired:
                    break
        self.model.train()
        return (float(np.mean(scores)) if scores else float("nan"), float(np.mean(losses)) if losses else float("nan"))


def load_trained_model(model_id, builder, path=None):
    """Load a checkpoint written by this notebook (or by the project's scripts)."""
    candidate = Path(path) if path else PATHS["checkpoints"] / model_id / "best.pt"
    if not candidate.exists():
        candidate = PATHS["checkpoints"] / model_id / "last.pt"
    if not candidate.exists():
        raise FileNotFoundError(
            f"no checkpoint for {model_id} at {PATHS['checkpoints'] / model_id}; run its training section first"
        )
    payload = torch.load(candidate, map_location=DEVICE, weights_only=False)
    model = builder()
    report = model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    return model.to(DEVICE), payload, candidate


print(
    "Trainer ready. Loss =",
    f"{RUN['dice_weight']}*Dice + {1 - RUN['dice_weight']:.1f}*BCE",
    "(BCEDiceLoss, the project's own class; from_logits set per architecture)",
    "| AMP:",
    AMP_ENABLED,
)
