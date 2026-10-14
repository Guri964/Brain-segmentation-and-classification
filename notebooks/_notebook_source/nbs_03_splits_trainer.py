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

import contextlib
import copy
import queue
import shutil as _shutil
import subprocess as _subprocess
import threading

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


def iterate_batches_legacy(patient_ids, *, batch_size, size, single_channel=False, shuffle=True, rng=None, max_steps=None):
    """The original generator, kept verbatim for the before/after benchmark.

    It called `load_patient()` once per patient (to count slices) and once per slice (to
    build the batch). Every one of those calls inflated the patient's whole cached archive.
    `iterate_batches(..., legacy=True)` runs this; nothing else does.
    """
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


def prepare_batch(chunk, *, store, size, single_channel):
    """Assemble one batch from `chunk` = [(volume_key, slice position), ...].

    One read per member per patient that appears in the chunk -- not one read per slice:
    the positions of that patient are requested together and indexed in a single step. The
    arrays, their order, the channel order (T1, T1ce, T2, FLAIR) and the mask handling are
    exactly what the previous implementation produced, so a batch is bit-for-bit the same
    (tests/test_data_pipeline.py asserts it against `iterate_batches_legacy`).
    """
    grouped: dict = {}
    for slot, (volume_key, position) in enumerate(chunk):
        grouped.setdefault(volume_key, []).append((slot, position))
    images: list = [None] * len(chunk)
    masks: list = [None] * len(chunk)
    image_member = "singles" if single_channel else "images"
    for volume_key, entries in grouped.items():
        positions = [position for _, position in entries]
        mask_rows = store.slices(volume_key, positions, member="masks")
        image_rows = store.slices(volume_key, positions, member=image_member)
        for row, (slot, _) in enumerate(entries):
            mask = mask_rows[row][None, :, :].astype(np.float32)
            if single_channel:
                image = image_rows[row].astype(np.float32)[None, :, :]
                if mask.shape[-1] != size:
                    mask = upsample_mask_nearest(mask, size)
            else:
                image = np.moveaxis(image_rows[row].astype(np.float32), -1, 0)
            images[slot] = image
            masks[slot] = mask
    return np.stack(images), np.stack(masks)


def iterate_batches(
    patient_ids,
    *,
    batch_size,
    size,
    single_channel=False,
    shuffle=True,
    rng=None,
    max_steps=None,
    store=None,
    legacy=False,
    prefetch=None,
):
    """Yield (images, masks) batches read from the cache on the fly.

    The batch composition, the slice ordering, the channel order (T1, T1ce, T2, FLAIR), the
    normalisation already baked into the cache, the mask resizing and the yielded dtypes are
    unchanged -- `legacy=True` runs the old implementation next to this one and the test
    suite asserts the two produce identical batches for the same seed.

    What changed is the number of times a cached patient is read:

      * the slice index is built from the cache index, i.e. from a header, so it costs no
        array reads at all. The old loop loaded a whole payload per patient for this.
      * each batch reads every member it needs once per patient appearing in it, and indexes
        the slices it wants from that. The old loop read the patient's entire archive for
        every single slice.
      * a 4-channel batch never opens `singles` and a single-channel batch never opens
        `images`; on a real BraTS volume that halves the bytes read.
    """
    if legacy:
        yield from iterate_batches_legacy(
            patient_ids,
            batch_size=batch_size,
            size=size,
            single_channel=single_channel,
            shuffle=shuffle,
            rng=rng,
            max_steps=max_steps,
        )
        return
    store = store or patient_store()
    rng = rng or np.random.default_rng(SEED)
    index = []
    for volume_key in patient_ids:
        count = store.slice_count(volume_key)
        index.extend((volume_key, position) for position in range(count))
    if shuffle:
        rng.shuffle(index)
    limit = len(index) if max_steps is None else min(len(index), int(max_steps) * batch_size)
    chunks = [index[start : start + batch_size] for start in range(0, limit, batch_size)]
    def make_batch(chunk):
        return prepare_batch(chunk, store=store, size=size, single_channel=single_channel)

    if prefetch is None:
        prefetch = bool(RUN.get("prefetch_batches", False))
    if prefetch:
        yield from prefetched_batches(make_batch, chunks)
        return
    for chunk in chunks:
        yield make_batch(chunk)


def prefetched_batches(make_batch, chunks, *, maxsize=2, join_timeout=2.0):
    """Prepare batches on one worker thread while the caller trains on the previous one.

    Threads rather than DataLoader processes: the arrays are read from memory maps, the GIL
    is released during the read, and nothing is pickled, so a batch costs no IPC and needs
    no worker start-up. It is only worth using when preparing a batch is a noticeable share
    of a training step; cell 18 measures both settings instead of assuming (`RUN[
    "prefetch_batches"]` is the default for the training loop).
    """
    end = object()
    slots = queue.Queue(maxsize=maxsize)
    stop = threading.Event()

    def produce():
        try:
            for chunk in chunks:
                if stop.is_set():
                    break
                slots.put(make_batch(chunk))
        except BaseException as exc:  # noqa: BLE001 - handed to the consumer, not swallowed
            with contextlib.suppress(Exception):
                slots.put(exc, timeout=5)
        finally:
            with contextlib.suppress(Exception):
                slots.put(end, timeout=5)

    worker = threading.Thread(target=produce, name="prefetch-batches", daemon=True)
    worker.start()
    try:
        while True:
            item = slots.get()
            if item is end:
                break
            if isinstance(item, BaseException):
                raise item
            yield item
    finally:
        stop.set()
        while True:                      # let a worker blocked on a full queue finish
            try:
                slots.get_nowait()
            except queue.Empty:
                break
        worker.join(timeout=join_timeout)


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


# =============================================================================
# SECTION 11c - Measuring the pipeline (where a training step's time goes)
# =============================================================================
# The trainer above is unchanged: same loss, same optimiser, same schedule, same AMP, same
# checkpoint format, same validation protocol. What is added here is measurement, because
# the reason the GPU was idle (0% utilisation observed, CPU at 100%) was never the model or
# the device placement -- it was 109.7 s of CPU work to prepare one batch of 8.
#
# Every GPU phase is timed with torch.cuda.synchronize() on both sides. Without the
# synchronise, a "fast" measurement only covers the kernel launch.
# =============================================================================


def synchronise(device=None):
    """`torch.cuda.synchronize()` on a CUDA device, a no-op anywhere else."""
    target = DEVICE if device is None else device
    kind = getattr(target, "type", None) or str(target).split(":")[0]
    if str(kind) == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(torch.device(target))


def store_activity(store, before):
    """The counters a single measurement added, not the process-wide totals."""
    report = store.stats_report()
    for key, value in before.items():
        if isinstance(value, int) and isinstance(report.get(key), int):
            report[key] = report[key] - value
    return report


def summarise_seconds(values):
    """Median / min / max of a list of durations, in seconds."""
    if not values:
        return {"n": 0, "median": None, "min": None, "max": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "n": int(array.size),
        "median": round(float(np.median(array)), 5),
        "min": round(float(array.min()), 5),
        "max": round(float(array.max()), 5),
    }


def autocast_dtype():
    """The dtype autocast actually uses, reported instead of assumed."""
    try:
        for candidate in ("cuda", "cpu"):
            try:
                return str(torch.get_autocast_dtype(candidate))
            except (AttributeError, TypeError, RuntimeError):
                continue
    except Exception:  # noqa: BLE001 - diagnostics must never break a run
        pass
    return "unknown"


class gpu_utilization_probe:
    """Sample the real `nvidia-smi` utilisation while the body of the `with` block runs.

    On CPU-only machines (and in CI) this reports `available: False` with the reason; it
    never invents a number.
    """

    def __init__(self, interval=0.2):
        self.interval = float(interval)
        self.samples: list = []
        self.reason = None
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        if not _shutil.which("nvidia-smi"):
            self.reason = "nvidia-smi is not on PATH"
            return self
        self._thread = threading.Thread(target=self._run, name="gpu-utilisation", daemon=True)
        self._thread.start()
        return self

    def _run(self):
        while not self._stop.is_set():
            try:
                result = _subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0:
                    self.samples.append([int(item) for item in result.stdout.split()])
            except Exception as exc:  # noqa: BLE001 - report, never raise from a probe
                self.reason = f"{type(exc).__name__}: {exc}"
                return
            self._stop.wait(self.interval)

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3)
        return self.report()

    def report(self):
        if not self.samples:
            return {"available": False, "reason": self.reason or "no sample collected"}
        values = np.asarray(self.samples, dtype=np.float64)
        return {
            "available": True,
            "samples": len(self.samples),
            "interval_seconds": self.interval,
            "mean_percent": round(float(values.mean()), 1),
            "max_percent": round(float(values.max()), 1),
            "first_gpu_mean_percent": round(float(values[:, 0].mean()), 1),
            "per_sample_first_gpu": [int(row[0]) for row in self.samples],
        }

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc_info):
        self.stop()
        return False


def benchmark_data_pipeline(
    patient_ids,
    *,
    batch_size,
    size,
    single_channel=False,
    steps=None,
    warmup=None,
    seed=None,
    legacy=False,
    prefetch=None,
    store=None,
):
    """Time batch preparation alone: no model, no GPU work, no other variable.

    `legacy=True` runs the previous implementation (one whole-archive read per slice) so the
    two can be compared on the same patients, the same cache state and the same batch size.
    """
    steps = int(RUN.get("benchmark_steps", 6) if steps is None else steps)
    warmup = int(RUN.get("benchmark_warmup", 2) if warmup is None else warmup)
    store = store or patient_store()
    seeds = int(SEED if seed is None else seed)
    activity_before = dict(store.stats)

    # The slice index the trainer rebuilds at the start of every epoch.
    started = time.perf_counter()
    if legacy:
        total_slices = sum(int(load_patient(key)["masks"].shape[0]) for key in patient_ids)
    else:
        total_slices = sum(patient_slice_count(key) for key in patient_ids)
    index_seconds = time.perf_counter() - started

    store.release()
    iterator = iterate_batches(
        patient_ids,
        batch_size=batch_size,
        size=size,
        single_channel=single_channel,
        shuffle=True,
        rng=np.random.default_rng(seeds),
        max_steps=warmup + steps,
        legacy=legacy,
        prefetch=prefetch,
        store=store,
    )
    durations, first_batch = [], None
    while True:
        started = time.perf_counter()
        try:
            images, masks = next(iterator)
        except StopIteration:
            break
        durations.append(time.perf_counter() - started)
        if first_batch is None:
            first_batch = durations[-1]
    # Drop the warm-up batches -- unless there were not enough of them, in which case every
    # batch that could be prepared is reported and `warmup_dropped` says so.
    warmup_dropped = warmup if len(durations) > warmup else 0
    measured = durations[warmup_dropped:]
    store = store or patient_store()
    report = {
        "pipeline": "legacy" if legacy else "cached-array",
        "prefetch": bool(prefetch if prefetch is not None else RUN.get("prefetch_batches", False)),
        "patients": len(patient_ids),
        "slices_available": int(total_slices),
        "batch_size": int(batch_size),
        "input_size": int(size),
        "channels": 1 if single_channel else 4,
        "index_seconds": round(index_seconds, 4),
        "index_seconds_per_patient": round(index_seconds / max(1, len(patient_ids)), 6),
        "first_batch_seconds": None if first_batch is None else round(first_batch, 4),
        "batch_seconds": summarise_seconds(measured),
        "warmup_dropped": warmup_dropped,
        "note": None if len(durations) > warmup else f"only {len(durations)} batch(es) were available; none dropped as warm-up",
        "slices_per_second": None
        if not measured or max(measured) == 0
        else round(batch_size * len(measured) / sum(measured), 1),
        "store": store_activity(store, activity_before),
    }
    return report


def benchmark_training_step(
    trainer,
    patient_ids,
    *,
    batch_size,
    size,
    single_channel=False,
    steps=None,
    warmup=None,
    seed=None,
    legacy=False,
    prefetch=None,
    probe_gpu=True,
    store=None,
):
    """Time the five phases of a real training step, separately and end to end.

    1. batch preparation (CPU, cache -> stacked numpy arrays)
    2. host-to-device transfer
    3. forward pass
    4. backward pass + optimiser step
    5. the whole step

    The measurement runs on an identical COPY of the model with a fresh AdamW, so the
    weights, the optimiser state and the step counter training is about to use are untouched.
    AMP, the loss, the gradient clipping and the autocast context are the trainer's own.
    """
    steps = int(RUN.get("benchmark_steps", 6) if steps is None else steps)
    warmup = int(RUN.get("benchmark_warmup", 2) if warmup is None else warmup)
    device = trainer.device
    model = copy.deepcopy(trainer.model).to(device)
    model.train()
    reference = trainer.optimizer.param_groups[0]
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(reference["lr"]), weight_decay=float(reference.get("weight_decay", 0.0))
    )
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=trainer.amp)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=trainer.amp)

    def one_step(images, masks):
        started = time.perf_counter()
        images_t = torch.from_numpy(images).to(device, non_blocking=True)
        masks_t = torch.from_numpy(masks).to(device, non_blocking=True)
        synchronise(device)
        transferred = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        with trainer.autocast():
            outputs = model(images_t)
            loss = trainer.criterion(outputs, masks_t)
        synchronise(device)
        forwarded = time.perf_counter()
        scaler.scale(loss).backward()
        if RUN["grad_clip"]:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), RUN["grad_clip"])
        scaler.step(optimizer)
        scaler.update()
        synchronise(device)
        finished = time.perf_counter()
        return (transferred - started, forwarded - transferred, finished - forwarded, finished - started), float(loss.detach())

    store = store or patient_store()
    activity_before = dict(store.stats)
    total_slices = sum(patient_slice_count(key) for key in patient_ids)
    store.release()
    iterator = iterate_batches(
        patient_ids,
        batch_size=batch_size,
        size=size,
        single_channel=single_channel,
        shuffle=True,
        rng=np.random.default_rng(int(SEED if seed is None else seed)),
        max_steps=warmup + steps,
        legacy=legacy,
        prefetch=prefetch,
        store=store,
    )
    seen_prepare, seen_transfer, seen_forward, seen_backward, seen_totals, seen_losses = [], [], [], [], [], []
    first_batch = None
    probe = gpu_utilization_probe() if probe_gpu else None
    if probe is not None:
        probe.start()
    try:
        while True:
            started = time.perf_counter()
            try:
                images, masks = next(iterator)
            except StopIteration:
                break
            prepared = time.perf_counter() - started
            phases, loss = one_step(images, masks)
            if first_batch is None:
                first_batch = prepared + phases[3]
            seen_prepare.append(prepared)
            seen_transfer.append(phases[0])
            seen_forward.append(phases[1])
            seen_backward.append(phases[2])
            seen_totals.append(phases[3])
            seen_losses.append(loss)
    finally:
        gpu = probe.stop() if probe is not None else None

    # Drop the warm-up steps -- unless there were not enough of them (a small smoke run or a
    # capped epoch), in which case every step that ran is reported instead of nothing.
    warmup_dropped = warmup if len(seen_totals) > warmup else 0
    prepare = seen_prepare[warmup_dropped:]
    transfer = seen_transfer[warmup_dropped:]
    forward = seen_forward[warmup_dropped:]
    backward = seen_backward[warmup_dropped:]
    totals = seen_totals[warmup_dropped:]
    losses = seen_losses[warmup_dropped:]

    report = {
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu_name": GPU_NAME,
        "amp": bool(trainer.amp),
        "autocast_dtype": autocast_dtype(),
        "model_id": trainer.model_id,
        "batch_size": int(batch_size),
        "input_size": int(size),
        "channels": 1 if single_channel else 4,
        "patients": len(patient_ids),
        "slices_available": int(total_slices),
        "pipeline": "legacy" if legacy else "cached-array",
        "prefetch": bool(prefetch if prefetch is not None else RUN.get("prefetch_batches", False)),
        "steps_measured": len(totals),
        "steps_seen": len(seen_totals),
        "steps_warmup": warmup,
        "warmup_dropped": warmup_dropped,
        "first_batch_seconds": None if first_batch is None else round(first_batch, 4),
        "phases_seconds": {
            "prepare": summarise_seconds(prepare),
            "cpu_to_gpu": summarise_seconds(transfer),
            "forward": summarise_seconds(forward),
            "backward_and_step": summarise_seconds(backward),
            "step_total": summarise_seconds(totals),
        },
        "loss_first": None if not losses else round(losses[0], 6),
        "loss_last": None if not losses else round(losses[-1], 6),
        "gpu_utilization": gpu,
        "store": store_activity(store, activity_before),
    }
    median_total = report["phases_seconds"]["step_total"]["median"]
    median_prepare = report["phases_seconds"]["prepare"]["median"]
    if median_total:
        report["prepare_share_of_step"] = round(median_prepare / median_total, 3)
        report["gpu_share_of_step"] = round(1 - median_prepare / median_total, 3)
    report["notes"] = []
    if not torch.cuda.is_available():
        report["notes"].append(
            "no CUDA device here: the forward/backward numbers are CPU timings. The GPU phases "
            "can only be judged on the T4 run."
        )
    elif report.get("prepare_share_of_step", 0) > 0.5:
        report["notes"].append(
            "the CPU still spends most of the step preparing batches, so the GPU cannot be busy: "
            f"{report['prepare_share_of_step'] * 100:.0f}% of the step is preparation"
        )
    else:
        report["notes"].append(
            f"preparation is {report.get('prepare_share_of_step', 0) * 100:.0f}% of a step, so the GPU is now the "
            "limiting factor -- utilisation can only be as high as the GPU work allows"
        )
    report["notes"].append(
        "measured on a copy of the model with a fresh optimiser: the real training state (weights, "
        "optimiser, step counter) is untouched by the benchmark"
    )
    if len(seen_totals) <= warmup:
        report["notes"].append(
            f"only {len(seen_totals)} step(s) could be prepared from the available slices, so no step was "
            "discarded as warm-up; the numbers include a cold first step"
        )
    if not totals:
        report["notes"].append("no step could be measured: no cached patient was available")
    return report


def verify_training_step(trainer, patient_ids, *, batch_size, size, single_channel=False, seed=None, store=None):
    """One batch through the real pipeline: shapes, dtypes, finite loss, real gradients, and a
    checkpoint that reloads.

    The forward/backward runs on a copy of the model WITHOUT an optimiser step, so nothing
    the training loop depends on moves. The checkpoint is written to `work/selftest/`, never
    to `checkpoints/<model_id>/`.
    """
    device = trainer.device
    model = copy.deepcopy(trainer.model).to(device)
    model.train()
    iterator = iterate_batches(
        patient_ids,
        batch_size=batch_size,
        size=size,
        single_channel=single_channel,
        shuffle=False,
        rng=np.random.default_rng(int(SEED if seed is None else seed)),
        max_steps=1,
        store=store or patient_store(),
    )
    images, masks = next(iterator)
    images_t = torch.from_numpy(images).to(device)
    masks_t = torch.from_numpy(masks).to(device)
    model.zero_grad(set_to_none=True)
    with trainer.autocast():
        outputs = model(images_t)
        loss = trainer.criterion(outputs, masks_t)
    loss.backward()
    logits = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    gradient_norm = float(torch.sqrt(sum((gradient.detach().float() ** 2).sum() for gradient in gradients)))

    payload = trainer.checkpoint_payload(metrics={"self_test": True}, config={"self_test": True}, epochs_planned=0)
    target = PATHS["work"] / "selftest" / f"{trainer.model_id}_selftest.pt"
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, target)
    reloaded = torch.load(target, map_location="cpu", weights_only=False)
    fresh = copy.deepcopy(trainer.model).to("cpu")
    load_report = fresh.load_state_dict(reloaded["state_dict"], strict=True)
    identical = all(
        torch.equal(fresh.state_dict()[key].cpu(), trainer.model.state_dict()[key].detach().cpu()) for key in fresh.state_dict()
    )
    resume_state = trainer.resume(allow=True, config={"self_test": True})
    model.zero_grad(set_to_none=True)

    return {
        "model_id": trainer.model_id,
        "device": str(device),
        "model_parameter_device": str(next(model.parameters()).device),
        "images_shape": list(images.shape),
        "masks_shape": list(masks.shape),
        "images_dtype": str(np.asarray(images).dtype),
        "masks_dtype": str(np.asarray(masks).dtype),
        "mask_values": sorted(int(value) for value in np.unique(masks)[:4]),
        "images_finite": bool(np.isfinite(np.asarray(images, dtype=np.float32)).all()),
        "masks_binary": bool(np.isin(np.unique(masks), [0, 1]).all()),
        "channel_order": "T1, T1ce, T2, FLAIR (cache contract; channel c is read as-is)",
        "output_shape": list(logits.shape),
        "output_device": str(logits.device),
        "loss": round(float(loss.detach()), 6),
        "loss_finite": bool(torch.isfinite(loss)),
        "amp_enabled": bool(trainer.amp),
        "autocast_dtype": autocast_dtype(),
        "gradients": len(gradients),
        "gradient_norm": round(gradient_norm, 6),
        "gradients_finite": bool(all(torch.isfinite(gradient).all() for gradient in gradients)),
        "gradients_nonzero": bool(any(float(gradient.abs().max()) > 0 for gradient in gradients)),
        "checkpoint_path": str(target),
        "checkpoint_keys": sorted(payload),
        "checkpoint_reload_ok": not load_report.missing_keys and not load_report.unexpected_keys and identical,
        "resume_state": resume_state,
    }


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
