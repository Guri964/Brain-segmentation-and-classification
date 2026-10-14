from __future__ import annotations
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence
import numpy as np
from _common import (
    Case,
    Dataset,
    code_revision,
    describe_environment,
    file_digest,
    save_checkpoint,
    seed_everything,
    utc_now,
    write_run_record,
)
@dataclass
class TrainingConfig:
    model_id: str
    architecture: str
    in_channels: int
    out_channels: int
    spatial_size: int
    normalisation: str
    threshold: float = 0.5
    epochs: int = 60
    batch_size: int = 4
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    patience: int = 12
    scheduler: str = "plateau"
    loss: str = "bce_dice"
    augment: bool = True
    seed: int = 20240101
    architecture_kwargs: dict[str, Any] | None = None
    def descriptor(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "display_name": self.model_id,
            "task": "segmentation",
            "architecture": self.architecture,
            "input_spec": {
                "channels": self.in_channels,
                "height": self.spatial_size,
                "width": self.spatial_size,
                "normalisation": self.normalisation,
            },
            "output_semantics": "probability",
            "fixed_threshold": self.threshold,
            "trainer": f"scripts/train_{self.architecture}.py",
            "architecture_kwargs": self.architecture_kwargs or {},
        }
def load_case_arrays(case: Case, size: int, channels: int, *, log: logging.Logger | None = None) -> tuple[np.ndarray, np.ndarray | None]:
    kind = case.source_kind
    if kind == "brats_hdf5_slice":
        return _load_brats_slice(case, size, channels, log=log)
    if kind == "brats_nifti_volume":
        return _load_brats_volume(case, size, channels, log=log)
    from app.utils.images import decode_image_bytes, resize_array
    planes = []
    for path in case.images[:channels]:
        raw = path.read_bytes()
        if path.suffix.lower() == ".npy":
            array = np.load(path)
        else:
            array = decode_image_bytes(raw).array
        array = np.asarray(array, dtype=np.float32)
        if array.ndim == 3:
            array = array.mean(axis=2)
        planes.append(resize_array(array, (size, size)))
    while len(planes) < channels:
        planes.append(np.zeros_like(planes[0]))
    image = np.stack(planes, axis=0)
    mask = None
    if case.mask is not None and case.mask.exists():
        raw = case.mask.read_bytes()
        if case.mask.suffix.lower() == ".npy":
            array = np.load(case.mask)
        else:
            array = decode_image_bytes(raw).array
        array = np.asarray(array, dtype=np.float32)
        if array.ndim == 3:
            array = array.max(axis=2)
        mask = (resize_array(array, (size, size)) > 0.5).astype(np.float32)
    if log is not None and mask is None:
        log.warning("Case %s has no mask; it will be skipped.", case.case_id)
    return image, mask


def _load_brats_slice(case: Case, size: int, channels: int, *, log: logging.Logger | None = None) -> tuple[np.ndarray, np.ndarray | None]:
    from app.utils.images import resize_array
    from brats import read_slice_file
    source = case.images[0]
    stored = read_slice_file(source, mask_channel_labels=case.options.get("mask_channel_labels"))
    image = stored.image
    if channels and channels != image.shape[0]:
        if channels < image.shape[0]:
            image = image[:channels]
        else:
            padded = np.zeros((channels, image.shape[1], image.shape[2]), dtype=image.dtype)
            padded[: image.shape[0]] = image
            image = padded
            if log is not None:
                log.warning(
                    "Case %s stores %d channel(s) but %d were requested; the missing channels are "
                    "zero-filled rather than duplicated.",
                    case.case_id,
                    stored.image.shape[0],
                    channels,
                )
    planes = [resize_array(plane, (size, size)) for plane in image]
    stacked = np.stack(planes, axis=0).astype(np.float32)
    whole = stored.whole_tumour().astype(np.float32)
    mask = (resize_array(whole, (size, size)) > 0.5).astype(np.float32)
    return stacked, mask


def _load_brats_volume(case: Case, size: int, channels: int, *, log: logging.Logger | None = None) -> tuple[np.ndarray, np.ndarray | None]:
    from app.utils.images import resize_array
    from brats import read_volume_directory
    directory = case.images[0].parent
    volume = read_volume_directory(directory)
    order = case.options.get("channel_order") or sorted(volume["modalities"])
    planes = []
    for name in order[:channels]:
        plane = volume["modalities"].get(name)
        if plane is None:
            continue
        planes.append(plane)
    if not planes:
        raise SystemExit(f"Case {case.case_id} has no readable modality in {directory}")
    middle = planes[0].shape[0] // 2
    stacked = np.stack([resize_array(plane[middle], (size, size)) for plane in planes], axis=0)
    segmentation = volume["segmentation"]
    if segmentation is None:
        return stacked.astype(np.float32), None
    mask = (resize_array((segmentation[middle] > 0).astype(np.float32), (size, size)) > 0.5).astype(np.float32)
    if log is not None:
        log.info(
            "Case %s uses the middle axial slice %d of the volume. To train across the whole "
            "volume, export the slices first (scripts/prepare_brats.py over an HDF5 slice tree, "
            "or slice the NIfTI volumes) and point --data at that manifest.",
            case.case_id,
            middle,
        )
    return stacked.astype(np.float32), mask


def normalise(image: np.ndarray, mode: str) -> np.ndarray:
    if mode == "zscore":
        mean = float(image.mean())
        std = float(image.std()) or 1.0
        return (image - mean) / std
    if mode == "scaling":
        peak = float(image.max()) or 1.0
        return image / peak
    return image
def augment_pair(image: np.ndarray, mask: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    if rng.random() < 0.5:
        image = image[:, :, ::-1]
        mask = mask[:, :, ::-1]
    if rng.random() < 0.5:
        image = image[:, ::-1, :]
        mask = mask[:, ::-1, :]
    if rng.random() < 0.5:
        shift = int(rng.integers(1, max(2, image.shape[-1] // 8)))
        axis = int(rng.integers(1, 3))
        image = np.roll(image, shift, axis=axis)
        mask = np.roll(mask, shift, axis=axis)
    if rng.random() < 0.3:
        scale = float(rng.uniform(0.9, 1.1))
        image = image * scale
    if rng.random() < 0.2:
        image = image + rng.normal(0.0, 0.02, size=image.shape).astype(np.float32)
    return image, mask
class OnDiskSegmentationDataset:
    def __init__(self, cases: Sequence[Case], config: TrainingConfig, *, augment: bool, seed: int, logger: logging.Logger | None = None):
        self.cases = list(cases)
        self.config = config
        self.augment = augment
        self.logger = logger
        self.rng = np.random.default_rng(seed)
        self._cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def __len__(self) -> int:
        return len(self.cases)

    def _load(self, case: Case) -> tuple[np.ndarray, np.ndarray] | None:
        if case.case_id in self._cache:
            return self._cache[case.case_id]
        image, mask = load_case_arrays(case, self.config.spatial_size, self.config.in_channels, log=self.logger)
        if mask is None:
            return None
        prepared = (normalise(image, self.config.normalisation).astype(np.float32), mask.astype(np.float32))
        if len(self._cache) < 512:
            self._cache[case.case_id] = prepared
        return prepared


class TorchSegmentationDataset:
    def __init__(self, loader: OnDiskSegmentationDataset):
        self.loader = loader

    def __len__(self) -> int:
        return len(self.loader.cases)

    def __getitem__(self, index: int):
        import torch
        case = self.loader.cases[index]
        prepared = self.loader._load(case)
        for attempt in range(1, min(8, len(self.loader.cases))):
            if prepared is not None:
                break
            case = self.loader.cases[(index + attempt) % len(self.loader.cases)]
            prepared = self.loader._load(case)
        image, mask = prepared
        mask = np.asarray(mask, dtype=np.float32)[None]
        if self.loader.augment:
            image, mask = augment_pair(image, mask, self.loader.rng)
        return torch.from_numpy(np.ascontiguousarray(image)), torch.from_numpy(np.ascontiguousarray(mask))


def build_dataset(dataset: Dataset, split: str, config: TrainingConfig, logger: logging.Logger, max_cases: int | None) -> tuple[np.ndarray, np.ndarray]:
    cases = dataset.cases_for(split)
    if max_cases:
        cases = cases[:max_cases]
    images: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    for case in cases:
        image, mask = load_case_arrays(case, config.spatial_size, config.in_channels, log=logger)
        if mask is None:
            continue
        images.append(normalise(image, config.normalisation))
        masks.append(mask)
    if not images:
        raise SystemExit(
            f"No usable case in the '{split}' split: every case either has no mask or references a "
            "missing file. Training segmentation needs image-mask pairs."
        )
    return np.stack(images).astype(np.float32), np.stack(masks)[:, None].astype(np.float32)
def train(
    dataset: Dataset,
    config: TrainingConfig,
    args,
    logger: logging.Logger,
    checkpoint_dir: Path,
    run_dir: Path,
) -> dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader
    from _common import torch_device
    from app.ml.losses import build_loss
    seed_everything(config.seed)
    device = torch_device(args.device)
    logger.info("Training %s on %s with %s", config.model_id, device, config.loss)
    train_cases = dataset.cases_for("train")
    validation_cases = dataset.cases_for("validation")
    if args.max_cases:
        train_cases = train_cases[: args.max_cases]
    if not train_cases:
        raise SystemExit("The training split is empty.")
    lazy = bool(getattr(args, "lazy_loading", False)) or any(
        case.source_kind.startswith("brats") for case in train_cases
    )
    if not validation_cases:
        validation_cases = train_cases[: max(1, min(len(train_cases) // 5, 32))]
        logger.warning(
            "No validation split is declared; using %d training case(s) for monitoring. Report "
            "these numbers as training-time monitoring only.",
            len(validation_cases),
        )
    if lazy:
        logger.info(
            "On-disk loading: %d training case(s), %d validation case(s) per pass",
            len(train_cases),
            min(len(validation_cases), getattr(args, "val_cases", 64)),
        )
        train_loader = DataLoader(
            TorchSegmentationDataset(
                OnDiskSegmentationDataset(train_cases, config, augment=config.augment, seed=config.seed, logger=logger)
            ),
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            drop_last=False,
        )
        validation_loader = DataLoader(
            TorchSegmentationDataset(
                OnDiskSegmentationDataset(
                    validation_cases[: getattr(args, "val_cases", 64)],
                    config,
                    augment=False,
                    seed=config.seed + 1,
                    logger=None,
                )
            ),
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
    else:
        train_images, train_masks = build_dataset(dataset, "train", config, logger, args.max_cases)
        validation_images, validation_masks = build_dataset(dataset, "validation", config, logger, args.max_cases or 64)
        logger.info("Train tensors %s · validation tensors %s", train_images.shape, validation_images.shape)
        train_loader = _ArrayLoader(train_images, train_masks, config.batch_size, shuffle=True, seed=config.seed, augment=config.augment)
        validation_loader = _ArrayLoader(validation_images, validation_masks, config.batch_size, shuffle=False, seed=config.seed)
    model = build_architecture(config)
    model.to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    criterion = build_loss(config.loss)
    scheduler = _build_scheduler(optimiser, config.scheduler)
    start_epoch = 1
    best_score = -1.0
    history: list[dict[str, float]] = []
    if args.resume is not None:
        state = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(state["model_state"])
        if "optimiser_state" in state:
            optimiser.load_state_dict(state["optimiser_state"])
        if scheduler is not None and state.get("scheduler_state"):
            scheduler.load_state_dict(state["scheduler_state"])
        start_epoch = int(state.get("epoch", 0)) + 1
        best_score = float(state.get("best_score", -1.0))
        history = list(state.get("history", []))
        logger.info("Resumed from %s at epoch %s (best score %.4f)", args.resume, start_epoch, best_score)
    checkpoint_path = checkpoint_dir / f"{config.model_id}.pt"
    epochs_without_improvement = 0
    started = time.perf_counter()
    for epoch in range(start_epoch, config.epochs + 1):
        model.train()
        running = 0.0
        seen = 0
        for images, masks in train_loader:
            tensor = images.to(device)
            target = masks.to(device)
            optimiser.zero_grad(set_to_none=True)
            logits = model(tensor)
            if isinstance(logits, (list, tuple)):
                logits = logits[0]
            loss = criterion(logits, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimiser.step()
            running += float(loss.detach()) * tensor.shape[0]
            seen += tensor.shape[0]
        train_loss = running / max(1, seen)
        validation_loss, validation_dice = _validate_loader(model, validation_loader, criterion, device)
        if scheduler is not None:
            if config.scheduler == "plateau":
                scheduler.step(validation_loss)
            else:
                scheduler.step()
        current_lr = optimiser.param_groups[0]["lr"]
        history.append(
            {
                "epoch": epoch,
                "train_loss": round(train_loss, 6),
                "val_loss": round(validation_loss, 6),
                "val_dice": round(validation_dice, 6),
                "learning_rate": round(current_lr, 8),
            }
        )
        logger.info(
            "epoch %3d | train %.4f | val %.4f | val dice %.4f | lr %.2e",
            epoch,
            train_loss,
            validation_loss,
            validation_dice,
            current_lr,
        )
        if validation_dice > best_score:
            best_score = validation_dice
            epochs_without_improvement = 0
            save_checkpoint(
                checkpoint_path,
                {
                    "model_state": model.state_dict(),
                    "optimiser_state": optimiser.state_dict(),
                    "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
                    "epoch": epoch,
                    "best_score": best_score,
                    "history": history,
                    "descriptor": config.descriptor(),
                    "config": config.__dict__,
                    "dataset": dataset.summary(),
                    "seed": config.seed,
                    "code_revision": code_revision(),
                    "trained_at": utc_now(),
                },
            )
            logger.info("  new best validation Dice %.4f -> %s", best_score, checkpoint_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                logger.info("Early stopping after %s epochs without improvement.", config.patience)
                break
    elapsed = time.perf_counter() - started
    record = {
        "model_id": config.model_id,
        "status": "completed",
        "architecture": config.architecture,
        "config": config.__dict__,
        "dataset": dataset.summary(),
        "splits_used": ["train", "validation"],
        "test_split_touched": False,
        "seed": config.seed,
        "epochs_run": len(history),
        "best_validation_dice": round(best_score, 6),
        "duration_seconds": round(elapsed, 2),
        "checkpoint": str(checkpoint_path),
        "checkpoint_digest": file_digest(checkpoint_path) if checkpoint_path.exists() else None,
        "history": history,
        "on_disk_loading": lazy,
        "train_cases": len(train_cases),
        "validation_cases": len(validation_cases),
        "environment": describe_environment(),
        "code_revision": code_revision(),
        "finished_at": utc_now(),
    }
    path = write_run_record(run_dir, record)
    logger.info("Run record written to %s", path)
    logger.info("Best validation Dice: %.4f (validation split only)", best_score)
    return record


class _ArrayLoader:
    def __init__(self, images: np.ndarray, masks: np.ndarray, batch_size: int, *, shuffle: bool, seed: int, augment: bool = False):
        self.images = images
        self.masks = masks
        self.batch_size = max(1, batch_size)
        self.shuffle = shuffle
        self.augment = augment
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return int(np.ceil(len(self.images) / self.batch_size))

    def __iter__(self):
        order = np.arange(len(self.images))
        if self.shuffle:
            self.rng.shuffle(order)
        for start in range(0, len(order), self.batch_size):
            index = order[start : start + self.batch_size]
            images = self.images[index]
            masks = self.masks[index]
            if self.augment:
                pairs = [
                    augment_pair(image, np.asarray(mask, dtype=np.float32)[None], self.rng)
                    for image, mask in zip(images, masks, strict=False)
                ]
                images = np.stack([pair[0] for pair in pairs]).astype(np.float32)
                masks = np.stack([pair[1][0] for pair in pairs]).astype(np.float32)
            yield images, masks


def _validate_loader(model, loader, criterion, device) -> tuple[float, float]:
    import torch
    from app.utils.metrics import dice as dice_metric
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    seen = 0
    with torch.no_grad():
        for images, masks in loader:
            tensor = images.to(device)
            target = masks.to(device)
            logits = model(tensor)
            if isinstance(logits, (list, tuple)):
                logits = logits[0]
            loss = criterion(logits, target)
            probability = _to_probability(logits, "probability")
            predicted = (probability > 0.5).detach().cpu().numpy().astype(np.uint8)
            truth = (target.detach().cpu().numpy() > 0.5).astype(np.uint8)
            batch = predicted.shape[0]
            total_dice += float(dice_metric(truth, predicted)) * batch
            total_loss += float(loss.detach()) * batch
            seen += batch
    if not seen:
        return float("nan"), 0.0
    return total_loss / seen, total_dice / seen


def _build_scheduler(optimiser, kind: str):
    from torch.optim import lr_scheduler
    if kind == "cosine":
        return lr_scheduler.CosineAnnealingLR(optimiser, T_max=50, eta_min=1e-6)
    if kind == "plateau":
        return lr_scheduler.ReduceLROnPlateau(optimiser, mode="min", factor=0.5, patience=5, min_lr=1e-6)
    return None
def build_architecture(config: TrainingConfig):
    from app.ml.architectures.attention_unet import RAUNet
    from app.ml.architectures.dynamic_unet import DynamicUNet
    from app.ml.architectures.segnet import SegNet
    kwargs = config.architecture_kwargs or {}
    if config.architecture == "segnet":
        return SegNet(in_channels=config.in_channels, out_channels=config.out_channels)
    if config.architecture == "dynamic_unet":
        return DynamicUNet(kwargs.get("filters", [16, 32, 64, 128, 256]), input_channels=config.in_channels, output_channels=config.out_channels)
    if config.architecture == "raunet":
        return RAUNet(input_channels=config.in_channels, output_channels=config.out_channels, **kwargs)
    raise SystemExit(f"Unknown architecture '{config.architecture}'.")
def _validate(model, images: np.ndarray, masks: np.ndarray, criterion, device, batch_size: int) -> tuple[float, float]:
    import torch
    from app.utils.metrics import dice as dice_metric
    model.eval()
    losses: list[float] = []
    dices: list[float] = []
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            tensor = torch.from_numpy(images[start : start + batch_size]).to(device)
            target = torch.from_numpy(masks[start : start + batch_size]).to(device)
            logits = model(tensor)
            if isinstance(logits, (list, tuple)):
                logits = logits[0]
            losses.append(float(criterion(logits, target)))
            probability = _to_probability(logits, getattr(model, "output_semantics", "probability"))
            probability = probability[:, 0].detach().cpu().numpy()
            for index in range(probability.shape[0]):
                dices.append(dice_metric(probability[index], masks[start + index, 0]))
    return (float(np.mean(losses)) if losses else float("nan"), float(np.mean(dices)) if dices else 0.0)
def _to_probability(logits, semantics: str):
    import torch
    if semantics == "logit":
        return torch.sigmoid(logits)
    return logits
def summarise_predictions(model, images: np.ndarray, device, batch_size: int) -> list[np.ndarray]:
    import torch
    model.eval()
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            tensor = torch.from_numpy(images[start : start + batch_size]).to(device)
            logits = model(tensor)
            if isinstance(logits, (list, tuple)):
                logits = logits[0]
            probability = _to_probability(logits, getattr(model, "output_semantics", "probability"))
            outputs.extend(probability[:, 0].detach().cpu().numpy())
    return outputs
def prepare_split(dataset: Dataset, split: str, config: TrainingConfig, logger: logging.Logger, max_cases: int | None) -> tuple[np.ndarray, np.ndarray]:
    return build_dataset(dataset, split, config, logger, max_cases)
def case_ids(dataset: Dataset, split: str) -> list[str]:
    return [case.case_id for case in dataset.cases_for(split)]
def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
def parse_size(value: str) -> int:
    size = int(value)
    if size % 32:
        raise SystemExit("--size must be a multiple of 32 so the encoder/decoder depths line up.")
    return size
def channel_default(mode: str) -> int:
    return 4 if mode == "four_channel" else 1
def resolve_config_defaults(config: TrainingConfig, dataset: Dataset) -> TrainingConfig:
    channels = sorted({case.channels for case in dataset.cases})
    if config.in_channels not in channels and channels:
        raise SystemExit(
            f"This dataset provides {channels}-channel cases, but {config.model_id} expects "
            f"{config.in_channels} channels. Train the architecture that matches the data, or supply "
            "four co-registered modalities for the four-channel model."
        )
    return config
def sample_batches(images: Sequence[np.ndarray], batch_size: int) -> list[np.ndarray]:
    return [images[start : start + batch_size] for start in range(0, len(images), batch_size)]
def assert_no_test_leakage(dataset: Dataset) -> None:
    seen: dict[str, str] = {}
    for split, ids in dataset.splits.items():
        for case_id in ids:
            if case_id in seen and seen[case_id] != split:
                raise SystemExit(
                    f"Case {case_id} appears in both '{seen[case_id]}' and '{split}'. Correct the "
                    "manifest before training: leaked cases invalidate every metric that follows."
                )
            seen[case_id] = split
    patients: dict[str, str] = {}
    for split, ids in dataset.splits.items():
        for case_id in ids:
            case = next((item for item in dataset.cases if item.case_id == case_id), None)
            if case is None:
                continue
            if case.patient_id in patients and patients[case.patient_id] != split:
                raise SystemExit(
                    f"Patient {case.patient_id} appears in both '{patients[case.patient_id]}' and "
                    f"'{split}'. Patient-level separation is required to avoid optimistic metrics."
                )
            patients[case.patient_id] = split
