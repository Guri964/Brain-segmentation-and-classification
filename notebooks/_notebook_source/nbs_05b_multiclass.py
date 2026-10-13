# %% [cell] 26. Multiclass dataset: ROI, radiomics features and schema check
# =============================================================================
# SECTION 12b - Radiomics for the second dataset (automatic brain-region ROI)
# =============================================================================
# The image-classification dataset has no lesion masks: each file is a single 2-D MRI slice
# with a class label. Radiomics therefore needs a region of interest that is *computed*, not
# annotated -- so one is derived per image (Otsu threshold, largest connected component,
# filling, one opening/closing pair) and every ROI is recorded together with its size. An ROI
# that ends up empty or degenerate disqualifies that image, and it is listed: no feature row is
# invented for it.
#
# The features come from the application's own extractor, in the same bare-named schema the
# serving path uses, so "extracted here" and "computed by the API" mean the same numbers.
# =============================================================================

MULTICLASS_FEATURE_TABLE = PATHS["artifacts"] / "classification_radiomics_features.csv"
MULTICLASS_FEATURES: dict = {"train": [], "validation": [], "test": []}
MULTICLASS_FEATURE_NAMES: list = []
MULTICLASS_ROI_REJECTS: list = []
MULTICLASS_EXTRACTED = False


def otsu_threshold(array):
    """Otsu's threshold on a uint8 image, in numpy only."""
    values = np.asarray(array, dtype=np.uint8).ravel()
    histogram = np.bincount(values, minlength=256).astype(np.float64)
    total = histogram.sum()
    if total <= 0:
        return 0
    probabilities = histogram / total
    omega = np.cumsum(probabilities)
    mu = np.cumsum(probabilities * np.arange(256))
    mu_total = mu[-1]
    denominator = omega * (1.0 - omega)
    denominator[denominator <= 0] = 1e-12
    between = (mu_total * omega - mu) ** 2 / denominator
    return int(np.argmax(between))


def largest_component(mask):
    """Largest 4-connected component of a boolean mask (scipy when available)."""
    try:
        from scipy import ndimage

        labels, count = ndimage.label(mask)
        if count <= 1:
            return mask
        sizes = ndimage.sum(mask, labels, index=range(1, count + 1))
        return labels == (int(np.argmax(sizes)) + 1)
    except Exception:  # pragma: no cover - scipy is a hard requirement in requirements.txt
        rows, columns = np.where(mask)
        if rows.size == 0:
            return mask
        return mask


def brain_roi(array, *, min_fraction=0.02):
    """Region of interest for a single grayscale slice, with its diagnostics.

    Returns (mask, diagnostics). The mask is the Otsu-foreground brain region, reduced to its
    largest connected component and morphologically cleaned. `min_fraction` guards against
    degenerate ROIs (a mostly-empty or almost-full mask).
    """
    array = np.asarray(array, dtype=np.uint8)
    try:
        from scipy import ndimage

        smooth = ndimage.gaussian_filter(array.astype(np.float32), sigma=1.2)
    except Exception:  # pragma: no cover
        smooth = array.astype(np.float32)
    threshold = otsu_threshold(smooth.astype(np.uint8))
    foreground = smooth > threshold
    # Keep the brain: drop the thin bright rim that some of these images carry.
    foreground = largest_component(foreground)
    try:
        from scipy import ndimage

        foreground = ndimage.binary_fill_holes(foreground)
        foreground = ndimage.binary_opening(foreground, iterations=2)
        foreground = ndimage.binary_closing(foreground, iterations=2)
        foreground = largest_component(foreground)
    except Exception:  # pragma: no cover
        pass
    fraction = float(foreground.mean())
    diagnostics = {"otsu_threshold": threshold, "roi_fraction": fraction, "roi_pixels": int(foreground.sum())}
    if fraction < min_fraction or fraction > 0.98:
        diagnostics["reason"] = f"degenerate ROI (foreground fraction {fraction:.4f})"
        diagnostics["accepted"] = False
        return None, diagnostics
    diagnostics["accepted"] = True
    return foreground.astype(bool), diagnostics


def multiclass_feature_row(path, class_name, split_name):
    """One feature row for one image, or (None, reason) when the extraction is impossible."""
    array = load_image_array(path, size=int(RUN["radiomics_image_size"]))
    mask, diagnostics = brain_roi(array)
    if mask is None:
        return None, {"image": str(path), "class": class_name, "reason": diagnostics.get("reason", "no ROI")}
    outcome = radiomics_service.extract(
        array.astype(np.float32),
        mask,
        spacing=None,  # no voxel spacing exists for these images: units stay pixels
        mask_source="auto_roi",
    )
    if not outcome.available:
        return None, {"image": str(path), "class": class_name, "reason": outcome.unavailable_reason}
    row = {name: value for name, value in numeric_features(outcome)}
    row.update(
        {
            "image_id": Path(path).stem,
            "image_relative_path": str(Path(path).name),
            "class_label": class_name,
            "split": split_name,
            "roi_pixels": diagnostics["roi_pixels"],
            "roi_fraction": diagnostics["roi_fraction"],
            "otsu_threshold": diagnostics["otsu_threshold"],
            "radiomics_backend": outcome.backend,
            "image_channel": "grayscale",
        }
    )
    return row, None


if not CLASSIFICATION_READY:
    print("Multiclass radiomics: skipped --", "the second dataset is not attached" if not CLASSIFICATION_IMAGES_AVAILABLE else "a split is empty")
else:
    limit = RUN["multiclass_radiomics_max_images"]
    rng_features = np.random.default_rng(SEED + 7)
    print(f"extracting radiomics on {RUN['radiomics_image_size']}px images with an automatic brain ROI")
    for split_name in ("train", "validation", "test"):
        for class_name in CLASSIFICATION_CLASSES:
            paths = [Path(path) for path in CLASSIFICATION_IMAGES[split_name].get(class_name, [])]
            if limit and len(paths) > limit:
                keep = rng_features.permutation(len(paths))[: int(limit)]
                paths = [paths[index] for index in sorted(keep.tolist())]
            for path in paths:
                row, rejection = multiclass_feature_row(path, class_name, split_name)
                if row is None:
                    MULTICLASS_ROI_REJECTS.append(rejection)
                    continue
                MULTICLASS_FEATURES[split_name].append(row)
    MULTICLASS_FEATURE_NAMES = list(MULTICLASS_FEATURES["train"][0]) if MULTICLASS_FEATURES["train"] else []
    MULTICLASS_FEATURE_NAMES = [
        name
        for name in MULTICLASS_FEATURE_NAMES
        if name
        not in {
            "image_id",
            "image_relative_path",
            "class_label",
            "split",
            "roi_pixels",
            "roi_fraction",
            "otsu_threshold",
            "radiomics_backend",
            "image_channel",
        }
    ]
    MULTICLASS_EXTRACTED = bool(MULTICLASS_FEATURES["train"] and MULTICLASS_FEATURES["validation"] and MULTICLASS_FEATURES["test"])

    rows_for_csv = [
        {**{name: row.get(name) for name in MULTICLASS_FEATURE_NAMES}, **{key: row[key] for key in (
            "image_id", "image_relative_path", "class_label", "split", "roi_pixels", "roi_fraction",
            "otsu_threshold", "radiomics_backend", "image_channel",
        )}}
        for split_name in ("train", "validation", "test")
        for row in MULTICLASS_FEATURES[split_name]
    ]
    if rows_for_csv:
        import csv as _csv

        with open(MULTICLASS_FEATURE_TABLE, "w", encoding="utf-8", newline="") as handle:
            writer = _csv.DictWriter(handle, fieldnames=list(rows_for_csv[0]))
            writer.writeheader()
            writer.writerows(rows_for_csv)

    roi_fraction_by_split = {
        split_name: (
            float(np.median([row["roi_fraction"] for row in rows])) if rows else None
        )
        for split_name, rows in MULTICLASS_FEATURES.items()
    }
    print("feature rows  :", {split: len(rows) for split, rows in MULTICLASS_FEATURES.items()})
    print("features      :", len(MULTICLASS_FEATURE_NAMES))
    print("median ROI fraction:", roi_fraction_by_split)
    print("rejected images:", len(MULTICLASS_ROI_REJECTS), "(first:", MULTICLASS_ROI_REJECTS[0] if MULTICLASS_ROI_REJECTS else "none", ")")

    # --- schema contract -------------------------------------------------------
    # The application's extractor defines the schema. Two things are checked here, both of them
    # about the mismatch that kept the shipped classifier disabled (21 expected features, 11 in
    # features_df.csv, no scaler, no feature order):
    #   1. the multiclass table and the BraTS table use the *same* feature names, because both
    #      come from the same extractor on the same kind of input, and
    #   2. every exported artefact records this exact ordered schema plus its scaler, so a
    #      served vector can never be silently padded or reordered.
    BRATS_SCHEMA = set(FEATURE_NAMES) if isinstance(FEATURE_NAMES, (list, tuple)) and FEATURE_NAMES else set()
    MULTICLASS_SCHEMA = set(MULTICLASS_FEATURE_NAMES)
    SCHEMA_CONTRACT = {
        "extractor": f"{radiomics_service.backend_status().get('selected_backend')} (app/ml/radiomics_native.py)",
        "multiclass_feature_count": len(MULTICLASS_FEATURE_NAMES),
        "brats_feature_count": len(BRATS_SCHEMA),
        "identical_to_brats_schema": bool(BRATS_SCHEMA) and BRATS_SCHEMA == MULTICLASS_SCHEMA,
        "missing_versus_brats": sorted(BRATS_SCHEMA - MULTICLASS_SCHEMA),
        "extra_versus_brats": sorted(MULTICLASS_SCHEMA - BRATS_SCHEMA),
        "legacy_checkpoint_note": (
            "radiomics_cnn_model.pth expects 21 unnamed features and shipped without a scaler or "
            "feature order, while the application's extractor produces the features listed above. "
            "It is never padded or reordered to fit: every artefact trained here carries its own "
            "ordered schema, scaler and label mapping, which is what makes it servable."
        ),
    }
    print("schema contract:", {key: SCHEMA_CONTRACT[key] for key in ("multiclass_feature_count", "identical_to_brats_schema")})
    write_json(PATHS["artifacts"] / "classification_schema_contract.json", SCHEMA_CONTRACT)

    # --- ROI figure ------------------------------------------------------------
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    preview_classes = CLASSIFICATION_CLASSES[:4]
    fig, axes = plt.subplots(2, len(preview_classes), figsize=(3.1 * max(1, len(preview_classes)), 6.4))
    axes = np.atleast_2d(axes)
    for index, class_name in enumerate(preview_classes):
        rows = [row for row in MULTICLASS_FEATURES["train"] if row["class_label"] == class_name]
        axes[0][index].set_title(class_name, fontsize=10)
        axes[0][index].axis("off")
        axes[1][index].axis("off")
        if not rows:
            continue
        source = next(
            (Path(path) for path in CLASSIFICATION_IMAGES["train"].get(class_name, []) if Path(path).stem == rows[0]["image_id"]),
            None,
        )
        if source is None:
            continue
        display = load_image_array(source, size=int(RUN["radiomics_image_size"]))
        mask, _ = brain_roi(display)
        axes[0][index].imshow(display, cmap="gray")
        axes[1][index].imshow(display, cmap="gray")
        if mask is not None:
            overlay = np.zeros(display.shape + (4,), dtype=np.float32)
            overlay[..., 0] = 1.0
            overlay[..., 3] = mask.astype(np.float32) * 0.45
            axes[1][index].imshow(overlay)
        axes[1][index].set_title(f"ROI {rows[0]['roi_fraction']:.2f} of the slice", fontsize=9)
    plt.tight_layout()
    roi_figure = PATHS["reports"] / "09_classification_roi.png"
    plt.savefig(roi_figure, dpi=130)
    plt.close(fig)
    print("figure saved  :", roi_figure)

# %% [cell] 27. Multiclass CNN training (from scratch, AMP, resumable)
# =============================================================================
# SECTION 13b - Image classifier for the second dataset
# =============================================================================
# A small convolutional network trained from scratch -- no pretrained weights, because the
# notebook has no internet access and weight downloads that cannot happen are not part of a
# reproducible run. Everything a long run needs is here: mixed precision, checkpoint/resume,
# progress with elapsed and ETA, early stopping, LR scheduling and validation-based selection.
# =============================================================================
MULTICLASS_CNN_TRAINED = False
MULTICLASS_CNN_HISTORY: list = []
MULTICLASS_CNN_SUMMARY: dict = {}


def classifier_architecture(channels, num_classes, dropout):
    """Small VGG-style 2-D classifier: three blocks, global average pooling, linear head."""
    import torch.nn as nn

    blocks = []
    in_channels = 1
    for width in channels:
        blocks += [
            nn.Conv2d(in_channels, width, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, width, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        ]
        in_channels = width
    return nn.Sequential(
        *blocks,
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Dropout(dropout),
        nn.Linear(in_channels, num_classes),
    )


class ImageClassificationTrainer:
    """Checkpointed training loop for the multiclass CNN (AMP, resume, ETA, early stopping)."""

    def __init__(self, model_id, model, *, num_classes, learning_rate, weight_decay, device=DEVICE, amp=AMP_ENABLED, dropout=0.3):
        import torch.nn as nn

        self.model_id = model_id
        self.model = model.to(device)
        self.device = device
        self.num_classes = int(num_classes)
        self.dropout = float(dropout)
        self.criterion = nn.CrossEntropyLoss()
        self.amp = bool(amp and device.type == "cuda")
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=50, eta_min=learning_rate / 20)
        try:
            self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp)
            self.autocast = lambda: torch.amp.autocast("cuda", enabled=self.amp)
            self.amp_api = "torch.amp"
        except (AttributeError, TypeError):  # pragma: no cover - older torch
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp)
            self.autocast = lambda: torch.cuda.amp.autocast(enabled=self.amp)
            self.amp_api = "torch.cuda.amp"
        self.directory = PATHS["checkpoints"] / model_id
        self.directory.mkdir(parents=True, exist_ok=True)
        self.history_path = self.directory / "history.json"
        self.state = {"epoch": 0, "step": 0, "best_macro_f1": -1.0, "best_epoch": None, "history": read_json(self.history_path, default=[]) or []}

    def save(self, *, metrics=None, config=None, epochs_planned=None, best=False):
        payload = {
            "state_dict": {key: value.detach().cpu() for key, value in self.model.state_dict().items()},
            "model_id": self.model_id,
            "task": "multiclass_classification",
            "classes": list(CLASSIFICATION_CLASSES),
            "architecture": "classifier_architecture(channels, num_classes, dropout)",
            "config": config or {},
            "epoch": self.state["epoch"],
            "step": self.state["step"],
            "best_macro_f1": self.state["best_macro_f1"],
            "history": self.state["history"],
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict() if self.amp else None,
            "numpy_rng": np.random.get_state(),
            "torch_rng": torch.get_rng_state(),
            "metrics": metrics or {},
            "epochs_planned": epochs_planned,
            "saved_at": utc_now(),
            "amp_api": self.amp_api,
        }
        torch.save(payload, self.directory / "last.pt")
        if best:
            torch.save(payload, self.directory / "best.pt")
        write_json(self.history_path, self.state["history"])
        return self.directory / ("best.pt" if best else "last.pt")

    def resume(self, *, allow=True):
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
            print(f"WARNING: optimiser state from {path.name} could not be restored ({exc}); continuing fresh")
        self.state["epoch"] = int(payload.get("epoch", 0))
        self.state["step"] = int(payload.get("step", 0))
        self.state["best_macro_f1"] = float(payload.get("best_macro_f1", -1.0))
        if isinstance(payload.get("history"), list):
            self.state["history"] = payload["history"]
        print(f"resumed {self.model_id} at epoch {self.state['epoch']} (best validation macro-F1 {self.state['best_macro_f1']:.4f})")
        return {"resumed": True, "epoch": self.state["epoch"], "best_macro_f1": self.state["best_macro_f1"]}

    # -- data ----------------------------------------------------------------
    @staticmethod
    def _tensor(array):
        """Per-image standardisation, then a channel dimension. Recorded in the manifest."""
        array = np.asarray(array, dtype=np.float32)
        array = (array - array.mean()) / max(float(array.std()), 1e-6)
        return torch.from_numpy(array[None])

    def batches(self, index, batch_size, rng, augment, max_steps=None):
        order = list(index)
        if augment:
            rng.shuffle(order)
        steps = 0
        for start in range(0, len(order), batch_size):
            chunk = order[start : start + batch_size]
            images, labels = [], []
            for path, class_index in chunk:
                images.append(self._tensor(load_image_array(path)))
                labels.append(class_index)
            batch = torch.stack(images)
            if augment:
                if rng.random() < 0.5:
                    batch = torch.flip(batch, dims=[3])
                angles = torch.from_numpy(rng.uniform(-10, 10, size=len(chunk)).astype(np.float32))
                scales = torch.from_numpy(rng.uniform(0.9, 1.1, size=len(chunk)).astype(np.float32))
                theta = torch.zeros(len(chunk), 2, 3)
                theta[:, 0, 0] = torch.cos(angles * np.pi / 180) / scales
                theta[:, 0, 1] = -torch.sin(angles * np.pi / 180) / scales
                theta[:, 1, 0] = torch.sin(angles * np.pi / 180) / scales
                theta[:, 1, 1] = torch.cos(angles * np.pi / 180) / scales
                grid = torch.nn.functional.affine_grid(theta, list(batch.shape), align_corners=False)
                batch = torch.nn.functional.grid_sample(batch, grid, align_corners=False, padding_mode="border")
            yield batch.to(self.device), torch.tensor(labels, dtype=torch.long, device=self.device)
            steps += 1
            if max_steps is not None and steps >= max_steps:
                return

    def predict_probabilities(self, paths, class_indices, batch_size):
        """Softmax probabilities for a list of images (used for the stacked ensemble)."""
        self.model.eval()
        outputs = []
        with torch.no_grad():
            for start in range(0, len(paths), batch_size):
                chunk = paths[start : start + batch_size]
                batch = torch.stack([self._tensor(load_image_array(path)) for path in chunk]).to(self.device)
                with self.autocast():
                    logits = self.model(batch)
                outputs.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
        return np.concatenate(outputs) if outputs else np.zeros((0, self.num_classes), dtype=np.float32)

    def evaluate(self, paths, class_indices, batch_size):
        from sklearn.metrics import accuracy_score, f1_score

        probabilities = self.predict_probabilities(paths, class_indices, batch_size)
        if not len(probabilities):
            return {"macro_f1": None, "accuracy": None, "samples": 0}
        predicted = probabilities.argmax(axis=1)
        return {
            "macro_f1": float(f1_score(class_indices, predicted, average="macro", zero_division=0)),
            "accuracy": float(accuracy_score(class_indices, predicted)),
            "samples": int(len(class_indices)),
        }

    # -- training -------------------------------------------------------------
    def train(self, *, train_index, val_index, epochs, batch_size, max_steps_per_epoch=None, early_stopping_patience=6, budget=None, config=None, validation_every=1):
        from tqdm.auto import tqdm

        rng = np.random.default_rng(SEED + 11)
        self.resume(allow=True)
        started = time.time()
        history = list(self.state["history"])
        best = self.state["best_macro_f1"]
        stale = 0
        epochs_planned = epochs
        last_path = None
        for epoch in range(self.state["epoch"] + 1, epochs + 1):
            self.model.train()
            if budget is not None and budget.expired:
                print("session budget reached; stopping cleanly (the next session resumes from last.pt)")
                break
            epoch_started = time.time()
            running, seen = 0.0, 0
            progress = tqdm(
                self.batches(train_index, batch_size, rng, augment=True, max_steps=max_steps_per_epoch),
                total=(min(max_steps_per_epoch, int(np.ceil(len(train_index) / batch_size))) if max_steps_per_epoch else int(np.ceil(len(train_index) / batch_size))),
                desc=f"{self.model_id} epoch {epoch}/{epochs}",
                leave=False,
            )
            for images, labels in progress:
                self.optimizer.zero_grad(set_to_none=True)
                with self.autocast():
                    loss = self.criterion(self.model(images), labels)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.state["step"] += 1
                running += float(loss.detach().cpu()) * len(labels)
                seen += len(labels)
                progress.set_postfix(loss=f"{running / max(seen, 1):.4f}", lr=f"{self.scheduler.get_last_lr()[0]:.2e}")
                if RUN["checkpoint_every_steps"] and self.state["step"] % int(RUN["checkpoint_every_steps"]) == 0:
                    self.save(metrics={"running_loss": running / max(seen, 1)}, config=config, epochs_planned=epochs_planned)
            progress.close()
            self.scheduler.step()
            self.state["epoch"] = epoch
            epoch_seconds = time.time() - epoch_started
            metrics = {"epoch_seconds": epoch_seconds, "train_loss": running / max(seen, 1), "samples": seen}
            if epoch % max(1, validation_every) == 0 or epoch == epochs:
                validation = self.evaluate([path for path, _ in val_index], np.asarray([index for _, index in val_index]), batch_size)
                metrics.update({f"validation_{key}": value for key, value in validation.items()})
            record = {"epoch": epoch, "lr": self.scheduler.get_last_lr()[0], **metrics, "elapsed_minutes": (time.time() - started) / 60}
            history.append(record)
            self.state["history"] = history
            improved = metrics.get("validation_macro_f1") is not None and metrics["validation_macro_f1"] > best
            if improved:
                best = metrics["validation_macro_f1"]
                self.state["best_macro_f1"] = best
                self.state["best_epoch"] = epoch
                stale = 0
            else:
                stale += 1
            last_path = self.save(metrics=metrics, config=config, epochs_planned=epochs_planned, best=improved)
            eta = ""
            if metrics.get("validation_macro_f1") is not None:
                eta = f" | val macro-F1 {metrics['validation_macro_f1']:.4f}{' (best)' if improved else ''}"
            print(
                f"{self.model_id} epoch {epoch}/{epochs}: loss {metrics['train_loss']:.4f} | "
                f"{epoch_seconds:.1f}s/epoch | elapsed {(time.time() - started) / 60:.1f} min | "
                f"step {self.state['step']}{eta}"
            )
            if early_stopping_patience and stale >= early_stopping_patience:
                print(f"early stopping: no validation improvement for {stale} epoch(s)")
                break
            if budget is not None and budget.require(5.0) is None and budget.expired:
                break
        return {
            "model_id": self.model_id,
            "epochs_completed": self.state["epoch"],
            "epochs_planned": epochs_planned,
            "steps": self.state["step"],
            "best_macro_f1": self.state["best_macro_f1"] if self.state["best_macro_f1"] >= 0 else None,
            "best_epoch": self.state["best_epoch"],
            "history": history,
            "minutes": (time.time() - started) / 60,
            "checkpoint": str(self.directory / ("best.pt" if self.state["best_epoch"] else "last.pt")),
            "resumable_from": str(self.directory / "last.pt"),
            "stopped_early": bool(early_stopping_patience and stale >= early_stopping_patience),
        }


if not (CLASSIFICATION_READY and MULTICLASS_EXTRACTED):
    print("Multiclass CNN: skipped --", "dataset missing" if not CLASSIFICATION_READY else "no radiomics rows")
else:
    MULTICLASS_LABEL_TO_INDEX = {name: index for index, name in enumerate(CLASSIFICATION_CLASSES)}
    MULTICLASS_INDEX_TO_LABEL = {str(index): name for name, index in MULTICLASS_LABEL_TO_INDEX.items()}

    def _index_for(split_name):
        return [
            (Path(path), MULTICLASS_LABEL_TO_INDEX[class_name])
            for class_name in CLASSIFICATION_CLASSES
            for path in CLASSIFICATION_IMAGES[split_name].get(class_name, [])
        ]

    TRAIN_INDEX = _index_for("train")
    VALIDATION_INDEX = _index_for("validation")
    TEST_INDEX = _index_for("test")
    print("train/validation/test images:", len(TRAIN_INDEX), len(VALIDATION_INDEX), len(TEST_INDEX))

    MULTICLASS_CNN_CONFIG = {
        "model_id": "brain_tumor_mri_cnn",
        "task": "multiclass_classification",
        "classes": CLASSIFICATION_CLASSES,
        "dataset": CLASSIFICATION_DATASET_NAME,
        "input": f"grayscale {RUN['classification_image_size']}x{RUN['classification_image_size']}, per-image standardisation",
        "channels": list(RUN["cnn_channels"]),
        "dropout": RUN["cnn_dropout"],
        "optimiser": f"AdamW lr={RUN['cnn_learning_rate']} weight_decay={RUN['cnn_weight_decay']}",
        "scheduler": "cosine annealing",
        "batch_size": RUN["cnn_batch_size"],
        "epochs": RUN["cnn_epochs"],
        "augmentation": "horizontal flip, rotation +/-10 deg, scale 0.9-1.1",
        "selection": "best validation macro-F1",
        "seed": SEED,
        "amp": AMP_ENABLED,
        "pretrained_weights": None,
    }

    cnn_model = classifier_architecture(tuple(RUN["cnn_channels"]), len(CLASSIFICATION_CLASSES), float(RUN["cnn_dropout"]))
    MULTICLASS_CNN_PARAMETERS = sum(parameter.numel() for parameter in cnn_model.parameters())
    print(f"CNN parameters: {MULTICLASS_CNN_PARAMETERS:,}")

    if not RUN.get("skip_multiclass_cnn", False):
        cnn_trainer = ImageClassificationTrainer(
            "brain_tumor_mri_cnn",
            cnn_model,
            num_classes=len(CLASSIFICATION_CLASSES),
            learning_rate=RUN["cnn_learning_rate"],
            weight_decay=RUN["cnn_weight_decay"],
            dropout=RUN["cnn_dropout"],
        )
        MULTICLASS_CNN_SUMMARY = cnn_trainer.train(
            train_index=TRAIN_INDEX,
            val_index=VALIDATION_INDEX,
            epochs=RUN["cnn_epochs"],
            batch_size=RUN["cnn_batch_size"],
            max_steps_per_epoch=RUN["cnn_max_steps_per_epoch"],
            early_stopping_patience=RUN["cnn_early_stopping_patience"],
            validation_every=RUN["cnn_validation_every_epochs"],
            budget=TimeBudget(RUN["session_budget_minutes"]),
            config=MULTICLASS_CNN_CONFIG,
        )
        MULTICLASS_CNN_SUMMARY["parameters"] = MULTICLASS_CNN_PARAMETERS
        MULTICLASS_CNN_HISTORY = MULTICLASS_CNN_SUMMARY["history"]
        MULTICLASS_CNN_TRAINED = True
        write_json(PATHS["artifacts"] / "classification_cnn_training_summary.json", MULTICLASS_CNN_SUMMARY)
        with open(PATHS["reports"] / "10_classification_cnn_curves.json", "w", encoding="utf-8") as handle:
            json.dump(MULTICLASS_CNN_HISTORY, handle, indent=2)
    else:
        print("CNN training skipped by configuration")

# %% [cell] 28. Radiomics classifiers on the same images
# =============================================================================
# SECTION 14b - Feature-based members for the second dataset
# =============================================================================
# Two feature-based classifiers on the radiomics table, fitted exactly like the segmentation-side
# classifiers: imputation and scaling from the training split only, validation macro-F1 for
# selection, test metrics reported once. Their probabilities are what the stacked ensemble
# consumes further down.
# =============================================================================
MULTICLASS_FEATURE_MODELS: dict = {}
MULTICLASS_PREPROCESSING: dict = {}
MULTICLASS_FEATURE_REPORTS: dict = {}

if not MULTICLASS_EXTRACTED:
    print("Radiomics classifiers: skipped -- no feature rows")
else:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.preprocessing import StandardScaler

    def _matrix(rows):
        return np.asarray([[row.get(name, np.nan) for name in MULTICLASS_FEATURE_NAMES] for row in rows], dtype=np.float64)

    X_train_raw = _matrix(MULTICLASS_FEATURES["train"])
    X_val_raw = _matrix(MULTICLASS_FEATURES["validation"])
    X_test_raw = _matrix(MULTICLASS_FEATURES["test"])
    y_train = np.asarray([MULTICLASS_LABEL_TO_INDEX[row["class_label"]] for row in MULTICLASS_FEATURES["train"]])
    y_val = np.asarray([MULTICLASS_LABEL_TO_INDEX[row["class_label"]] for row in MULTICLASS_FEATURES["validation"]])
    y_test = np.asarray([MULTICLASS_LABEL_TO_INDEX[row["class_label"]] for row in MULTICLASS_FEATURES["test"]])

    multiclass_imputer = SimpleImputer(strategy="median").fit(X_train_raw)
    multiclass_scaler = StandardScaler().fit(multiclass_imputer.transform(X_train_raw))
    X_train = multiclass_scaler.transform(multiclass_imputer.transform(X_train_raw))
    X_val = multiclass_scaler.transform(multiclass_imputer.transform(X_val_raw))
    X_test = multiclass_scaler.transform(multiclass_imputer.transform(X_test_raw))
    MULTICLASS_PREPROCESSING = {"imputer": multiclass_imputer, "scaler": multiclass_scaler}

    constant_features = [name for name, spread in zip(MULTICLASS_FEATURE_NAMES, np.nanstd(X_train, axis=0)) if spread <= 1e-9]
    print(f"features: {len(MULTICLASS_FEATURE_NAMES)} | constant in the training split (kept, zeroed by scaling): {len(constant_features)}")

    multiclass_candidates = {
        "radiomics_logistic_regression": LogisticRegression(max_iter=3000, class_weight="balanced", random_state=SEED),
        "radiomics_hist_gradient_boosting": HistGradientBoostingClassifier(random_state=SEED, max_iter=300, learning_rate=0.08),
    }
    for name, estimator in multiclass_candidates.items():
        estimator.fit(X_train, y_train)
        entry = {"model": name, "feature_source": "auto_roi"}
        for split_name, X, y in (("validation", X_val, y_val), ("test", X_test, y_test)):
            probabilities = estimator.predict_proba(X)
            predicted = probabilities.argmax(axis=1)
            entry[split_name] = {
                "accuracy": float(accuracy_score(y, predicted)),
                "macro_f1": float(f1_score(y, predicted, average="macro", zero_division=0)),
                "samples": int(len(y)),
            }
        MULTICLASS_FEATURE_MODELS[name] = estimator
        MULTICLASS_FEATURE_REPORTS[name] = entry
    for name, entry in MULTICLASS_FEATURE_REPORTS.items():
        print(f"{name:<34} validation macro-F1 {entry['validation']['macro_f1']:.4f} | test {entry['test']['macro_f1']:.4f}")

# %% [cell] 29. Stacked ensemble: compatibility guard, stacking layer, honest comparison
# =============================================================================
# SECTION 17b - Stacked ensemble for the multiclass task
# =============================================================================
# Members are combined only when they are actually compatible: same dataset, same label space,
# same class order, same images, and outputs that are proper probability distributions. Members
# that fail any of those checks are excluded *with the reason printed* -- which is why the
# BraTS HGG/LGG classifier, a different dataset with a different label space, can never end up
# in this ensemble.
#
# The stacking layer (multinomial logistic regression) is fitted on the validation split's
# member probabilities and scored once on the held-out test split. The comparison against the
# best single member is reported as measured: no superiority is claimed beyond what the numbers
# on that split show.
# =============================================================================
MULTICLASS_STACK_RESULT: dict = {"state": "not_available", "reason": "the multiclass pipeline did not run"}


def stack_member(name, classes, dataset_id, probabilities, *, fitted_on):
    return {
        "name": name,
        "classes": tuple(classes),
        "dataset_id": dataset_id,
        "probabilities": probabilities,
        "fitted_on": fitted_on,
    }


def check_stackable(members):
    """Return (compatible_members, rejections). A member is rejected with a reason, never patched."""
    accepted, rejections = [], []
    if not members:
        return accepted, [{"name": None, "reason": "no members supplied"}]
    reference = members[0]
    for member in members:
        problems = []
        if member["classes"] != reference["classes"]:
            problems.append(f"label space differs: {member['classes']} vs {reference['classes']}")
        if member["dataset_id"] != reference["dataset_id"]:
            problems.append(f"dataset differs: {member['dataset_id']!r} vs {reference['dataset_id']!r}")
        probabilities = member["probabilities"]
        expected_classes = len(reference["classes"])
        for split_name, matrix in probabilities.items():
            if matrix.ndim != 2 or matrix.shape[1] != expected_classes:
                problems.append(f"{split_name}: shape {matrix.shape} does not match {expected_classes} classes")
                continue
            if not np.all(np.isfinite(matrix)):
                problems.append(f"{split_name}: contains NaN or infinite values")
            sums = matrix.sum(axis=1)
            if matrix.size and not np.allclose(sums, 1.0, atol=1e-3):
                problems.append(f"{split_name}: rows do not sum to 1 (max deviation {float(np.abs(sums - 1).max()):.4f})")
            if matrix.min() < -1e-6 or matrix.max() > 1 + 1e-6:
                problems.append(f"{split_name}: values outside [0, 1]")
        if problems:
            rejections.append({"name": member["name"], "reason": "; ".join(problems)})
        else:
            accepted.append(member)
    # every accepted member must describe the same images in the same order
    if len(accepted) > 1:
        sizes = {name: matrix.shape[0] for name, matrix in accepted[0]["probabilities"].items()}
        for member in accepted[1:]:
            for split_name, matrix in member["probabilities"].items():
                if matrix.shape[0] != sizes.get(split_name):
                    rejections.append(
                        {"name": member["name"], "reason": f"{split_name}: {matrix.shape[0]} rows, expected {sizes.get(split_name)}"}
                    )
                    accepted = [entry for entry in accepted if entry["name"] != member["name"]]
    return accepted, rejections


if not (MULTICLASS_CNN_TRAINED and MULTICLASS_FEATURE_MODELS):
    print("Stacked ensemble: skipped -- need the CNN and the feature-based members")
    MULTICLASS_STACK_RESULT = {"state": "skipped", "reason": "a required member was not trained"}
else:
    from sklearn.linear_model import LogisticRegression as _StackingLogisticRegression
    from sklearn.metrics import accuracy_score, f1_score

    MULTICLASS_DATASET_ID = "brain_tumor_mri_dataset::" + ",".join(CLASSIFICATION_CLASSES)

    def _paths_for(split_name):
        return [path for path, _ in _index_for(split_name)]

    def _labels_for(split_name):
        return np.asarray([index for _, index in _index_for(split_name)])

    cnn_probabilities = {
        split_name: cnn_trainer.predict_probabilities(_paths_for(split_name), _labels_for(split_name), RUN["cnn_batch_size"])
        for split_name in ("train", "validation", "test")
    }
    candidates = [
        stack_member("cnn", CLASSIFICATION_CLASSES, MULTICLASS_DATASET_ID, cnn_probabilities, fitted_on="train"),
        stack_member(
            "radiomics_logistic_regression",
            CLASSIFICATION_CLASSES,
            MULTICLASS_DATASET_ID,
            {split: MULTICLASS_FEATURE_MODELS["radiomics_logistic_regression"].predict_proba(X) for split, X in (("train", X_train), ("validation", X_val), ("test", X_test))},
            fitted_on="train",
        ),
        stack_member(
            "radiomics_hist_gradient_boosting",
            CLASSIFICATION_CLASSES,
            MULTICLASS_DATASET_ID,
            {split: MULTICLASS_FEATURE_MODELS["radiomics_hist_gradient_boosting"].predict_proba(X) for split, X in (("train", X_train), ("validation", X_val), ("test", X_test))},
            fitted_on="train",
        ),
    ]
    # Deliberately offered as a candidate so the guard can demonstrate the refusal: the BraTS
    # HGG/LGG classifier is a different dataset with a different label space.
    if CLASSIFICATION_EVALUATION and CLASSIFICATION_EVALUATION.get("comparison"):
        candidates.append(
            stack_member(
                "brats_hgg_lgg_classifier",
                ("HGG", "LGG"),
                "brats2020::HGG,LGG",
                {split: np.zeros((len(_paths_for("validation") if split == "validation" else _paths_for("test")), 2), dtype=np.float32) for split in ("validation", "test")},
                fitted_on="brats patient-level train split",
            )
        )
    accepted, rejections = check_stackable(candidates)
    print("members accepted :", [member["name"] for member in accepted])
    for rejection in rejections:
        print(f"member rejected  : {rejection['name']} -- {rejection['reason']}")

    if len(accepted) < 2:
        MULTICLASS_STACK_RESULT = {"state": "skipped", "reason": "fewer than two compatible members", "rejections": rejections}
        print("Not enough compatible members to stack: reporting the single member instead.")
    else:
        y_validation = _labels_for("validation")
        y_test = _labels_for("test")
        stack_features_validation = np.concatenate([member["probabilities"]["validation"] for member in accepted], axis=1)
        stack_features_test = np.concatenate([member["probabilities"]["test"] for member in accepted], axis=1)
        stacker = _StackingLogisticRegression(max_iter=3000, class_weight="balanced", random_state=SEED)
        stacker.fit(stack_features_validation, y_validation)
        stack_probabilities = {
            "validation": stacker.predict_proba(stack_features_validation),
            "test": stacker.predict_proba(stack_features_test),
        }
        average_probabilities = {
            "validation": np.mean([member["probabilities"]["validation"] for member in accepted], axis=0),
            "test": np.mean([member["probabilities"]["test"] for member in accepted], axis=0),
        }

        def _score(probabilities, labels):
            predicted = probabilities.argmax(axis=1)
            return {
                "accuracy": float(accuracy_score(labels, predicted)),
                "macro_f1": float(f1_score(labels, predicted, average="macro", zero_division=0)),
                "samples": int(len(labels)),
            }

        comparison = {
            member["name"]: {
                "validation": _score(member["probabilities"]["validation"], y_validation),
                "test": _score(member["probabilities"]["test"], y_test),
            }
            for member in accepted
        }
        comparison["simple_average"] = {
            "validation": _score(average_probabilities["validation"], y_validation),
            "test": _score(average_probabilities["test"], y_test),
        }
        comparison["stacked_logistic_regression"] = {
            "validation": _score(stack_probabilities["validation"], y_validation),
            "test": _score(stack_probabilities["test"], y_test),
        }
        best_single = max(
            (name for name in comparison if name not in {"simple_average", "stacked_logistic_regression"}),
            key=lambda name: comparison[name]["test"]["macro_f1"],
        )
        stacked_test = comparison["stacked_logistic_regression"]["test"]["macro_f1"]
        best_test = comparison[best_single]["test"]["macro_f1"]
        MULTICLASS_STACK_RESULT = {
            "state": "fitted_and_evaluated",
            "members": [member["name"] for member in accepted],
            "rejections": rejections,
            "stacking_layer": "multinomial logistic regression",
            "fitted_on": "validation split member probabilities (the test split is never used to fit it)",
            "selection_split": "validation",
            "test_split_used_for_tuning": False,
            "comparison": comparison,
            "best_single_member": best_single,
            "stacked_minus_best_single_macro_f1": float(stacked_test - best_test),
            "statement": (
                f"On the held-out test split the stacking layer scores macro-F1 {stacked_test:.4f} and the best "
                f"single member ({best_single}) scores {best_test:.4f}. "
                + (
                    "The stacking layer is better on this split."
                    if stacked_test > best_test
                    else "The stacking layer is not better on this split; the ensemble is reported as measured, "
                    "not as an improvement."
                )
            ),
            "stacker": stacker,
            "probabilities": {"validation": stack_probabilities["validation"], "test": stack_probabilities["test"]},
            "average_probabilities": average_probabilities,
            "labels": {"validation": y_validation, "test": y_test},
            "classes": CLASSIFICATION_CLASSES,
        }
        write_json(
            PATHS["artifacts"] / "classification_multiclass_stack.json",
            {key: value for key, value in MULTICLASS_STACK_RESULT.items() if key not in {"stacker", "probabilities", "average_probabilities", "labels"}},
        )
        print("\nheld-out comparison (macro-F1 on the test split):")
        for name, entry in sorted(comparison.items(), key=lambda item: -item[1]["test"]["macro_f1"]):
            print(f"  {name:<34} validation {entry['validation']['macro_f1']:.4f} | test {entry['test']['macro_f1']:.4f}")
        print(MULTICLASS_STACK_RESULT["statement"])

# %% [cell] 30. Multiclass held-out evaluation (metrics, confusion matrix, ROC-AUC)
# =============================================================================
# SECTION 18b - Everything the report needs, on data the models never saw
# =============================================================================
MULTICLASS_EVALUATION: dict = {"state": "not_available", "reason": "the multiclass pipeline did not run"}

if MULTICLASS_STACK_RESULT.get("state") != "fitted_and_evaluated":
    print("Multiclass evaluation: skipped --", MULTICLASS_STACK_RESULT.get("reason", "no fitted ensemble"))
else:
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_recall_fscore_support,
        roc_auc_score,
    )

    y_test = MULTICLASS_STACK_RESULT["labels"]["test"]
    model_probabilities = {
        **{member: None for member in MULTICLASS_STACK_RESULT["members"]},
        "simple_average": MULTICLASS_STACK_RESULT["average_probabilities"]["test"],
        "stacked_logistic_regression": MULTICLASS_STACK_RESULT["probabilities"]["test"],
    }
    for member in accepted:
        model_probabilities[member["name"]] = member["probabilities"]["test"]

    reports = {}
    for name, probabilities in model_probabilities.items():
        if probabilities is None:
            continue
        predicted = probabilities.argmax(axis=1)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_test, predicted, labels=list(range(len(CLASSIFICATION_CLASSES))), zero_division=0
        )
        entry = {
            "accuracy": float(accuracy_score(y_test, predicted)),
            "macro_f1": float(f1_score(y_test, predicted, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(y_test, predicted, average="weighted", zero_division=0)),
            "macro_precision": float(precision.mean()),
            "macro_recall": float(recall.mean()),
            "per_class": {
                name_class: {
                    "precision": float(precision[index]),
                    "recall": float(recall[index]),
                    "f1": float(f1[index]),
                    "support": int(support[index]),
                }
                for index, name_class in enumerate(CLASSIFICATION_CLASSES)
            },
            "confusion_matrix": confusion_matrix(y_test, predicted, labels=list(range(len(CLASSIFICATION_CLASSES)))).tolist(),
            "confusion_matrix_order": list(CLASSIFICATION_CLASSES),
            "samples": int(len(y_test)),
            "class_distribution": {
                name_class: int((y_test == index).sum()) for index, name_class in enumerate(CLASSIFICATION_CLASSES)
            },
        }
        if len(CLASSIFICATION_CLASSES) == 2:
            entry["roc_auc"] = float(roc_auc_score(y_test, probabilities[:, 1]))
            entry["roc_auc_average"] = f"binary (positive class = {CLASSIFICATION_CLASSES[1]})"
        else:
            try:
                entry["roc_auc"] = float(roc_auc_score(y_test, probabilities, multi_class="ovr", average="macro"))
                entry["roc_auc_average"] = "macro one-vs-rest"
            except ValueError as exc:
                entry["roc_auc"] = None
                entry["roc_auc_error"] = str(exc)
        reports[name] = entry

    MULTICLASS_EVALUATION = {
        "state": "evaluated",
        "task": "multiclass_classification",
        "evaluated_at": utc_now(),
        "dataset": {
            "name": CLASSIFICATION_DATASET_NAME,
            "root": CLASSIFICATION_DATASET.get("root"),
            "classes": CLASSIFICATION_CLASSES,
            "images": {name: len(groups) and sum(len(paths) for paths in groups.values()) for name, groups in CLASSIFICATION_IMAGES.items()},
            "synthetic": CLASSIFICATION_IS_SYNTHETIC,
            "provenance_note": CLASSIFICATION_PROVENANCE_NOTE,
        },
        "protocol": {
            "split_source": CLASSIFICATION_SPLIT_SOURCE,
            "split_method": "stratified by class; no patient identifiers exist in this dataset",
            "patient_level_split_possible": False,
            "selection_split": "validation",
            "test_split_used_for_tuning": False,
            "near_duplicate_audit": CLASSIFICATION_AUDIT,
            "preprocessing": "grayscale, resize to RUN['classification_image_size'], per-image standardisation (CNN) or train-fit median imputation + standardisation (radiomics)",
        },
        "models": reports,
        "stacked_ensemble": {
            "members": MULTICLASS_STACK_RESULT["members"],
            "rejected_members": MULTICLASS_STACK_RESULT["rejections"],
            "best_single_member": MULTICLASS_STACK_RESULT["best_single_member"],
            "statement": MULTICLASS_STACK_RESULT["statement"],
        },
        "caveats": [
            "The classes are the dataset's own annotations (glioma / meningioma / pituitary / no tumour). "
            "They are not a malignancy verdict, a stage or a prognosis.",
            "The dataset carries no patient identifiers, so a patient-level split is impossible; "
            "near-duplicates of training images are detected and removed from the evaluation splits instead.",
            "Radiomics here is measured in an automatically derived brain-region ROI without voxel spacing, "
            "so the units are pixel-domain and the ROI is an estimate, not an annotation.",
        ],
    }
    write_json(PATHS["artifacts"] / "classification_multiclass_evaluation.json", MULTICLASS_EVALUATION)

    # --- figures ---------------------------------------------------------------
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    stacked = reports["stacked_logistic_regression"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    matrix = np.asarray(stacked["confusion_matrix"])
    image = axes[0].imshow(matrix, cmap="Blues")
    axes[0].set_xticks(range(len(CLASSIFICATION_CLASSES)))
    axes[0].set_xticklabels(CLASSIFICATION_CLASSES, rotation=20, ha="right")
    axes[0].set_yticks(range(len(CLASSIFICATION_CLASSES)))
    axes[0].set_yticklabels(CLASSIFICATION_CLASSES)
    axes[0].set_xlabel("predicted")
    axes[0].set_ylabel("actual")
    axes[0].set_title(f"Stacked ensemble, held-out test (n={stacked['samples']})")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axes[0].text(column, row, str(matrix[row, column]), ha="center", va="center", fontsize=9)
    fig.colorbar(image, ax=axes[0], shrink=0.8)
    names = list(reports)
    axes[1].bar(names, [reports[name]["macro_f1"] for name in names], color="tab:green")
    axes[1].set_ylabel("macro-F1 (test split)")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].set_title("Members and ensemble on the same held-out images")
    plt.tight_layout()
    evaluation_figure = PATHS["reports"] / "11_classification_evaluation.png"
    plt.savefig(evaluation_figure, dpi=130)
    plt.close(fig)

    print("held-out metrics (test split):")
    for name, entry in sorted(reports.items(), key=lambda item: -item[1]["macro_f1"]):
        print(
            f"  {name:<34} accuracy {entry['accuracy']:.4f} | macro-F1 {entry['macro_f1']:.4f} | "
            f"ROC-AUC {entry['roc_auc'] if entry.get('roc_auc') is None else round(entry['roc_auc'], 4)} | n={entry['samples']}"
        )
    print("confusion matrix (stacked):", stacked["confusion_matrix"], "order", stacked["confusion_matrix_order"])
    print("figure saved:", evaluation_figure)
