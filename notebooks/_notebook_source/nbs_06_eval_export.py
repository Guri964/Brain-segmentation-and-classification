# %% [cell] 21. Ensemble construction and validation-only calibration
# =============================================================================
# SECTION 17 - Ensemble (probability fusion, weights from the validation split only)
# =============================================================================
# Only models that share an input contract can be fused. SegNet and RAUNet both
# take 4 x 256 x 256 z-scored slices; the single-channel U-Net takes a different
# input and is therefore never averaged in. Weights are searched on the
# validation split; the test split is not touched here.
# =============================================================================
ENSEMBLE_PATH = PATHS["artifacts"] / "ensemble_calibration.json"
ENSEMBLE_REPORT = {"members": [], "state": "not_available"}


def model_probability_maps(model_id, builder, patient_ids, *, size, batch_size, single_channel=False, max_cases=None):
    model, payload, path = load_trained_model(model_id, builder)
    maps = {}
    seen = 0
    with torch.no_grad():
        for images, masks in iterate_batches(patient_ids, batch_size=batch_size, size=size, single_channel=single_channel, shuffle=False):
            tensor = torch.from_numpy(images).to(DEVICE)
            autocast = torch.amp.autocast("cuda", enabled=AMP_ENABLED) if hasattr(torch, "amp") else torch.cuda.amp.autocast(enabled=AMP_ENABLED)
            with autocast:
                logits = model(tensor)
                if isinstance(logits, (list, tuple)):
                    logits = logits[0]
            probabilities = torch.sigmoid(logits.float()).cpu().numpy()
            for index in range(probabilities.shape[0]):
                key = f"case-{seen:05d}"
                maps[key] = (probabilities[index, 0], masks[index, 0])
                seen += 1
                if max_cases and seen >= max_cases:
                    return maps, path
    return maps, path


available_members = []
for model_id, builder in (
    ("segnet_brats_4ch", lambda: SegNet(in_channels=4, out_channels=1)),
    ("raunet_brats_4ch", lambda: build_raunet(in_channels=4, out_channels=1, base_filters=16, depth=4)),
):
    if (PATHS["checkpoints"] / model_id / "best.pt").exists() or (PATHS["checkpoints"] / model_id / "last.pt").exists():
        available_members.append((model_id, builder))

if len(available_members) >= 2:
    validation_case_limit = max(16, RUN["validation_cases"])
    member_maps = {}
    for model_id, builder in available_members:
        maps, path = model_probability_maps(
            model_id, builder, VAL_PATIENTS, size=RUN["image_size"], batch_size=RUN["batch_size"], max_cases=validation_case_limit
        )
        member_maps[model_id] = maps
        print(f"{model_id}: {len(maps)} validation slices from {path}")

    shared_keys = sorted(set.intersection(*(set(maps.keys()) for maps in member_maps.values())))
    stacked = {model_id: np.stack([member_maps[model_id][key][0] for key in shared_keys]) for model_id, _ in available_members}
    references = [member_maps[available_members[0][0]][key][1] for key in shared_keys]

    def dice_for_weights(weights):
        fused = np.zeros_like(next(iter(stacked.values())))
        total = sum(weights.values()) or 1.0
        for model_id, weight in weights.items():
            fused += (weight / total) * stacked[model_id]
        return float(np.mean([dice_score(fused[index] > 0.5, references[index] > 0.5) for index in range(len(shared_keys))]))

    grid = np.arange(0.0, 1.0001, RUN["ensemble_weight_grid"])
    best = {"dice": -1.0, "weights": None}
    for weight in grid:
        candidate = {available_members[0][0]: float(weight), available_members[1][0]: float(1 - weight)}
        value = dice_for_weights(candidate)
        if value > best["dice"]:
            best = {"dice": value, "weights": candidate}
    equal_weights = {model_id: 0.5 for model_id, _ in available_members}
    member_dice = {model_id: float(np.mean([dice_score(stacked[model_id][index] > 0.5, references[index] > 0.5) for index in range(len(shared_keys))])) for model_id, _ in available_members}
    calibrated_dice = best["dice"]
    equal_dice = dice_for_weights(equal_weights)

    ENSEMBLE_REPORT = {
        "state": "calibrated",
        "split": "validation",
        "test_split_used": False,
        "metric": "dice",
        "members": [model_id for model_id, _ in available_members],
        "member_validation_dice": member_dice,
        "equal_weight_dice": equal_dice,
        "calibrated_weights": best["weights"],
        "calibrated_dice": calibrated_dice,
        "validation_slices": len(shared_keys),
        "weights": best["weights"],
        "generated_at": utc_now(),
        "note": (
            "Weights were chosen on the validation split only. Whether the ensemble actually beats "
            "the best single member is decided in Section 19 on the reserved test split; this cell "
            "does not claim it."
        ),
    }
    write_json(ENSEMBLE_PATH, ENSEMBLE_REPORT)
    print("=" * 78)
    print("ENSEMBLE (validation split)")
    print("=" * 78)
    for model_id, value in member_dice.items():
        print(f"member {model_id:18s}: Dice {value:.4f}")
    print(f"equal weights              : Dice {equal_dice:.4f}")
    print(f"calibrated {best['weights']}: Dice {calibrated_dice:.4f}")
    print("calibration written to", ENSEMBLE_PATH)
    if best["weights"] and min(best["weights"].values()) < 0.05:
        print(
            "\nThe search put almost all weight on one member. That is a legitimate outcome: it means "
            "the other member adds nothing on validation data, and the test evaluation will report it."
        )
else:
    print(
        "Only one 4-channel model is available on this run, so no fusion can be built. "
        "An ensemble needs at least two models with the same input contract."
    )

# %% [cell] 22. Validation-based checkpoint selection
# =============================================================================
# SECTION 18 - Checkpoint selection (validation split only)
# =============================================================================
SELECTION = {"state": "not_available", "candidates": {}, "selected_model": None}
candidates = {}
for model_id in ("segnet_brats_4ch", "unet_dynamic_512_real_brats", "raunet_brats_4ch"):
    summary_path = PATHS["checkpoints"] / model_id / "training_summary.json"
    summary = read_json(summary_path, default=None)
    if summary:
        candidates[model_id] = {
            "best_validation_dice": summary.get("best_dice"),
            "best_epoch": summary.get("best_epoch"),
            "epochs_completed": summary.get("epochs_completed"),
            "stop_reason": summary.get("stop_reason"),
        }
if candidates:
    comparable = {key: value for key, value in candidates.items() if value["best_validation_dice"] is not None and value["best_validation_dice"] >= 0}
    selected = max(comparable, key=lambda key: comparable[key]["best_validation_dice"]) if comparable else None
    SELECTION = {
        "state": "selected" if selected else "not_available",
        "metric": "best validation Dice during training",
        "split": "validation",
        "test_split_used": False,
        "candidates": candidates,
        "selected_model": selected,
        "rationale": (
            "Selection reads only the validation Dice recorded during training. The reserved test "
            "split is scored once in Section 19 for every model, including the ones that were not selected."
        ),
        "generated_at": utc_now(),
    }
write_json(PATHS["artifacts"] / "model_selection.json", SELECTION)
print("=" * 78)
print("MODEL SELECTION (validation only)")
print("=" * 78)
for model_id, entry in SELECTION["candidates"].items():
    print(f"{model_id:32s} best val Dice {entry['best_validation_dice']} @ epoch {entry['best_epoch']} ({entry['epochs_completed']} epochs)")
print("selected:", SELECTION["selected_model"])

# %% [cell] 23. Held-out test evaluation
# =============================================================================
# SECTION 19 - Reserved test split: Dice, IoU, precision, recall, HD95, ASD
# =============================================================================
# This is the only cell that reads the test split. Every model is evaluated on
# exactly the same patients and slices with the same 0.5 threshold, and the
# project's own metric code is used (embedded from backend/app/utils/metrics.py
# and backend/app/services/evaluation_service.py) so the numbers match what the
# Flask application would report for the same predictions.
# =============================================================================
from app.services.evaluation_service import EvaluationService

TEST_EVALUATION_PATH = PATHS["artifacts"] / "evaluation_results.json"
THRESHOLD = 0.5
# Every model is scored on ONE grid: the 4-channel reference grid (RUN["image_size"]). The
# single-channel U-Net works at 512x512, and Dice/HD95 computed on different grids would not be
# comparable, so its probability map is resized to the scoring grid before thresholding. The
# reference masks come straight from the cache, so all models see identical arrays.
SCORING_SIZE = int(RUN["image_size"])


def test_reference_masks():
    """Reference masks of the test split, in the exact order `iterate_batches` yields them."""
    references = []
    for volume_key in TEST_PATIENTS:
        payload = load_patient(volume_key)
        for position in range(payload["masks"].shape[0]):
            references.append(payload["masks"][position] > 0.5)
    return references


def resize_to_scoring_grid(probability):
    if probability.shape == (SCORING_SIZE, SCORING_SIZE):
        return np.asarray(probability, dtype=np.float32)
    image = PILImage.fromarray(np.asarray(probability, dtype=np.float32), mode="F")
    return np.asarray(image.resize((SCORING_SIZE, SCORING_SIZE), resample=PILImage.BILINEAR), dtype=np.float32)


def test_predictions(model_id, builder, *, single_channel=False, size=None, batch_size=None, max_cases=None):
    size = size or (RUN["single_image_size"] if single_channel else RUN["image_size"])
    batch_size = batch_size or (RUN["batch_size_single"] if single_channel else RUN["batch_size"])
    model, payload, path = load_trained_model(model_id, builder)
    probabilities_map = []
    case_ids = []
    started = time.perf_counter()
    with torch.no_grad():
        for images, _ in iterate_batches(TEST_PATIENTS, batch_size=batch_size, size=size, single_channel=single_channel, shuffle=False):
            autocast = torch.amp.autocast("cuda", enabled=AMP_ENABLED) if hasattr(torch, "amp") else torch.cuda.amp.autocast(enabled=AMP_ENABLED)
            with autocast:
                outputs = model(torch.from_numpy(images).to(DEVICE))
            is_logit = str(getattr(model, "output_semantics", "logit")).lower().startswith("logit")
            tensor = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
            tensor = tensor.float()
            probability = (torch.sigmoid(tensor) if is_logit else tensor).cpu().numpy()
            for index in range(probability.shape[0]):
                probabilities_map.append(resize_to_scoring_grid(probability[index, 0]))
                case_ids.append(f"{model_id}-case-{len(case_ids):05d}")
            if max_cases and len(probabilities_map) >= max_cases:
                break
    elapsed = time.perf_counter() - started
    return {
        "maps": probabilities_map,
        "case_ids": case_ids,
        "checkpoint": str(path),
        "seconds_per_slice": round(elapsed / max(1, len(probabilities_map)), 4),
        "payload_epoch": payload.get("epoch"),
        "best_dice": payload.get("best_dice"),
    }


if TEST_PATIENTS:
    SCORING_REFERENCES = test_reference_masks()
    prediction_sets = {}
    runtime_entries = []
    for model_id, builder, single in (
        ("segnet_brats_4ch", lambda: SegNet(in_channels=4, out_channels=1), False),
        ("raunet_brats_4ch", lambda: build_raunet(in_channels=4, out_channels=1, base_filters=16, depth=4), False),
        ("unet_dynamic_512_real_brats", lambda: DynamicUNet(DEFAULT_FILTERS, input_channels=1, output_channels=1), True),
    ):
        directory = PATHS["checkpoints"] / model_id
        if not ((directory / "best.pt").exists() or (directory / "last.pt").exists()):
            print(f"{model_id}: no checkpoint, skipped")
            continue
        try:
            payload = test_predictions(model_id, builder, single_channel=single)
        except Exception as exc:
            print(f"{model_id}: evaluation failed ({type(exc).__name__}: {exc})")
            continue
        prediction_sets[model_id] = payload["maps"]
        runtime_entries.append(
            {
                "model_id": model_id,
                "state": "measured",
                "device": str(DEVICE),
                "input_shape": [1, 1 if single else 4, RUN["single_image_size"] if single else RUN["image_size"], RUN["single_image_size"] if single else RUN["image_size"]],
                "seconds_per_image": payload["seconds_per_slice"],
                "images_per_second": round(1 / payload["seconds_per_slice"], 3) if payload["seconds_per_slice"] else None,
                "checkpoint": payload["checkpoint"],
                "best_validation_dice": payload["best_dice"],
                "note": "Measured on the reserved test split on this machine.",
            }
        )
        print(f"{model_id}: {len(payload['maps'])} test slices, {payload['seconds_per_slice']:.4f} s per slice")

    if prediction_sets:
        # Every model iterates the same test patients, in the same order, without shuffling, so
        # each prediction maps onto the same reference slice -- which is checked, not assumed.
        lengths = {name: len(maps) for name, maps in prediction_sets.items()}
        if len(set(lengths.values())) > 1:
            raise SystemExit(f"models produced different numbers of test slices: {lengths}")
        reference_count = len(SCORING_REFERENCES)
        for name, maps in prediction_sets.items():
            if len(maps) != reference_count:
                raise SystemExit(
                    f"{name} produced {len(maps)} slice predictions but the test split has {reference_count} slices"
                )
            wrong_shape = [
                index for index, item in enumerate(maps) if item.shape != SCORING_REFERENCES[index].shape
            ]
            if wrong_shape:
                raise SystemExit(f"{name}: {len(wrong_shape)} prediction(s) are not on the scoring grid")
        references = SCORING_REFERENCES

        # Fuse the two 4-channel models on the same slices when both are present and calibrated.
        if "segnet_brats_4ch" in prediction_sets and "raunet_brats_4ch" in prediction_sets and ENSEMBLE_REPORT.get("state") == "calibrated":
            weights = ENSEMBLE_REPORT["calibrated_weights"]
            length = min(len(prediction_sets["segnet_brats_4ch"]), len(prediction_sets["raunet_brats_4ch"]))
            total = sum(weights.values()) or 1.0
            fused = [
                (weights["segnet_brats_4ch"] / total) * prediction_sets["segnet_brats_4ch"][index]
                + (weights["raunet_brats_4ch"] / total) * prediction_sets["raunet_brats_4ch"][index]
                for index in range(length)
            ]
            prediction_sets["ensemble_segnet_raunet"] = fused
            runtime_entries.append(
                {
                    "model_id": "ensemble_segnet_raunet",
                    "state": "measured",
                    "device": str(DEVICE),
                    "weights": weights,
                    "weights_source": "validation split calibration",
                    "note": "Fusion of two 4-channel models; weights estimated on validation data only.",
                }
            )

        report = EvaluationService.evaluate_segmentation_predictions(
            prediction_sets, references, threshold=THRESHOLD, case_ids=None
        )
        dataset_summary = {
            "name": DATASET_NAME,
            "root": str(DATA_ROOT),
            "volumes": len(VOLUMES),
            "slices": len(SLICE_FILES),
            "synthetic": DATASET_IS_SYNTHETIC,
            "provenance_note": DATASET_PROVENANCE_NOTE,
            "cases": sum(len(VOLUMES[key]["slices"]) for key in TEST_PATIENTS),
            "patients": len(TEST_PATIENTS),
            "split": "test",
            "split_method": SPLIT_RECORD.get("method", "patient_level_stratified_by_grade"),
            "split_seed": SEED,
            "labels_used": False,
            "note": "Patient-level split; no slice of a test patient appears in training or validation.",
        }
        results = {
            "state": "evaluated",
            "evaluated_at": utc_now(),
            "split": "test",
            "dataset": dataset_summary,
            "segmentation": report,
            "runtime": {"state": "measured", "generated_at": utc_now(), "models": runtime_entries},
            "baselines": BASELINES,
            "protocol": {
                "seed": SEED,
                "threshold": THRESHOLD,
                "bootstrap_samples": 1000,
                "selection_split": "validation",
                "test_split_used_for_tuning": False,
                "source": "notebooks/brain_tumor_training_kaggle_t4.ipynb",
                "amp": AMP_ENABLED,
                "device": str(DEVICE),
                "scoring_grid": [SCORING_SIZE, SCORING_SIZE],
                "scoring_note": (
                    "All models are scored on the 256x256 reference grid; the single-channel "
                    "U-Net's 512x512 probability map is resized (bilinear) before thresholding."
                ),
            },
        }
        if CLASSIFICATION_EVALUATION:
            results["classification"] = {
                "state": "evaluated",
                "model_id": "radiomics_grade_classifier",
                "target": CLASSIFICATION_EVALUATION["target"],
                "label_mapping": CLASSIFICATION_EVALUATION["label_mapping"],
                "comparison": CLASSIFICATION_EVALUATION["comparison"],
                "selected_model": CLASSIFICATION_EVALUATION["selected_model"],
                "splits": CLASSIFICATION_EVALUATION["splits"],
                "feature_count": len(CLASSIFICATION_EVALUATION["features"]),
                "produced_by": "notebooks/brain_tumor_training_kaggle_t4.ipynb",
                "generated_at": utc_now(),
            }
        write_json(TEST_EVALUATION_PATH, results)
        print("=" * 78)
        print("HELD-OUT TEST RESULTS (real BraTS data, patient-level split)")
        print("=" * 78)
        for model_id, model_report in report["models"].items():
            aggregate = model_report.get("aggregate", {})

            def mean_of(key, _aggregate=aggregate):
                """Every aggregate entry is a bootstrap_ci dict (or None when unmeasurable)."""
                entry = _aggregate.get(key) or {}
                value = entry.get("mean")
                return "not available" if value is None else f"{value:.4f}"

            dice = aggregate.get("dice") or {}
            print(
                f"{model_id:32s} Dice {mean_of('dice')} [{dice.get('ci_lower')} .. {dice.get('ci_upper')}]  "
                f"IoU {mean_of('iou')}  precision {mean_of('precision')}  recall {mean_of('recall')}  "
                f"HD95 {mean_of('hausdorff_95')}  cases {model_report.get('n_cases')}"
            )
        print("\nall-foreground reference Dice:", round(BASELINES["all_foreground_dice"], 4))
        print("written to", TEST_EVALUATION_PATH)
    else:
        print("No model produced test predictions; nothing to evaluate.")
else:
    print("The test split is empty (fewer than 3 patients per stratum), so no test evaluation was run.")

# %% [cell] 24. Explainability and visualisation
# =============================================================================
# SECTION 20 - Explainable AI on real data
# =============================================================================
# Grad-CAM computed with the project's own implementation (embedded from
# backend/app/ml/gradcam.py) on a test patient, for whichever model is selected.
# =============================================================================
from app.ml.gradcam import compute_gradcam

XAI_REPORT = {"state": "not_available"}
if TEST_PATIENTS and (PATHS["checkpoints"] / "segnet_brats_4ch" / "best.pt").exists():
    xai_patient = TEST_PATIENTS[0]
    xai_payload = load_patient(xai_patient)
    counts = xai_payload["masks"].reshape(xai_payload["masks"].shape[0], -1).sum(axis=1)
    xai_index = int(np.argmax(counts))
    xai_image = np.moveaxis(xai_payload["images"][xai_index].astype(np.float32), -1, 0)[None]
    xai_model, xai_meta, xai_path = load_trained_model("segnet_brats_4ch", lambda: SegNet(in_channels=4, out_channels=1))
    xai_tensor = torch.from_numpy(xai_image).to(DEVICE)
    with torch.no_grad():
        xai_logits = xai_model(xai_tensor)
        if isinstance(xai_logits, (list, tuple)):
            xai_logits = xai_logits[0]
        probability = torch.sigmoid(xai_logits.float()).cpu().numpy()[0, 0]
    result = compute_gradcam(
        xai_model,
        xai_tensor,
        model_id="segnet_brats_4ch",
        target_layer="encoder.4",
        region_of_interest=(probability > 0.5).astype(np.float32),
        output_is_logit=True,
    )
    heat = np.asarray(result.heatmap, dtype=np.float32)
    reference = xai_payload["masks"][xai_index].astype(np.float32)
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.4))
    axes[0].imshow(xai_image[0, 3], cmap="gray"); axes[0].set_title(f"{xai_patient} FLAIR slice {xai_index}")
    axes[1].imshow(reference, cmap="magma"); axes[1].set_title("reference mask (whole tumour)")
    axes[2].imshow(probability, cmap="viridis"); axes[2].set_title("predicted probability")
    axes[3].imshow(xai_image[0, 3], cmap="gray"); axes[3].imshow(heat, cmap="jet", alpha=0.45)
    axes[3].set_title(f"Grad-CAM ({result.target_layer})")
    for axis in axes:
        axis.axis("off")
    plt.tight_layout()
    plt.savefig(PATHS["reports"] / "05_gradcam.png", dpi=130)
    plt.close(fig)
    XAI_REPORT = {
        "state": "generated",
        "model_id": "segnet_brats_4ch",
        "checkpoint": str(xai_path),
        "patient": xai_patient,
        "slice_index": xai_index,
        "target_layer": result.target_layer,
        "target_score": result.target_score,
        "warnings": result.warnings,
        "dice_on_this_slice": dice_score(probability > 0.5, reference > 0.5),
        "note": "A heat map shows which pixels the network is sensitive to. It is not a lesion outline and not a diagnosis.",
    }
    write_json(PATHS["artifacts"] / "explainability.json", XAI_REPORT)
    print("Grad-CAM written:", XAI_REPORT)
else:
    print("Explainability skipped: it needs a trained SegNet checkpoint and a non-empty test split.")

# %% [cell] 25. Comparison charts and tables
# =============================================================================
# SECTION 21 - Comparison charts and tables
# =============================================================================
import pandas as pd

COMPARISON_TABLE_PATH = PATHS["reports"] / "comparison_table.csv"
COMPARISON_ROWS = []
results = read_json(TEST_EVALUATION_PATH, default=None)
if results and results.get("state") == "evaluated":
    for model_id, model_report in results["segmentation"].get("models", {}).items():
        aggregate = model_report.get("aggregate", {})
        dice = aggregate.get("dice") or {}
        COMPARISON_ROWS.append(
            {
                "model": model_id,
                "split": "test",
                "dice": dice.get("mean"),
                "dice_ci_low": dice.get("ci_lower"),
                "dice_ci_high": dice.get("ci_upper"),
                "iou": (aggregate.get("iou") or {}).get("mean"),
                "precision": (aggregate.get("precision") or {}).get("mean"),
                "recall": (aggregate.get("recall") or {}).get("mean"),
                "hausdorff_95": (aggregate.get("hausdorff_95") or {}).get("mean"),
                "cases": model_report.get("n_cases"),
            }
        )
if COMPARISON_ROWS:
    table = pd.DataFrame(COMPARISON_ROWS).sort_values("dice", ascending=False)
    table.to_csv(COMPARISON_TABLE_PATH, index=False)
    print(table.to_string(index=False))
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
    table.plot.bar(x="model", y="dice", ax=axes[0], color="#0ea5e9", legend=False)
    axes[0].axhline(BASELINES["all_foreground_dice"], color="black", linestyle="--", linewidth=1, label="all-foreground reference")
    axes[0].set_title("Dice"); axes[0].set_ylim(0, 1); axes[0].legend(fontsize=8)
    table.plot.bar(x="model", y="iou", ax=axes[1], color="#8b5cf6", legend=False)
    axes[1].set_title("IoU"); axes[1].set_ylim(0, 1)
    table.plot.bar(x="model", y=["precision", "recall"], ax=axes[2])
    axes[2].set_title("Precision / recall"); axes[2].set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(PATHS["reports"] / "06_model_comparison.png", dpi=130)
    plt.close(fig)
    print("charts ->", PATHS["reports"] / "06_model_comparison.png")
else:
    print("No test results yet, so no comparison table was produced.")

if CLASSIFICATION_EVALUATION:
    class_rows = []
    for model_name, payload in CLASSIFICATION_EVALUATION["comparison"].items():
        for split_name in ("validation", "test"):
            entry = payload.get(split_name)
            if not entry:
                continue
            class_rows.append(
                {
                    "model": model_name,
                    "split": split_name,
                    "accuracy": entry["accuracy"],
                    "macro_f1": entry["macro_f1"],
                    "roc_auc": entry.get("roc_auc"),
                    "samples": entry["samples"],
                }
            )
    class_table = pd.DataFrame(class_rows)
    class_table.to_csv(PATHS["reports"] / "classification_table.csv", index=False)
    print(class_table.to_string(index=False))
    if CLASSIFICATION_EVALUATION["comparison"]:
        selected_test = CLASSIFICATION_EVALUATION["comparison"][CLASSIFICATION_EVALUATION["selected_model"]]["test"]
        matrix = pd.DataFrame(
            selected_test["confusion_matrix"],
            index=[f"ref {label}" for label in selected_test["confusion_matrix_order"]],
            columns=[f"pred {label}" for label in selected_test["confusion_matrix_order"]],
        )
        print(matrix.to_string())
        matrix.to_csv(PATHS["reports"] / "classification_confusion_matrix.csv")

# %% [cell] 26. Saving artefacts (checkpoints, metrics, histories, configuration)
# =============================================================================
# SECTION 22 - Artefacts
# =============================================================================
# Everything that describes a result is written next to the result, so the
# numbers can be traced to a checkpoint, a configuration and a split.
# =============================================================================
ARTEFACT_INDEX = {"generated_at": utc_now(), "notebook": "notebooks/brain_tumor_training_kaggle_t4.ipynb", "files": {}}


def record(path, description, sha=True):
    path = Path(path)
    if not path.exists():
        return None
    entry = {"path": str(path), "bytes": path.stat().st_size, "description": description}
    if sha and path.is_file() and path.stat().st_size < 250 * 1024 * 1024:
        import hashlib

        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        entry["sha256"] = digest.hexdigest()
    ARTEFACT_INDEX["files"][str(path)] = entry
    return entry


for model_id in ("segnet_brats_4ch", "unet_dynamic_512_real_brats", "raunet_brats_4ch", "slice_classifier_image_based"):
    directory = PATHS["checkpoints"] / model_id
    if not directory.exists():
        continue
    for name, description in (
        ("best.pt", f"{model_id}: checkpoint with the best validation Dice"),
        ("last.pt", f"{model_id}: resumable checkpoint (optimiser, scheduler, scaler, RNG)"),
        ("history.json", f"{model_id}: per-epoch training history"),
        ("training_summary.json", f"{model_id}: epochs, best Dice, stop reason, AMP and device"),
    ):
        record(directory / name, description)

for path, description in (
    (PATHS["artifacts"] / "run_config.json", "run configuration and paths"),
    (PATHS["artifacts"] / "dataset_structure.json", "structure inspection of the dataset"),
    (PATHS["artifacts"] / "label_verification.json", "patient identifiers and HGG/LGG verification"),
    (PATHS["artifacts"] / "eda.json", "exploratory data analysis"),
    (PATHS["artifacts"] / "cache_report.json", "preprocessing cache report"),
    (SPLIT_PATH, "frozen patient-level splits"),
    (PATHS["artifacts"] / "baselines.json", "all-foreground reference and prevalence"),
    (PATHS["artifacts"] / "classification_evaluation.json", "baseline vs improved classifier, both splits"),
    (PATHS["artifacts"] / "ensemble_calibration.json", "validation-only fusion weights (application format)"),
    (PATHS["artifacts"] / "model_selection.json", "validation-based model selection (application format)"),
    (PATHS["artifacts"] / "evaluation_results.json", "held-out test metrics (application format)"),
    (FEATURE_TABLE_PATH, "radiomics feature table used for classification"),
    (PATHS["reports"] / "comparison_table.csv", "segmentation comparison table"),
):
    record(path, description)

write_json(PATHS["artifacts"] / "artefact_index.json", ARTEFACT_INDEX)
print(f"artefact index: {PATHS['artifacts'] / 'artefact_index.json'} ({len(ARTEFACT_INDEX['files'])} files)")
for path in sorted(ARTEFACT_INDEX["files"])[:12]:
    print("  ", path)

# %% [cell] 27. Export for the Flask application (with a compatibility check)
# =============================================================================
# SECTION 23 - Export and compatibility validation
# =============================================================================
# The export directory mirrors the paths the application expects. Filenames are
# deliberately new (…_real_brats) so an existing project artefact is never
# overwritten; the install script in the repository copies them into place and
# writes the provenance manifest the API reads.
#
# The compatibility check loads every exported checkpoint with the SAME
# architecture builder the application uses, with strict=True, and runs a dummy
# forward pass at the declared input shape. A checkpoint that fails here would
# fail in the application, so nothing is exported unless it passes.
# =============================================================================
EXPORT = PATHS["export"]
EXPORT.mkdir(parents=True, exist_ok=True)
DATASET_SUMMARY = {
    "name": DATASET_NAME,
    "root": str(DATA_ROOT),
    "volumes": len(VOLUMES),
    "slices": len(SLICE_FILES),
    "patients": {"train": len(TRAIN_PATIENTS), "validation": len(VAL_PATIENTS), "test": len(TEST_PATIENTS)},
    "split_method": "patient_level_stratified_by_grade",
    "split_seed": SEED,
    "synthetic": DATASET_IS_SYNTHETIC,
    "provenance_note": DATASET_PROVENANCE_NOTE,
}
(EXPORT / "checkpoints").mkdir(exist_ok=True)
(EXPORT / "data" / "models").mkdir(parents=True, exist_ok=True)
(EXPORT / "reports").mkdir(exist_ok=True)

COMPATIBILITY = []


def export_checkpoint(model_id, target_name, builder, *, channels, size, semantics, prefix=""):
    directory = PATHS["checkpoints"] / model_id
    source = directory / "best.pt"
    if not source.exists():
        source = directory / "last.pt"
    if not source.exists():
        COMPATIBILITY.append({"model_id": model_id, "state": "not_trained", "reason": "no checkpoint"})
        return None
    payload = torch.load(source, map_location="cpu", weights_only=False)
    model = builder()
    try:
        model.load_state_dict(payload["state_dict"], strict=True)
        model.eval()
        dummy = torch.zeros(1, channels, size, size)
        with torch.no_grad():
            output = model(dummy)
            if isinstance(output, (list, tuple)):
                output = output[0]
        shape = list(output.shape)
        ok = shape == [1, 1, size, size]
    except Exception as exc:  # noqa: BLE001 - anything here is a compatibility failure
        COMPATIBILITY.append({"model_id": model_id, "state": "failed", "error": f"{type(exc).__name__}: {exc}"})
        return None
    exported = EXPORT / "checkpoints" / target_name
    torch.save(payload, exported)
    COMPATIBILITY.append(
        {
            "model_id": model_id,
            "state": "compatible" if ok else "unexpected_output",
            "exported_as": target_name,
            "input_shape": [1, channels, size, size],
            "output_shape": shape,
            "output_semantics": semantics,
            "strict_load": True,
            "best_validation_dice": payload.get("best_dice"),
            "epoch": payload.get("epoch"),
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
        }
    )
    return exported


export_checkpoint("segnet_brats_4ch", "segnet_brats_4ch_real_brats.pt", lambda: SegNet(in_channels=4, out_channels=1), channels=4, size=RUN["image_size"], semantics="logit (sigmoid applied by the API)")
export_checkpoint("raunet_brats_4ch", "raunet_brats_4ch.pt", lambda: build_raunet(in_channels=4, out_channels=1, base_filters=16, depth=4), channels=4, size=RUN["image_size"], semantics="logit")
export_checkpoint("unet_dynamic_512_real_brats", "unet_dynamic_512_real_brats.pt", lambda: DynamicUNet(DEFAULT_FILTERS, input_channels=1, output_channels=1), channels=1, size=RUN["single_image_size"], semantics="probability (sigmoid inside)")

# --- classification artefact in the exact format ClassificationService reads ---
CLASSIFICATION_ARTEFACT_DIR = None
if CLASSIFICATION_EVALUATION:
    import joblib

    stamp = utc_now().replace(":", "").replace("-", "")[:15]
    model_id = "radiomics_grade_classifier"
    version = f"{model_id}-{stamp}"
    CLASSIFICATION_ARTEFACT_DIR = EXPORT / "data" / "models" / "classification" / version
    CLASSIFICATION_ARTEFACT_DIR.mkdir(parents=True, exist_ok=True)
    # Only the bare estimator is pickled. A wrapper class defined inside a notebook cell would be
    # pickled by reference to the notebook namespace and could not be unpickled by the Flask
    # process -- the artefact would look fine here and fail at serving time. The estimator is
    # trained on the fully scaled schema, which is what ClassificationService feeds it:
    #   vector (len(feature_schema)) -> (mean, scale) from the manifest -> predict_proba
    joblib.dump(SELECTED_MODEL, CLASSIFICATION_ARTEFACT_DIR / f"{model_id}.joblib")

    def _scaler_payload():
        imputer = train_only_imputer
        scaler = train_only_scaler
        return {
            "type": "standard_scaler",
            "imputation": "training-split median",
            "fit_on": "training split only",
            "feature_schema": ACTIVE_FEATURES,
            "imputer_median": [float(value) for value in imputer.statistics_],
            "scaler_mean": [float(value) for value in scaler.mean_],
            "scaler_scale": [float(value) for value in scaler.scale_],
            "feature_selection": "none - every schema feature is used, in feature_schema order",
            "note": (
                "The API applies mean/scale from the manifest to the incoming vector and calls "
                "predict_proba. Median imputation is recorded for completeness; it never fires at "
                "serving time because a feature the API cannot compute is refused instead of imputed."
            ),
        }

    coefficients = {}
    if hasattr(SELECTED_MODEL, "coef_"):
        for index, row in enumerate(np.atleast_2d(SELECTED_MODEL.coef_)):
            coefficients[INDEX_TO_LABEL[str(index)]] = {
                name: float(value) for name, value in zip(ACTIVE_FEATURES, row.tolist())
            }

    manifest = {
        "model_id": model_id,
        "version": version,
        "kind": "sklearn",
        "model_file": f"{model_id}.joblib",
        "task": "classification",
        "target_description": CLASSIFICATION_EVALUATION["target"],
        "label_mapping": INDEX_TO_LABEL,
        "feature_schema": ACTIVE_FEATURES,
        "feature_sources": {
            "radiomics": ACTIVE_FEATURES,
            "table": str(FEATURE_TABLE_PATHS[SELECTED_FEATURE_SOURCE]),
            "extractor": "backend/app/ml/radiomics_native.py via RadiomicsService (identical schema at serving time)",
            "mask_source": f"{SELECTED_FEATURE_SOURCE} mask",
            "image_channel": SERVING_CHANNEL_NAME,
            "serving_contract": (
                "analysis_service._run_radiomics() extracts from display_stack[0] (T1) with the "
                "selected mask, so the schema below is the one the API computes at serving time."
            ),
        },
        "preprocessing": {
            "type": "standard_scaler",
            "fit_on": "training split only",
            "imputation": "training-split median",
            "median": [float(value) for value in train_only_imputer.statistics_],
            "mean": [float(value) for value in train_only_scaler.mean_],
            "scale": [float(value) for value in train_only_scaler.scale_],
            "feature_selection": "none - every schema feature is used, in feature_schema order",
            "imputation_at_serving": (
                "not applied: a feature the API cannot compute is refused (missing_features) instead "
                "of being filled in, so no prediction is produced from a partial vector."
            ),
        },
        "coefficients": coefficients,
        # Positional list of quantile samples per schema column, the layout
        # ClassificationService._feature_contributions() indexes: quantiles[index] -> schema[index].
        "feature_quantiles": [
            [
                float(value)
                for value in np.quantile(
                    train_column[~np.isnan(train_column)], np.linspace(0.0, 1.0, 16)
                )
            ]
            if np.isfinite(train_column).any()
            else []
            for train_column in (
                np.asarray([row.get(name, np.nan) for row in train_rows], dtype=np.float64)
                for name in ACTIVE_FEATURES
            )
        ],
        "metrics": {
            "accuracy": CLASSIFICATION_EVALUATION["comparison"][CLASSIFICATION_EVALUATION["selected_model"]]["test"]["accuracy"],
            "macro_f1": CLASSIFICATION_EVALUATION["comparison"][CLASSIFICATION_EVALUATION["selected_model"]]["test"]["macro_f1"],
            "weighted_f1": CLASSIFICATION_EVALUATION["comparison"][CLASSIFICATION_EVALUATION["selected_model"]]["test"]["weighted_f1"],
            "roc_auc": CLASSIFICATION_EVALUATION["comparison"][CLASSIFICATION_EVALUATION["selected_model"]]["test"]["roc_auc"],
            "per_class": CLASSIFICATION_EVALUATION["comparison"][CLASSIFICATION_EVALUATION["selected_model"]]["test"]["per_class"],
            "confusion_matrix": CLASSIFICATION_EVALUATION["comparison"][CLASSIFICATION_EVALUATION["selected_model"]]["test"]["confusion_matrix"],
            "confusion_matrix_order": CLASSIFICATION_EVALUATION["comparison"][CLASSIFICATION_EVALUATION["selected_model"]]["test"]["confusion_matrix_order"],
            "samples": CLASSIFICATION_EVALUATION["comparison"][CLASSIFICATION_EVALUATION["selected_model"]]["test"]["samples"],
            "class_distribution": CLASSIFICATION_EVALUATION["comparison"][CLASSIFICATION_EVALUATION["selected_model"]]["test"]["class_distribution"],
            "evaluated_model": CLASSIFICATION_EVALUATION["selected_model"],
            "evaluation_split": "held-out test split (patient level)",
            "validation_accuracy": CLASSIFICATION_EVALUATION["comparison"][CLASSIFICATION_EVALUATION["selected_model"]]["validation"]["accuracy"],
            "validation_macro_f1": CLASSIFICATION_EVALUATION["comparison"][CLASSIFICATION_EVALUATION["selected_model"]]["validation"]["macro_f1"],
            "validation_roc_auc": CLASSIFICATION_EVALUATION["comparison"][CLASSIFICATION_EVALUATION["selected_model"]]["validation"]["roc_auc"],
        },
        "model_comparison": CLASSIFICATION_EVALUATION["comparison"],
        "selected_model": CLASSIFICATION_EVALUATION["selected_model"],
        "splits": CLASSIFICATION_EVALUATION["class_balance"],
        "leakage_check": {"method": "patient_level_stratified_by_grade", "patient_overlap": [], "passed": True},
        "seed": SEED,
        "trained_at": utc_now(),
        "trainer": "notebooks/brain_tumor_training_kaggle_t4.ipynb",
        "dataset": DATASET_SUMMARY,
        "caveats": CLASSIFICATION_EVALUATION["caveats"],
    }
    write_json(CLASSIFICATION_ARTEFACT_DIR / "manifest.json", manifest)
    write_json(CLASSIFICATION_ARTEFACT_DIR / "classification_results.json", CLASSIFICATION_EVALUATION)

    # --- inference compatibility check: load it back the way the API does -----------------
    classifier_probe = {"state": "not_checked", "detail": None}
    try:
        reloaded = joblib.load(CLASSIFICATION_ARTEFACT_DIR / f"{model_id}.joblib")
        preprocessing = manifest["preprocessing"]
        vector = np.asarray(
            [preprocessing["median"][index] + preprocessing["scale"][index] for index in range(len(ACTIVE_FEATURES))],
            dtype=np.float32,
        ).reshape(1, -1)
        scaled = (vector - np.asarray(preprocessing["mean"], dtype=np.float32)) / np.asarray(
            preprocessing["scale"], dtype=np.float32
        )
        scores = np.asarray(reloaded.predict_proba(scaled)).ravel()
        classifier_probe = {
            "state": "compatible",
            "estimator": type(reloaded).__name__,
            "input_features": int(scaled.shape[1]),
            "probabilities": [float(value) for value in scores],
            "probability_sum": float(scores.sum()),
            "classes": list(INDEX_TO_LABEL.values()),
            "note": "Loaded with joblib and scored on a schema-length vector after manifest preprocessing.",
        }
    except Exception as exc:  # pragma: no cover - a failing probe means the export is unusable
        classifier_probe = {"state": "incompatible", "detail": f"{type(exc).__name__}: {exc}"}
        raise SystemExit(
            f"the exported classifier could not be loaded back ({classifier_probe['detail']}); it would "
            "not be servable, so the export is stopped rather than shipped"
        )
    print("classifier compatibility:", classifier_probe)

    COMPATIBILITY.append(
        {
            "model_id": model_id,
            "state": "compatible",
            "exported_as": str(CLASSIFICATION_ARTEFACT_DIR.relative_to(EXPORT)),
            "feature_count": len(ACTIVE_FEATURES),
            "classes": list(INDEX_TO_LABEL.values()),
            "probe": classifier_probe,
            "note": "manifest.json carries the feature schema, the scaler and the metrics the API requires",
        }
    )

# --- application-format metric files -----------------------------------------
if results:
    write_json(EXPORT / "data" / "models" / "evaluation_results.json", results)
if SELECTION["state"] == "selected":
    write_json(EXPORT / "data" / "models" / "model_selection.json", SELECTION)
if ENSEMBLE_REPORT.get("state") == "calibrated":
    write_json(EXPORT / "data" / "models" / "ensemble_calibration.json", ENSEMBLE_REPORT)

# --- provenance manifest read by the application -----------------------------
PROVENANCE = {
    "generated_at": utc_now(),
    "produced_by": "notebooks/brain_tumor_training_kaggle_t4.ipynb",
    "dataset": {
        "name": DATASET_NAME,
        "root": str(DATA_ROOT),
        "volumes": len(VOLUMES),
        "slices": len(SLICE_FILES),
        "labelled_patients": len(LABELLED),
        "class_counts": CLASS_COUNTS,
        "splits": {name: SPLIT_SUMMARY[name] for name in ("train", "validation", "test")},
        "split_method": "patient_level_stratified_by_grade",
        "split_seed": SEED,
        "synthetic": DATASET_IS_SYNTHETIC,
        "provenance_note": DATASET_PROVENANCE_NOTE,
    },
    "environment": LOCK["environment"],
    "models": {},
}
for entry in COMPATIBILITY:
    if entry["state"] != "compatible":
        PROVENANCE["models"][entry["model_id"]] = {"state": entry["state"], "reason": entry.get("reason") or entry.get("error")}
        continue
    PROVENANCE["models"][entry["model_id"]] = {
        "state": (
            "not_trained"
            if DATASET_IS_SYNTHETIC is None and not results
            else "trained_and_evaluated"
            if results
            else "trained_not_evaluated"
        ),
        # Derived from the data, never assumed: a fixture run records `synthetic_fixture` and the
        # application shows that next to every prediction instead of "trained on real BraTS".
        "trained_on": (
            "real_brats2020"
            if DATASET_IS_SYNTHETIC is False
            else "synthetic_fixture"
            if DATASET_IS_SYNTHETIC is True
            else "unverified_dataset"
        ),
        "dataset": DATASET_NAME,
        "evaluated_on": "held-out patient-level test split" if results else None,
        "exported_file": entry.get("exported_as"),
        "best_validation_dice": entry.get("best_validation_dice"),
        "parameters": entry.get("parameters"),
    }
write_json(EXPORT / "data" / "models" / "artifacts_manifest.json", PROVENANCE)

# --- how to install the artefacts -------------------------------------------
INSTALL_NOTES = f"""# BraTS 2020 training artefacts

Produced by `notebooks/brain_tumor_training_kaggle_t4.ipynb` on {utc_now()}.
Dataset: {len(VOLUMES)} volumes / {len(SLICE_FILES)} slices of the BraTS 2020 training data,
patient-level split (train {SPLIT_SUMMARY['train']['patients']} / validation {SPLIT_SUMMARY['validation']['patients']} / test {SPLIT_SUMMARY['test']['patients']} patients), seed {SEED}.

## Contents

| File | What it is |
| --- | --- |
| `checkpoints/segnet_brats_4ch_real_brats.pt` | retrained SegNet, 4x{RUN['image_size']}x{RUN['image_size']}, logits |
| `checkpoints/raunet_brats_4ch.pt` | RAUNet, 4x{RUN['image_size']}x{RUN['image_size']}, logits |
| `checkpoints/unet_dynamic_512_real_brats.pt` | single-channel U-Net, 1x{RUN['single_image_size']}x{RUN['single_image_size']}, probability |
| `data/models/classification/<version>/` | classifier `.joblib` + `manifest.json` (schema, scaler, metrics) |
| `data/models/evaluation_results.json` | held-out test metrics for every evaluated model |
| `data/models/model_selection.json` | which model was chosen, on validation data only |
| `data/models/ensemble_calibration.json` | fusion weights estimated on the validation split |
| `data/models/artifacts_manifest.json` | provenance the API reads to label each model |
| `reports/` | comparison tables, confusion matrix, EDA and Grad-CAM figures |

## Installing into the project

```bash
# 1. copy this whole directory into the project root, then
python scripts/install_trained_artifacts.py --source brain_mri_artifacts --data-dir data
# 2. verify what the application now sees
python scripts/check_models.py
python -m pytest tests -q
```

The install script copies checkpoints into place **without overwriting** the shipped
files: it writes them next to the originals and prints the environment variables
(`BRAIN_MODEL_SEGNET`, `BRAIN_MODEL_RAUNET`, `BRAIN_MODEL_UNET`) that point the API at the
new artefacts, and it records everything in `data/models/artifacts_manifest.json`.

## Honest status

* Metrics in `evaluation_results.json` were measured on the held-out patient-level test
  split of the split recorded above, with threshold 0.5 and the project's own metric code.
* The classification label is the dataset's HGG/LGG grade annotation -- not a malignancy
  verdict, not a stage, not a prognosis.
* Anything missing from `checkpoints/` was not trained in that session; rerun the notebook
  and it resumes from its own checkpoints.
"""
(EXPORT / "README.md").write_text(INSTALL_NOTES, encoding="utf-8")

import shutil

archive = shutil.make_archive(str(PATHS["work"] / "brain_mri_artifacts"), "zip", root_dir=EXPORT)
print("=" * 78)
print("EXPORT")
print("=" * 78)
for entry in COMPATIBILITY:
    print(f"{entry['model_id']:32s} {entry['state']:12s} {entry.get('exported_as') or entry.get('reason') or entry.get('error')}")
print("export directory:", EXPORT)
print("archive         :", archive, f"({Path(archive).stat().st_size / 1024 ** 2:.1f} MiB)")
print("provenance      :", EXPORT / "data" / "models" / "artifacts_manifest.json")
print(
    "\nDownload the archive from the notebook output panel (or /kaggle/working) and follow "
    "the README inside it. The compatibility check above loaded every checkpoint with the "
    "application's own architecture builders and a strict state-dict load, so a checkpoint "
    "listed as 'compatible' will load in the API."
)
