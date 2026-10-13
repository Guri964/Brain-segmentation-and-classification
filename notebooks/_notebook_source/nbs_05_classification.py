# %% [cell] 22. Classification dataset: radiomics features in the application's schema
# =============================================================================
# SECTION 14 / 16 - Classification data (radiomics features in the application's schema)
# =============================================================================
# Order note: the request lists classification training before radiomics
# extraction. Radiomics features ARE the classification input here, so extraction
# has to happen first; Section 16 later adds the feature-quality analysis
# (reference mask vs predicted mask, slice-to-slice stability).
#
# The feature names come from the application's own extractor (embedded verbatim
# from backend/app/ml/radiomics_native.py), so the table a model is trained on has
# exactly the schema the API computes at serving time. That is the fix for the
# original "21 features expected, 11 available, no scaler" mismatch.
#
# What the serving path actually does (read from the code, not assumed):
#   * analysis_service._run_radiomics() extracts from `display_reference`, which is
#     preprocessed.display_stack[0][1] -- the FIRST channel of a four-channel upload;
#   * preprocessing_service.MODALITY_ORDER_BRATS = ("t1", "t1ce", "t2", "flair"),
#     so that plane is T1;
#   * the ROI is one slice's mask, and the feature names carry no channel prefix.
# The tables below therefore hold bare feature names on the T1 plane of the slice
# with the largest tumour area, which is the closest training-side equivalent of a
# served analysis. Features are computed for two regions of interest so the effect
# of the mask source can be measured instead of guessed:
#   * "reference" - the BraTS ground-truth mask (what the annotation says),
#   * "predicted" - the mask of the best segmentation checkpoint chosen on the
#     validation split (what the API will actually use).
# =============================================================================
import csv
from types import SimpleNamespace

from app.services.radiomics_service import RadiomicsService

RADIOMICS_CONFIG = SimpleNamespace(
    RADIOMICS_BACKEND="native",       # PyRadiomics is used when installed; the schema is identical
    RADIOMICS_BIN_COUNT=32,
    RADIOMICS_BIN_WIDTH=0.0,
    RADIOMICS_DISCRETIZATION="bin_count",
)
radiomics_service = RadiomicsService(RADIOMICS_CONFIG)
print("radiomics backend:", radiomics_service.backend_status()["selected_backend"])

SERVING_CHANNEL = 0
SERVING_CHANNEL_NAME = ("t1", "t1ce", "t2", "flair")[SERVING_CHANNEL]
MASK_SOURCES = ("reference", "predicted")

FEATURE_TABLE_PATHS = {
    "reference": PATHS["artifacts"] / "radiomics_features.csv",
    "predicted": PATHS["artifacts"] / "radiomics_features_predicted.csv",
}
FEATURE_TABLE_PATH = FEATURE_TABLE_PATHS["reference"]
ROW_METADATA_COLUMNS = (
    "case_id",
    "patient_id",
    "label",
    "split",
    "mask_source",
    "foreground_pixels",
    "radiomics_backend",
    "image_channel",
    "slice_index",
)
# The extractor also reports diagnostics next to the features ('shape_units': 'pixels',
# 'diagnostics_dimensionality': '2D', ...). They are not measurements, so they are kept out
# of the schema -- and the names are recorded so the exclusion is visible in the manifest.
NON_NUMERIC_FEATURE_NAMES: set[str] = set()


def numeric_features(outcome):
    """Yield (name, value) for the numeric, finite features only."""
    for name, value in outcome.features.items():
        if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
            NON_NUMERIC_FEATURE_NAMES.add(name)
            continue
        number = float(value)
        if not np.isfinite(number):
            continue
        yield name, number


def focus_slice_index(masks):
    """Index and pixel count of the slice with the largest tumour area."""
    counts = masks.reshape(masks.shape[0], -1).sum(axis=1)
    return int(np.argmax(counts)), int(counts.max())


def segmentation_for_classification():
    """Segmentation model with the best *validation* Dice (validation split only)."""
    ranking = []
    for model_id in ("segnet_brats_4ch", "unet_dynamic_512_real_brats", "raunet_brats_4ch"):
        summary = read_json(PATHS["checkpoints"] / model_id / "training_summary.json", default=None)
        if not summary or summary.get("best_dice") is None:
            continue
        ranking.append((float(summary["best_dice"]), model_id))
    ranking.sort(reverse=True)
    return ranking[0][1] if ranking else None


SEGMENTATION_FOR_FEATURES = segmentation_for_classification()
print(
    "segmentation checkpoint used for the predicted-mask table (best validation Dice):",
    SEGMENTATION_FOR_FEATURES,
)

SEGMENTATION_BUILDERS = {
    "segnet_brats_4ch": lambda: SegNet(in_channels=4, out_channels=1),
    "unet_dynamic_512_real_brats": lambda: DynamicUNet(
        DEFAULT_FILTERS, input_channels=1, output_channels=1
    ),
    "raunet_brats_4ch": lambda: build_raunet(in_channels=4, out_channels=1, base_filters=16, depth=4),
}
_SEGMENTATION_CACHE: dict = {}


def predict_mask_probability(volume_key, index, model_id):
    """Probability map of `model_id` for one cached slice, at the cached mask resolution."""
    if model_id not in _SEGMENTATION_CACHE:
        model, _, path = load_trained_model(model_id, SEGMENTATION_BUILDERS[model_id])
        is_logit = str(getattr(model, "output_semantics", "logit")).lower().startswith("logit")
        _SEGMENTATION_CACHE[model_id] = (model, is_logit, path)
        print(f"loaded {model_id} from {path} for the predicted-mask features")
    model, is_logit, _ = _SEGMENTATION_CACHE[model_id]
    payload = load_patient(volume_key)
    if model_id == "unet_dynamic_512_real_brats":
        image = payload["singles"][index].astype(np.float32)[None, None, :, :]
    else:
        image = np.moveaxis(payload["images"][index].astype(np.float32), -1, 0)[None]
    with torch.no_grad():
        output = model(torch.from_numpy(image).to(DEVICE))
        if isinstance(output, (list, tuple)):
            output = output[0]
        tensor = output.float()
        probability = torch.sigmoid(tensor) if is_logit else tensor
        probability = probability.cpu().numpy()[0, 0]
    target = tuple(payload["masks"].shape[1:])
    if probability.shape != target:
        probability = np.asarray(
            PILImage.fromarray(probability.astype(np.float32), mode="F").resize(
                (target[1], target[0]), resample=PILImage.BILINEAR
            ),
            dtype=np.float32,
        )
    return probability, model_id


def patient_feature_row(volume_key, *, mask_source):
    """One feature row per patient: the focus slice on the serving channel (T1).

    The application extracts radiomics from a single analysed plane, so the training row is
    built from a single plane too. Averaging several slices would produce a schema the API can
    never feed.
    """
    payload = load_patient(volume_key)
    images = payload["images"].astype(np.float32)      # (S, 256, 256, 4)
    masks = payload["masks"].astype(np.float32)        # (S, 256, 256)
    if masks.ndim != 3 or masks.shape[0] == 0:
        return None, {"volume": volume_key, "mask_source": mask_source, "reason": "no cached slice"}
    focus, _ = focus_slice_index(masks)
    extra = {}
    if mask_source == "reference":
        mask = masks[focus] > 0.5
    else:
        if not SEGMENTATION_FOR_FEATURES:
            return None, {
                "volume": volume_key,
                "mask_source": mask_source,
                "reason": "no segmentation checkpoint is available for a predicted mask",
            }
        probability, model_id = predict_mask_probability(volume_key, focus, SEGMENTATION_FOR_FEATURES)
        mask = probability > 0.5
        extra = {
            "roi_mask_dice": dice_score(mask, masks[focus] > 0.5),
            "segmentation_model": model_id,
        }
    if not mask.any():
        return None, {
            "volume": volume_key,
            "mask_source": mask_source,
            "reason": "the region of interest is empty, so no region-based feature exists",
        }
    outcome = radiomics_service.extract(
        images[focus, :, :, SERVING_CHANNEL],
        mask,
        spacing=None,          # the Kaggle mirror carries no voxel spacing: units stay pixels
        mask_source=mask_source,
    )
    if not outcome.available:
        return None, {
            "volume": volume_key,
            "mask_source": mask_source,
            "reason": outcome.unavailable_reason,
        }
    row = {name: value for name, value in numeric_features(outcome)}
    row.update(
        {
            "case_id": f"{volume_key}_focus{focus}",
            "patient_id": volume_key,
            "label": payload["grade"],
            "split": next(
                (name for name in ("train", "validation", "test") if volume_key in SPLITS[name]),
                "unassigned",
            ),
            "mask_source": mask_source,
            "foreground_pixels": int(mask.sum()),
            "radiomics_backend": outcome.backend,
            "image_channel": SERVING_CHANNEL_NAME,
            "slice_index": focus,
        }
    )
    row.update(extra)
    return row, {
        "volume": volume_key,
        "mask_source": mask_source,
        "focus_slice": focus,
        "foreground_pixels": int(mask.sum()),
    }


FEATURE_TABLES: dict = {}
FEATURE_NAMES_BY_SOURCE: dict = {}
FEATURE_SKIPPED = {source: [] for source in MASK_SOURCES}
FEATURE_ROWS: list = []
FEATURE_NAMES: list = []

if CLASSIFICATION_POSSIBLE:
    if RUN["smoke_test"]:
        # Smoke run: the classifier needs rows in train, validation AND test, so extract for the
        # patients of those splits instead of the first N sorted volumes (which could all sit in
        # one split and make the comparison impossible).
        candidates = sorted(
            {key for key in (*TRAIN_PATIENTS, *VAL_PATIENTS, *TEST_PATIENTS) if key in LABELLED}
        )
    else:
        candidates = sorted(LABELLED)
        if RUN["radiomics_max_patients"]:
            candidates = candidates[: RUN["radiomics_max_patients"]]
    print(f"extracting features for {len(candidates)} labelled patient(s) x {len(MASK_SOURCES)} mask source(s)")
    for source in MASK_SOURCES:
        rows: list = []
        skipped: list = []
        started = time.time()
        for index, volume_key in enumerate(candidates, start=1):
            row, note = patient_feature_row(volume_key, mask_source=source)
            if row is None:
                skipped.append(note)
            else:
                rows.append(row)
            if index % 20 == 0 or index == len(candidates):
                elapsed = time.time() - started
                print(f"  {source:9s} {index}/{len(candidates)} patients ({elapsed:.0f}s, {elapsed / index:.3f}s per patient)")
        # A schema is only usable if every row carries the value: a name that is missing for
        # one patient would make the API refuse that case instead of scoring it.
        all_names = sorted({name for row in rows for name in row} - set(ROW_METADATA_COLUMNS) - {"roi_mask_dice", "segmentation_model"})
        schema = [
            name
            for name in all_names
            if all(isinstance(row.get(name), float) and np.isfinite(row[name]) for row in rows)
        ]
        dropped = [name for name in all_names if name not in schema]
        metadata = [column for column in ROW_METADATA_COLUMNS if column not in {"image_channel"}] + ["image_channel"]
        if source == "predicted":
            metadata = metadata + ["roi_mask_dice", "segmentation_model"]
        path = FEATURE_TABLE_PATHS[source]
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=[*metadata, *schema])
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key, "") if row.get(key) is not None else "" for key in writer.fieldnames})
        FEATURE_TABLES[source] = rows
        FEATURE_NAMES_BY_SOURCE[source] = schema
        FEATURE_SKIPPED[source] = skipped
        write_json(
            PATHS["artifacts"] / f"radiomics_features_{source}_report.json",
            {
                "mask_source": source,
                "patients_requested": len(candidates),
                "rows": len(rows),
                "feature_count": len(schema),
                "features_dropped_for_missing_values": dropped,
                "non_numeric_extractor_outputs": sorted(NON_NUMERIC_FEATURE_NAMES),
                "skipped": skipped[:20],
                "table": str(path),
                "image_channel": SERVING_CHANNEL_NAME,
                "serving_note": (
                    "Schema matches analysis_service._run_radiomics(): one plane "
                    "(display_stack[0], T1), bare feature names, no channel prefix."
                ),
            },
        )

    FEATURE_ROWS = FEATURE_TABLES.get("reference", [])
    FEATURE_NAMES = FEATURE_NAMES_BY_SOURCE.get("reference", [])

    print("=" * 78)
    print("CLASSIFICATION FEATURES")
    print("=" * 78)
    for source in MASK_SOURCES:
        rows = FEATURE_TABLES.get(source, [])
        print(
            f"{source:9s} mask: {len(rows):4d} row(s) | {len(FEATURE_NAMES_BY_SOURCE.get(source, [])):3d} feature(s) "
            f"| skipped {len(FEATURE_SKIPPED[source])} | {FEATURE_TABLE_PATHS[source].name}"
        )
        by_split: dict = {}
        for row in rows:
            by_split.setdefault(row["split"], {}).setdefault(row["label"], 0)
            by_split[row["split"]][row["label"]] += 1
        print(f"            rows per split/class: {by_split}")
    print(f"serving channel: {SERVING_CHANNEL} ({SERVING_CHANNEL_NAME}) | non-numeric extractor outputs excluded: {sorted(NON_NUMERIC_FEATURE_NAMES)}")
else:
    print(
        "No grade table was found, so no labelled feature table could be built. Classification is "
        "skipped: the platform must not train on an invented label."
    )

# %% [cell] 23. Baseline and improved classification
# =============================================================================
# SECTION 15 - Baseline and improved classifiers on the same held-out split
# =============================================================================
# For each mask source (reference, predicted):
#   * same patient-level split as segmentation (train / validation / test),
#   * imputation, scaling and constant-feature removal fitted on the training split only,
#   * baseline = logistic regression, improved = gradient boosting (or MLP),
#   * selection by validation macro-F1; test metrics reported once, for every candidate.
# The exported artefact carries the ordered feature schema, the scaler, the label mapping and
# the metrics -- exactly what ClassificationService requires to serve a prediction.
# =============================================================================
LABELS: list = []
LABEL_TO_INDEX: dict = {}
INDEX_TO_LABEL: dict = {}
train_only_imputer = None
train_only_scaler = None
ACTIVE_FEATURES: list = []
CONSTANT_FEATURES: list = []
SELECTED_MODEL = None
train_rows: list = []
CLASSIFICATION_EVALUATION = None

if CLASSIFICATION_POSSIBLE and FEATURE_ROWS:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_recall_fscore_support,
        roc_auc_score,
    )
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler

    LABELS = sorted({row["label"] for row in FEATURE_ROWS})
    LABEL_TO_INDEX = {label: index for index, label in enumerate(LABELS)}
    INDEX_TO_LABEL = {str(index): label for label, index in LABEL_TO_INDEX.items()}
    def evaluate_classifier(name, model, X_train, y_train, X_val, y_val, X_test, y_test):
        model.fit(X_train, y_train)
        report = {"model": name}
        for split_name, X, y in (("validation", X_val, y_val), ("test", X_test, y_test)):
            probabilities = model.predict_proba(X)
            predicted = probabilities.argmax(axis=1)
            precision, recall, f1, support = precision_recall_fscore_support(
                y, predicted, labels=list(range(len(LABELS))), zero_division=0
            )
            entry = {
                "accuracy": float(accuracy_score(y, predicted)),
                "macro_f1": float(f1_score(y, predicted, average="macro", zero_division=0)),
                "weighted_f1": float(f1_score(y, predicted, average="weighted", zero_division=0)),
                "per_class": {
                    INDEX_TO_LABEL[str(index)]: {
                        "precision": float(precision[index]),
                        "recall": float(recall[index]),
                        "f1": float(f1[index]),
                        "support": int(support[index]),
                    }
                    for index in range(len(LABELS))
                },
                "confusion_matrix": confusion_matrix(y, predicted, labels=list(range(len(LABELS)))).tolist(),
                "confusion_matrix_order": [INDEX_TO_LABEL[str(index)] for index in range(len(LABELS))],
                "samples": int(len(y)),
                "class_distribution": {INDEX_TO_LABEL[str(index)]: int((y == index).sum()) for index in range(len(LABELS))},
            }
            try:
                if len(LABELS) == 2:
                    entry["roc_auc"] = float(roc_auc_score(y, probabilities[:, 1]))
                    entry["roc_auc_average"] = f"binary (positive class = {LABELS[1]})"
                else:
                    entry["roc_auc"] = float(roc_auc_score(y, probabilities, multi_class="ovr", average="macro"))
                    entry["roc_auc_average"] = "macro one-vs-rest"
            except ValueError as exc:
                entry["roc_auc"] = None
                entry["roc_auc_error"] = str(exc)
            report[split_name] = entry
        return report

    def train_feature_source(source, rows, feature_names):
        """Fit both candidates on one feature table; return everything needed to export."""
        source_train = [row for row in rows if row["split"] == "train"]
        source_val = [row for row in rows if row["split"] == "validation"]
        source_test = [row for row in rows if row["split"] == "test"]
        missing = [
            name
            for name, subset in (("train", source_train), ("validation", source_val), ("test", source_test))
            if not subset
        ]
        if missing:
            raise SystemExit(
                f"these splits have no radiomics rows for the {source} mask: {missing}. Without a "
                "held-out patient-level test split no classification metric can be reported. Increase "
                "the number of patients, or check that the grade table covers enough volumes."
            )
        matrix = lambda subset: np.asarray(  # noqa: E731 - short local helper
            [[row.get(name, np.nan) for name in feature_names] for row in subset], dtype=np.float64
        )
        X_train_raw, X_val_raw, X_test_raw = matrix(source_train), matrix(source_val), matrix(source_test)
        y_train = np.asarray([LABEL_TO_INDEX[row["label"]] for row in source_train])
        y_val = np.asarray([LABEL_TO_INDEX[row["label"]] for row in source_val])
        y_test = np.asarray([LABEL_TO_INDEX[row["label"]] for row in source_test])

        imputer = SimpleImputer(strategy="median").fit(X_train_raw)
        scaler = StandardScaler().fit(imputer.transform(X_train_raw))
        X_train = scaler.transform(imputer.transform(X_train_raw))
        X_val = scaler.transform(imputer.transform(X_val_raw))
        X_test = scaler.transform(imputer.transform(X_test_raw))

        # ClassificationService applies median imputation and standardisation from the manifest and
        # then calls predict_proba on a vector of exactly len(feature_schema) values. Dropping
        # columns here would produce a model the API could never feed, so features that are constant
        # in the training split are *reported*, not removed. Standardisation turns them into zeros,
        # so they cannot influence the linear model either way.
        variances = np.nanstd(X_train, axis=0)
        if np.all(variances <= 1e-9):
            raise SystemExit(f"every {source}-mask feature is constant in the training split; check the extraction step")
        constant = [name for name, variance in zip(feature_names, variances) if variance <= 1e-9]
        active = list(feature_names)
        print(
            f"[{source}] features: {len(active)} | constant in the training split (kept: zero after "
            f"standardisation, no signal): {len(constant)}"
        )

        baseline_model = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED)
        if RUN["classifier_improved"] == "mlp":
            improved_model = MLPClassifier(
                hidden_layer_sizes=(64, 32), max_iter=1500, random_state=SEED, early_stopping=True
            )
        else:
            improved_model = HistGradientBoostingClassifier(random_state=SEED, max_iter=400, learning_rate=0.06)

        reports = {
            "baseline_logistic_regression": evaluate_classifier(
                "baseline_logistic_regression", baseline_model, X_train, y_train, X_val, y_val, X_test, y_test
            ),
            f"improved_{RUN['classifier_improved']}": evaluate_classifier(
                f"improved_{RUN['classifier_improved']}", improved_model, X_train, y_train, X_val, y_val, X_test, y_test
            ),
        }
        estimators = {
            "baseline_logistic_regression": baseline_model,
            f"improved_{RUN['classifier_improved']}": improved_model,
        }
        return {
            "source": source,
            "rows": {"train": source_train, "validation": source_val, "test": source_test},
            "imputer": imputer,
            "scaler": scaler,
            "features": active,
            "constant_features": constant,
            "reports": reports,
            "estimators": estimators,
        }

    FITTED = {}
    for source in MASK_SOURCES:
        rows = FEATURE_TABLES.get(source) or []
        names = FEATURE_NAMES_BY_SOURCE.get(source) or []
        if not rows or not names:
            continue
        FITTED[source] = train_feature_source(source, rows, names)

    if not FITTED:
        raise SystemExit("no feature table had rows and a usable schema; classification cannot be trained")

    comparison = {}
    for source, fitted in FITTED.items():
        for model_name, report in fitted["reports"].items():
            comparison[f"{source}::{model_name}"] = {
                "feature_source": source,
                "model": model_name,
                "validation": report["validation"],
                "test": report["test"],
            }

    selection_metric = "macro_f1"
    selected_key = max(comparison, key=lambda key: comparison[key]["validation"][selection_metric])
    selected_source = comparison[selected_key]["feature_source"]
    selected_fitted = FITTED[selected_source]

    # Bind the exported artefacts to the selected candidate.
    train_only_imputer = selected_fitted["imputer"]
    train_only_scaler = selected_fitted["scaler"]
    ACTIVE_FEATURES = selected_fitted["features"]
    CONSTANT_FEATURES = selected_fitted["constant_features"]
    SELECTED_MODEL = selected_fitted["estimators"][comparison[selected_key]["model"]]
    train_rows = selected_fitted["rows"]["train"]
    SELECTED_FEATURE_SOURCE = selected_source
    SELECTED_ESTIMATOR_NAME = comparison[selected_key]["model"]

    CLASSIFICATION_EVALUATION = {
        "target": "BraTS 2020 HGG vs LGG (WHO grade group recorded in name_mapping.csv)",
        "label_mapping": INDEX_TO_LABEL,
        "selection_metric": selection_metric,
        "selection_split": "validation",
        "selected_model": selected_key,
        "selected_estimator_name": SELECTED_ESTIMATOR_NAME,
        "selected_feature_source": SELECTED_FEATURE_SOURCE,
        "comparison": comparison,
        "feature_sources": {
            source: {
                "feature_count": len(fitted["features"]),
                "schema": fitted["features"],
                "constant_features_in_training_split": fitted["constant_features"],
                "table": str(FEATURE_TABLE_PATHS[source]),
                "mask_source": source,
                "rows": {split: len(rows) for split, rows in fitted["rows"].items()},
            }
            for source, fitted in FITTED.items()
        },
        "features": ACTIVE_FEATURES,
        "constant_features_in_training_split": CONSTANT_FEATURES,
        "non_numeric_extractor_outputs": sorted(NON_NUMERIC_FEATURE_NAMES),
        "image_channel": SERVING_CHANNEL_NAME,
        "splits": {
            split: {
                "rows": len(selected_fitted["rows"][split]),
                "patients": len({row["patient_id"] for row in selected_fitted["rows"][split]}),
            }
            for split in ("train", "validation", "test")
        },
        "class_balance": {
            split: {
                label: sum(1 for row in selected_fitted["rows"][split] if row["label"] == label)
                for label in LABELS
            }
            for split in ("train", "validation", "test")
        },
        "caveats": [
            "The label is the dataset's own grade annotation (WHO grade group). It is not a malignancy verdict, a stage or a prognosis.",
            "Features are computed on slices resized to 256x256 with no voxel spacing, so they describe the stored image, not physical anatomy.",
            "The number of patients per class bounds how much these numbers can mean; one split with few patients has wide intervals.",
            "Selection uses the validation split only. The test split is scored once here, after selection.",
        ],
        "generated_at": utc_now(),
    }
    write_json(PATHS["artifacts"] / "classification_evaluation.json", CLASSIFICATION_EVALUATION)

    print("=" * 78)
    print("CLASSIFICATION (patient-level split, features from the app's own extractor)")
    print("=" * 78)
    for key, entry in sorted(comparison.items()):
        validation, test = entry["validation"], entry["test"]
        print(
            f"{key:52s} val macro-F1 {validation['macro_f1']:.4f} (acc {validation['accuracy']:.4f}) | "
            f"test macro-F1 {test['macro_f1']:.4f} (acc {test['accuracy']:.4f}, AUC {test['roc_auc']})"
        )
    print(f"selected by validation {selection_metric}:", selected_key)
    print("test confusion matrix (rows reference, columns predicted):")
    print(comparison[selected_key]["test"]["confusion_matrix"])
    print("classes:", LABELS, "| test rows:", len(selected_fitted["rows"]["test"]))
else:
    print("Classification skipped: no labelled feature table (see the section above for the reason).")

# %% [cell] 24. Image-based classifier (optional, second view on the same split)
# =============================================================================
# SECTION 15b - Image-based classifier on the same patient-level split
# =============================================================================
# A compact 2-D CNN classifier on slices, pooled per patient at inference. It exists so the
# radiomics numbers can be compared with an image-based model on identical splits; it is
# deliberately small, trained only when a GPU is present, and it is not exported as a servable
# artefact (the API's classification path consumes radiomics features).
# =============================================================================
class SliceClassifier(nn.Module):
    def __init__(self, channels=4, width=24, n_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channels, width, 3, padding=1), nn.BatchNorm2d(width), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(width, width * 2, 3, padding=1), nn.BatchNorm2d(width * 2), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(width * 2, width * 4, 3, padding=1), nn.BatchNorm2d(width * 4), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(width * 4, n_classes)

    def forward(self, x):
        return self.head(self.features(x).flatten(1))


IMAGE_CLASSIFIER_SUMMARY = None
if CLASSIFICATION_POSSIBLE and FEATURE_ROWS and RUN["image_classifier_enabled"] and CUDA_AVAILABLE:
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score

    def classifier_batches(patient_ids, *, slices_per_patient, batch_size, shuffle, rng=None):
        items = []
        for volume_key in patient_ids:
            payload = load_patient(volume_key)
            masks = payload["masks"]
            counts = masks.reshape(masks.shape[0], -1).sum(axis=1)
            focus = int(np.argmax(counts))
            half = slices_per_patient // 2
            for index in range(max(0, focus - half), min(len(counts), focus + half + 1)):
                if counts[index] == 0:
                    continue
                items.append((volume_key, index))
        if shuffle:
            rng = rng or np.random.default_rng(SEED)
            rng.shuffle(items)
        for start in range(0, len(items), batch_size):
            chunk = items[start : start + batch_size]
            images, labels = [], []
            for volume_key, index in chunk:
                payload = load_patient(volume_key)
                images.append(np.moveaxis(payload["images"][index].astype(np.float32), -1, 0))
                labels.append(LABEL_TO_INDEX[payload["grade"]])
            yield np.stack(images), np.asarray(labels, dtype=np.int64), [item[0] for item in chunk]

    def image_autocast():
        try:
            return torch.amp.autocast("cuda", enabled=AMP_ENABLED)
        except (AttributeError, TypeError):
            return torch.cuda.amp.autocast(enabled=AMP_ENABLED)

    image_classifier = SliceClassifier(channels=4).to(DEVICE)
    image_optimizer = torch.optim.AdamW(image_classifier.parameters(), lr=3e-4)
    try:
        image_scaler = torch.amp.GradScaler("cuda", enabled=AMP_ENABLED)
    except (AttributeError, TypeError):
        image_scaler = torch.cuda.amp.GradScaler(enabled=AMP_ENABLED)
    image_loss = nn.CrossEntropyLoss()
    IMAGE_HISTORY = []
    image_started = time.time()
    for epoch in range(1, RUN["image_classifier_epochs"] + 1):
        image_classifier.train()
        losses = []
        for images, labels, _ in classifier_batches(
            TRAIN_PATIENTS, slices_per_patient=RUN["image_classifier_slices_per_patient"], batch_size=RUN["batch_size"], shuffle=True
        ):
            batch_images = torch.from_numpy(images).to(DEVICE).to(torch.float32)
            batch_labels = torch.from_numpy(labels).to(DEVICE)
            image_optimizer.zero_grad(set_to_none=True)
            with image_autocast():
                loss = image_loss(image_classifier(batch_images), batch_labels)
            image_scaler.scale(loss).backward()
            image_scaler.step(image_optimizer)
            image_scaler.update()
            losses.append(float(loss.detach()))
        IMAGE_HISTORY.append({"epoch": epoch, "train_loss": float(np.mean(losses)) if losses else None})
        print(f"image classifier epoch {epoch}/{RUN['image_classifier_epochs']}  loss {IMAGE_HISTORY[-1]['train_loss']:.4f}")

    def patient_predictions(patient_ids):
        image_classifier.eval()
        per_patient = {}
        with torch.no_grad():
            for images, _, volume_keys in classifier_batches(
                patient_ids, slices_per_patient=RUN["image_classifier_slices_per_patient"], batch_size=RUN["batch_size"], shuffle=False
            ):
                with image_autocast():
                    logits = image_classifier(torch.from_numpy(images).to(DEVICE).to(torch.float32))
                probabilities = torch.softmax(logits.float(), dim=1).cpu().numpy()
                for row, volume_key in enumerate(volume_keys):
                    per_patient.setdefault(volume_key, []).append(probabilities[row])
        return {volume_key: np.mean(np.stack(stack), axis=0) for volume_key, stack in per_patient.items()}

    image_metrics = {}
    for split_name, patient_ids in (("validation", VAL_PATIENTS), ("test", TEST_PATIENTS)):
        probabilities = patient_predictions(patient_ids)
        usable = [key for key in patient_ids if key in probabilities and VOLUMES[key]["grade"] in LABEL_TO_INDEX]
        if not usable:
            image_metrics[split_name] = {"state": "not_evaluated", "reason": "no patient had both a prediction and a label"}
            continue
        truth = np.asarray([LABEL_TO_INDEX[VOLUMES[key]["grade"]] for key in usable])
        scores = np.stack([probabilities[key] for key in usable])
        predicted = scores.argmax(axis=1)
        image_metrics[split_name] = {
            "accuracy": float(accuracy_score(truth, predicted)),
            "macro_f1": float(f1_score(truth, predicted, average="macro", zero_division=0)),
            "patients": len(usable),
            "roc_auc": float(roc_auc_score(truth, scores[:, 1]))
            if len(LABELS) == 2 and len(set(truth.tolist())) > 1
            else None,
            "confusion_matrix": confusion_matrix(truth, predicted, labels=list(range(len(LABELS)))).tolist(),
        }

    torch.save(
        {
            "state_dict": {key: value.detach().cpu() for key, value in image_classifier.state_dict().items()},
            "architecture": "SliceClassifier(channels=4, width=24)",
            "config": {"epochs": RUN["image_classifier_epochs"]},
            "label_mapping": INDEX_TO_LABEL,
            "history": IMAGE_HISTORY,
        },
        PATHS["checkpoints"] / "slice_classifier_image_based.pt",
    )
    IMAGE_CLASSIFIER_SUMMARY = {
        "architecture": "SliceClassifier(channels=4, width=24), slice predictions averaged per patient",
        "split": "same patient-level split as the radiomics classifiers",
        "history": IMAGE_HISTORY,
        "metrics": image_metrics,
        "elapsed_minutes": round((time.time() - image_started) / 60, 2),
        "exported": False,
        "note": (
            "Comparison only: the API's classification path consumes radiomics features, so this "
            "checkpoint is stored in the notebook output but not exported as a servable classifier."
        ),
    }
    write_json(PATHS["artifacts"] / "image_classifier_summary.json", IMAGE_CLASSIFIER_SUMMARY)
    print("=" * 78)
    print("IMAGE-BASED CLASSIFIER (same patient-level split)")
    print("=" * 78)
    for split_name, entry in image_metrics.items():
        print(f"{split_name:11s}: {entry}")
else:
    print(
        "Image-based classifier skipped "
        f"(labels={CLASSIFICATION_POSSIBLE}, rows={len(FEATURE_ROWS)}, "
        f"enabled={RUN['image_classifier_enabled']}, cuda={CUDA_AVAILABLE})."
    )
    print(
        "It needs a GPU to be worth running; the radiomics comparison in Section 15 still holds, "
        "and no image-based number is reported in its place."
    )

# %% [cell] 25. Radiomics quality: reference mask vs predicted mask
# =============================================================================
# SECTION 16 - Radiomics extraction detail and quality
# =============================================================================
# Three things matter for honesty here:
#   1. features computed on a predicted mask inherit the segmentation error,
#   2. features computed on resized 256x256 slices without voxel spacing are
#      pixel-domain numbers, not physical ones,
#   3. a single-slice region is a small sample: neighbouring slices disagree.
# This cell measures 1 and 3 on the data instead of asserting anything about them.
# =============================================================================
def radiomics_quality_report(reference_rows, predicted_rows, schema, *, comparison_patients=8):
    """Measure the mask-source drift and the slice-to-slice stability of the feature table.

    Written as a function over the two tables rather than as inline code so the test suite can
    exercise it on tables built by the notebook itself.
    """
    reference_by_patient = {row["patient_id"]: row for row in reference_rows}
    predicted_by_patient = {row["patient_id"]: row for row in predicted_rows}
    shared_patients = sorted(set(reference_by_patient) & set(predicted_by_patient))
    shared_schema = [name for name in schema if name in set(predicted_rows[0])] if predicted_rows else []
    per_feature = {}
    for name in shared_schema:
        relative = []
        for patient in shared_patients:
            reference_value = float(reference_by_patient[patient][name])
            predicted_value = float(predicted_by_patient[patient][name])
            relative.append(abs(predicted_value - reference_value) / (abs(reference_value) + 1e-9))
        per_feature[name] = float(np.median(relative))

    # Slice-to-slice variability of the reference-mask features around the focus slice.
    slice_variability = {}
    for patient in shared_patients[:comparison_patients]:
        payload = load_patient(patient)
        masks = payload["masks"].astype(np.float32)
        images = payload["images"].astype(np.float32)
        focus, _ = focus_slice_index(masks)
        focus_features = dict(
            numeric_features(
                radiomics_service.extract(
                    images[focus, :, :, SERVING_CHANNEL], masks[focus] > 0.5, spacing=None, mask_source="reference"
                )
            )
        )
        for offset in (-1, 1):
            neighbour = focus + offset
            if not 0 <= neighbour < masks.shape[0] or not (masks[neighbour] > 0.5).any():
                continue
            neighbour_features = dict(
                numeric_features(
                    radiomics_service.extract(
                        images[neighbour, :, :, SERVING_CHANNEL],
                        masks[neighbour] > 0.5,
                        spacing=None,
                        mask_source="reference",
                    )
                )
            )
            for name, value in neighbour_features.items():
                if name in focus_features:
                    slice_variability.setdefault(name, []).append(
                        abs(value - focus_features[name]) / (abs(focus_features[name]) + 1e-9)
                    )
    per_feature_variability = {name: float(np.median(values)) for name, values in slice_variability.items()}

    mask_dice_values = [
        float(row["roi_mask_dice"]) for row in predicted_rows if isinstance(row.get("roi_mask_dice"), float)
    ]
    return {
        "state": "measured",
        "patients_compared": len(shared_patients),
        "shared_feature_count": len(shared_schema),
        "reference_vs_predicted_mask": {
            "median_relative_difference_over_features": float(np.median(list(per_feature.values()))) if per_feature else None,
            "p90_relative_difference_over_features": float(np.percentile(list(per_feature.values()), 90)) if per_feature else None,
            "per_feature_median_relative_difference": per_feature,
            "roi_mask_dice": {
                "median": float(np.median(mask_dice_values)) if mask_dice_values else None,
                "min": float(np.min(mask_dice_values)) if mask_dice_values else None,
                "max": float(np.max(mask_dice_values)) if mask_dice_values else None,
                "patients": len(mask_dice_values),
                "model": SEGMENTATION_FOR_FEATURES,
                "note": "Dice of the predicted mask against the BraTS reference mask on the focus slice.",
            },
        },
        "slice_to_slice_variability": {
            "patients": min(len(shared_patients), comparison_patients),
            "features_compared": len(per_feature_variability),
            "median_relative_difference_over_features": float(np.median(list(per_feature_variability.values())))
            if per_feature_variability
            else None,
            "per_feature_median_relative_difference": per_feature_variability,
            "note": "Focus slice against its immediate neighbours with the reference mask, same extractor and settings.",
        },
        "spacing_available": False,
        "dimensionality": "2D (single slice) features on 256x256 resized images",
        "caveats": [
            "Physical-unit features (the *_mm entries) are None everywhere: the Kaggle slice mirror carries no voxel spacing, so those are excluded from the schema rather than approximated.",
            "A classifier trained on reference regions and served on predicted regions inherits the difference measured above; the measured number is the size of that inheritance, not a correction for it.",
            "Slice-to-slice variability is reported because the training row is a single slice, like a served analysis.",
        ],
        "generated_at": utc_now(),
    }


RADIOMICS_QUALITY: dict = {"state": "not_evaluated"}
if CLASSIFICATION_POSSIBLE and FEATURE_TABLES.get("reference") and FEATURE_TABLES.get("predicted"):
    RADIOMICS_QUALITY = radiomics_quality_report(
        FEATURE_TABLES["reference"],
        FEATURE_TABLES["predicted"],
        FEATURE_NAMES_BY_SOURCE["reference"],
    )
    write_json(PATHS["artifacts"] / "radiomics_quality.json", RADIOMICS_QUALITY)
    print("=" * 78)
    print("RADIOMICS QUALITY (measured, not assumed)")
    print("=" * 78)
    print("patients compared                        :", RADIOMICS_QUALITY["patients_compared"])
    print("mask Dice (predicted vs reference, focus):", RADIOMICS_QUALITY["reference_vs_predicted_mask"]["roi_mask_dice"]["median"])
    print("mask source drift (median over features) :", RADIOMICS_QUALITY["reference_vs_predicted_mask"]["median_relative_difference_over_features"])
    print("slice-to-slice drift (median)            :", RADIOMICS_QUALITY["slice_to_slice_variability"]["median_relative_difference_over_features"])
else:
    reasons = []
    if not CLASSIFICATION_POSSIBLE:
        reasons.append("no grade table, so no labelled row exists")
    if not FEATURE_TABLES.get("reference"):
        reasons.append("the reference-mask table is empty")
    if not FEATURE_TABLES.get("predicted"):
        reasons.append(
            "the predicted-mask table is empty ("
            + "; ".join(sorted({str(note.get("reason")) for note in FEATURE_SKIPPED.get("predicted", []) if note.get("reason")})[:3])
            + ")"
        )
    print("Radiomics quality comparison skipped: " + "; ".join(reasons) + ".")
    print(
        "Nothing is reported in its place: an unmeasured difference is not a measured one. Re-run "
        "after the segmentation models converge and the predicted regions stop being empty."
    )
