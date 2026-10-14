from __future__ import annotations
import argparse
import csv
import json
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (
    BACKEND_ROOT,
    code_revision,
    configure_logging,
    describe_environment,
    resolve_data_dir,
    seed_everything,
    torch_device,
    utc_now,
    write_run_record,
)
sys.path.insert(0, str(BACKEND_ROOT))
IDENTIFIER_COLUMNS = {
    "case_id",
    "case",
    "id",
    "patient",
    "patient_id",
    "subject",
    "subject_id",
    "scan_id",
    "study",
    "study_id",
    "name",
    "filename",
    "path",
    "label",
    "target",
    "grade",
    "class",
    "diagnosis",
    "split",
    "group",
}
def main() -> int:
    parser = argparse.ArgumentParser(description="Train the radiomics classifier from a labelled feature table")
    parser.add_argument("--features", type=Path, required=True, help="CSV or JSON table with one row per case")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--patient-column", default=None, help="Column with patient identifiers (patient-level split)")
    parser.add_argument("--id-column", default=None, help="Column with case identifiers")
    parser.add_argument("--feature-columns", default=None, help="Comma-separated feature column names (default: inferred)")
    parser.add_argument("--architecture", default="mlp", choices=["linear", "mlp", "feature_fusion"])
    parser.add_argument("--hidden-dims", default="128,64")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20240101)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--target-description", default=None, help="What the label means, recorded in the manifest")
    parser.add_argument("--model-id", default="radiomics_classifier")
    parser.add_argument("--version", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    logger = configure_logging(args.verbose)
    seed_everything(args.seed)
    rows, columns = _read_table(args.features)
    if args.label_column not in columns:
        raise SystemExit(
            f"Column '{args.label_column}' is not in {args.features}. Available columns: "
            f"{', '.join(sorted(columns)[:20])}{' ...' if len(columns) > 20 else ''}"
        )
    feature_columns = (
        [item.strip() for item in args.feature_columns.split(",")]
        if args.feature_columns
        else [name for name in columns if name not in IDENTIFIER_COLUMNS]
    )
    if not feature_columns:
        raise SystemExit("No feature columns were found; pass --feature-columns explicitly.")
    labels: list[str] = []
    features: list[list[float]] = []
    groups: list[str] = []
    skipped = 0
    for index, row in enumerate(rows):
        label = str(row.get(args.label_column, "")).strip()
        if not label:
            skipped += 1
            continue
        values: list[float] = []
        complete = True
        for name in feature_columns:
            raw = row.get(name, "")
            try:
                values.append(float(raw))
            except (TypeError, ValueError):
                complete = False
                break
        if not complete:
            skipped += 1
            continue
        labels.append(label)
        features.append(values)
        group = (
            str(row.get(args.patient_column, "")).strip()
            if args.patient_column
            else str(row.get(args.id_column, "")).strip() if args.id_column else f"row-{index:05d}"
        )
        groups.append(group or f"row-{index:05d}")
    if not features:
        raise SystemExit("No complete labelled row was found; nothing can be trained.")
    if skipped:
        logger.warning("Skipped %d row(s) with a missing label or a missing feature value.", skipped)
    matrix = np.asarray(features, dtype=np.float32)
    classes = sorted(set(labels))
    label_mapping = {str(index): name for index, name in enumerate(classes)}
    y = np.asarray([classes.index(label) for label in labels], dtype=np.int64)
    logger.info(
        "Table: %d usable row(s) · %d feature(s) · %d class(es) %s",
        len(matrix),
        len(feature_columns),
        len(classes),
        classes,
    )
    if len(classes) < 2:
        raise SystemExit("A classifier needs at least two classes; the table contains one.")
    if len(matrix) < 10 * len(feature_columns):
        logger.warning(
            "Only %d rows for %d features. With this ratio a classifier will fit the sample, not "
            "the biology; treat any metric from this run as provisional and report the size.",
            len(matrix),
            len(feature_columns),
        )
    from sklearn.model_selection import GroupShuffleSplit, train_test_split
    if args.patient_column or args.id_column:
        gss_test = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.seed)
        train_val_idx, test_idx = next(gss_test.split(matrix, y, groups=np.asarray(groups)))
        remaining_groups = np.asarray(groups)[train_val_idx]
        gss_val = GroupShuffleSplit(n_splits=1, test_size=args.val_size, random_state=args.seed)
        train_idx_rel, val_idx_rel = next(gss_val.split(matrix[train_val_idx], y[train_val_idx], groups=remaining_groups))
        train_idx = train_val_idx[train_idx_rel]
        val_idx = train_val_idx[val_idx_rel]
        strategy = "patient_level_group_shuffle"
    else:
        train_val_idx, test_idx = train_test_split(
            np.arange(len(matrix)), test_size=args.test_size, random_state=args.seed, stratify=y
        )
        train_idx, val_idx = train_test_split(
            train_val_idx, test_size=args.val_size, random_state=args.seed, stratify=y[train_val_idx]
        )
        strategy = "row_level_stratified"
        logger.warning(
            "No patient column was given, so the split is row-level. Slices or repeated measurements "
            "from one patient can then appear in both training and test data, which inflates the "
            "metrics. Record this as a limitation."
        )
    logger.info("Split (%s): train %d · validation %d · test %d", strategy, len(train_idx), len(val_idx), len(test_idx))
    if min(len(train_idx), len(val_idx), len(test_idx)) == 0:
        raise SystemExit("A split came out empty; provide more cases or adjust --test-size/--val-size.")
    mean = matrix[train_idx].mean(axis=0)
    scale = matrix[train_idx].std(axis=0)
    scale[scale == 0] = 1.0
    standardised = (matrix - mean) / scale
    if args.dry_run:
        logger.info("Dry run: data validated, no model trained.")
        logger.info("Features (%d): %s", len(feature_columns), ", ".join(feature_columns[:8]) + (" ..." if len(feature_columns) > 8 else ""))
        return 0
    model, val_report = _fit_torch(
        args,
        standardised,
        y,
        train_idx,
        val_idx,
        test_idx,
        len(feature_columns),
        len(classes),
        logger,
    )
    from app.services.evaluation_service import EvaluationService
    probabilities = model(test_idx)
    predicted = probabilities.argmax(axis=1)
    test_report = EvaluationService.evaluate_classification(
        [int(value) for value in y[test_idx]],
        [int(value) for value in predicted],
        [[float(component) for component in row] for row in probabilities],
        n_classes=len(classes),
        label_mapping=label_mapping,
    )
    test_report["split_sizes"] = {"train": len(train_idx), "validation": len(val_idx), "test": len(test_idx)}
    test_report["split_strategy"] = strategy
    test_report["validation"] = val_report
    test_report["test_rows"] = int(len(test_idx))
    data_dir = resolve_data_dir(args.data_dir)
    artifact_dir = data_dir / "models" / "classification"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    version = args.version or utc_now().replace(":", "").replace("-", "")
    model_file = f"{args.model_id}-{version}.pt"
    import torch
    torch.save(
        {
            "state_dict": model.module.state_dict(),
            "architecture": args.architecture,
            "architecture_kwargs": {
                "n_classes": len(classes),
                "hidden_dims": [int(item) for item in args.hidden_dims.split(",")],
                "dropout": args.dropout,
            },
        },
        artifact_dir / model_file,
    )
    manifest = {
        "model_id": args.model_id,
        "version": version,
        "kind": "torch",
        "model_file": model_file,
        "architecture": args.architecture,
        "architecture_kwargs": {
            "n_classes": len(classes),
            "hidden_dims": [int(item) for item in args.hidden_dims.split(",")],
            "dropout": args.dropout,
        },
        "feature_schema": feature_columns,
        "label_mapping": label_mapping,
        "target_description": args.target_description
        or f"Column '{args.label_column}' with classes {classes}",
        "preprocessing": {
            "type": "standard_scaler",
            "mean": [float(value) for value in mean],
            "scale": [float(value) for value in scale],
            "fit_on": "training split only",
        },
        "split": {
            "strategy": strategy,
            "sizes": {"train": len(train_idx), "validation": len(val_idx), "test": len(test_idx)},
            "seed": args.seed,
            "test_used_for_tuning": False,
        },
        "metrics": {
            "validation": val_report,
            "test": {key: value for key, value in test_report.items() if key != "reliability_curve"},
        },
        "training": {
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "patience": args.patience,
            "batch_size": args.batch_size,
        },
        "source_table": str(args.features),
        "trained_at": utc_now(),
        "code_revision": code_revision(),
        "environment": describe_environment(),
        "note": (
            "Predictions from this artefact restate a statistical association between radiomics "
            "features and the recorded label. They are not a diagnosis, a grade or a prognosis, and "
            "they are meaningful only for inputs produced the same way as the training table."
        ),
    }
    (artifact_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    (artifact_dir / "evaluation.json").write_text(json.dumps(test_report, indent=2, default=str), encoding="utf-8")
    logger.info("Artefact written to %s", artifact_dir / model_file)
    logger.info("Test metrics: %s", {key: test_report.get(key) for key in ("accuracy", "balanced_accuracy", "f1_macro", "roc_auc")})
    write_run_record(
        data_dir / "models" / "training",
        {
            "model_id": args.model_id,
            "status": "completed",
            "kind": "classification",
            "feature_count": len(feature_columns),
            "classes": classes,
            "split": manifest["split"],
            "metrics": manifest["metrics"],
            "code_revision": code_revision(),
        },
    )
    logger.info(
        "The classification branch is now active. It will appear in analyses only for inputs whose "
        "feature names match the recorded schema."
    )
    return 0
def _read_table(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    if path.suffix.lower() == ".csv":
        with open(path, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            rows = [dict(row) for row in reader]
            columns = list(reader.fieldnames or [])
        return rows, columns
    if path.suffix.lower() in {".json", ".jsonl"}:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            payload = payload.get("rows") or payload.get("records") or []
        rows = [dict(row) for row in payload]
        columns = sorted({key for row in rows for key in row})
        return rows, columns
    raise SystemExit(f"Unsupported table format: {path.suffix}. Use CSV or JSON.")
def _fit_torch(args, standardised, y, train_idx, val_idx, test_idx, n_features, n_classes, logger):
    import torch
    from app.ml.classifier_models import build_classifier
    device = torch_device(args.device)
    module = build_classifier(
        args.architecture,
        input_dim=n_features,
        n_classes=n_classes,
        hidden_dims=[int(item) for item in args.hidden_dims.split(",")],
        dropout=args.dropout,
    ).to(device)
    optimiser = torch.optim.AdamW(module.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = torch.nn.CrossEntropyLoss()
    train_x = torch.from_numpy(standardised[train_idx]).to(device)
    train_y = torch.from_numpy(y[train_idx]).to(device)
    val_x = torch.from_numpy(standardised[val_idx]).to(device)
    val_y = torch.from_numpy(y[val_idx]).to(device)
    best_state = None
    best_loss = float("inf")
    patience = 0
    history: list[dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        module.train()
        permutation = torch.randperm(len(train_x), device=device)
        total = 0.0
        for start in range(0, len(train_x), args.batch_size):
            batch = permutation[start : start + args.batch_size]
            optimiser.zero_grad(set_to_none=True)
            logits = module(train_x[batch])
            loss = criterion(logits, train_y[batch])
            loss.backward()
            optimiser.step()
            total += float(loss.detach()) * len(batch)
        module.eval()
        with torch.no_grad():
            val_logits = module(val_x)
            val_loss = float(criterion(val_logits, val_y))
            val_accuracy = float((val_logits.argmax(dim=1) == val_y).float().mean())
        history.append({"epoch": epoch, "train_loss": round(total / len(train_x), 6), "val_loss": round(val_loss, 6), "val_accuracy": round(val_accuracy, 4)})
        if val_loss < best_loss - 1e-5:
            best_loss = val_loss
            best_state = {key: value.detach().clone() for key, value in module.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= args.patience:
                logger.info("Early stopping at epoch %d after %d epochs without improvement.", epoch, args.patience)
                break
    if best_state is not None:
        module.load_state_dict(best_state)
    module.eval()
    with torch.no_grad():
        val_probabilities = torch.softmax(module(val_x), dim=1).cpu().numpy()
    class Trained:
        def __init__(self):
            self.module = module
        def __call__(self, indices) -> np.ndarray:
            with torch.no_grad():
                tensor = torch.from_numpy(standardised[indices]).to(device)
                return torch.softmax(module(tensor), dim=1).cpu().numpy()
    from app.services.evaluation_service import EvaluationService
    val_report = EvaluationService.evaluate_classification(
        [int(value) for value in y[val_idx]],
        [int(value) for value in val_probabilities.argmax(axis=1)],
        [[float(component) for component in row] for row in val_probabilities],
        n_classes=n_classes,
        label_mapping={str(index): f"class-{index}" for index in range(n_classes)},
    )
    val_report["history"] = history
    val_report["best_validation_loss"] = round(best_loss, 6)
    return Trained(), val_report
if __name__ == "__main__":
    raise SystemExit(main())
