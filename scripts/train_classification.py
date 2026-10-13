from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _classification import (
    assign_patient_splits,
    build_matrix,
    class_weights,
    declared_splits,
    evaluate,
    feature_names,
    impute,
    linear_coefficients,
    quantiles,
    select_rows,
    split_summary,
    standardise,
    text_report,
)
from _common import (
    BACKEND_ROOT,
    code_revision,
    configure_logging,
    describe_environment,
    resolve_data_dir,
    seed_everything,
    utc_now,
)

sys.path.insert(0, str(BACKEND_ROOT))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Train and compare the tumour-grade classifier (for example BraTS HGG vs LGG) on a "
            "labelled radiomics table. A baseline and an improved model are fitted on the same "
            "training rows and scored on the same held-out test split."
        )
    )
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--patient-column", default="patient_id")
    parser.add_argument("--id-column", default="case_id")
    parser.add_argument("--split-column", default="split", help="Respect this column when it holds train/validation/test")
    parser.add_argument("--feature-columns", default=None)
    parser.add_argument("--exclude-columns", default=None, help="Extra columns to keep out of the feature set")
    parser.add_argument("--deep-features", type=Path, default=None, help="Optional second table (deep features) merged on case_id")
    parser.add_argument("--train", type=float, default=0.70)
    parser.add_argument("--val", type=float, default=0.15)
    parser.add_argument("--test", type=float, default=0.15)
    parser.add_argument("--target-description", default=None)
    parser.add_argument("--model-id", default="radiomics_grade_classifier")
    parser.add_argument("--version", default=None)
    parser.add_argument("--improved", choices=["mlp", "gradient_boosting", "random_forest"], default="mlp")
    parser.add_argument("--hidden-dims", default="64,32")
    parser.add_argument("--max-iter", type=int, default=600)
    parser.add_argument("--class-weight", choices=["none", "balanced"], default="balanced")
    parser.add_argument("--seed", type=int, default=20240101)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--no-artefact", action="store_true", help="Compare models without writing a servable artefact")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    logger = configure_logging(args.verbose)
    seed_everything(args.seed)
    rows, columns = _read_table(args.features)
    if not rows:
        raise SystemExit(f"{args.features} is empty")
    if args.label_column not in columns:
        raise SystemExit(
            f"Column '{args.label_column}' is not in {args.features}. Columns: {', '.join(columns[:20])}"
        )

    excluded = {item.strip() for item in (args.exclude_columns or "").split(",") if item.strip()}
    explicit = [item.strip() for item in args.feature_columns.split(",")] if args.feature_columns else None
    names = feature_names(rows, [name for name in columns if name not in excluded], explicit)

    deep_names: list[str] = []
    if args.deep_features:
        deep_rows, _ = _read_table(args.deep_features)
        deep_names = [
            name
            for name in (deep_rows[0].keys() if deep_rows else [])
            if name not in {"case_id", "patient_id", "split"}
        ]
        by_case = {str(row.get("case_id", "")): row for row in deep_rows}
        merged = 0
        for row in rows:
            match = by_case.get(str(row.get("case_id", "")))
            if not match:
                continue
            for name in deep_names:
                row[name] = match.get(name, "")
            merged += 1
        if merged == 0:
            raise SystemExit(
                f"No case_id in {args.features} matched {args.deep_features}; the deep feature table "
                "must be produced from the same manifest."
            )
        names = names + deep_names
        logger.info("Merged %d deep-feature row(s); %d feature(s) in total", merged, len(names))

    usable, rejected = select_rows(
        rows,
        label_column=args.label_column,
        patient_column=args.patient_column,
        id_column=args.id_column,
    )
    if rejected:
        reasons = ", ".join(f"{item['case_id']}: {item['reason']}" for item in rejected[:3])
        logger.warning("%d row(s) rejected (%s)", len(rejected), reasons)
    if not usable:
        raise SystemExit("No usable labelled row; nothing can be trained.")

    labels_seen = sorted({row["label"] for row in usable})
    if len(labels_seen) < 2:
        raise SystemExit(f"Only one class present ({labels_seen}); a classifier cannot be trained or evaluated.")
    label_mapping = {str(index): name for index, name in enumerate(labels_seen)}
    index_of = {name: index for index, name in enumerate(labels_seen)}

    declared = declared_splits(usable) if args.split_column else None
    if declared:
        splits = declared
        method = f"taken from the '{args.split_column}' column of {args.features}"
    else:
        splits = assign_patient_splits(
            usable, train=args.train, val=args.val, test=args.test, seed=args.seed, logger=logger
        )
        method = f"assigned here, patient-level, stratified, seed {args.seed}"
    summary = split_summary(usable, splits)
    for name in ("train", "validation", "test"):
        entry = summary[name]
        logger.info(
            "%-10s %5d row(s) · %4d patient(s) · classes %s",
            name,
            entry["rows"],
            entry["patients"],
            entry["class_distribution"],
        )
    if not summary["leakage_free"]:
        raise SystemExit(
            "Patients appear in more than one split "
            f"({summary['patient_overlap'][:5]}). Training would leak; fix the split column or unset it."
        )
    if summary["test"]["rows"] == 0:
        raise SystemExit("The test split is empty; nothing can be reported on held-out data.")
    logger.info("Splits: %s", method)

    matrix, _ = build_matrix(usable, names)
    filled, medians = impute(matrix, np.isnan(matrix), splits["train"])
    scaled, means, scales = standardise(filled, splits["train"])
    y = np.asarray([index_of[row["label"]] for row in usable], dtype=int)
    n_classes = len(labels_seen)

    if args.dry_run:
        logger.info(
            "Dry run: %d feature(s), %d class(es), splits validated, no model fitted.",
            len(names),
            n_classes,
        )
        return 0

    from sklearn.linear_model import LogisticRegression

    if args.class_weight == "none":
        weights = None
    else:
        weights = class_weights(y[splits["train"]], n_classes)[y[splits["train"]]]
    baseline = LogisticRegression(max_iter=args.max_iter, random_state=args.seed)
    baseline.fit(scaled[splits["train"]], y[splits["train"]], sample_weight=weights)

    improved = _build_improved(args, n_classes)
    improved.fit(scaled[splits["train"]], y[splits["train"]], sample_weight=weights)

    predictions = {
        "baseline_logistic_regression": _score(baseline, scaled, y, splits, n_classes, labels_seen),
        f"improved_{args.improved}": _score(improved, scaled, y, splits, n_classes, labels_seen),
    }
    for name, payload in predictions.items():
        logger.info(
            "%-32s validation macro-F1 %.4f · test accuracy %.4f · test macro-F1 %.4f · test ROC-AUC %s",
            name,
            payload["validation"]["macro_f1"] if payload["validation"] else float("nan"),
            payload["test"]["accuracy"],
            payload["test"]["macro_f1"],
            payload["test"]["roc_auc"],
        )
    logger.info(
        "Model selection uses the validation split only; the test numbers above are reported for "
        "both models and neither was tuned on them."
    )
    best_name = max(
        predictions,
        key=lambda key: (predictions[key]["validation"] or {"macro_f1": -1})["macro_f1"],
    )
    logger.info("Selected by validation macro-F1: %s", best_name)

    for name, payload in predictions.items():
        if payload["test"]["samples"]:
            logger.info("Test classification report for %s:\n%s", name, payload["report"])

    evaluation = {
        "target": args.target_description
        or f"Dataset label '{args.label_column}' with classes {', '.join(labels_seen)}",
        "label_mapping": label_mapping,
        "class_distribution": {str(index): int((y == index).sum()) for index in range(n_classes)},
        "split_strategy": method,
        "splits": {name: {key: value for key, value in summary[name].items() if key != "patient_ids"} for name in ("train", "validation", "test")},
        "leakage_check": {"patient_overlap": summary["patient_overlap"], "passed": summary["leakage_free"]},
        "comparison": {
            name: {
                "validation": payload["validation"],
                "test": payload["test"],
            }
            for name, payload in predictions.items()
        },
        "selected_model": best_name,
        "feature_count": len(names),
        "feature_schema": names,
        "deep_features_used": bool(deep_names),
        "preprocessing": {
            "imputation": "training-split median",
            "standardisation": "training-split mean and standard deviation",
        },
        "seed": args.seed,
        "generated_at": utc_now(),
        "code_revision": code_revision(),
        "environment": describe_environment(),
        "caveats": [
            "The label is the dataset's own annotation (for example a WHO grade recorded by the "
            "dataset). It is not a malignancy or staging label and must not be described as one.",
            "Metrics come from a single split; with few patients per class the confidence intervals "
            "are wide and are not reported here.",
            "A classifier trained on radiomics from predicted masks inherits the segmentation error.",
        ],
    }

    data_dir = resolve_data_dir(args.data_dir)
    output_dir = args.output_dir or data_dir / "models" / "classification" / _artefact_name(args, data_dir)
    if not args.no_artefact:
        output_dir.mkdir(parents=True, exist_ok=True)
    evaluation_path = (output_dir if not args.no_artefact else data_dir / "models") / "classification_results.json"
    evaluation_path.parent.mkdir(parents=True, exist_ok=True)
    evaluation_path.write_text(json.dumps(evaluation, indent=2, default=str), encoding="utf-8")
    logger.info("Evaluation written to %s", evaluation_path)

    if not args.no_artefact:
        import joblib

        selected = baseline if best_name == "baseline_logistic_regression" else improved
        model_file = f"{args.model_id}.joblib"
        joblib.dump(selected, output_dir / model_file)
        manifest = {
            "model_id": args.model_id,
            "version": args.version or output_dir.name,
            "kind": "sklearn",
            "model_file": model_file,
            "task": "classification",
            "target_description": evaluation["target"],
            "label_mapping": label_mapping,
            "feature_schema": names,
            "feature_sources": {
                "radiomics": [name for name in names if name not in deep_names],
                "deep_features": deep_names,
                "table": str(args.features),
                "deep_table": str(args.deep_features) if args.deep_features else None,
            },
            "preprocessing": {
                "type": "standard_scaler",
                "mean": means,
                "scale": scales,
                "median": medians,
                "imputation": "training-split median",
                "fit_on": "training split only",
            },
            "coefficients": linear_coefficients(selected, n_classes),
            "feature_quantiles": quantiles(matrix, splits["train"]),
            "metrics": _flat_metrics(predictions, best_name),
            "model_comparison": {
                name: {"validation": payload["validation"], "test": payload["test"]}
                for name, payload in predictions.items()
            },
            "selected_model": best_name,
            "splits": {name: summary[name]["class_distribution"] for name in ("train", "validation", "test")},
            "leakage_check": {
                "method": method,
                "patient_overlap": summary["patient_overlap"],
                "passed": summary["leakage_free"],
            },
            "seed": args.seed,
            "trained_at": utc_now(),
            "code_revision": code_revision(),
            "environment": describe_environment(),
            "trainer": "scripts/train_classification.py",
            "caveats": evaluation["caveats"],
        }
        (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
        results_index = data_dir / "models" / "evaluation_results.json"
        merged = {}
        if results_index.exists():
            try:
                merged = json.loads(results_index.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                merged = {}
        merged.setdefault("state", "evaluated")
        merged["classification"] = {
            "state": "evaluated",
            "model_id": args.model_id,
            "version": manifest["version"],
            "target": evaluation["target"],
            "label_mapping": label_mapping,
            "comparison": evaluation["comparison"],
            "selected_model": best_name,
            "splits": evaluation["splits"],
            "feature_count": len(names),
            "baseline": "logistic regression on radiomics features",
            "improved": args.improved,
            "generated_at": evaluation["generated_at"],
            "produced_by": "scripts/train_classification.py",
            "artefact": str(output_dir),
        }
        results_index.parent.mkdir(parents=True, exist_ok=True)
        results_index.write_text(json.dumps(merged, indent=2, default=str), encoding="utf-8")
        logger.info("Artefact written to %s and indexed in %s", output_dir, results_index)
        logger.info(
            "The application serves this artefact only because it carries the feature schema, the "
            "scaler and the label mapping; keep the directory next to the API process."
        )
    return 0


def _flat_metrics(predictions: dict, best_name: str) -> dict:
    selected = predictions[best_name]
    test = selected.get("test") or {}
    validation = selected.get("validation") or {}
    payload = {
        "accuracy": test.get("accuracy"),
        "macro_f1": test.get("macro_f1"),
        "weighted_f1": test.get("weighted_f1"),
        "roc_auc": test.get("roc_auc"),
        "roc_auc_average": test.get("roc_auc_average"),
        "per_class": test.get("per_class") or {},
        "confusion_matrix": test.get("confusion_matrix") or [],
        "confusion_matrix_order": test.get("confusion_matrix_order") or [],
        "samples": test.get("samples"),
        "class_distribution": test.get("class_distribution") or {},
        "evaluated_model": best_name,
        "evaluation_split": "held-out test split",
        "validation_accuracy": validation.get("accuracy"),
        "validation_macro_f1": validation.get("macro_f1"),
        "validation_roc_auc": validation.get("roc_auc"),
    }
    if test.get("per_class"):
        precision = [item["precision"] for item in test["per_class"].values()]
        recall = [item["recall"] for item in test["per_class"].values()]
        payload["precision_macro"] = round(sum(precision) / len(precision), 6)
        payload["recall_macro"] = round(sum(recall) / len(recall), 6)
    for name, entry in predictions.items():
        test_entry = entry.get("test") or {}
        payload[f"{name}_test_accuracy"] = test_entry.get("accuracy")
        payload[f"{name}_test_macro_f1"] = test_entry.get("macro_f1")
    payload["baseline_test_metrics"] = (predictions.get("baseline_logistic_regression") or {}).get("test")
    improved_name = next((key for key in predictions if key.startswith("improved_")), None)
    payload["improved_test_metrics"] = (predictions.get(improved_name) or {}).get("test") if improved_name else None
    return payload


def _build_improved(args, n_classes: int):
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.neural_network import MLPClassifier

    if args.improved == "gradient_boosting":
        return GradientBoostingClassifier(random_state=args.seed)
    if args.improved == "random_forest":
        return RandomForestClassifier(n_estimators=400, random_state=args.seed, class_weight="balanced")
    hidden = tuple(int(item) for item in args.hidden_dims.split(",") if item.strip())
    return MLPClassifier(
        hidden_layer_sizes=hidden,
        max_iter=args.max_iter,
        random_state=args.seed,
        early_stopping=False,
        learning_rate_init=1e-3,
    )


def _score(model, scaled: np.ndarray, y: np.ndarray, splits: dict, n_classes: int, labels: list[str]) -> dict:
    payload: dict = {"validation": None, "test": None, "report": ""}
    for name in ("validation", "test"):
        indices = list(splits.get(name, []))
        if not indices:
            continue
        probabilities = _probabilities(model, scaled[indices], n_classes)
        payload[name] = evaluate(y[indices], probabilities, n_classes)
        if name == "test":
            payload["report"] = text_report(y[indices], probabilities, labels)
    return payload


def _probabilities(model, matrix: np.ndarray, n_classes: int) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(matrix), dtype=float)
    decision = np.asarray(model.decision_function(matrix), dtype=float)
    if decision.ndim == 1:
        stacked = np.stack([-decision, decision], axis=1)
    else:
        stacked = decision
    exponent = np.exp(stacked - stacked.max(axis=1, keepdims=True))
    return exponent / exponent.sum(axis=1, keepdims=True)


def _read_table(path: Path) -> tuple[list[dict], list[str]]:
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload.get("rows") if isinstance(payload, dict) else payload
        if not rows:
            raise SystemExit(f"{path} contains no rows")
        return [dict(row) for row in rows], list(rows[0].keys())
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [dict(row) for row in reader]
        columns = list(reader.fieldnames or [])
    return rows, columns


def _artefact_name(args, data_dir: Path) -> str:
    stamp = utc_now().replace(":", "").replace("-", "")[:15]
    return f"{args.model_id}-{stamp}"


if __name__ == "__main__":
    raise SystemExit(main())
