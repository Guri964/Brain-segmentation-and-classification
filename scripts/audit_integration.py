#!/usr/bin/env python
"""Audit every model artefact this application can serve, and prove what actually runs.

Run it on the machine that holds the weights::

    python scripts/audit_integration.py            # human-readable report
    python scripts/audit_integration.py --json     # the same evidence as JSON

For each artefact the script reports where it was found (through the central resolver in
``backend/app/config.py``), whether it loads, and whether a **real forward pass** succeeds with
the input contract the notebook trained it on. Where a checkpoint is absent the section says so
and names the file, the directory and the environment override - it never substitutes a
prediction, a metric or a placeholder number.

Two honest limits of this script, stated up front:

* a successful forward pass proves the artefact is loadable and contract-compatible, **not**
  that the model is accurate. Accuracy needs ground-truth data and is measured elsewhere
  (``scripts/train_*.py`` + ``/api/evaluation``); nothing here reproduces a paper number;
* the three metrics a checkpoint's own metadata records are printed *as recorded by the
  training run* and are labelled as such.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backend"))

FORECAST_ONLY = (
    "Execution check only: the tensor is synthetic, so this proves the checkpoint loads and "
    "accepts the documented input, not that its predictions are correct."
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def describe_file(path: Path | None) -> dict:
    if path is None or not Path(path).is_file():
        return {"present": False}
    file = Path(path)
    return {
        "present": True,
        "path": str(file),
        "bytes": file.stat().st_size,
        "sha256": sha256(file),
    }


def audit_torch_models(config) -> list[dict]:
    """Segmentation + radiomics checkpoints: load through the registry and execute."""

    import numpy as np

    from app.ml.registry import ModelRegistry

    registry = ModelRegistry(config)
    findings: list[dict] = []
    for model_id in ("segnet_brats_4ch", "unet_dynamic_512", "raunet_brats_4ch"):
        descriptor = registry.get_descriptor(model_id)
        spec = descriptor.input_spec
        entry = {
            "model_id": model_id,
            "display_name": descriptor.display_name,
            "task": descriptor.task,
            "architecture": descriptor.architecture,
            "provenance": descriptor.provenance,
            "input_spec": spec.to_dict(),
            "output_semantics": descriptor.output_semantics,
            "file": describe_file(Path(descriptor.path)),
        }
        if not entry["file"]["present"]:
            entry["state"] = "checkpoint_missing"
            entry["reason"] = (
                f"{Path(descriptor.path).name} was not found. Expected in "
                f"{Path(descriptor.path).parent}."
            )
            findings.append(entry)
            continue
        try:
            loaded = registry.load_model(model_id)
        except Exception as exc:  # noqa: BLE001 - the audit reports, it never raises
            entry["state"] = "load_failed"
            entry["reason"] = f"{type(exc).__name__}: {exc}"
            findings.append(entry)
            continue
        entry["state"] = "ready"
        entry["parameter_count"] = int(sum(p.numel() for p in loaded.module.parameters()))
        entry["state_dict_tensors"] = len(loaded.module.state_dict())
        import torch

        channels = spec.channels
        tensor = torch.zeros((1, channels, spec.height, spec.width), dtype=torch.float32)
        # A zero tensor would only prove the shapes line up, so drive it with a smooth pattern
        # resembling a brain-ish intensity gradient. This is still synthetic data.
        grid = np.linspace(-1.0, 1.0, spec.height, dtype=np.float32)[:, None]
        pattern = np.repeat(grid, spec.width, axis=1)
        for channel in range(channels):
            tensor[0, channel] = torch.from_numpy(pattern * (1.0 + 0.1 * channel))
        try:
            loaded.module.eval()
            with torch.no_grad():
                output = loaded.module(tensor)
            logits = output[0] if isinstance(output, (list, tuple)) else output
            semantics = getattr(loaded.module, "output_semantics", descriptor.output_semantics)
            if semantics == "logit":
                probabilities = torch.sigmoid(logits)
            elif semantics == "class_logits":
                probabilities = torch.softmax(logits, dim=1)
            else:
                probabilities = logits
            entry["forward_pass"] = {
                "output_semantics": semantics,
                "input_shape": list(tensor.shape),
                "raw_output_shape": list(np.asarray(output).shape),
                "output_min": float(np.asarray(output).min()),
                "output_max": float(np.asarray(output).max()),
                "probability_min": float(np.asarray(probabilities).min()),
                "probability_max": float(np.asarray(probabilities).max()),
                "note": FORECAST_ONLY,
            }
        except Exception as exc:  # noqa: BLE001
            entry["forward_pass"] = {"error": f"{type(exc).__name__}: {exc}"}
        findings.append(entry)
    return findings


def audit_radiomics_classifier(config) -> dict:
    """``radiomics_cnn_model.pth``: is the feature table it was trained on available?"""

    entry: dict = {
        "model_id": "radiomics_mlp_v1",
        "display_name": "Radiomics feature classifier",
        "file": describe_file(Path(config.MODEL_PATHS["radiomics_mlp_v1"])),
        "trainer": "Copy_of_Ensemble_Framework_for_Brain_Abnormality.ipynb",
    }
    if not entry["file"]["present"]:
        entry["state"] = "checkpoint_missing"
        entry["reason"] = "radiomics_cnn_model.pth was not found in the searched locations."
        return entry
    try:
        import torch

        state = torch.load(
            config.MODEL_PATHS["radiomics_mlp_v1"], map_location="cpu", weights_only=False
        )
        if not isinstance(state, dict):
            state = getattr(state, "state_dict", lambda: {})()
        entry["layers"] = {
            key: list(value.shape)
            for key, value in state.items()
            if hasattr(value, "shape") and len(getattr(value, "shape", ())) == 2
        }
        expected = None
        for key, value in state.items():
            if key.endswith("weight") and getattr(value, "ndim", 0) == 2:
                expected = int(value.shape[1])
                break
        entry["expected_feature_count"] = expected
    except Exception as exc:  # noqa: BLE001
        entry["state"] = "load_failed"
        entry["reason"] = f"{type(exc).__name__}: {exc}"
        return entry

    table = REPO_ROOT / "features_df.csv"
    entry["available_feature_table"] = describe_file(table)
    if table.is_file():
        header = table.read_text(encoding="utf-8").splitlines()[0]
        columns = [name.strip() for name in header.split(",") if name.strip()]
        entry["available_feature_count"] = len(columns)
        entry["available_feature_names"] = columns
        if expected is not None and len(columns) != expected:
            entry["state"] = "present_but_unusable"
            entry["reason"] = (
                f"The checkpoint's first layer expects {expected} features but features_df.csv "
                f"has {len(columns)} columns, and the checkpoint ships no scaler, no column-order "
                "record and no label mapping. Serving it would require reordering/padding "
                "features that were never part of its training input, which would produce "
                "meaningless output - so the model stays disabled."
            )
            return entry
    entry["state"] = "ready_for_offline_use"
    entry["reason"] = (
        "The feature table matches the checkpoint width. Serving still requires the notebook's "
        "column order and label mapping to be recorded; see models/README.md."
    )
    return entry


def audit_keras_classifier(config) -> dict:
    from app.services.image_classification_service import ImageClassificationService

    service = ImageClassificationService(config)
    status = service.status(force=True)
    model = status["models"][0]
    return {
        "model_id": model["model_id"],
        "display_name": model["display_name"],
        "state": model["state"],
        "reason": model["reason"],
        "file": describe_file(Path(model["model_path"])),
        "metadata": describe_file(Path(model["metadata_file"])),
        "framework": "TensorFlow/Keras",
        "framework_available": model["framework_available"],
        "framework_version": model["framework_version"],
        "architecture": model.get("architecture"),
        "input_size": model.get("input_size"),
        "classes": model.get("classes"),
        "preprocessing": model.get("preprocessing"),
        "reported_metrics": model.get("reported_metrics"),
        "metrics_provenance": model.get("metrics_provenance"),
        "install_note": status["install_note"],
    }


def audit_ensemble(config) -> dict:
    from app.ml.registry import ModelRegistry
    from app.services.ensemble_service import EnsembleService
    from app.services.segmentation_service import SegmentationService

    registry = ModelRegistry(config)
    ensemble = EnsembleService(registry, config, SegmentationService(registry, config))
    compatibility = ensemble.compatibility()
    return {
        "available": compatibility.get("available"),
        "groups": compatibility.get("groups", []),
        "reasons": compatibility.get("reasons", []),
        "ensemblable": [
            group["members"] for group in compatibility.get("groups", []) if group.get("ensemblable")
        ],
        "rule": (
            "Only models whose declared input contract is identical (channels, height, width and "
            "normalisation) are averaged. Predictions from different contracts are never fused."
        ),
        "note": compatibility.get("note"),
    }


def audit_radiomics(config) -> dict:
    from app.ml import radiomics_native
    from app.services.radiomics_service import RadiomicsService

    import numpy as np

    status = RadiomicsService(config).backend_status()
    # Counts come from an actual extraction on a synthetic region, not from a constant.
    image = np.zeros((48, 48), dtype=np.float32)
    image[8:40, 10:38] = np.linspace(0.0, 100.0, 32, dtype=np.float32)[:, None]
    mask = np.zeros((48, 48), dtype=bool)
    mask[14:34, 16:32] = True
    features = radiomics_native.extract_features(image, mask)
    categories: dict[str, int] = {}
    for name in features:
        category = radiomics_native.categorise(name)
        categories[category] = categories.get(category, 0) + 1
    return {
        "backend": status.get("selected_backend"),
        "pyradiomics_available": status.get("pyradiomics_available"),
        "preference": status.get("preference"),
        "feature_count": len(features),
        "categories": categories,
        "note": status.get("note"),
    }


def audit_stacked_ensemble(config) -> dict:
    """The notebook's stacked ensemble: artefacts, contracts and one real end-to-end run.

    The pipeline is executed on a synthetic four-channel slice, so a successful call proves that
    all five head artefacts load, that the 46-feature schema matches the scaler and the CNN, that
    the ten meta features can be built, and that the head returns a label. It proves nothing about
    accuracy: the notebook's split, masks and labels are not shipped with the weights.
    """

    import numpy as np

    from app.ml.registry import ModelRegistry
    from app.services.ensemble_bundle_service import EnsembleBundleService
    from app.services.segmentation_service import SegmentationService

    registry = ModelRegistry(config)
    bundle = EnsembleBundleService(
        config, registry, SegmentationService(registry, config)
    )
    status = bundle.status(force=True)
    entry: dict = {
        "model_id": "stacked_ensemble_brats",
        "display_name": "Stacked ensemble (FINAL ENSEMBLE NB.ipynb)",
        "state": status["state"],
        "available": status["available"],
        "reason": status["reason"],
        "task_description": status["task_description"],
        "input_requirements": status["input_requirements"],
        "meta_features": status["meta_features"],
        "target_semantics": status["target_semantics"],
        "leakage_note": status["leakage_note"],
        "reported_metrics": status["reported_metrics"],
        "metrics_provenance": status["metrics_provenance"],
        "install_note": status["install_note"],
        "artifacts": [
            {
                "key": artifact["key"],
                "file": artifact["file"],
                "role": artifact["role"],
                "present": artifact["present"],
                "path": artifact["path"],
                "bytes": artifact["size_bytes"],
                "sha256": (
                    sha256(Path(artifact["path"]))[:16] if artifact["present"] else None
                ),
            }
            for artifact in status["artifacts"]
        ],
        "members": [
            {
                "model_id": member["model_id"],
                "status": member["status"],
                "available": member["available"],
                "architecture": member["architecture"],
                "parameter_count": member["parameter_count"],
                "path": member["path"],
                "sha256": (
                    sha256(Path(member["path"]))[:16]
                    if Path(member["path"]).is_file()
                    else None
                ),
            }
            for member in status["members"]
        ],
    }
    if not status["available"]:
        entry["reason_for_no_run"] = (
            "The pipeline is not executed while an artefact or a member is missing: a partial "
            "run would have to impute a meta-feature the head was not trained on."
        )
        return entry
    entry["contracts"] = status["contracts"]
    # A smooth phantom with a bright blob, so the members' consensus mask is not empty. This is
    # synthetic input: the numbers it produces are execution evidence, never a performance claim.
    size = 240
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    blob = np.exp(-(((xx - 120.0) ** 2 + (yy - 130.0) ** 2) / (2 * 22.0**2)))
    tensor = np.zeros((1, 4, size, size), dtype=np.float32)
    for channel in range(4):
        ramp = (xx + yy) / (2.0 * size)
        tensor[0, channel] = (0.4 * ramp + 0.6 * blob) * (1.0 + 0.05 * channel)
        channel_values = tensor[0, channel]
        tensor[0, channel] = (channel_values - channel_values.mean()) / (
            channel_values.std() + 1e-6
        )
    try:
        prediction = bundle.predict(tensor, threshold=0.5)
        entry["pipeline_run"] = {
            "input_shape": list(tensor.shape),
            "note": FORECAST_ONLY,
            "members": [
                {
                    "model_id": member["model_id"],
                    "max_probability": member["max_probability"],
                    "mean_probability": member["mean_probability"],
                    "dice_vs_consensus": member["dice_vs_consensus"],
                    "reused_from_analysis": member["reused_from_analysis"],
                }
                for member in prediction["member_statistics"]
            ],
            "consensus": prediction["consensus"],
            "radiomics_features": prediction["radiomics"]["feature_count"],
            "radiomics_score": prediction["radiomics"]["score"],
            "meta_features": prediction["meta_features"],
            "meta_classes": prediction["meta_classes"],
            "prediction": prediction["prediction"],
            "class_labels": prediction["class_labels"],
            "class_probabilities": prediction["class_probabilities"],
            "dice_reference": prediction["dice_reference"],
            "feature_mask_source": prediction["feature_mask_source"],
            "artifact_digests": prediction["artifact_digests"],
            "inference_seconds": prediction["inference_seconds"],
        }
    except Exception as exc:  # noqa: BLE001 - the audit reports, it never raises
        entry["pipeline_run"] = {"error": f"{type(exc).__name__}: {exc}"}
    return entry


def audit_frontend() -> dict:
    source = REPO_ROOT / "frontend" / "src"
    dist = REPO_ROOT / "frontend" / "dist"
    pages = sorted(path.name for path in (source / "pages").glob("*.tsx"))
    images = sorted(path.name for path in (source / "components").glob("*.tsx"))
    return {
        "pages": pages,
        "components": images,
        "built_bundle": bool(dist.exists() and (dist / "index.html").is_file()),
        "framework": "React + TypeScript + Tailwind (Vite)",
    }


def collect(config) -> dict:
    from app.config import weights_directory

    return {
        "repository": str(REPO_ROOT),
        "weights_directory": str(weights_directory()),
        "search_order": [
            "models/weights/ (or $BRAIN_WEIGHTS_DIR)",
            "repository root",
            "data/models/",
        ],
        "torch_models": audit_torch_models(config),
        "keras_image_classifier": audit_keras_classifier(config),
        "radiomics_classifier": audit_radiomics_classifier(config),
        "ensemble": audit_ensemble(config),
        "stacked_ensemble": audit_stacked_ensemble(config),
        "radiomics": audit_radiomics(config),
        "frontend": audit_frontend(),
    }


def _wrap(text: str, prefix: str, width: int = 86) -> list[str]:
    """Wrap a long sentence under a label so the plain-text report stays readable."""

    words = str(text).split()
    lines: list[str] = []
    current = prefix
    for word in words:
        if len(current) + len(word) + 1 > width and current.strip() != prefix.strip():
            lines.append(current.rstrip())
            current = " " * len(prefix)
        current += word + " "
    lines.append(current.rstrip())
    return lines


def render(report: dict) -> str:
    lines: list[str] = []
    add = lines.append
    add("=" * 90)
    add("MODEL AND PIPELINE AUDIT - what this deployment can actually execute")
    add("=" * 90)
    add(f"repository       : {report['repository']}")
    add(f"weights directory: {report['weights_directory']}")
    add("search order     : " + " -> ".join(report["search_order"]))
    add("")

    add("-" * 90)
    add("SEGMENTATION CHECKPOINTS")
    add("-" * 90)
    for entry in report["torch_models"]:
        add(f"{entry['model_id']:20s} {entry['state']}")
        add(f"  architecture : {entry['architecture']}")
        add(f"  source       : {entry['provenance']}")
        spec = entry["input_spec"]
        add(
            "  input        : "
            f"{spec['channels']} x {spec['height']} x {spec['width']} "
            f"({spec['normalisation']})"
        )
        add(f"  output       : {entry['output_semantics']}")
        file = entry["file"]
        if file.get("present"):
            add(f"  file         : {file['path']} ({file['bytes']:,} bytes)")
            add(f"  sha256       : {file['sha256'][:16]}...")
        if entry.get("parameter_count"):
            add(
                f"  parameters   : {entry['parameter_count']:,} "
                f"({entry['state_dict_tensors']} tensors)"
            )
        if entry.get("forward_pass"):
            forward = entry["forward_pass"]
            if "error" in forward:
                add(f"  forward pass : FAILED - {forward['error']}")
            else:
                add(
                    "  forward pass : OK  "
                    f"out {forward['raw_output_shape']} "
                    f"range [{forward['output_min']:.4f}, {forward['output_max']:.4f}] "
                    f"-> probability [{forward['probability_min']:.4f}, "
                    f"{forward['probability_max']:.4f}]"
                )
        if entry.get("reason"):
            add(f"  blocked by   : {entry['reason']}")
        add("")

    keras = report["keras_image_classifier"]
    add("-" * 90)
    add("IMAGE CLASSIFIER (four-class tumour type, firstnb.ipynb)")
    add("-" * 90)
    add(f"  model_id     : {keras['model_id']}")
    add(f"  state        : {keras['state']}")
    add(f"  reason       : {keras['reason']}")
    add(f"  file         : {keras['file'].get('path', keras['install_note'])}")
    if keras["file"].get("present"):
        add(f"  bytes        : {keras['file']['bytes']:,}")
    add(f"  metadata     : {keras['metadata'].get('path', 'metadata.json not found')}")
    add(
        f"  framework    : {keras['framework']} available="
        f"{keras['framework_available']} version={keras['framework_version']}"
    )
    if keras.get("classes"):
        add(f"  classes      : {', '.join(keras['classes'])}")
    if keras.get("input_size"):
        add(f"  input        : {keras['input_size']}")
    if keras.get("reported_metrics"):
        add(f"  metrics      : {keras['reported_metrics']} (as recorded by the training run)")
        add(f"                 {keras['metrics_provenance']}")
    add("")

    radiomics_classifier = report["radiomics_classifier"]
    add("-" * 90)
    add("RADIOMICS FEATURE CLASSIFIER (radiomics_cnn_model.pth)")
    add("-" * 90)
    add(f"  state        : {radiomics_classifier['state']}")
    add(f"  reason       : {radiomics_classifier.get('reason', '')}")
    if radiomics_classifier.get("expected_feature_count"):
        add(f"  expects      : {radiomics_classifier['expected_feature_count']} features")
    if radiomics_classifier.get("available_feature_count"):
        add(f"  table offers : {radiomics_classifier['available_feature_count']} features")
    add("")

    ensemble = report["ensemble"]
    add("-" * 90)
    add("ENSEMBLE")
    add("-" * 90)
    for group in ensemble["groups"]:
        members = ", ".join(group["members"])
        add(
            f"  {group['channels']}x{group['height']}x{group['width']}: {members} "
            f"-> ensemblable={group.get('ensemblable')} "
            f"({len(group['members'])} available member(s))"
        )
    add(f"  rule         : {ensemble['rule']}")
    for reason in ensemble.get("reasons") or []:
        add(f"  why not      : {reason}")
    add("")

    stacked = report["stacked_ensemble"]
    add("-" * 90)
    add("STACKED ENSEMBLE (FINAL ENSEMBLE NB.ipynb)")
    add("-" * 90)
    add(f"  state        : {stacked['state']} (available={stacked['available']})")
    add(f"  reason       : {stacked['reason']}")
    requirements = stacked.get("input_requirements") or {}
    if requirements:
        add(
            "  input        : "
            f"{requirements['channels']} x {requirements['spatial_size'][0]} x "
            f"{requirements['spatial_size'][1]} ({requirements['normalisation']})"
        )
        add(f"  modalities   : {', '.join(requirements['modality_order'])}")
    add("  artefacts    :")
    for artifact in stacked["artifacts"]:
        marker = "present" if artifact["present"] else "MISSING"
        size = f"{artifact['bytes']:,} bytes" if artifact["bytes"] else "-"
        add(f"    [{marker:7s}] {artifact['file']:32s} {size:>16s}  {artifact['role']}")
        if artifact["present"]:
            add(f"               sha256 {artifact['sha256']}...  {artifact['path']}")
    add("  members      :")
    for member in stacked["members"]:
        parameters = (
            f"{member['parameter_count']:,} params" if member["parameter_count"] else "-"
        )
        add(
            f"    {member['model_id']:22s} {member['status']:16s} {parameters:>18s}  "
            f"{member['architecture']}"
        )
    if stacked.get("contracts"):
        contracts = stacked["contracts"]
        add(
            f"  contracts    : radiomics features={contracts.get('radiomics_features')} "
            f"meta features={contracts.get('meta_features')} "
            f"classes={contracts.get('meta_classes')}"
        )
    add(f"  meta columns : {', '.join(stacked['meta_features'])}")
    run = stacked.get("pipeline_run")
    if run and "error" in run:
        add(f"  pipeline run : FAILED - {run['error']}")
    elif run:
        add(
            "  pipeline run : OK on a synthetic 4-channel slice "
            f"input={run['input_shape']} consensus={run['consensus']['foreground_pixels']} px"
        )
        for member in run["members"]:
            add(
                f"                 {member['model_id']:22s} dice_vs_consensus="
                f"{member['dice_vs_consensus']:.4f} max_p={member['max_probability']:.4f} "
                f"mean_p={member['mean_probability']:.4f}"
            )
        add(
            f"                 radiomics {run['radiomics_features']} features -> score "
            f"{run['radiomics_score']:.6f}"
        )
        add(
            f"                 head prediction={run['prediction']} "
            f"labels={run['class_labels']} probabilities={run['class_probabilities']}"
        )
        add(f"                 {run['note']}")
    elif stacked.get("reason_for_no_run"):
        add(f"  pipeline run : not attempted - {stacked['reason_for_no_run']}")
    for line in _wrap(stacked["target_semantics"], "  target       : "):
        add(line)
    for line in _wrap(stacked["leakage_note"], "  leakage      : "):
        add(line)
    add(f"  metrics      : {stacked['reported_metrics']}")
    for line in _wrap(stacked["metrics_provenance"], "                 "):
        add(line)
    for line in _wrap(stacked["install_note"], "  install      : "):
        add(line)
    add("")

    radiomics = report["radiomics"]
    add("-" * 90)
    add("RADIOMICS EXTRACTION")
    add("-" * 90)
    add(f"  backend      : {radiomics['backend']}")
    add(f"  pyradiomics  : {radiomics['pyradiomics_available']}")
    breakdown = ", ".join(f"{name} {count}" for name, count in sorted(radiomics["categories"].items()))
    add(f"  features     : {radiomics['feature_count']} ({breakdown})")
    add("")

    frontend = report["frontend"]
    add("-" * 90)
    add("INTERFACE")
    add("-" * 90)
    add(f"  stack        : {frontend['framework']}")
    add(f"  built bundle : {frontend['built_bundle']}")
    add(f"  pages        : {', '.join(frontend['pages'])}")
    add("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="print the audit as JSON")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit non-zero when one of the documented artefacts is missing",
    )
    arguments = parser.parse_args()

    os.environ.setdefault("BRAIN_ENV", "testing")
    from app.config import get_config  # noqa: PLC0415 - after sys.path setup

    config = get_config()
    report = collect(config)

    if arguments.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(render(report))

    if arguments.strict:
        missing = []
        for entry in report["torch_models"]:
            if entry["state"] == "checkpoint_missing":
                missing.append(entry["model_id"])
        if report["keras_image_classifier"]["state"] == "checkpoint_missing":
            missing.append(report["keras_image_classifier"]["model_id"])
        # The stacked ensemble is part of the documented deliverable, so --strict counts its
        # artefacts too: an operator who copied seven of the eight files gets a non-zero exit.
        for artifact in report["stacked_ensemble"]["artifacts"]:
            if not artifact["present"]:
                missing.append(artifact["file"])
        if missing:
            print("missing artefacts: " + ", ".join(missing), file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
