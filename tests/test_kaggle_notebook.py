"""Tests for notebooks/brain_tumor_training_kaggle_t4.ipynb.

Three things are verified here, all without a GPU and without BraTS data:

1. **Structure and freshness.** Every required section is present, every code cell is valid
   Python, and the project sources embedded in the notebook are byte-identical to
   `backend/app/**` (so a checkpoint trained in the notebook uses exactly the architectures,
   loss, metrics and radiomics extractor the API uses).

2. **Executability.** The notebook's own cells are executed in order against both synthetic
   fixtures (`scripts/make_brats_fixture.py` for segmentation, `scripts/make_classification_fixture.py`
   for the multiclass task) in smoke mode: preprocessing, splits, trainer, radiomics feature
   tables, classifiers, ensemble calibration, held-out evaluation, XAI, artefact export and the
   compatibility check all run for real. A notebook that only looks right would fail these tests.

3. **Servability.** The exported classification directory is opened with the real
   `ClassificationService`, and the exported checkpoints with the real `ModelRegistry`, to
   prove the artefacts the notebook writes are the artefacts the Flask application loads.

Every number produced here comes from random fixture arrays. None of it is a result and none
of it is asserted as one -- only shapes, keys, states and orderings are checked.
"""
from __future__ import annotations

import ast
import hashlib
import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = REPO_ROOT / "notebooks" / "brain_tumor_training_kaggle_t4.ipynb"
BUILDER = REPO_ROOT / "notebooks" / "build_kaggle_notebook.py"

REQUIRED_SECTIONS = [
    "GPU verification",
    "dataset discovery",
    "Structure inspection",
    "label verification",
    "Exploratory data analysis",
    "MRI visualisation",
    "mask visualisation",
    "preprocessing",
    "SPLITS",
    "baselines",
    "Trainer",
    "SEGNET TRAINING",
    "U-NET TRAINING",
    "RAUNET TRAINING",
    "CLASSIFICATION FEATURES",
    "CLASSIFICATION (patient-level split",
    "CLASSIFICATION DATASET",
    "Image-based classifier",
    "Radiomics quality",
    "MULTICLASS DATASET",
    "MULTICLASS CNN",
    "RADIOMICS CLASSIFIERS",
    "STACKED ENSEMBLE",
    "MULTICLASS EVALUATION",
    "ENSEMBLE",
    "MODEL SELECTION",
    "HELD-OUT TEST RESULTS",
    "Grad-CAM written",
    "comparison charts",
    "artefact index",
    "EXPORT",
]

EMBEDDED_FILES = [
    "backend/app/ml/architectures/segnet.py",
    "backend/app/ml/architectures/dynamic_unet.py",
    "backend/app/ml/architectures/attention_unet.py",
    "backend/app/ml/losses.py",
    "backend/app/ml/gradcam.py",
    "backend/app/ml/radiomics_native.py",
    "backend/app/services/evaluation_service.py",
    "backend/app/services/radiomics_service.py",
    "backend/app/utils/metrics.py",
    "backend/app/utils/images.py",
    "backend/app/utils/io.py",
]


def load_notebook() -> dict:
    if not NOTEBOOK.exists():
        pytest.skip(f"{NOTEBOOK.name} has not been generated yet")
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def code_cells(notebook: dict) -> list[str]:
    cells = []
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = cell["source"]
        cells.append(source if isinstance(source, str) else "".join(source))
    return cells


def test_notebook_covers_every_required_section():
    notebook = load_notebook()
    text = "\n".join(
        "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
        for cell in notebook["cells"]
    )
    missing = [section for section in REQUIRED_SECTIONS if section.lower() not in text.lower()]
    assert not missing, f"the notebook does not cover: {missing}"


def test_every_code_cell_is_valid_python():
    for index, source in enumerate(code_cells(load_notebook())):
        try:
            ast.parse(source)
        except SyntaxError as exc:  # pragma: no cover - a syntax error is a hard failure
            raise AssertionError(f"notebook code cell {index} does not parse: {exc}") from exc


def test_embedded_project_sources_match_the_repository():
    notebook = load_notebook()
    recorded = notebook.get("_embedded_hashes")
    assert recorded, "the notebook does not record the hashes of the embedded sources"
    for relative in EMBEDDED_FILES:
        path = REPO_ROOT / relative
        assert path.exists(), f"{relative} is referenced by the notebook but missing"
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert recorded.get(relative) == digest, (
            f"{relative} changed since the notebook was generated; run "
            "python notebooks/build_kaggle_notebook.py"
        )


def test_notebook_is_not_stale():
    """build_kaggle_notebook.py --check must pass, i.e. the committed file matches its sources."""
    import subprocess

    result = subprocess.run(
        [sys.executable, str(BUILDER), "--check"],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_notebook_never_promises_a_result_before_training():
    """Guard against the notebook quietly reporting fixture numbers as findings."""
    notebook = load_notebook()
    text = "\n".join(
        "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
        for cell in notebook["cells"]
    )
    for forbidden in ("placeholder metric", "dummy metric", "expected_dice =", "assume dice"):
        assert forbidden not in text
    assert "synthetic" in text.lower(), "the notebook must state that the dataset is real, not synthetic"


@pytest.fixture(scope="module")
def notebook_namespace(tmp_path_factory):
    """Execute the notebook's cells in order against the synthetic BraTS fixture (CPU, smoke mode)."""
    pytest.importorskip("torch")
    pytest.importorskip("h5py")
    pytest.importorskip("sklearn")
    pytest.importorskip("matplotlib")
    pytest.importorskip("joblib")

    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from make_brats_fixture import main as fixture_main
    from make_classification_fixture import main as classification_fixture_main

    workdir = tmp_path_factory.mktemp("kaggle-run")
    fixture_root = workdir / "brats_fixture"
    assert (
        fixture_main(
            [
                "--out", str(fixture_root),
                "--volumes", "8",
                "--slices-per-volume", "3",
                "--empty-every", "4",
            ]
        )
        == 0
    )

    # Second dataset: the image-classification fixture stands in for the Brain Tumor MRI Dataset,
    # with the same directory shape (Training/<class>, Testing/<class>) and a marker that says it
    # is synthetic, so the notebook has to record `synthetic: true` next to every artefact.
    classification_fixture = workdir / "classification_fixture"
    assert (
        classification_fixture_main(
            [
                "--out", str(classification_fixture),
                "--train-per-class", "6",
                "--test-per-class", "4",
                "--size", "72",
            ]
        )
        == 0
    )

    namespace: dict = {"__name__": "kaggle_notebook", "__file__": str(NOTEBOOK)}
    executed: list[int] = []
    previous_cwd = Path.cwd()
    import os

    os.chdir(workdir)
    try:
        for index, source in enumerate(code_cells(load_notebook())):
            buffer = io.StringIO()
            try:
                with redirect_stdout(buffer):
                    exec(compile(source, f"<notebook cell {index}>", "exec"), namespace)
            except SystemExit as exc:
                raise AssertionError(
                    f"notebook cell {index} called SystemExit({exc.code!r}).\n{buffer.getvalue()[-2000:]}"
                ) from exc
            except Exception as exc:  # noqa: BLE001 - surface the failing cell
                tail = buffer.getvalue()[-2000:]
                raise AssertionError(
                    f"notebook cell {index} failed: {type(exc).__name__}: {exc}\n--- cell output ---\n{tail}"
                ) from exc
            executed.append(index)
            if "WORK = Path(" in source:
                # the notebook created its work directories: point it at the fixture instead of
                # scanning /kaggle/input, and make the run small so the test stays fast.
                namespace["RUN"]["dataset_roots"] = [str(fixture_root)]
                namespace["RUN"]["classification_dataset_roots"] = [str(classification_fixture)]
                namespace["RUN"]["smoke_test"] = True
                namespace["apply_smoke_overrides"]()
            if "TRAIN_PATIENTS, VAL_PATIENTS, TEST_PATIENTS = TRAIN_PATIENTS" in source:
                # smoke mode trimmed the splits; the fixture has patients in every class, so use
                # the full split lists to exercise every section.
                namespace["TRAIN_PATIENTS"] = list(namespace["SPLITS"]["train"])
                namespace["VAL_PATIENTS"] = list(namespace["SPLITS"]["validation"])
                namespace["TEST_PATIENTS"] = list(namespace["SPLITS"]["test"])
    finally:
        os.chdir(previous_cwd)

    assert len(executed) == len(code_cells(load_notebook()))
    return namespace


def _import_real_backend(namespace) -> None:
    """Make `import app...` resolve to the repository package, not the notebook's embedded copy.

    Executing the notebook puts `<work>/_embedded/backend` on sys.path and imports `app` from
    there. The integration tests need the real backend, so the embedded path is dropped and any
    already-imported `app` module that came from it is removed before importing again.
    """
    work = str(Path(namespace["PATHS"]["work"]))
    sys.path[:] = [entry for entry in sys.path if not entry.startswith(work)]
    for name in [name for name in sys.modules if name == "app" or name.startswith("app.")]:
        module = sys.modules[name]
        if work in str(getattr(module, "__file__", "") or ""):
            del sys.modules[name]
    backend_root = REPO_ROOT / "backend"
    if str(backend_root) not in sys.path:
        sys.path.insert(0, str(backend_root))


def test_notebook_runs_end_to_end(notebook_namespace):
    namespace = notebook_namespace
    workdir = Path(namespace["PATHS"]["work"])

    # --- preprocessing and splits -----------------------------------------
    assert namespace["SPLIT_SUMMARY"]["leakage_free"] is True
    assert namespace["SPLIT_SUMMARY"]["patients_assigned"] == namespace["SPLIT_SUMMARY"]["patients_total"]
    assert namespace["SPLIT_SUMMARY"]["patient_overlap"] == []
    cached = sorted((workdir / "cache").glob("*.npz"))
    assert len(cached) == namespace["SPLIT_SUMMARY"]["patients_total"]

    # --- training produced resumable checkpoints ---------------------------
    for model_id in ("segnet_brats_4ch", "raunet_brats_4ch", "unet_dynamic_512_real_brats"):
        directory = workdir / "checkpoints" / model_id
        assert (directory / "last.pt").exists(), f"{model_id} has no resumable checkpoint"
        assert (directory / "best.pt").exists(), f"{model_id} has no best checkpoint"
        payload = namespace["torch"].load(directory / "last.pt", map_location="cpu", weights_only=False)
        for key in ("state_dict", "optimizer", "scheduler", "epoch", "best_dice", "numpy_rng", "torch_rng"):
            assert key in payload, f"{model_id} checkpoint is missing '{key}' and could not be resumed"
        history = json.loads((directory / "history.json").read_text(encoding="utf-8"))
        assert history and history[0]["train_loss"] is not None

    # --- the radiomics table carries the schema the API computes ------------
    header = (workdir / "artifacts" / "radiomics_features.csv").read_text(encoding="utf-8").splitlines()[0].split(",")
    assert {"case_id", "patient_id", "label", "split", "mask_source", "image_channel"}.issubset(header)
    # analysis_service extracts from the first display plane (T1), so the names are bare, not
    # channel-prefixed: a "ch3__..." schema could never be fed by the API.
    assert not any(name.startswith("ch") and "__" in name for name in header), header[:10]
    assert any(name.startswith("firstorder_") for name in header)
    assert any(name.startswith("glcm_") for name in header)
    assert "shape_units" not in header, "string diagnostics must not enter the feature schema"

    # --- the table is consumable by scripts/train_classification.py ---------
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    import _classification

    with (workdir / "artifacts" / "radiomics_features.csv").open(newline="", encoding="utf-8") as handle:
        import csv

        rows = list(csv.DictReader(handle))
    columns = list(rows[0].keys())
    names = _classification.feature_names(rows, columns, None)
    assert names == namespace["FEATURE_NAMES"], "the notebook schema and the CLI schema disagree"

    # --- classification: both candidates on the same split, metrics recorded -
    evaluation = json.loads((workdir / "artifacts" / "classification_evaluation.json").read_text(encoding="utf-8"))
    assert evaluation["selection_split"] == "validation"
    assert evaluation["label_mapping"] == {"0": "HGG", "1": "LGG"}
    assert evaluation["comparison"], "no classifier was evaluated"
    assert evaluation["selected_model"] in evaluation["comparison"]
    for key, entry in evaluation["comparison"].items():
        assert entry["feature_source"] in {"reference", "predicted"}
        assert key.endswith(entry["model"])
        for split in ("validation", "test"):
            metrics = entry[split]
            assert 0.0 <= metrics["accuracy"] <= 1.0
            assert metrics["confusion_matrix"]
            assert metrics["per_class"]
            assert metrics["samples"] > 0
    assert evaluation["feature_sources"]["reference"]["schema"] == namespace["FEATURE_NAMES"]

    # --- ensemble weights come from validation data only -------------------
    ensemble_path = workdir / "artifacts" / "ensemble_calibration.json"
    ensemble = json.loads(ensemble_path.read_text(encoding="utf-8"))
    assert ensemble["split"] == "validation"
    assert ensemble["test_split_used"] is False
    assert abs(sum(ensemble["weights"].values()) - 1.0) < 1e-6

    # --- held-out evaluation in the application's own format ---------------
    results = json.loads((workdir / "artifacts" / "evaluation_results.json").read_text(encoding="utf-8"))
    assert results["state"] == "evaluated"
    assert results["split"] == "test"
    assert results["protocol"]["test_split_used_for_tuning"] is False
    assert results["protocol"]["selection_split"] == "validation"
    models = results["segmentation"]["models"]
    assert "segnet_brats_4ch" in models
    for model_id, entry in models.items():
        aggregate = entry["aggregate"]
        for key in ("dice", "iou", "precision", "recall"):
            value = aggregate.get(key)
            assert value and value.get("mean") is not None, f"{model_id} has no {key}"
            assert 0.0 <= value["mean"] <= 1.0
            assert value["ci_lower"] is not None and value["ci_upper"] is not None
    assert results["classification"]["selected_model"] == evaluation["selected_model"]
    assert results["baselines"]["all_foreground_dice"] >= 0.0
    assert results["runtime"]["models"], "no run time was measured"

    # --- explainability and figures ---------------------------------------
    xai = json.loads((workdir / "artifacts" / "explainability.json").read_text(encoding="utf-8"))
    assert xai["state"] == "generated"
    assert xai["target_layer"] == "encoder.4"
    assert (workdir / "reports" / "05_gradcam.png").exists()
    assert (workdir / "reports" / "04_training_curves.png").exists()
    assert (workdir / "reports" / "06_model_comparison.png").exists()

    # --- export: checkpoints load with the application's builders ----------
    compatibility = {entry["model_id"]: entry for entry in namespace["COMPATIBILITY"]}
    for model_id, expected_shape in (
        ("segnet_brats_4ch", [1, 4, 256, 256]),
        ("raunet_brats_4ch", [1, 4, 256, 256]),
        ("unet_dynamic_512_real_brats", [1, 1, 512, 512]),
    ):
        entry = compatibility[model_id]
        assert entry["state"] == "compatible", entry
        assert entry["input_shape"] == expected_shape
        assert entry["output_shape"] == [1, 1, expected_shape[-1], expected_shape[-1]]
        assert entry["strict_load"] is True
    classifier_entry = compatibility["radiomics_grade_classifier"]
    assert classifier_entry["probe"]["state"] == "compatible"
    assert classifier_entry["probe"]["input_features"] == len(namespace["FEATURE_NAMES"])
    assert abs(classifier_entry["probe"]["probability_sum"] - 1.0) < 1e-6

    export_root = workdir / "brain_mri_artifacts"
    for relative in (
        "checkpoints/segnet_brats_4ch_real_brats.pt",
        "checkpoints/raunet_brats_4ch.pt",
        "checkpoints/unet_dynamic_512_real_brats.pt",
        "data/models/evaluation_results.json",
        "data/models/model_selection.json",
        "data/models/artifacts_manifest.json",
        "README.md",
    ):
        assert (export_root / relative).exists(), f"the export is missing {relative}"
    assert list((export_root / "data" / "models" / "classification").glob("*/manifest.json"))

    # --- the shipped artefacts are never overwritten ------------------------
    provenance = json.loads((export_root / "data" / "models" / "artifacts_manifest.json").read_text(encoding="utf-8"))
    assert provenance["produced_by"].endswith(".ipynb")
    # This run is on the synthetic fixture, and the export has to say exactly that: the flag is
    # derived from the HDF5 marker the fixture writes, so a notebook run on made-up images can
    # never be labelled as trained on real BraTS data.
    assert provenance["dataset"]["synthetic"] is True
    assert "ynthetic" in provenance["dataset"]["name"]
    assert "not established" not in provenance["dataset"]["provenance_note"]
    for model_id, entry in provenance["models"].items():
        assert entry["state"] in {"trained_and_evaluated", "not_trained", "incompatible"}, entry
        assert entry["trained_on"] in {"synthetic_fixture", "not_trained", "unverified_dataset"}, entry
    classifier_manifests = list((export_root / "data" / "models" / "classification").glob("*/manifest.json"))
    exported_classifier = json.loads(max(classifier_manifests, key=lambda path: path.stat().st_mtime).read_text(encoding="utf-8"))
    assert exported_classifier["dataset"]["synthetic"] is True
    # nothing the repository ships may be named like an export target
    for name in ("segnet_model.pth", "raunet_brats_4ch.pt", "UNet-[16, 32, 64, 128, 256].pt"):
        assert not (export_root / name).exists(), f"the export must not shadow {name}"


def test_classification_dataset_discovery_reports_a_missing_dataset(notebook_namespace):
    """A missing second dataset is reported, never substituted -- and does not crash the run."""
    import tempfile

    with tempfile.TemporaryDirectory() as empty:
        result = notebook_namespace["discover_classification_dataset"]([empty])
    assert result["found"] is False
    assert result["searched"] == [empty]
    assert "brain-tumor-mri-dataset" in result["hint"]
    # the function that decides what the data *is* keeps three distinguishable answers
    resolve = notebook_namespace["resolve_classification_provenance"]
    assert resolve("auto", {"synthetic": True}, 40, ["a", "b"])[0] is True
    assert resolve("auto", None, 7200, ["a", "b", "c", "d"])[0] is False
    assert resolve("auto", None, 40, ["a", "b"])[0] is None
    assert resolve("real", None, 10, ["a", "b"])[0] is False


def test_multiclass_dataset_is_read_from_its_own_directories(notebook_namespace):
    """Classes and counts come from the tree on disk, not from an assumption about the dataset."""
    assert notebook_namespace["CLASSIFICATION_IMAGES_AVAILABLE"] is True
    classes = notebook_namespace["CLASSIFICATION_CLASSES"]
    assert classes == ["glioma", "meningioma", "notumor", "pituitary"], classes

    structure = json.loads(
        (Path(notebook_namespace["PATHS"]["work"]) / "artifacts" / "classification_structure.json").read_text()
    )
    assert structure["state"] == "inspected"
    assert structure["class_counts"] == {name: 10 for name in classes}
    assert structure["extensions"] == {".png": 40}
    assert structure["modes"] == {"L": 40}
    assert structure["synthetic"] is True, "the fixture marker must be detected"
    assert "not substituted" not in structure["provenance_note"]

    splits = notebook_namespace["CLASSIFICATION_SPLIT_RECORD"]
    assert splits["split_source"].startswith("the dataset's own")
    assert splits["splits"]["train"] == {name: 4 for name in classes}
    assert splits["splits"]["test"] == {name: 4 for name in classes}
    assert splits["splits"]["validation"] == {name: 2 for name in classes}


def test_leakage_audit_measures_before_it_removes(notebook_namespace):
    """The audit reports candidates and confirmations separately, and cannot gut a split."""
    audit = notebook_namespace["CLASSIFICATION_AUDIT"]
    assert audit["state"] in {"measured", "measured_and_enforced"}
    assert audit["patient_identifiers_available"] is False
    assert audit["patient_level_split_possible"] is False
    for split_name in ("validation", "test"):
        entry = audit["split_audits"][split_name]
        assert entry["images_before"] > 0
        assert entry["images_after"] == entry["images_before"] - len(entry["excluded"])
        assert entry["hash_candidates"] >= entry["confirmed_near_duplicates"]
        # the fixtures are distinct images: nothing may be removed from the evaluation splits
        assert entry["confirmed_near_duplicates"] == 0, entry["examples"]
    # the guard exists: a rule that would remove more than the allowed share is refused
    assert audit["max_exclusion_fraction"] <= 0.2


def test_multiclass_artefacts_are_exported_with_task_and_dataset(notebook_namespace):
    workdir = Path(notebook_namespace["PATHS"]["work"])
    export_root = workdir / "brain_mri_artifacts"

    # the CNN checkpoint and the stacking ensemble
    assert (export_root / "checkpoints" / "brain_tumor_mri_cnn.pt").exists()
    artefact_dirs = list((export_root / "data" / "models" / "classification_multiclass").glob("*/manifest.json"))
    assert len(artefact_dirs) == 1, artefact_dirs
    manifest = json.loads(artefact_dirs[0].read_text())
    assert manifest["task"] == "multiclass_classification"
    assert manifest["classes"] == notebook_namespace["CLASSIFICATION_CLASSES"]
    assert manifest["dataset"]["synthetic"] is True
    assert manifest["feature_schema"] == notebook_namespace["MULTICLASS_FEATURE_NAMES"]
    preprocessing = manifest["provenance_contract"]["preprocessing"]
    assert preprocessing["fit_on"] == "training split only"
    assert len(preprocessing["mean"]) == len(manifest["feature_schema"])
    assert len(preprocessing["scale"]) == len(manifest["feature_schema"])
    assert manifest["selected_model"] == "stacked_logistic_regression"
    for name in ("stacked_logistic_regression", "simple_average", "cnn"):
        assert name in manifest["metrics"], f"{name} has no recorded metrics"
        assert manifest["metrics"][name]["samples"] > 0
    assert (artefact_dirs[0].parent / "stacked_ensemble.joblib").exists()

    # every exported artefact is identifiable by task, label space and dataset
    index = json.loads((export_root / "data" / "models" / "models_index.json").read_text())
    by_name = {entry["name"]: entry for entry in index["artefacts"]}
    assert by_name["brain_tumor_mri_classifier"]["task"] == "multiclass_classification"
    assert by_name["brain_tumor_mri_classifier"]["dataset_synthetic"] is True
    assert by_name["brain_tumor_mri_classifier"]["served_by_application"] is False
    assert by_name["segnet_brats_4ch"]["task"] == "segmentation"
    assert by_name["radiomics_grade_classifier"]["task"] == "classification"
    assert "HGG" in by_name["radiomics_grade_classifier"]["label_space"]
    assert "glioma" in by_name["brain_tumor_mri_cnn"]["label_space"]


def test_stacked_ensemble_refuses_incompatible_members(notebook_namespace):
    """Members from another dataset/label space are rejected, with the reason recorded."""
    stack = notebook_namespace["MULTICLASS_STACK_RESULT"]
    assert stack["state"] == "fitted_and_evaluated"
    rejected = {entry["name"]: entry["reason"] for entry in stack["rejections"]}
    assert "brats_hgg_lgg_classifier" in rejected
    reason = rejected["brats_hgg_lgg_classifier"]
    assert "label space differs" in reason and "dataset differs" in reason
    # the BraTS grade model is never a member
    assert "brats_hgg_lgg_classifier" not in stack["members"]
    assert stack["test_split_used_for_tuning"] is False
    assert stack["selection_split"] == "validation"
    # superiority is claimed only when the measured numbers support it
    stacked = stack["comparison"]["stacked_logistic_regression"]["test"]["macro_f1"]
    best = stack["comparison"][stack["best_single_member"]]["test"]["macro_f1"]
    assert abs(stack["stacked_minus_best_single_macro_f1"] - (stacked - best)) < 1e-9
    if stacked <= best:
        assert "not better" in stack["statement"] or "not an improvement" in stack["statement"]


def test_multiclass_evaluation_is_on_a_held_out_split(notebook_namespace):
    workdir = Path(notebook_namespace["PATHS"]["work"])
    evaluation = json.loads((workdir / "artifacts" / "classification_multiclass_evaluation.json").read_text())
    assert evaluation["state"] == "evaluated"
    protocol = evaluation["protocol"]
    assert protocol["selection_split"] == "validation"
    assert protocol["test_split_used_for_tuning"] is False
    assert protocol["patient_level_split_possible"] is False
    assert evaluation["dataset"]["synthetic"] is True
    assert evaluation["dataset"]["classes"] == notebook_namespace["CLASSIFICATION_CLASSES"]
    metrics = evaluation["models"]["stacked_logistic_regression"]
    for key in ("accuracy", "macro_f1", "weighted_f1", "macro_precision", "macro_recall", "per_class", "confusion_matrix"):
        assert key in metrics, key
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["macro_f1"] <= 1.0
    assert len(metrics["per_class"]) == len(notebook_namespace["CLASSIFICATION_CLASSES"])
    assert metrics["samples"] > 0
    # the honest limits travel with the numbers
    caveats = " ".join(evaluation["caveats"]).lower()
    assert "not a malignancy" in caveats
    assert "patient identifiers" in caveats


def test_predicted_mask_features_and_quality_report(notebook_namespace):
    """The predicted-mask path runs, and the quality report measures it from the tables.

    The smoke run's segmentation models are trained for a few steps on random arrays, so their
    predicted regions are mostly empty and the notebook correctly refuses to build that table.
    Here a stand-in scorer that predicts a large positive logit lets the same code path produce
    the table, so the *reporting* code is covered even when the models are useless.
    """
    torch = pytest.importorskip("torch")
    namespace = notebook_namespace

    class AlwaysForeground(torch.nn.Module):
        output_semantics = "logit"

        def forward(self, x):
            return torch.full((x.shape[0], 1, x.shape[-2], x.shape[-1]), 8.0)

    previous_selector = namespace["SEGMENTATION_FOR_FEATURES"]
    namespace["_SEGMENTATION_CACHE"]["stand_in"] = (AlwaysForeground(), True, Path("stand-in"))
    namespace["SEGMENTATION_FOR_FEATURES"] = "stand_in"
    try:
        labelled = sorted(namespace["LABELLED"])
        predicted_rows = []
        for volume_key in labelled:
            row, note = namespace["patient_feature_row"](volume_key, mask_source="predicted")
            assert row is not None, note
            assert isinstance(row["roi_mask_dice"], float)
            predicted_rows.append(row)
    finally:
        namespace["SEGMENTATION_FOR_FEATURES"] = previous_selector

    reference_rows = namespace["FEATURE_TABLES"]["reference"]
    assert reference_rows, "the reference-mask table is empty"
    assert set(namespace["FEATURE_NAMES"]) <= set(predicted_rows[0]), "same schema, different columns"

    report = namespace["radiomics_quality_report"](
        reference_rows, predicted_rows, namespace["FEATURE_NAMES"], comparison_patients=2
    )
    assert report["state"] == "measured"
    assert report["patients_compared"] == len(predicted_rows)
    assert 0.0 <= report["reference_vs_predicted_mask"]["roi_mask_dice"]["median"] <= 1.0
    assert report["reference_vs_predicted_mask"]["median_relative_difference_over_features"] is not None
    assert report["slice_to_slice_variability"]["features_compared"] > 0
    assert report["spacing_available"] is False


def test_exported_classifier_is_served_by_the_application(notebook_namespace):
    """The classification artefact the notebook writes must satisfy ClassificationService."""
    from dataclasses import dataclass

    namespace = notebook_namespace
    export_classification = Path(namespace["PATHS"]["work"]) / "brain_mri_artifacts" / "data" / "models" / "classification"

    _import_real_backend(namespace)
    from app.config import DevelopmentConfig
    from app.services.classification_service import ClassificationService

    @dataclass
    class Config(DevelopmentConfig):
        CLASSIFIER_ARTIFACT_DIR: object = export_classification

    service = ClassificationService(Config())
    status = service.status()
    assert status["available"] is True, status.get("reason")
    assert status["feature_count"] == len(namespace["FEATURE_NAMES"])
    assert status["label_mapping"] == {"0": "HGG", "1": "LGG"}
    assert status["metrics"]["evaluation_split"].startswith("held-out test split")
    assert status["preprocessing"]["type"] == "standard_scaler"
    schema = status["feature_schema"]
    outcome = service.classify({name: 1.0 for name in schema}, explain=True).to_dict()
    assert outcome["available"] is True, outcome.get("reason")
    assert outcome["predicted_label"] in {"HGG", "LGG"}
    assert outcome["feature_contributions"] or outcome["metrics"]
    incomplete = service.classify({name: 1.0 for name in schema[:-1]}, explain=False).to_dict()
    assert incomplete["available"] is False
    assert incomplete["missing_features"]


def test_exported_checkpoints_load_in_the_application_registry(notebook_namespace):
    """The exported SegNet checkpoint must load through the real ModelRegistry, not just Torch."""
    from dataclasses import dataclass

    namespace = notebook_namespace
    export_root = Path(namespace["PATHS"]["work"]) / "brain_mri_artifacts"
    segnet_path = export_root / "checkpoints" / "segnet_brats_4ch_real_brats.pt"

    _import_real_backend(namespace)
    from app.config import DevelopmentConfig
    from app.ml.registry import ModelRegistry

    @dataclass
    class Config(DevelopmentConfig):
        MODEL_PATHS: object = None
        DEVICE_PREFERENCE: str = "cpu"
        TORCH_THREADS: int = 2
        MODEL_CACHE_ENABLED: bool = False

    registry = ModelRegistry(
        Config(
            MODEL_PATHS={
                "segnet_brats_4ch": str(segnet_path),
                "unet_dynamic_512": str(REPO_ROOT / "UNet-[16, 32, 64, 128, 256].pt"),
                "raunet_brats_4ch": str(export_root / "checkpoints" / "raunet_brats_4ch.pt"),
                "radiomics_mlp_v1": str(REPO_ROOT / "radiomics_cnn_model.pth"),
            }
        )
    )
    segnet = registry.load_model("segnet_brats_4ch")
    assert segnet.is_ready, segnet.message
    assert segnet.parameter_count and segnet.parameter_count > 100_000
    raunet = registry.load_model("raunet_brats_4ch")
    assert raunet.is_ready, raunet.message
