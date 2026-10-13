"""The four-class Keras image classifier: state machine, contract checks and serving path.

Two kinds of test live here:

* tests that always run - the state machine, the metadata contract and the NumPy preprocessing,
  none of which need TensorFlow;
* tests that need TensorFlow. The real ``brain_tumor_xception_final.keras`` is not in the
  repository (weights are never committed), so those tests run ``tests/tf_image_classifier_driver.py``
  in a **subprocess**: it builds a miniature model with the real notebook contract
  (299x299x3 RGB -> four softmax outputs), points the central resolver at a temporary weights
  directory and drives the HTTP endpoint. That validates the plumbing - paths, metadata,
  preprocessing, probabilities, error handling - and makes no claim about the accuracy of the
  supplied checkpoint, which cannot be measured without its test split. The subprocess is not a
  stylistic choice: importing TensorFlow and then ``torch.compile`` (``torch._dynamo`` ->
  ``triton``) crashes the interpreter in this environment, so the two stacks must not share a
  process. See docs/LIMITATIONS.md.
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app import config as config_module
from app.errors import ModelUnavailableError, ValidationError
from app.ml import image_classifier as module
from app.ml.image_classifier import KerasImageClassifier, xception_preprocess

CLASSES = ["Glioma", "Meningioma", "Pituitary", "No Tumor"]


def metadata_payload(**overrides) -> dict:
    payload = {
        "model_name": "Brain_Tumor_Detection",
        "architecture": "Xception",
        "framework": "TensorFlow/Keras",
        "framework_version": "2.15.0",
        "input_size": [299, 299, 3],
        "classes": list(CLASSES),
        "class_to_index": {name: index for index, name in enumerate(CLASSES)},
        "preprocessing": "keras.applications.xception.preprocess_input",
        "model_artifact": "brain_tumor_xception_final.keras",
        "training_config": {"batch_size": 32},
        "metrics": {"test_accuracy": 0.9091, "macro_f1": 0.9067},
        "generated_at": "2025-01-31T01:20:00+00:00",
    }
    payload.update(overrides)
    return payload


@pytest.fixture(autouse=True)
def no_framework_probe(monkeypatch):
    """Keep TensorFlow out of the test process, and pin the framework-missing behaviour.

    The module-level import seam is what ``_framework_status`` calls, so replacing it makes the
    "checkpoint present, framework absent" branch deterministic instead of environment-dependent.
    ``test_the_driver_does_not_import_tensorflow_into_the_test_process`` fails if this ever
    regresses.
    """

    from app.errors import DependencyMissingError

    def unavailable():
        raise DependencyMissingError("TensorFlow is not importable in this environment.")

    monkeypatch.setattr(module, "_import_tensorflow", unavailable)


@pytest.fixture()
def weights_dir(tmp_path, monkeypatch):
    """Redirect the central resolver at a temporary project so no real file is touched."""

    root = tmp_path / "project"
    directory = root / "models" / "weights"
    directory.mkdir(parents=True)
    monkeypatch.setattr(config_module, "BASE_DIR", root)
    monkeypatch.delenv("BRAIN_WEIGHTS_DIR", raising=False)
    monkeypatch.delenv("BRAIN_MODEL_XCEPTION", raising=False)
    monkeypatch.delenv("BRAIN_MODEL_XCEPTION_METADATA", raising=False)
    return directory


def write_metadata(directory: Path, **overrides) -> Path:
    path = directory / "metadata.json"
    path.write_text(json.dumps(metadata_payload(**overrides), indent=2))
    return path


def write_checkpoint(directory: Path) -> Path:
    path = directory / "brain_tumor_xception_final.keras"
    path.write_bytes(b"not a real keras archive")
    return path


def png_bytes(size: tuple[int, int] = (320, 260), seed: int = 3) -> bytes:
    rng = np.random.default_rng(seed)
    array = (rng.random((size[1], size[0], 3)) * 255).astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return buffer.getvalue()


def upload(client, payload: bytes, filename: str = "slice.png"):
    return client.post(
        "/api/classification/image",
        data={"file": (io.BytesIO(payload), filename)},
        content_type="multipart/form-data",
    )


# --------------------------------------------------------------------------- state machine
def test_without_the_checkpoint_the_model_is_reported_as_missing(weights_dir):
    from app.config import TestingConfig

    classifier = KerasImageClassifier(TestingConfig)
    status = classifier.status()
    assert status["state"] == "checkpoint_missing"
    assert status["available"] is False
    assert status["model_file"] == "brain_tumor_xception_final.keras"
    assert str(weights_dir) in status["reason"]
    assert "models/README.md" in status["reason"]


def test_a_missing_checkpoint_is_reported_without_importing_the_framework(weights_dir):
    """The framework probe is skipped when there is nothing to execute: fast and quiet."""

    from app.config import TestingConfig

    status = KerasImageClassifier(TestingConfig).status()
    assert status["framework_available"] is None
    assert status["framework_version"] is None


def test_checkpoint_without_metadata_is_refused_not_guessed(weights_dir):
    from app.config import TestingConfig

    write_checkpoint(weights_dir)
    status = KerasImageClassifier(TestingConfig).status()
    assert status["state"] == "metadata_missing"
    assert status["available"] is False
    assert "which class" in status["reason"]


def test_inconsistent_metadata_is_refused(weights_dir):
    from app.config import TestingConfig

    write_checkpoint(weights_dir)
    write_metadata(weights_dir, class_to_index={"Glioma": 0, "Meningioma": 1})
    status = KerasImageClassifier(TestingConfig).status()
    assert status["state"] == "metadata_missing"
    assert "class_to_index" in status["reason"]


def test_unknown_preprocessing_is_refused_rather_than_substituted(weights_dir):
    from app.config import TestingConfig

    write_checkpoint(weights_dir)
    write_metadata(weights_dir, preprocessing="imagenet_mode")
    status = KerasImageClassifier(TestingConfig).status()
    assert status["state"] == "metadata_missing"
    assert "imagenet_mode" in status["reason"]


def test_two_channel_input_is_refused(weights_dir):
    from app.config import TestingConfig

    write_checkpoint(weights_dir)
    write_metadata(weights_dir, input_size=[299, 299, 2])
    status = KerasImageClassifier(TestingConfig).status()
    assert status["state"] == "metadata_missing"
    assert "channel" in status["reason"]


def test_a_present_checkpoint_without_tensorflow_reports_framework_missing(weights_dir):
    from app.config import TestingConfig

    write_checkpoint(weights_dir)
    write_metadata(weights_dir)
    status = KerasImageClassifier(TestingConfig).status()
    assert status["state"] == "framework_missing"
    assert status["available"] is False
    assert status["framework_available"] is False
    assert "TensorFlow" in status["reason"]
    assert status["classes"] == CLASSES  # the metadata is still reported as read


@pytest.mark.parametrize(
    "payload, expected",
    [
        ({}, "'classes'"),
        ({"classes": ["only-one"]}, "'classes'"),
        ({"classes": CLASSES}, "'class_to_index'"),
        (
            {"classes": CLASSES, "class_to_index": {"Glioma": 0}},
            "'class_to_index'",
        ),
        (
            {"classes": CLASSES, "class_to_index": {"Glioma": 0, "Meningioma": 1, "Pituitary": 2, "X": 3}},
            "class_to_index",
        ),
    ],
)
def test_metadata_contract_is_enforced(payload, expected):
    problem = KerasImageClassifier._validate_metadata(payload)
    assert problem is not None
    assert expected in problem


def test_valid_metadata_passes_the_contract():
    status = KerasImageClassifier._validate_metadata(metadata_payload())
    assert KerasImageClassifier._validate_metadata(metadata_payload()) is None or True
    assert status is None


def test_predicting_without_the_checkpoint_raises_an_actionable_error(weights_dir):
    from app.config import TestingConfig

    classifier = KerasImageClassifier(TestingConfig)
    with pytest.raises(ModelUnavailableError) as error:
        classifier.predict(np.zeros((299, 299, 3), dtype=np.uint8))
    message = str(error.value)
    assert "brain_tumor_xception_final.keras" in message
    assert "metadata.json" in message
    assert error.value.details["expected_files"] == [
        "brain_tumor_xception_final.keras",
        "metadata.json",
    ]
    assert error.value.details["state"] == "checkpoint_missing"


def test_predicting_needs_three_channels(weights_dir, monkeypatch):
    from app.config import TestingConfig

    write_checkpoint(weights_dir)
    write_metadata(weights_dir)
    classifier = KerasImageClassifier(TestingConfig)
    monkeypatch.setattr(
        classifier, "ensure_loaded", lambda: (object(), classifier.read_metadata() or {})
    )
    with pytest.raises(ValidationError):
        classifier.predict(np.zeros((299, 299), dtype=np.uint8))


# --------------------------------------------------------------------------- preprocessing
def test_preprocessing_is_the_documented_affine_scaling():
    values = np.array([[[0.0, 127.5, 255.0]]], dtype=np.float32)
    result = xception_preprocess(values)
    assert result.dtype == np.float32
    np.testing.assert_allclose(result, [[[-1.0, 0.0, 1.0]]], atol=1e-6)


# The comparison against ``keras.applications.xception.preprocess_input`` itself needs the
# framework, so it lives in the subprocess driver and is asserted by
# ``test_the_served_preprocessing_equals_the_keras_function``.

def test_prepare_resizes_to_the_metadata_size_and_scales():
    prepared = KerasImageClassifier._prepare(np.full((64, 32, 3), 255, dtype=np.uint8), 299, 299)
    assert prepared.shape == (299, 299, 3)
    assert prepared.dtype == np.float32
    np.testing.assert_allclose(prepared, 1.0, atol=1e-6)


def test_probability_helper_detects_a_distribution():
    assert module.is_probability_vector(np.array([0.2, 0.8])) is True
    assert module.is_probability_vector(np.array([2.0, -1.0])) is False
    assert module.is_probability_vector(np.array([np.nan, 1.0])) is False
    np.testing.assert_allclose(module.softmax(np.array([0.0, 0.0])), [0.5, 0.5])


# --------------------------------------------------------------------------- HTTP surface
def test_image_models_endpoint_describes_the_missing_model(client):
    response = client.get("/api/classification/image/models")
    assert response.status_code == 200
    models = response.get_json()["models"]
    assert len(models) == 1
    assert models[0]["model_id"] == "brain_tumor_xception"
    assert models[0]["task"] == "image_classification"
    assert models[0]["state"] in {"ready", "checkpoint_missing", "metadata_missing", "framework_missing"}
    assert models[0]["reported_metrics"] is None or isinstance(models[0]["reported_metrics"], dict)


def test_image_classification_status_is_part_of_the_classification_payload(client):
    payload = client.get("/api/classification").get_json()
    assert "image_classification" in payload
    summary = payload["image_classification"]
    assert summary["task"] == "image_classification"
    assert "distinct_from_grade_model" in summary
    assert isinstance(summary["accepted_formats"], list)


def test_capabilities_expose_the_image_classifier_state(client):
    summary = client.get("/api/capabilities").get_json()["image_classification"]
    assert summary["model_id"] == "brain_tumor_xception"
    assert isinstance(summary["available"], bool)
    assert summary["reason"]
    assert summary["metrics_provenance"]


def test_upload_without_a_file_is_a_validation_error(client):
    response = client.post("/api/classification/image", data={}, content_type="multipart/form-data")
    assert response.status_code == 422
    assert response.get_json()["error"]["details"]["field"] == "file"


def test_a_non_image_upload_is_rejected_by_extension(client):
    response = upload(client, b"radiomics table", filename="features.csv")
    assert response.status_code == 415
    assert "features.csv" in response.get_json()["error"]["message"]


def test_a_corrupt_image_is_rejected_after_decoding(client):
    response = upload(client, b"not really a png", filename="slice.png")
    assert response.status_code == 415
    assert "could not be decoded" in response.get_json()["error"]["message"]


def test_without_the_checkpoint_the_endpoint_answers_409_and_never_a_label(client):
    response = upload(client, png_bytes())
    assert response.status_code == 409
    error = response.get_json()["error"]
    assert error["code"] == "model_unavailable"
    assert "brain_tumor_xception_final.keras" in error["message"]
    assert "probability" not in error


# --------------------------------------------------------------------- end to end with a model
DRIVER = Path(__file__).resolve().parent / "tf_image_classifier_driver.py"


@pytest.fixture(scope="module")
def driver_report():
    """Run the TensorFlow serving path in its own process, once, and return its JSON."""

    import importlib.util
    import subprocess
    import sys

    # find_spec, not import: TensorFlow must never enter this process (see the module docstring)
    if importlib.util.find_spec("tensorflow") is None:  # pragma: no cover - environment dependent
        pytest.skip("TensorFlow is not installed in this environment")

    completed = subprocess.run(
        [sys.executable, str(DRIVER)],
        capture_output=True,
        text=True,
        timeout=900,
        cwd=str(Path(__file__).resolve().parent.parent),
    )
    if completed.returncode != 0:  # pragma: no cover - the driver reports its own failures
        pytest.fail(
            f"the TensorFlow driver exited with {completed.returncode}\n"
            f"stdout: {completed.stdout[-2000:]}\nstderr: {completed.stderr[-2000:]}"
        )
    payload = [line for line in completed.stdout.splitlines() if line.startswith("{")]
    assert payload, f"the driver printed no JSON report: {completed.stdout[-2000:]}"
    return json.loads(payload[-1])


def test_end_to_end_inference_returns_metadata_ordered_probabilities(driver_report):
    assert driver_report["state"] == "ready"
    assert driver_report["classes"] == CLASSES
    assert driver_report["inference_status"] == 200
    body = driver_report["inference"]
    assert body["predicted_label"] == CLASSES[2]
    assert body["predicted_index"] == 2
    assert len(body["probabilities"]) == 4
    assert abs(sum(body["probabilities"]) - 1.0) < 1e-4
    assert body["classes_source"].endswith("metadata.json")
    assert body["original_shape"] == [260, 320, 3]
    assert body["model_input_size"] == [299, 299, 3]
    assert body["preprocessing"] == "keras.applications.xception.preprocess_input"
    assert body["preprocessing_implementation"] == "RGB [0, 255] -> x / 127.5 - 1"
    assert body["calibrated"] is False
    assert "not calibrated" in body["calibration_note"]
    assert body["metrics_provenance"].startswith(
        "These numbers are the metrics the training notebook"
    )
    assert body["task"] == "image_classification"
    assert body["inference_seconds"] >= 0


def test_the_served_preprocessing_equals_the_keras_function(driver_report):
    assert driver_report["preprocessing_max_abs_diff"] <= 1e-6


def test_the_class_order_comes_from_the_metadata(driver_report):
    assert driver_report["renamed_classes"]["classes"] == ["A", "B", "C", "D"]
    assert driver_report["renamed_classes"]["predicted_label"] == "C"


def test_a_checkpoint_that_disagrees_with_the_metadata_is_refused(driver_report):
    outcome = driver_report["class_count_mismatch"]
    assert outcome["status"] == 409
    assert "declares 2 classes" in outcome["message"]


def test_a_corrupt_upload_is_rejected(driver_report):
    assert driver_report["corrupt"]["status"] == 415


def test_a_tiny_image_is_rejected_before_inference(driver_report):
    assert driver_report["tiny"]["status"] == 422
    assert "at least 16" in driver_report["tiny"]["message"]


def test_grayscale_uploads_are_converted_to_rgb(driver_report):
    assert driver_report["grayscale"]["status"] == 200
    assert driver_report["grayscale"]["original_shape"] == [40, 40, 3]
    assert driver_report["grayscale"]["probabilities"] == 4


def test_the_classifier_never_reports_a_grade_or_a_malignancy(driver_report):
    semantics = driver_report["inference"]["label_semantics"].lower()
    assert "grade" in semantics
    assert "malignancy" in semantics
    assert all("HGG" not in name for name in driver_report["inference"]["classes"])


def test_reported_metrics_are_labelled_as_reported(driver_report):
    assert driver_report["inference"]["reported_metrics"]["test_accuracy"] == pytest.approx(0.9091)
    assert "not been re-measured" in driver_report["inference"]["metrics_provenance"]


def test_each_app_instance_resolves_the_checkpoint_freshly(driver_report):
    assert driver_report["second_instance_label"] == CLASSES[2]


def test_the_driver_does_not_import_tensorflow_into_the_test_process():
    """The serving path runs out of process, so the suite never mixes TF with PyTorch."""

    import sys

    assert "tensorflow" not in sys.modules
