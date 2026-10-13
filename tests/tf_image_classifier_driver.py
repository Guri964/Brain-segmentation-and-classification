"""Subprocess driver for the TensorFlow-dependent image-classifier tests.

It is deliberately **not** named ``test_*``: pytest must not collect it, because importing
TensorFlow and PyTorch into one process is what the tests are avoiding. In this workspace
``import tensorflow`` followed by ``torch.compile`` (which imports ``torch._dynamo`` ->
``triton``) segfaults, an incompatibility between the two native stacks, not an application bug.
Running the Keras serving path in its own process keeps the main test process - and any
deployment that serves both model families - out of that trap.

Usage::

    python tests/tf_image_classifier_driver.py            # prints one JSON object

The driver builds a miniature model with the **real** notebook signature
(299x299x3 RGB -> four softmax outputs) into a temporary weights directory, points the central
resolver at it and drives the HTTP endpoint. It proves the serving path; it proves nothing about
the accuracy of the supplied checkpoint, which is absent here and is never simulated.
"""

from __future__ import annotations

import io
import json
import os
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backend"))

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


def build_tiny_model(directory: Path, favourite: int = 2) -> Path:
    """A stand-in with the real input/output signature: 299x299x3 RGB -> 4-way softmax."""

    import numpy as np
    import tensorflow as tf
    from keras import layers

    inputs = tf.keras.Input(shape=(299, 299, 3), name="image")
    x = layers.Conv2D(4, 3, strides=8, activation="relu", name="probe_conv")(inputs)
    x = layers.GlobalAveragePooling2D(name="probe_pool")(x)
    outputs = layers.Dense(4, activation="softmax", dtype="float32", name="predictions")(x)
    model = tf.keras.Model(inputs, outputs, name="brain_tumor_detection_standin")
    model.get_layer("predictions").set_weights(
        [np.zeros((4, 4), dtype=np.float32), np.eye(4, dtype=np.float32)[favourite]]
    )
    path = directory / "brain_tumor_xception_final.keras"
    model.save(path)
    return path


def png_bytes(size=(320, 260), mode="RGB") -> bytes:
    import numpy as np
    from PIL import Image

    rng = np.random.default_rng(3)
    if mode == "RGB":
        array = (rng.random((size[1], size[0], 3)) * 255).astype(np.uint8)
    else:
        array = (rng.random((size[1], size[0])) * 255).astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return buffer.getvalue()


def fresh_client(data_dir: Path):
    from app import create_app
    from app.config import TestingConfig

    class SandboxConfig(TestingConfig):
        DATA_DIR = data_dir
        LOG_DIR = data_dir / "logs"

    SandboxConfig.init_directories()
    application = create_app(config_object=SandboxConfig, env="testing")
    application.config.update(TESTING=True)
    return application.test_client()


def upload(client, payload: bytes, filename: str = "slice.png"):
    return client.post(
        "/api/classification/image",
        data={"file": (io.BytesIO(payload), filename)},
        content_type="multipart/form-data",
    )


def main() -> int:
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    root = Path(tempfile.mkdtemp(prefix="tf-image-classifier-"))
    weights = root / "weights"
    weights.mkdir(parents=True)
    data_dir = root / "data"
    data_dir.mkdir()
    os.environ["BRAIN_WEIGHTS_DIR"] = str(weights)
    os.environ["BRAIN_ENV"] = "testing"
    os.environ["BRAIN_DATA_DIR"] = str(data_dir)
    os.environ.pop("BRAIN_MODEL_XCEPTION", None)
    os.environ.pop("BRAIN_MODEL_XCEPTION_METADATA", None)

    results: dict = {"driver": "tf_image_classifier_driver"}
    build_tiny_model(weights)
    (weights / "metadata.json").write_text(json.dumps(metadata_payload(), indent=2))
    image = png_bytes()
    results["model_bytes"] = (weights / "brain_tumor_xception_final.keras").stat().st_size

    # 1. preprocessing must equal the Keras function
    import numpy as np
    from keras.applications import xception

    from app.ml.image_classifier import xception_preprocess

    rng = np.random.default_rng(0)
    sample = (rng.random((2, 16, 16, 3)) * 255).astype(np.float32)
    reference = np.asarray(xception.preprocess_input(sample.copy()))
    results["preprocessing_max_abs_diff"] = float(
        np.abs(xception_preprocess(sample) - reference).max()
    )

    # 2. status and one real request through the HTTP endpoint
    client = fresh_client(data_dir)
    status = client.get("/api/classification").get_json()["image_classification"]
    results["state"] = status["state"]
    results["classes"] = status["models"][0]["classes"]
    response = upload(client, image)
    results["inference_status"] = response.status_code
    body = response.get_json()
    results["inference"] = {
        key: body.get(key)
        for key in (
            "predicted_index",
            "predicted_label",
            "classes",
            "probabilities",
            "classes_source",
            "output_transform",
            "calibrated",
            "calibration_note",
            "preprocessing",
            "preprocessing_implementation",
            "metrics_provenance",
            "label_semantics",
            "reported_metrics",
            "task",
            "inference_seconds",
        )
    }
    results["inference"]["original_shape"] = body["input"]["original_shape"]
    results["inference"]["model_input_size"] = body["input"]["model_input_size"]

    # 3. the class order comes from metadata, not from the code
    renamed = metadata_payload(classes=["A", "B", "C", "D"])
    renamed["class_to_index"] = {name: index for index, name in enumerate(renamed["classes"])}
    (weights / "metadata.json").write_text(json.dumps(renamed))
    renamed_body = upload(fresh_client(data_dir), image).get_json()
    results["renamed_classes"] = {
        "classes": renamed_body["classes"],
        "predicted_label": renamed_body["predicted_label"],
    }

    # 4. a checkpoint that disagrees with the metadata is refused, never relabelled
    short = metadata_payload(classes=["Glioma", "Meningioma"],
                             class_to_index={"Glioma": 0, "Meningioma": 1})
    (weights / "metadata.json").write_text(json.dumps(short))
    mismatch = upload(fresh_client(data_dir), image)
    results["class_count_mismatch"] = {
        "status": mismatch.status_code,
        "message": mismatch.get_json()["error"]["message"],
    }

    # 5. a corrupt image, a tiny image and a grayscale image
    (weights / "metadata.json").write_text(json.dumps(metadata_payload()))
    corrupt = upload(fresh_client(data_dir), b"not really a png")
    results["corrupt"] = {"status": corrupt.status_code}

    small = io.BytesIO()
    from PIL import Image

    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(small, format="PNG")
    tiny = upload(fresh_client(data_dir), small.getvalue(), filename="tiny.png")
    results["tiny"] = {"status": tiny.status_code, "message": tiny.get_json()["error"]["message"]}

    grayscale = upload(
        fresh_client(data_dir), png_bytes(size=(40, 40), mode="L"), filename="gray.png"
    )
    results["grayscale"] = {
        "status": grayscale.status_code,
        "original_shape": grayscale.get_json()["input"]["original_shape"],
        "probabilities": len(grayscale.get_json()["probabilities"]),
    }

    # 6. a second app instance must reuse nothing stale from the first
    reused = upload(fresh_client(data_dir), image).get_json()
    results["second_instance_label"] = reused["predicted_label"]

    print(json.dumps(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
