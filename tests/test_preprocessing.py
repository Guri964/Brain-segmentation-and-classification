from __future__ import annotations
import numpy as np
import pytest
def test_single_channel_pipeline_shape_and_range(brain_png):
    from app.services.preprocessing_service import preprocess_single_channel
    prepared = preprocess_single_channel(brain_png.read_bytes(), brain_png.name, size=256)
    assert prepared.tensor.shape == (1, 1, 256, 256)
    assert prepared.channels == 1
    assert prepared.pipeline.startswith("single_channel")
    values = prepared.tensor.numpy()
    assert values.min() >= 0.0 and values.max() <= 1.0
def test_single_channel_rejects_rgb_silently_converted(brain_png, tmp_path):
    from PIL import Image
    from app.services.preprocessing_service import preprocess_single_channel
    rgb = tmp_path / "rgb.png"
    Image.open(brain_png).convert("RGB").save(rgb)
    prepared = preprocess_single_channel(rgb.read_bytes(), rgb.name, size=64)
    assert prepared.tensor.shape == (1, 1, 64, 64)
def test_four_channel_requires_four_distinct_images(four_channel_pngs):
    from app.services.preprocessing_service import preprocess_four_channel
    payloads = [(path.name, path.read_bytes()) for path in four_channel_pngs]
    prepared = preprocess_four_channel(payloads, size=128, modality_names=["T1", "T1ce", "T2", "FLAIR"])
    assert prepared.tensor.shape == (1, 4, 128, 128)
    assert prepared.channels == 4
def test_four_channel_refuses_duplicated_channel(brain_png):
    from app.errors import UnsupportedInputError
    from app.services.preprocessing_service import preprocess_four_channel
    raw = brain_png.read_bytes()
    payloads = [(f"same_{index}.png", raw) for index in range(4)]
    with pytest.raises(UnsupportedInputError):
        preprocess_four_channel(payloads, size=64)
def test_four_channel_rejects_wrong_count(four_channel_pngs):
    from app.errors import UnsupportedInputError
    from app.services.preprocessing_service import preprocess_four_channel
    payloads = [(path.name, path.read_bytes()) for path in four_channel_pngs[:2]]
    with pytest.raises(UnsupportedInputError):
        preprocess_four_channel(payloads, size=64)
def test_channel_count_mismatch_is_refused(brain_png):
    from app.errors import UnsupportedInputError
    from app.services.preprocessing_service import build_input_for_model
    from app.ml.registry import InputSpec, ModelDescriptor
    descriptor = ModelDescriptor(
        model_id="four_channel_model",
        display_name="four",
        task="segmentation",
        architecture="segnet",
        path="",
        output_semantics="logit",
        trainer="scripts/train_segnet.py",
        input_spec=InputSpec(channels=4, height=64, width=64, normalisation="zscore", dtype="float32"),
    )
    with pytest.raises(UnsupportedInputError):
        build_input_for_model(descriptor, [(brain_png.name, brain_png.read_bytes())])
