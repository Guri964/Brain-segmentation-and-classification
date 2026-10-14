from __future__ import annotations
import numpy as np
import pytest
def _prepared(brain_png, size=512):
    from app.services.preprocessing_service import preprocess_single_channel
    return preprocess_single_channel(brain_png.read_bytes(), brain_png.name, size=size)
def test_predict_returns_probability_and_mask(services, brain_png):
    prepared = _prepared(brain_png, 512)
    result = services["segmentation"].predict("unet_dynamic_512", prepared, threshold=0.5)
    assert result.probability.shape == (512, 512)
    assert result.probability.min() >= 0.0 and result.probability.max() <= 1.0
    assert result.mask.dtype == bool
    assert result.threshold == 0.5
    assert result.foreground_pixels == int(result.mask.sum())
    assert result.summary()["foreground_fraction"] == pytest.approx(result.foreground_fraction, abs=1e-6)
def test_threshold_changes_mask_not_probability(services, brain_png):
    prepared = _prepared(brain_png, 512)
    low = services["segmentation"].predict("unet_dynamic_512", prepared, threshold=0.2)
    high = services["segmentation"].predict("unet_dynamic_512", prepared, threshold=0.8)
    assert np.allclose(low.probability, high.probability)
    assert low.foreground_pixels >= high.foreground_pixels
def test_empty_mask_reports_none_not_zero(services, brain_png):
    prepared = _prepared(brain_png, 512)
    result = services["segmentation"].predict("unet_dynamic_512", prepared, threshold=0.99999)
    assert result.foreground_pixels == 0
    assert result.probability_mean_in_mask is None
    assert any("empty mask" in warning for warning in result.warnings)
    assert result.summary()["foreground_pixels"] == 0
def test_wrong_channel_count_is_refused(services, brain_png):
    from app.errors import InferenceError
    prepared = _prepared(brain_png, 512)
    with pytest.raises(InferenceError):
        services["segmentation"].predict("segnet_brats_4ch", prepared)
def test_ensemble_reports_why_it_cannot_run(services):
    compatibility = services["ensemble"].compatibility(["segnet_brats_4ch", "unet_dynamic_512"])
    assert compatibility["available"] is False
    assert compatibility["reasons"]
    assert any(
        "different inputs" in reason or "fewer than two" in reason.lower() or "fail the reference" in reason
        for reason in compatibility["reasons"]
    )
def test_ensemble_groups_describe_compatibility(services):
    compatibility = services["ensemble"].compatibility(["segnet_brats_4ch", "unet_dynamic_512"])
    groups = compatibility["groups"]
    assert len(groups) >= 1
    assert all(group["ensemblable"] is False for group in groups)
    shapes = {(group["channels"], group["height"], group["width"]) for group in groups}
    assert (1, 512, 512) in shapes
    for group in groups:
        assert group["ensemblable"] is False or len(group["members"]) > 1
    status = {entry["model_id"]: entry["status"] for entry in compatibility["model_status"]}
    assert "segnet_brats_4ch" in status
    assert "unet_dynamic_512" in status
def test_equal_weight_fusion_is_the_mean(services):
    from app.services.segmentation_service import SegmentationResult
    def member(model_id, value):
        probability = np.full((8, 8), value, dtype=np.float32)
        return SegmentationResult(
            model_id=model_id,
            model_display_name=model_id,
            device="cpu",
            threshold=0.5,
            probability=probability,
            mask=probability > 0.5,
            foreground_pixels=int((probability > 0.5).sum()),
            foreground_fraction=float((probability > 0.5).mean()),
            probability_mean_in_mask=None,
            input_shape=[1, 1, 8, 8],
            output_semantics="probability",
            inference_seconds=0.0,
            tta_used=False,
            mc_dropout_used=False,
            n_passes=1,
        )
    outcome = services["ensemble"].predict(
        {"a": member("a", 0.2), "b": member("b", 0.6)}, strategy="equal", threshold=0.5
    )
    assert outcome.available is True
    assert outcome.probability is not None
    assert float(outcome.probability.mean()) == pytest.approx(0.4, abs=1e-6)
    assert outcome.foreground_pixels == 0
    above = services["ensemble"].predict(
        {"a": member("a", 0.4), "b": member("b", 0.9)}, strategy="equal", threshold=0.5
    )
    assert float(above.probability.mean()) == pytest.approx(0.65, abs=1e-6)
    assert above.foreground_pixels == 64
    assert above.weights["a"] == pytest.approx(above.weights["b"])
    assert above.disagreement is not None
def test_single_member_ensemble_is_refused(services):
    from app.services.segmentation_service import SegmentationResult
    probability = np.full((8, 8), 0.9, dtype=np.float32)
    single = SegmentationResult(
        model_id="only",
        model_display_name="only",
        device="cpu",
        threshold=0.5,
        probability=probability,
        mask=probability > 0.5,
        foreground_pixels=64,
        foreground_fraction=1.0,
        probability_mean_in_mask=0.9,
        input_shape=[1, 1, 8, 8],
        output_semantics="probability",
        inference_seconds=0.0,
        tta_used=False,
        mc_dropout_used=False,
        n_passes=1,
    )
    outcome = services["ensemble"].predict({"only": single}, strategy="equal", threshold=0.5)
    assert outcome.available is False
    assert outcome.reasons
