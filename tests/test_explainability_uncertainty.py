from __future__ import annotations
import numpy as np
def _prepared(brain_png, size=512):
    from app.services.preprocessing_service import preprocess_single_channel
    return preprocess_single_channel(brain_png.read_bytes(), brain_png.name, size=size)
def test_gradcam_returns_heatmap_and_overlay(services, brain_png):
    prepared = _prepared(brain_png, 512)
    result = services["segmentation"].predict("unet_dynamic_512", prepared, threshold=0.5)
    payload = services["explainability"].explain_segmentation(
        "unet_dynamic_512",
        prepared.tensor,
        display_image=prepared.display_stack[0][1],
        predicted_mask=result.mask,
        include_activation_map=False,
    )
    assert payload["method"]
    assert payload["target_layer"]
    assert payload["images"]["heatmap_rgb"].shape[-1] == 3
    assert payload["activation_shape"]
    assert payload["heatmap_mean"] is not None
def test_gradcam_heatmap_is_normalised_and_finite(services, brain_png):
    prepared = _prepared(brain_png, 512)
    payload = services["explainability"].explain_segmentation(
        "unet_dynamic_512",
        prepared.tensor,
        display_image=prepared.display_stack[0][1],
        predicted_mask=None,
        include_activation_map=False,
    )
    heatmap = np.asarray(payload["images"]["heatmap_gray"], dtype=np.float32)
    assert np.isfinite(heatmap).all()
    assert 0.0 <= heatmap.min() and heatmap.max() <= 1.0 + 1e-6
def test_gradcam_tracks_the_prediction_it_explains(services, brain_png):
    prepared = _prepared(brain_png, 512)
    result = services["segmentation"].predict("unet_dynamic_512", prepared, threshold=0.3)
    payload = services["explainability"].explain_segmentation(
        "unet_dynamic_512",
        prepared.tensor,
        display_image=prepared.display_stack[0][1],
        predicted_mask=result.mask,
        include_activation_map=False,
    )
    if result.foreground_pixels == 0:
        return
    heatmap = np.asarray(payload["images"]["heatmap_gray"], dtype=np.float32)
    mask = result.mask.astype(np.float32)
    inside = heatmap[mask > 0].mean()
    outside = heatmap[mask == 0].mean()
    assert inside >= outside * 0.5
def test_uncertainty_uses_the_input_tensor(services, brain_png):
    prepared = _prepared(brain_png, 512)
    result = services["segmentation"].predict("unet_dynamic_512", prepared, threshold=0.5, tta=True)
    assert result.tta_used is True
    assert result.n_passes > 1
    module = services["registry"].load_model("unet_dynamic_512").module
    assert module is not None
    passes = services["segmentation"]._tta_passes(module, prepared.tensor, "cpu")
    assert len(passes) >= 2
    outcome = services["uncertainty"].from_probability_stack(
        [*passes, result.probability], method="test_time_augmentation", threshold=0.5
    )
    assert outcome.available is True
    assert outcome.n_passes >= 2
    assert outcome.mean_dispersion is not None
    assert outcome.per_pixel_map is not None
def test_uncertainty_not_offered_from_a_single_pass(services):
    import numpy as np
    outcome = services["uncertainty"].from_probability_stack(
        [np.zeros((8, 8), dtype=np.float32)], method="test_time_augmentation", threshold=0.5
    )
    assert outcome.available is False
    assert outcome.unavailable_reason
    assert outcome.n_passes == 1
def test_mc_dropout_refused_for_models_without_dropout(services, brain_png):
    prepared = _prepared(brain_png, 512)
    result = services["segmentation"].predict("unet_dynamic_512", prepared, threshold=0.5, mc_dropout=True)
    assert result.mc_dropout_used is False
    assert any("dropout" in warning for warning in result.warnings)
