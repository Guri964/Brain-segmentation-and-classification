from __future__ import annotations
import numpy as np
import pytest
def test_native_extractor_reports_known_shape_measures(services, reference_mask):
    image = np.where(reference_mask > 0, 0.8, 0.1).astype(np.float32)
    payload = services["radiomics"].extract(image, reference_mask, spacing=None).to_dict()
    assert payload["available"] is True
    assert payload["feature_count"] > 20
    assert payload["features"]["shape_pixel_area"] == pytest.approx(float(reference_mask.sum()), rel=1e-7)
    assert payload["spacing_available"] is False
def test_physical_measurements_need_spacing(services, reference_mask):
    image = np.where(reference_mask > 0, 0.8, 0.1).astype(np.float32)
    without = services["radiomics"].extract(image, reference_mask, spacing=None).to_dict()
    with_spacing = services["radiomics"].extract(image, reference_mask, spacing=(2.0, 2.0)).to_dict()
    assert without["spacing_available"] is False
    assert with_spacing["spacing_available"] is True
    assert any(name.startswith("shape_pixel") for name in without["features"])
    assert "shape_pixel_area" in without["features"]
def test_empty_mask_yields_unavailable_not_zero(services):
    empty = np.zeros((32, 32), dtype=np.float32)
    payload = services["radiomics"].extract(np.zeros((32, 32), dtype=np.float32), empty, spacing=None).to_dict()
    assert payload["available"] is False
    assert payload["unavailable_reason"]
    assert payload["features"] == {}
def test_shape_mismatch_is_refused(services):
    payload = services["radiomics"].extract(np.zeros((16, 16), dtype=np.float32), np.ones((8, 8)), spacing=None).to_dict()
    assert payload["available"] is False
    assert "grid" in payload["unavailable_reason"]
def test_features_are_finite_numbers(services, reference_mask):
    image = np.where(reference_mask > 0, 0.7, 0.2).astype(np.float32)
    payload = services["radiomics"].extract(image, reference_mask, spacing=(1.0, 1.0)).to_dict()
    assert payload["available"] is True
    numeric = {name: value for name, value in payload["features"].items() if isinstance(value, (int, float))}
    assert len(numeric) >= 40, "the extractor should report the full feature set it implements"
    for name, value in numeric.items():
        assert np.isfinite(value), f"{name} is not finite"
    unavailable = [name for name, value in payload["features"].items() if value is None]
    for name in unavailable:
        assert name.startswith("shape_") or name.startswith("diagnostics_"), (
            f"{name} is missing although it is not a spacing-dependent shape feature"
        )
    diagnostics = {name: value for name, value in payload["features"].items() if name.startswith("diagnostics_")}
    assert diagnostics.get("diagnostics_dimensionality") == "2D"
def test_backend_status_declares_which_extractor_runs(services):
    status = services["radiomics"].backend_status()
    assert status["selected_backend"] in {"pyradiomics", "native"}
    assert status["note"]
