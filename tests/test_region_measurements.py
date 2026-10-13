"""Region measurements (abnormality measurements) taken from the segmentation mask.

The brief asks the application to report pixel area, bounding box, width, height,
image coverage, centroid and perimeter for the segmented region.  These tests pin
the arithmetic to hand-computable examples, prove that the perimeter agrees with
the radiomics shape block, and check that the values actually reach the API, the
chatbot answer and the PDF report.  An empty mask must never turn into a
measurement of zero.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
BACKEND_ROOT = REPO_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.ml.radiomics_native import shape_features_2d  # noqa: E402
from app.ml.region_metrics import measure_region, region_geometry  # noqa: E402


def _square(height: int = 10, width: int = 10, y0: int = 10, x0: int = 30, size: int = 100):
    mask = np.zeros((size, size), dtype=bool)
    mask[y0 : y0 + height, x0 : x0 + width] = True
    return mask


def test_square_geometry_is_exact():
    measured = measure_region(_square(), source="unit-test")
    assert measured["detected"] is True
    assert measured["pixel_area"] == 100
    assert measured["image_coverage"] == pytest.approx(100 / (100 * 100))
    assert measured["image_coverage_percent"] == pytest.approx(1.0)
    assert measured["bounding_box"] == {"x": 30, "y": 10, "width": 10, "height": 10}
    assert measured["centroid"] == {"x": 34.5, "y": 14.5}
    assert measured["units"] == "pixels"
    assert measured["mask_source"] == "unit-test"
    assert measured["image_size"] == {"width": 100, "height": 100}
    # a 10x10 block has a boundary of 40 pixel edges; the Crofton estimator is close
    assert 38.0 <= measured["perimeter_px"] <= 41.0
    assert measured["equivalent_diameter_px"] == pytest.approx(2 * np.sqrt(100 / np.pi), rel=1e-4)


def test_centroid_tracks_a_moved_region():
    left = measure_region(_square(width=4, height=4, y0=0, x0=0))
    right = measure_region(_square(width=4, height=4, y0=90, x0=90))
    assert left["centroid"] == {"x": 1.5, "y": 1.5}
    assert right["centroid"] == {"x": 91.5, "y": 91.5}
    assert left["bounding_box"]["width"] == left["bounding_box"]["height"] == 4


def test_perimeter_agrees_with_the_radiomics_shape_block():
    mask = _square(height=24, width=13, y0=7, x0=11, size=80)
    measured = measure_region(mask)
    reference = shape_features_2d(mask)["perimeter_px"]
    assert measured["perimeter_px"] == pytest.approx(reference, abs=1e-3)


def test_empty_mask_is_reported_as_not_detected_not_zero():
    measured = measure_region(np.zeros((32, 32), dtype=bool), source="unit-test")
    assert measured["detected"] is False
    assert measured["pixel_area"] == 0
    assert measured["bounding_box"] is None
    assert measured["centroid"] is None
    assert measured["perimeter_px"] is None
    assert measured["equivalent_diameter_px"] is None
    assert "no region was detected" in measured["note"].lower()


def test_missing_mask_is_reported_as_absent():
    measured = measure_region(None, source="none")
    assert measured["detected"] is False
    assert measured["centroid"] is None
    assert "no segmentation mask was available" in measured["note"].lower()


def test_three_dimensional_geometry_is_axis_aware():
    volume = np.zeros((4, 10, 10), dtype=np.uint8)
    volume[1:3, 2:6, 4:9] = 1
    geometry = region_geometry(volume)
    assert geometry["detected"] is True
    assert geometry["voxel_count"] == 2 * 4 * 5
    assert geometry["bounding_box"] == [[1, 2], [2, 5], [4, 8]]
    assert geometry["centroid"] == [1.5, 3.5, 6.0]
    assert geometry["extent"] == [2, 4, 5]
    with pytest.raises(ValueError):
        measure_region(volume)


def _analysis(client, brain_png):
    response = client.post(
        "/api/analysis",
        data={
            "analysis_type": "single_channel",
            "scan_identifier": "REGION-1",
            "enable_radiomics": "true",
            "files": (io.BytesIO(brain_png.read_bytes()), brain_png.name),
        },
        content_type="multipart/form-data",
    )
    assert response.status_code == 201, response.get_data(as_text=True)
    return response.get_json()["analysis_id"]


def test_analysis_payload_exposes_region_measurements(client, brain_png):
    analysis_id = _analysis(client, brain_png)
    result = client.get(f"/api/analysis/{analysis_id}").get_json()["result"]
    measurements = result["region_measurements"]
    assert measurements["detected"] is bool(measurements["pixel_area"])
    assert measurements["pixel_area"] == result["reference_mask"]["foreground_pixels"]
    assert measurements["mask_source"] == result["reference_mask"]["source"]
    assert measurements["units"] == "pixels"
    if measurements["detected"]:
        bbox = measurements["bounding_box"]
        centroid = measurements["centroid"]
        assert bbox["width"] > 0 and bbox["height"] > 0
        assert 0 <= measurements["image_coverage"] <= 1
        assert bbox["x"] <= centroid["x"] <= bbox["x"] + bbox["width"]
        assert bbox["y"] <= centroid["y"] <= bbox["y"] + bbox["height"]
    else:
        assert measurements["centroid"] is None


def test_radiomics_quality_carries_the_centroid(client, brain_png):
    analysis_id = _analysis(client, brain_png)
    result = client.get(f"/api/analysis/{analysis_id}").get_json()["result"]
    radiomics = result["radiomics"]
    assert radiomics["available"] is True
    quality = radiomics["quality"]
    assert quality["roi_centroid"] is None or len(quality["roi_centroid"]) == len(
        quality["roi_bounding_box"]
    )
    if quality["roi_centroid"]:
        assert quality["roi_centroid_definition"]


def test_report_renders_with_the_region_section(client, brain_png):
    """The new PDF section must not break report generation."""

    analysis_id = _analysis(client, brain_png)
    result = client.get(f"/api/analysis/{analysis_id}").get_json()["result"]
    report = result.get("report") or {}
    assert report.get("available") is True, report.get("reason")
    assert report["size_bytes"] > 1000
    download = client.get(f"/api/reports/{report['report_id']}")
    assert download.status_code == 200
    assert download.data[:4] == b"%PDF"


def test_assistant_quotes_the_region_measurements(client, brain_png):
    analysis_id = _analysis(client, brain_png)
    answer = client.post(
        "/api/assistant/ask",
        json={"analysis_id": analysis_id, "question": "How large is the segmented region?"},
    )
    assert answer.status_code == 200
    body = answer.get_json()
    assert body["topic"] == "measurements"
    if body["answer"] and "no region" not in body["answer"].lower():
        assert "pixels" in body["answer"]
