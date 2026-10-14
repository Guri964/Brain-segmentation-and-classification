from __future__ import annotations
import io
def _analysis(client, brain_png) -> dict:
    response = client.post(
        "/api/analysis",
        data={
            "analysis_type": "single_channel",
            "scan_identifier": "REPORT-TEST",
            "enable_radiomics": "true",
            "files": (io.BytesIO(brain_png.read_bytes()), brain_png.name),
        },
        content_type="multipart/form-data",
    )
    assert response.status_code == 201
    return response.get_json()
def test_report_is_generated_and_served(client, brain_png):
    payload = _analysis(client, brain_png)
    report = payload.get("report")
    assert report is not None
    if not report.get("available"):
        assert report.get("reason")
        return
    assert report["size_bytes"] > 10_000
    download = client.get(f"/api/reports/{report['report_id']}")
    assert download.status_code == 200
    assert download.data.startswith(b"%PDF")
def test_report_lists_and_regenerates(client, brain_png):
    payload = _analysis(client, brain_png)
    analysis_id = payload["analysis_id"]
    listing = client.get("/api/reports")
    assert any(item["analysis_id"] == analysis_id for item in listing.get_json()["items"])
    regenerated = client.post(f"/api/reports/analysis/{analysis_id}")
    assert regenerated.status_code in {200, 201}
    assert regenerated.get_json()["size_bytes"] > 0
