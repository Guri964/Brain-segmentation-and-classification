from __future__ import annotations
import io
import json
def test_health_and_capabilities(client):
    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.get_json()["status"] == "ok"
    capabilities = client.get("/api/capabilities")
    assert capabilities.status_code == 200
    payload = capabilities.get_json()
    assert payload["models"]
    assert "classification" in payload and "ensemble" in payload
    assert payload["disclaimer"].startswith("Research prototype")
def test_model_endpoints(client):
    models = client.get("/api/models")
    assert models.status_code == 200
    assert models.get_json()["models"]
    health = client.get("/api/models/unet_dynamic_512/health")
    assert health.status_code == 200
    assert health.get_json()["verdict"]
    checks = client.post("/api/models/health-check", json={"force": False})
    assert checks.status_code == 200
    assert "responsive_models" in checks.get_json()
    missing = client.get("/api/models/does-not-exist")
    assert missing.status_code == 404
    assert missing.get_json()["error"]["code"]
def test_analysis_requires_a_file(client):
    response = client.post("/api/analysis", data={"analysis_type": "single_channel"})
    assert response.status_code == 422
    assert response.get_json()["error"]["code"] == "validation_error"
def test_analysis_rejects_unknown_extension(client):
    response = client.post(
        "/api/analysis",
        data={
            "analysis_type": "single_channel",
            "files": (io.BytesIO(b"not an image"), "scan.tiff"),
        },
        content_type="multipart/form-data",
    )
    assert response.status_code in {415, 422}
    assert response.get_json()["error"]["message"]
def test_four_channel_analysis_refuses_duplicated_channels(client, brain_png):
    raw = brain_png.read_bytes()
    data = {
        "analysis_type": "four_channel",
        "modalities_confirmed": "true",
        "files": [(io.BytesIO(raw), f"m{index}.png") for index in range(4)],
    }
    response = client.post("/api/analysis", data=data, content_type="multipart/form-data")
    assert response.status_code in {415, 422}
    message = response.get_json()["error"]["message"].lower()
    assert "same image" in message or "different" in message
def test_single_channel_analysis_end_to_end(client, brain_png):
    response = client.post(
        "/api/analysis",
        data={
            "analysis_type": "single_channel",
            "scan_identifier": "PYTEST-1",
            "enable_radiomics": "true",
            "enable_explainability": "true",
            "enable_ensemble": "true",
            "files": (io.BytesIO(brain_png.read_bytes()), brain_png.name),
        },
        content_type="multipart/form-data",
    )
    assert response.status_code == 201, response.get_data(as_text=True)
    payload = response.get_json()
    analysis_id = payload["analysis_id"]
    detail = client.get(f"/api/analysis/{analysis_id}")
    assert detail.status_code == 200
    body = detail.get_json()
    assert body["payload_source"]
    assert body["result"]["analysis_id"] == analysis_id
    result = body["result"]
    assert result["preprocessing"]["channels"] == 1
    assert result["inputs"]
    available = [model_id for model_id, entry in result["models"].items() if entry["available"]]
    assert available, "no model produced output for the test input"
    first = result["models"][available[0]]
    assert first["summary"]["foreground_pixels"] >= 0
    if first["summary"]["foreground_pixels"] == 0:
        assert first["summary"]["probability_mean_in_mask"] is None
    assert "ensemble" in result and isinstance(result["ensemble"]["reasons"], list)
    assert result["reference_mask"]["model_id"] in available or result["reference_mask"]["source"]
    assert result["model_comparison"]["state"] in {"unlabelled", "evaluated"}
    history = client.get("/api/history")
    assert history.status_code == 200
    assert any(item["analysis_id"] == analysis_id for item in history.get_json()["items"])
    statistics = client.get("/api/history/statistics")
    assert statistics.status_code == 200
    assert statistics.get_json()["totals"]["analyses"] >= 1
    export = client.get(f"/api/history/{analysis_id}/export")
    assert export.status_code == 200
    exported = json.loads(export.data.decode("utf-8"))
    assert exported["analysis_id"] == analysis_id
    assert exported["segmentation"]
    deleted = client.delete(f"/api/analysis/{analysis_id}")
    assert deleted.status_code in {200, 204}
    assert client.get(f"/api/analysis/{analysis_id}").status_code == 404
def test_unknown_analysis_is_404(client):
    response = client.get("/api/analysis/an-does-not-exist")
    assert response.status_code == 404
def test_evaluation_endpoint_reports_not_evaluated_honestly(services, client):
    stored = services["evaluation"].stored_results()
    response = client.get("/api/evaluation")
    assert response.status_code == 200
    payload = response.get_json()
    if stored["state"] != "evaluated":
        assert payload["evaluation"]["state"] == "not_evaluated"
        assert payload["evaluation"]["reason"]
        assert payload["evaluation"]["produced_by"].endswith("evaluate_models.py")
def test_radiomics_catalogue_and_missing_analysis(client):
    catalogue = client.get("/api/radiomics/catalogue")
    assert catalogue.status_code == 200
    assert catalogue.get_json()["categories"]
    missing = client.get("/api/radiomics/analysis/an-not-there")
    assert missing.status_code == 404
def test_assistant_answers_and_refuses(client, brain_png):
    status = client.get("/api/assistant/status")
    assert status.status_code == 200
    assert status.get_json()["backend"] in {"local", "llm"}
    refused = client.post("/api/assistant/ask", json={"question": "Is this tumour malignant?"})
    assert refused.status_code == 200
    body = refused.get_json()
    assert body["topic"] == "out_of_scope"
    answer = body["answer"].lower()
    assert "cannot" in answer or "not able" in answer or "outside" in answer
    empty = client.post("/api/assistant/ask", json={"question": ""})
    assert empty.status_code == 422
def test_reports_endpoint_shape(client):
    response = client.get("/api/reports")
    assert response.status_code == 200
    assert "items" in response.get_json()
def test_plan_endpoint_explains_what_would_run(client):
    response = client.post("/api/analysis/plan", json={"analysis_type": "single_channel", "enable_radiomics": True})
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["steps"]
    assert "runnable" in payload["steps"][0]
def test_error_payloads_are_structured(client):
    response = client.get("/api/analysis/an-not-there")
    body = response.get_json()
    assert set(body) == {"error"}
    assert {"code", "message"} <= set(body["error"])
