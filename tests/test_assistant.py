from __future__ import annotations
import io
def _analysis(client, brain_png) -> dict:
    response = client.post(
        "/api/analysis",
        data={
            "analysis_type": "single_channel",
            "enable_radiomics": "true",
            "files": (io.BytesIO(brain_png.read_bytes()), brain_png.name),
        },
        content_type="multipart/form-data",
    )
    assert response.status_code == 201
    return response.get_json()
def test_assistant_quotes_the_analysis(client, brain_png):
    payload = _analysis(client, brain_png)
    analysis_id = payload["analysis_id"]
    answer = client.post(
        "/api/assistant/ask",
        json={"analysis_id": analysis_id, "question": "How large is the segmented region?"},
    )
    assert answer.status_code == 200
    body = answer.get_json()
    assert body["analysis_id"] == analysis_id
    assert body["backend"] in {"local", "llm"}
    assert body["sources"] or body["answer"]
    assert body["disclaimers"]
def test_assistant_without_an_analysis_stays_general(client):
    answer = client.post("/api/assistant/ask", json={"question": "What does the platform do?"})
    assert answer.status_code == 200
    assert answer.get_json()["analysis_id"] is None
def test_assistant_history_is_recorded(client, brain_png):
    payload = _analysis(client, brain_png)
    analysis_id = payload["analysis_id"]
    client.post("/api/assistant/ask", json={"analysis_id": analysis_id, "question": "Which models ran?"})
    history = client.get(f"/api/assistant/history/{analysis_id}")
    assert history.status_code == 200
    messages = history.get_json()["messages"]
    assert any(message["role"] == "user" for message in messages)
    assert any(message["role"] == "assistant" for message in messages)
