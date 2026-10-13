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


def test_assistant_answers_label_questions_and_still_refuses_clinical_ones(client, brain_png):
    payload = _analysis(client, brain_png)
    analysis_id = payload["analysis_id"]

    label_question = client.post(
        "/api/assistant/ask",
        json={"analysis_id": analysis_id, "question": "Which grade group did the classifier predict?"},
    )
    assert label_question.status_code == 200
    body = label_question.get_json()
    assert body["topic"] == "classification"
    # No classifier artefact is installed in the test deployment, so the answer must say so
    # rather than produce a label.
    assert "no class was predicted" in body["answer"]
    assert "inventing a result" in body["answer"]

    refusal = client.post(
        "/api/assistant/ask",
        json={"analysis_id": analysis_id, "question": "Is this tumour malignant, and what is the prognosis?"},
    )
    assert refusal.status_code == 200
    assert refusal.get_json()["topic"] == "out_of_scope"
