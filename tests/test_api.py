from fastapi.testclient import TestClient

from dronedocs import api


def test_ask_endpoint(make_pipeline):
    api.set_pipeline(make_pipeline("Auto disarm after landing defaults to 2 s [px4.md#c000]."))
    client = TestClient(api.app)
    r = client.post("/ask", json={"question": "PX4 auto disarm timeout?", "include_debug": True})
    assert r.status_code == 200
    body = r.json()
    assert body["refused"] is False
    assert body["citations"][0]["source"] == "px4.md"
    assert "total" in body["latency_ms"] and body["debug"]["usage"]["prompt_tokens"] == 100


def test_ask_validation_error():
    client = TestClient(api.app)
    assert client.post("/ask", json={"question": ""}).status_code == 422


def test_health_and_ui():
    client = TestClient(api.app)
    h = client.get("/health").json()
    assert h["status"] in {"ok", "degraded"} and "index_ready" in h
    assert "DroneDocs" in client.get("/").text
