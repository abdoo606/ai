"""HTTP boundary, authentication, validation, and rate-limit tests."""

from fastapi.testclient import TestClient

from zehnora.config import load_settings
from zehnora.server import create_app


class FakeTools:
    def __init__(self):
        self.tools = {}
        self.calls = []

    def call(self, name, arguments):
        self.calls.append((name, arguments))
        if name == "list_topics":
            return "physics"
        return f"{name} result"


class FakeAssistant:
    def __init__(self):
        self.tools = FakeTools()
        self.messages = []
        self.resets = 0

    def ask(self, message):
        self.messages.append(message)
        return f"reply: {message}"

    def reset(self):
        self.resets += 1


def settings(tmp_path, **extra):
    env = {
        "ZEHNORA_HOME": str(tmp_path / "home"),
        "ZEHNORA_WORKSPACE": str(tmp_path / "workspace"),
        **extra,
    }
    return load_settings(env)


def test_health_is_public_and_security_headers_are_present(tmp_path):
    client = TestClient(create_app(FakeAssistant(), settings(tmp_path, ZEHNORA_ACCESS_TOKEN="secret")))
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["cache-control"] == "no-store"


def test_optional_authentication_protects_api_and_accepts_two_header_styles(tmp_path):
    assistant = FakeAssistant()
    client = TestClient(create_app(assistant, settings(tmp_path, ZEHNORA_ACCESS_TOKEN="secret")))
    assert client.get("/api/status").status_code == 401
    assert client.get("/api/status", headers={"X-Zehnora-Token": "wrong"}).status_code == 401
    status = client.get("/api/status", headers={"Authorization": "Bearer secret"})
    assert status.status_code == 200
    assert status.json()["authentication_enabled"] is True
    response = client.post("/api/chat", json={"message": "hello"},
                           headers={"X-Zehnora-Token": "secret"})
    assert response.json() == {"reply": "reply: hello"}


def test_authentication_failures_are_rate_limited(tmp_path):
    client = TestClient(create_app(FakeAssistant(), settings(tmp_path, ZEHNORA_ACCESS_TOKEN="secret")))
    for _ in range(10):
        assert client.get("/api/status", headers={"X-Zehnora-Token": "wrong"}).status_code == 401
    blocked = client.get("/api/status", headers={"X-Zehnora-Token": "wrong"})
    assert blocked.status_code == 429
    assert blocked.headers["retry-after"] == "60"


def test_no_authentication_is_required_when_token_is_not_configured(tmp_path):
    response = TestClient(create_app(FakeAssistant(), settings(tmp_path))).get("/api/status")
    assert response.status_code == 200
    assert response.json()["authentication_enabled"] is False


def test_request_validation_rejects_empty_oversized_and_invalid_learning_jobs(tmp_path):
    client = TestClient(create_app(FakeAssistant(), settings(tmp_path)))
    assert client.post("/api/chat", json={"message": ""}).status_code == 422
    assert client.post("/api/chat", json={"message": "x" * 10_001}).status_code == 422
    assert client.post("/api/learn", json={"topic": "physics", "max_sources": 101}).status_code == 422
    assert client.post("/api/learn", json={"topic": "physics", "depth": 0}).status_code == 422
    assert client.get("/api/recall?q=").status_code == 422


def test_learning_endpoint_passes_bounded_arguments_to_tool(tmp_path):
    assistant = FakeAssistant()
    client = TestClient(create_app(assistant, settings(tmp_path)))
    response = client.post("/api/learn", json={"topic": "physics", "max_sources": 10, "depth": 3})
    assert response.status_code == 200
    assert assistant.tools.calls[-1] == (
        "learn_deep", {"topic": "physics", "max_sources": 10, "depth": 3},
    )


def test_chat_rate_limit_is_enforced_per_client(tmp_path):
    client = TestClient(create_app(
        FakeAssistant(), settings(tmp_path, ZEHNORA_RATE_LIMIT_PER_MINUTE="1")
    ))
    assert client.post("/api/chat", json={"message": "first"}).status_code == 200
    limited = client.post("/api/chat", json={"message": "second"})
    assert limited.status_code == 429
    assert limited.headers["retry-after"] == "60"


def test_reset_calls_assistant_reset(tmp_path):
    assistant = FakeAssistant()
    client = TestClient(create_app(assistant, settings(tmp_path)))
    assert client.post("/api/reset", json={}).status_code == 200
    assert assistant.resets == 1
