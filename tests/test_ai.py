"""Choosing the AI provider per case (OpenAI or Gemini), without network calls."""

import base64
import json
import os
import tempfile

import pytest

os.environ.setdefault("EARWITNESS_CASES", tempfile.mkdtemp(prefix="ew-ai-"))

from fastapi.testclient import TestClient  # noqa: E402

from earwitness import ai  # noqa: E402
from earwitness.ai import gemini  # noqa: E402
from earwitness.modules import brief  # noqa: E402
from earwitness.modules import geo  # noqa: E402
from earwitness.modules import vision  # noqa: E402
from earwitness.server import app as server  # noqa: E402
from earwitness.server import auth  # noqa: E402

KEYS = ("OPENAI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "EARWITNESS_AI_PROVIDER")


@pytest.fixture
def env(monkeypatch):
    for key in KEYS:
        monkeypatch.delenv(key, raising=False)

    return monkeypatch


def test_no_keys_means_no_ai(env):
    assert ai.available() == []
    assert ai.default() is None
    assert ai.for_case({"ai_provider": "openai"}) is None


def test_single_key_is_the_default(env):
    env.setenv("GEMINI_API_KEY", "g")

    assert ai.available() == ["gemini"]
    assert ai.for_case({}) is gemini
    # Asking for a provider the server has no key for falls back to the default.
    assert ai.choice({"ai_provider": "openai"}) == "gemini"


def test_google_api_key_also_enables_gemini(env):
    env.setenv("GOOGLE_API_KEY", "g")

    assert ai.available() == ["gemini"]


def test_case_choice_and_server_default(env):
    env.setenv("OPENAI_API_KEY", "o")
    env.setenv("GEMINI_API_KEY", "g")

    assert ai.default() == "openai"
    assert ai.choice({"ai_provider": "gemini"}) == "gemini"

    env.setenv("EARWITNESS_AI_PROVIDER", "Gemini")

    assert ai.default() == "gemini"
    assert ai.choice({"ai_provider": "openai"}) == "openai"
    assert ai.choice({"ai_provider": "nonsense"}) == "gemini"


def test_neutral_content_becomes_gemini_parts():
    jpeg = b"\xff\xd8\xff\xe0fake-jpeg"
    parts = gemini.parts_for_gemini(
        [
            {"type": "input_text", "text": "what is this?"},
            {
                "type": "input_image",
                "image_url": "data:image/jpeg;base64," + base64.b64encode(jpeg).decode(),
            },
        ]
    )

    assert parts[0].text == "what is this?"
    assert parts[1].inline_data.mime_type == "image/jpeg"
    assert parts[1].inline_data.data == jpeg

    with pytest.raises(ValueError):
        gemini.parts_for_gemini([{"type": "input_image", "image_url": "https://x/y.jpg"}])


def test_module_schemas_convert_for_gemini():
    from google.genai import types

    for schema in (
        vision._SCHEMA,
        geo._PROPOSE_SCHEMA,
        geo._VERIFY_SCHEMA,
        brief._SCHEMA,
        brief._DOSSIER_SCHEMA,
    ):
        converted = gemini.schema_for_gemini(schema)
        # Same schema (these use no nullable type lists) and the SDK accepts it.
        assert converted == schema
        types.GenerateContentConfig(
            response_mime_type="application/json", response_json_schema=converted
        )

    nullable = {"type": "object", "properties": {"a": {"type": ["string", "null"]}}}

    assert gemini.schema_for_gemini(nullable)["properties"]["a"] == {
        "anyOf": [{"type": "string"}, {"type": "null"}]
    }


@pytest.fixture
def client(env):
    env.setenv("OPENAI_API_KEY", "o")
    env.setenv("GEMINI_API_KEY", "g")
    env.setattr(server.store, "start", lambda case, only=None: None)
    server.app.dependency_overrides[auth.require_user] = lambda: auth.User(
        "auth0|tester", {}
    )

    yield TestClient(server.app)

    server.app.dependency_overrides.clear()


def test_health_lists_providers(client):
    body = client.get("/api/health").json()

    assert body["ai"] == {"providers": ["openai", "gemini"], "default": "openai"}


def test_upload_rejects_unknown_provider(client):
    response = client.post(
        "/api/cases",
        params={"ai_provider": "bogus"},
        files={"file": ("clip.mp4", b"not really a video")},
    )

    assert response.status_code == 400


def test_upload_and_rerun_store_the_choice(client):
    response = client.post(
        "/api/cases",
        params={"ai_provider": "gemini"},
        files={"file": ("clip.mp4", b"not really a video")},
    )
    assert response.status_code == 200, response.text
    case_id = response.json()["id"]
    assert response.json()["ai_provider"] == "gemini"

    response = client.post(f"/api/cases/{case_id}/rerun", params={"ai_provider": "openai"})
    assert response.status_code == 200, response.text
    assert response.json()["ai_provider"] == "openai"

    case = server.store.cases[case_id]

    with open(os.path.join(case.workdir, "case.json")) as handle:
        assert json.load(handle)["options"]["ai_provider"] == "openai"

    client.delete(f"/api/cases/{case_id}")


def _quota_error(delay: str = "27603s", per: str = "Day"):
    from google.genai import errors

    return errors.ClientError(
        429,
        {
            "error": {
                "code": 429,
                "message": f"Generate content requests per {per.lower()} limit exceeded.",
                "status": "RESOURCE_EXHAUSTED",
                "details": [
                    {
                        "@type": "type.googleapis.com/google.rpc.QuotaFailure",
                        "violations": [
                            {"quotaId": f"GenerateRequestsPer{per}PerProjectPerModel-FreeTier"}
                        ],
                    },
                    {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": delay},
                ],
            }
        },
    )


def test_daily_quota_moves_on_without_waiting(monkeypatch):
    """A spent daily quota is not retried; every model is tried once, then a plain message."""
    import asyncio

    calls: list[str] = []

    class Models:
        async def generate_content(self, model, contents, config):
            calls.append(model)
            raise _quota_error()

    class Client:
        class aio:
            models = Models()

    async def no_sleep(_):
        raise AssertionError("must not wait on a daily quota")

    monkeypatch.setattr(gemini, "_client", lambda: Client())
    monkeypatch.setattr(gemini.asyncio, "sleep", no_sleep)

    with pytest.raises(RuntimeError, match="Gemini quota used up.*about 8 h.*OpenAI"):
        asyncio.run(
            gemini._generate(["a", "b", "c"], instructions="x", parts=[], schema=None)
        )

    assert calls == ["a", "b", "c"]
    # A per-minute limit with a short wait is worth retrying.
    per_minute = _quota_error("12.5s", per="Minute")
    assert gemini._retry_after_s(per_minute) == 12.5
    assert not gemini._quota_spent(per_minute)


def test_working_model_still_falls_back(monkeypatch):
    monkeypatch.setattr(gemini, "_working_model", gemini._MODELS[1])

    assert gemini._models()[0] == gemini._MODELS[1]
    assert sorted(gemini._models()) == sorted(gemini._MODELS)
