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
from earwitness.ai import ollama  # noqa: E402
from earwitness.modules import brief  # noqa: E402
from earwitness.modules import geo  # noqa: E402
from earwitness.modules import vision  # noqa: E402
from earwitness.server import app as server  # noqa: E402
from earwitness.server import auth  # noqa: E402

KEYS = (
    "OPENAI_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "EARWITNESS_OLLAMA_URL",
    "EARWITNESS_OLLAMA_MODEL",
    "EARWITNESS_AI_PROVIDER",
)


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


def _fake_ollama(monkeypatch, reply: dict, status: int = 200) -> list[dict]:
    """Route the Ollama provider's HTTP calls to a stand-in server; returns the requests."""
    import httpx

    seen: list[dict] = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append({"url": str(request.url), "body": json.loads(request.content)})

        return httpx.Response(status, json=reply)

    real = httpx.AsyncClient
    monkeypatch.setattr(
        ollama.httpx,
        "AsyncClient",
        lambda **kw: real(transport=httpx.MockTransport(handle), **kw),
    )

    return seen


def test_ollama_is_a_provider(env):
    env.setenv("EARWITNESS_OLLAMA_URL", "http://10.8.96.3:11434/")

    assert ai.available() == ["ollama"]
    assert ai.for_case({}) is ollama
    assert ollama.model() == "gemma4"


def test_ollama_falls_back_when_format_is_unsupported(env, tmp_path):
    """Ollama's MLX runner answers 501 to ``format``: the schema moves into the prompt."""
    import asyncio

    import httpx

    env.setenv("EARWITNESS_OLLAMA_URL", "http://vps:11434")
    env.setattr(ollama, "_constrained", None)
    seen: list[dict] = []

    def handle(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(body)

        if "format" in body:
            return httpx.Response(501, json={"error": "structured output is unavailable"})

        return httpx.Response(
            200, json={"message": {"content": 'Sure:\n```json\n{"summary": "ok"}\n```'}}
        )

    real = httpx.AsyncClient
    env.setattr(
        ollama.httpx, "AsyncClient", lambda **kw: real(transport=httpx.MockTransport(handle), **kw)
    )
    schema = {"type": "object", "properties": {"summary": {"type": "string"}}, "required": ["summary"]}
    ask = lambda: ollama.structured(  # noqa: E731
        str(tmp_path / str(len(seen))), instructions="Sum up.",
        content=[{"type": "input_text", "text": "x"}], schema=schema, name="n",
    )  # fmt: skip

    assert asyncio.run(ask())["summary"] == "ok"
    assert "format" in seen[0] and "format" not in seen[1]
    assert '"required": ["summary"]' in seen[1]["messages"][0]["content"]

    # The server's answer is remembered: no second 501 round trip.
    asyncio.run(ask())
    assert len(seen) == 3 and "format" not in seen[2]


def test_ollama_structured_request(env, tmp_path):
    import asyncio

    env.setenv("EARWITNESS_OLLAMA_URL", "http://10.8.96.3:11434/")
    env.setattr(ollama, "_constrained", None)
    seen = _fake_ollama(env, {"message": {"role": "assistant", "content": '{"summary": "a street"}'}})
    jpeg = base64.b64encode(b"\xff\xd8fake").decode()
    schema = {"type": "object", "properties": {"summary": {"type": "string"}}, "required": ["summary"]}

    result = asyncio.run(
        ollama.structured(
            str(tmp_path),
            instructions="Describe it.",
            content=[
                {"type": "input_text", "text": "Frame 1:"},
                {"type": "input_image", "image_url": "data:image/jpeg;base64," + jpeg},
            ],
            schema=schema,
            name="report",
        )
    )

    assert result == {"summary": "a street", "_model": "gemma4"}
    assert seen[0]["url"] == "http://10.8.96.3:11434/api/chat"
    body = seen[0]["body"]
    assert body["model"] == "gemma4"
    assert body["format"] == schema
    assert body["stream"] is False
    assert body["options"]["num_ctx"] >= 16384
    assert body["messages"][0] == {"role": "system", "content": "Describe it."}
    assert body["messages"][1]["images"] == [jpeg]
    assert body["messages"][1]["content"] == "Frame 1:\n\n[image 1]"

    # A second identical call is answered from the case's cache.
    asyncio.run(
        ollama.structured(
            str(tmp_path), instructions="Describe it.",
            content=[
                {"type": "input_text", "text": "Frame 1:"},
                {"type": "input_image", "image_url": "data:image/jpeg;base64," + jpeg},
            ],
            schema=schema, name="report",
        )
    )  # fmt: skip
    assert len(seen) == 1


def test_ollama_missing_model_is_explained(env, tmp_path):
    import asyncio

    env.setenv("EARWITNESS_OLLAMA_URL", "http://vps:11434")
    env.setenv("EARWITNESS_OLLAMA_MODEL", "gemma4:27b")
    env.setattr(ollama, "_constrained", None)
    _fake_ollama(env, {"error": "model 'gemma4:27b' not found"}, status=404)

    with pytest.raises(RuntimeError, match="ollama pull gemma4:27b"):
        asyncio.run(
            ollama.structured(
                str(tmp_path), instructions="x",
                content=[{"type": "input_text", "text": "y"}], schema={"type": "object"}, name="n",
            )
        )  # fmt: skip


def test_ollama_audio_is_sent_as_wav(env, tmp_path):
    """Ollama rejects MP3 ("unrecognized audio format"): the transcript converts to WAV."""
    import asyncio
    import subprocess

    mp3 = tmp_path / "speech.mp3"
    subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
         str(mp3)],
        check=True,
    )  # fmt: skip
    env.setenv("EARWITNESS_OLLAMA_URL", "http://vps:11434")
    env.setattr(ollama, "_constrained", True)
    seen = _fake_ollama(env, {"message": {"content": '{"segments": [{"start": 0, "end": 1, "text": "hello"}]}'}})

    result = asyncio.run(ollama.transcribe(str(tmp_path), str(mp3)))

    assert result["text"] == "hello" and result["model"] == "gemma4"
    audio = base64.b64decode(seen[0]["body"]["messages"][1]["images"][0])
    assert audio[:4] == b"RIFF" and audio[8:12] == b"WAVE"

    with pytest.raises(RuntimeError, match="only one model"):
        asyncio.run(ollama.transcribe(str(tmp_path), str(mp3), "check"))
