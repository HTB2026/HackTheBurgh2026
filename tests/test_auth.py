"""Auth0 gate on write endpoints, with tokens signed by a local test key."""

import os
import tempfile
import time

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa

os.environ.setdefault("EARWITNESS_CASES", tempfile.mkdtemp(prefix="ew-auth-"))

from fastapi.testclient import TestClient  # noqa: E402

from earwitness.server import app as server  # noqa: E402
from earwitness.server import auth  # noqa: E402

DOMAIN = "earwitness-test.eu.auth0.com"
AUDIENCE = "https://earwitness.api"
KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)
OTHER_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("AUTH0_DOMAIN", DOMAIN)
    monkeypatch.setenv("AUTH0_AUDIENCE", AUDIENCE)
    monkeypatch.setenv("AUTH0_CLIENT_ID", "test-client")
    monkeypatch.setattr(auth, "_signing_key", lambda config, token: KEY.public_key())

    # No `with`: the lifespan (database startup) is not needed for these checks.
    return TestClient(server.app)


def token(key=KEY, **overrides) -> str:
    now = int(time.time())
    claims = {
        "iss": f"https://{DOMAIN}/",
        "aud": AUDIENCE,
        "sub": "auth0|tester",
        "iat": now,
        "exp": now + 600,
        **overrides,
    }

    return jwt.encode(claims, key, algorithm="RS256")


def from_path(client, bearer: str | None):
    headers = {"Authorization": f"Bearer {bearer}"} if bearer else {}

    return client.post(
        "/api/cases/from-path",
        json={"path": "/nonexistent/clip.mp4"},
        headers=headers,
    )


def test_write_needs_login(client):
    response = from_path(client, None)

    assert response.status_code == 401
    assert response.headers["www-authenticate"].startswith("Bearer")


def test_valid_token_passes_the_gate(client):
    # Past the gate the handler runs: the made-up path is then a 400, not a 401.
    assert from_path(client, token()).status_code == 400


@pytest.mark.parametrize(
    "bad",
    [
        {"key": OTHER_KEY},
        {"aud": "https://someone-else.api"},
        {"iss": "https://evil.example/"},
        {"exp": int(time.time()) - 60},
    ],
    ids=["wrong-key", "wrong-audience", "wrong-issuer", "expired"],
)
def test_bad_tokens_are_refused(client, bad):
    key = bad.pop("key", KEY)

    assert from_path(client, token(key, **bad)).status_code == 401


def test_delete_and_rerun_are_gated(client):
    assert client.delete("/api/cases/nope").status_code == 401
    assert client.post("/api/cases/nope/rerun").status_code == 401
    assert (
        client.delete(
            "/api/cases/nope", headers={"Authorization": f"Bearer {token()}"}
        ).status_code
        == 200
    )


def test_reading_stays_open(client):
    assert client.get("/api/cases").status_code == 200


def test_unconfigured_server_refuses_writes(client, monkeypatch):
    monkeypatch.delenv("AUTH0_DOMAIN")

    assert from_path(client, token()).status_code == 503
