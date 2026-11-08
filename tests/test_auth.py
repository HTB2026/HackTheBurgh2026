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
    # No network: /userinfo knows nothing unless a test says otherwise.
    monkeypatch.setattr(auth, "_fetch_profile", lambda config, token: {})
    monkeypatch.setattr(auth, "_profiles", {})

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


def upload(client, bearer: str | None):
    headers = {"Authorization": f"Bearer {bearer}"} if bearer else {}

    # An unsupported AI provider makes a request that passes the gate fail with
    # 400 before any case is created.
    return client.post(
        "/api/cases?ai_provider=nonexistent",
        files={"file": ("clip.mp4", b"not a video")},
        headers=headers,
    )


def test_write_needs_login(client):
    response = upload(client, None)

    assert response.status_code == 401
    assert response.headers["www-authenticate"].startswith("Bearer")


def test_valid_token_passes_the_gate(client):
    # Past the gate the handler runs: the unknown AI provider is then a 400, not a 401.
    assert upload(client, token()).status_code == 400


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

    assert upload(client, token(key, **bad)).status_code == 401


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

    assert upload(client, token()).status_code == 503


ALICE = "auth0|alice"
BOB = "auth0|bob"


def bearer(sub: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token(sub=sub)}"}


@pytest.fixture
def profiles(monkeypatch):
    """Auth0 /userinfo answers, by user id, without the network."""
    known = {
        ALICE: {"sub": ALICE, "name": "Alice Example", "email": "alice@example.com",
                "email_verified": True, "picture": "https://example.com/a.png"},
        BOB: {"sub": BOB, "name": "bob@example.com", "nickname": "bob",
              "email": "bob@example.com", "email_verified": False},
    }  # fmt: skip
    monkeypatch.setattr(auth, "_profiles", {})
    monkeypatch.setattr(
        auth,
        "_fetch_profile",
        lambda config, raw: known[
            jwt.decode(raw, options={"verify_signature": False})["sub"]
        ],
    )
    monkeypatch.delenv("EARWITNESS_ADMINS", raising=False)

    return known


def owned_case(sub: str | None):
    """A case on disk as an upload by ``sub`` would leave it (no analysis run)."""
    options = {"uploaded_by": sub} if sub else {}
    case = server.store.create("clip.mp4", options)

    if sub:
        case.owner = {
            "id": auth.public_id(sub),
            "name": "Alice Example",
            "picture": None,
        }
        case.record("uploaded", case.owner)

    server.store.save_meta(case)

    return case


def test_me_names_the_user_without_their_email(client, profiles):
    response = client.get("/api/me", headers=bearer(ALICE))

    assert response.status_code == 200
    assert response.json() == {
        "id": auth.public_id(ALICE),
        "name": "Alice Example",
        "picture": "https://example.com/a.png",
        "admin": False,
    }
    # A name that is really an email address is not shown.
    assert client.get("/api/me", headers=bearer(BOB)).json()["name"] == "bob"


def test_only_the_owner_can_delete(client, profiles):
    case = owned_case(ALICE)

    refused = client.delete(f"/api/cases/{case.id}", headers=bearer(BOB))

    assert refused.status_code == 403
    assert "Alice Example" in refused.json()["detail"]
    assert case.id in server.store.cases

    assert client.delete(f"/api/cases/{case.id}", headers=bearer(ALICE)).json() == {
        "deleted": True
    }


def test_only_the_owner_can_rerun(client, profiles):
    case = owned_case(ALICE)

    assert (
        client.post(f"/api/cases/{case.id}/rerun", headers=bearer(BOB)).status_code
        == 403
    )
    assert [h["action"] for h in case.history] == ["uploaded"]


def test_admins_can_change_any_case(client, profiles, monkeypatch):
    case = owned_case(ALICE)
    # By id, or by a verified email (Bob's is unverified, so naming it is not enough).
    monkeypatch.setenv("EARWITNESS_ADMINS", "bob@example.com")

    assert (
        client.delete(f"/api/cases/{case.id}", headers=bearer(BOB)).status_code == 403
    )

    monkeypatch.setenv("EARWITNESS_ADMINS", f"someone@else.org, {BOB}")

    assert (
        client.delete(f"/api/cases/{case.id}", headers=bearer(BOB)).status_code == 200
    )


def test_cases_from_before_ownership_stay_open(client, profiles):
    case = owned_case(None)

    assert (
        client.delete(f"/api/cases/{case.id}", headers=bearer(BOB)).status_code == 200
    )


def test_owner_and_record_survive_a_restart(client, profiles):
    case = owned_case(ALICE)
    # A case with no evidence file counts as a cut-off upload; give it one.
    open(case.evidence_path, "wb").close()
    reloaded = type(server.store)(server.store.root)

    again = reloaded.cases[case.id]
    assert again.owner == case.owner
    assert again.summary()["history"][0]["action"] == "uploaded"
    assert "uploaded_by" not in again.summary()


def test_legacy_owner_without_a_name_is_still_the_owner(client, profiles):
    case = server.store.create("clip.mp4", {"uploaded_by": ALICE})
    open(case.evidence_path, "wb").close()
    reloaded = type(server.store)(server.store.root)
    server.store.cases[case.id] = reloaded.cases[case.id]

    assert reloaded.cases[case.id].owner == {
        "id": auth.public_id(ALICE),
        "name": None,
        "picture": None,
    }
    refused = client.delete(f"/api/cases/{case.id}", headers=bearer(BOB))
    assert refused.status_code == 403
    assert "the person who uploaded it" in refused.json()["detail"]


def test_userinfo_failure_does_not_block_the_owner(client, profiles, monkeypatch):
    case = owned_case(ALICE)

    def down(config, raw):
        raise auth.httpx.ConnectError("auth0 down")

    monkeypatch.setattr(auth, "_fetch_profile", down)

    assert (
        client.delete(f"/api/cases/{case.id}", headers=bearer(ALICE)).status_code == 200
    )
