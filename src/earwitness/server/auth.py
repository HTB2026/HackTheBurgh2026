"""Auth0 login for actions that change evidence (upload, rerun, delete).

Viewing stays open. Write endpoints depend on ``require_user``, which checks an
Auth0 access token (RS256) for the API named by AUTH0_AUDIENCE, issued by
AUTH0_DOMAIN. If Auth0 is not configured, write endpoints refuse with 503
rather than silently letting anyone in.

Each case belongs to whoever uploaded it: only they, or an admin listed in
EARWITNESS_ADMINS (Auth0 user ids or verified emails, comma-separated), may
rerun or delete it. Access tokens carry only the user id, so the display name
comes from Auth0's /userinfo endpoint (cached per user). Pages are public, so
what a case shows of its owner is a name, a picture and a hashed id, never the
email or the raw Auth0 id.
"""

import asyncio
import hashlib
import os
import time
from dataclasses import dataclass
from dataclasses import field
from typing import Any

import httpx
import jwt
from fastapi import Header
from fastapi import HTTPException


@dataclass(frozen=True, slots=True)
class Settings:
    domain: str
    audience: str
    client_id: str

    @property
    def issuer(self) -> str:
        return f"https://{self.domain}/"


@dataclass(frozen=True, slots=True)
class User:
    sub: str
    claims: dict[str, Any]
    profile: dict[str, Any] = field(default_factory=dict)

    @property
    def id(self) -> str:
        return public_id(self.sub)

    @property
    def name(self) -> str:
        return display_name(self.profile)

    @property
    def admin(self) -> bool:
        admins = {
            a.strip().lower()
            for a in os.environ.get("EARWITNESS_ADMINS", "").split(",")
            if a.strip()
        }
        email = str(self.profile.get("email") or "").lower()
        verified = email and self.profile.get("email_verified") is True

        return self.sub.lower() in admins or bool(verified and email in admins)

    def public(self) -> dict[str, Any]:
        """What a public page may show about this user."""
        return {
            "id": self.id,
            "name": self.name,
            "picture": self.profile.get("picture"),
        }


_jwks: dict[str, jwt.PyJWKClient] = {}
_profiles: dict[str, tuple[float, dict[str, Any]]] = {}
_PROFILE_TTL = 3600.0


def public_id(sub: str) -> str:
    """A stable id for a user that does not reveal their Auth0 / Google id."""
    return hashlib.sha256(sub.encode()).hexdigest()[:16]


def display_name(profile: dict[str, Any]) -> str:
    """A name to show publicly; never the email address."""
    for key in ("name", "nickname"):
        value = str(profile.get(key) or "").strip()

        if value and "@" not in value:
            return value

    email = str(profile.get("email") or "")

    return email.partition("@")[0] or "Signed-in user"


def settings() -> Settings | None:
    domain = os.environ.get("AUTH0_DOMAIN", "").strip().removeprefix("https://")
    audience = os.environ.get("AUTH0_AUDIENCE", "").strip()
    client_id = os.environ.get("AUTH0_CLIENT_ID", "").strip()

    if not (domain and audience and client_id):
        return None

    return Settings(domain.rstrip("/"), audience, client_id)


def public_config() -> dict[str, str] | None:
    """What the browser needs to start a login (all public values)."""
    config = settings()

    if config is None:
        return None

    return {
        "domain": config.domain,
        "client_id": config.client_id,
        "audience": config.audience,
    }


def _signing_key(config: Settings, token: str) -> Any:
    client = _jwks.setdefault(
        config.domain,
        jwt.PyJWKClient(f"https://{config.domain}/.well-known/jwks.json"),
    )

    return client.get_signing_key_from_jwt(token).key


def verify(config: Settings, token: str) -> User:
    key = _signing_key(config, token)
    claims = jwt.decode(
        token,
        key,
        algorithms=["RS256"],
        audience=config.audience,
        issuer=config.issuer,
        options={"require": ["exp", "iat", "sub"]},
    )

    return User(
        sub=claims["sub"], claims=claims, profile=profile(config, token, claims["sub"])
    )


def _fetch_profile(config: Settings, token: str) -> dict[str, Any]:
    response = httpx.get(
        f"https://{config.domain}/userinfo",
        headers={"Authorization": f"Bearer {token}"},
        timeout=5.0,
    )
    response.raise_for_status()

    return response.json()


def profile(config: Settings, token: str, sub: str) -> dict[str, Any]:
    """Name, email and picture from Auth0, cached; empty if Auth0 will not say."""
    cached = _profiles.get(sub)

    if cached is not None and time.monotonic() - cached[0] < _PROFILE_TTL:
        return cached[1]

    try:
        found = _fetch_profile(config, token)
    except httpx.HTTPError, ValueError:
        # Not worth refusing the action over: the user is still verified.
        return cached[1] if cached is not None else {}

    if found.get("sub") != sub:
        return {}

    _profiles[sub] = (time.monotonic(), found)

    return found


def can_change(user: User, case: Any) -> bool:
    """Whether ``user`` may rerun or delete ``case``.

    Cases from before ownership was recorded have no owner and stay open to
    anyone signed in, so nothing becomes impossible to clean up.
    """
    owner = case.options.get("uploaded_by")

    return owner is None or owner == user.sub or user.admin


async def require_user(authorization: str | None = Header(default=None)) -> User:
    config = settings()

    if config is None:
        raise HTTPException(
            503, "login is not configured on this server (set AUTH0_* in .env)"
        )

    scheme, _, token = (authorization or "").partition(" ")

    if scheme.lower() != "bearer" or not token:
        raise HTTPException(
            401, "log in to do this", headers={"WWW-Authenticate": "Bearer"}
        )

    try:
        # The JWKS fetch is blocking network I/O (cached after the first call).
        return await asyncio.to_thread(verify, config, token.strip())
    except (jwt.PyJWTError, jwt.PyJWKClientError) as error:
        raise HTTPException(
            401,
            f"invalid login token: {error}",
            headers={"WWW-Authenticate": 'Bearer error="invalid_token"'},
        )
