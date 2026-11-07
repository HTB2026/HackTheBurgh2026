"""Auth0 login for actions that change evidence (upload, rerun, delete).

Viewing stays open. Write endpoints depend on ``require_user``, which checks an
Auth0 access token (RS256) for the API named by AUTH0_AUDIENCE, issued by
AUTH0_DOMAIN. If Auth0 is not configured, write endpoints refuse with 503
rather than silently letting anyone in.
"""

import asyncio
import os
from dataclasses import dataclass
from typing import Any

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


_jwks: dict[str, jwt.PyJWKClient] = {}


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

    return User(sub=claims["sub"], claims=claims)


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
