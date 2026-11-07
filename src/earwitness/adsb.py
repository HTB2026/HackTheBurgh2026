import asyncio
import json
import time
from typing import Protocol

import httpx

from earwitness import logging
from earwitness._common import ForensicError
from earwitness._common import OnSuccess
from earwitness.flights import Origin

logger = logging.get_logger(__name__)

_USER_AGENT = "earwitness/0.1 (hackathon research tool)"
_TIMEOUT_S = 10.0


class FeedError(ForensicError):
    UNAVAILABLE = "unavailable"
    BAD_STATUS = "bad_status"


class ImplementsAircraftFeed(Protocol):
    async def fetch(
        self, origin: Origin, radius_nm: int
    ) -> OnSuccess[dict[str, object]]: ...


class PointFeed:
    """Any ADSBExchange-v2-compatible /v2/point endpoint (airplanes.live, adsb.lol)."""

    __slots__ = ("_base_url", "_client")

    def __init__(self, base_url: str, client: httpx.AsyncClient) -> None:
        self._base_url = base_url
        self._client = client

    async def fetch(
        self, origin: Origin, radius_nm: int
    ) -> OnSuccess[dict[str, object]]:
        point = f"{origin.latitude}/{origin.longitude}/{radius_nm}"
        url = f"{self._base_url}/v2/point/{point}"

        try:
            response = await self._client.get(url)
        # NOTE: httpx has no non-raising request API.
        except httpx.HTTPError:
            return FeedError.UNAVAILABLE

        if response.status_code != 200:
            return FeedError.BAD_STATUS

        return response.json()


async def record(
    feed: ImplementsAircraftFeed,
    path: str,
    origin: Origin,
    *,
    radius_nm: int,
    interval_s: float,
) -> None:
    """Append one JSON line per poll until cancelled."""
    while True:
        started = time.monotonic()
        response = await feed.fetch(origin, radius_nm)

        if isinstance(response, ForensicError):
            logger.warning("Aircraft feed poll failed.", extra={"error": str(response)})
        else:
            line = json.dumps({"poll_time": time.time(), "response": response})
            await asyncio.to_thread(_append, path, line)
            logger.info(
                "Recorded aircraft.", extra={"count": len(response.get("ac", []))}
            )

        await asyncio.sleep(max(0.0, interval_s - (time.monotonic() - started)))


def _append(path: str, line: str) -> None:
    with open(path, "a") as handle:
        handle.write(line + "\n")


def default(*, base_url: str = "https://api.airplanes.live") -> PointFeed:
    client = httpx.AsyncClient(
        headers={"User-Agent": _USER_AGENT},
        timeout=_TIMEOUT_S,
    )

    return PointFeed(base_url, client)
