import asyncio
import json
import math
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from earwitness._common import ForensicError
from earwitness._common import OnSuccess

_EARTH_RADIUS_M = 6_371_000.0
_FEET_TO_M = 0.3048
_MAX_GAP_S = 60.0


class FlightError(ForensicError):
    EMPTY_LOG = "empty_log"


@dataclass(frozen=True, slots=True)
class Origin:
    latitude: float
    longitude: float


@dataclass(frozen=True, slots=True)
class Track:
    """Positions in local metres (east, north, up above the receiver's ground)."""

    identifier: str
    callsign: str | None
    times: np.ndarray
    positions: np.ndarray


def to_local(origin: Origin, latitude: float, longitude: float) -> tuple[float, float]:
    east = (
        math.radians(longitude - origin.longitude)
        * _EARTH_RADIUS_M
        * math.cos(math.radians(origin.latitude))
    )
    north = math.radians(latitude - origin.latitude) * _EARTH_RADIUS_M

    return east, north


def to_geographic(origin: Origin, east: float, north: float) -> tuple[float, float]:
    latitude = origin.latitude + math.degrees(north / _EARTH_RADIUS_M)
    longitude = origin.longitude + math.degrees(
        east / (_EARTH_RADIUS_M * math.cos(math.radians(origin.latitude))),
    )

    return latitude, longitude


def _altitude_m(aircraft: dict[str, object]) -> float | None:
    for key in ("alt_geom", "alt_baro"):
        value = aircraft.get(key)

        if isinstance(value, int | float):
            return float(value) * _FEET_TO_M

    return None


def _read_lines(path: str) -> list[str]:
    with open(path) as handle:
        return handle.readlines()


def _split_on_gaps(
    identifier: str, callsign: str | None, rows: np.ndarray
) -> list[Track]:
    breaks = np.flatnonzero(np.diff(rows[:, 0]) > _MAX_GAP_S) + 1
    tracks: list[Track] = []

    for number, part in enumerate(np.split(rows, breaks)):
        if len(part) < 2:
            continue

        tracks.append(
            Track(
                identifier=f"{identifier}#{number}" if number else identifier,
                callsign=callsign,
                times=part[:, 0],
                positions=part[:, 1:],
            ),
        )

    return tracks


async def load_log(
    path: str,
    origin: Origin,
    *,
    ground_height_m: float,
) -> OnSuccess[list[Track]]:
    """Read a JSONL log of ADSBExchange-v2-style responses into local tracks.

    Each position is timestamped at the response time minus its seen_pos age.
    ground_height_m is the receiver's ground height on the same datum as the
    aircraft altitudes (GNSS altitude where reported, otherwise pressure altitude).
    """
    lines = await asyncio.to_thread(_read_lines, path)
    samples: dict[str, dict[float, tuple[float, float, float]]] = defaultdict(dict)
    callsigns: dict[str, str] = {}

    for line in lines:
        entry = json.loads(line)
        response = entry["response"]
        now = response.get("now", entry["poll_time"] * 1000) / 1000

        for aircraft in response.get("ac", []):
            altitude = _altitude_m(aircraft)

            if "lat" not in aircraft or altitude is None:
                continue

            identifier = aircraft["hex"]
            moment = round(now - float(aircraft.get("seen_pos", 0.0)), 1)
            east, north = to_local(origin, aircraft["lat"], aircraft["lon"])
            samples[identifier][moment] = (east, north, altitude - ground_height_m)

            if flight := str(aircraft.get("flight", "")).strip():
                callsigns[identifier] = flight

    tracks: list[Track] = []

    for identifier, points in samples.items():
        rows = np.array([(t, *p) for t, p in sorted(points.items())])
        tracks.extend(_split_on_gaps(identifier, callsigns.get(identifier), rows))

    if not tracks:
        return FlightError.EMPTY_LOG

    return tracks
