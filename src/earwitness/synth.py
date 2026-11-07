import asyncio
import json
import os
from dataclasses import dataclass
from datetime import datetime
from datetime import timedelta
from datetime import timezone

import numpy as np
from scipy import signal
from scipy.io import wavfile

from earwitness import doppler
from earwitness import flights
from earwitness import grid
from earwitness.flights import Origin
from earwitness.flights import Track

_SAMPLE_RATE = 16_000
_DURATION_S = 300
_GROUND_HEIGHT_M = 100.0
_FEET_TO_M = 0.3048
_ORIGIN = Origin(55.9533, -3.1883)
_TRUTH = np.array([1_234.0, -567.0, 0.0])
_START = datetime(2026, 11, 7, 14, 0, tzinfo=timezone.utc)


@dataclass(frozen=True, slots=True)
class Aircraft:
    identifier: str
    callsign: str
    track: Track
    fundamental_hz: float
    harmonics: int
    audible: bool


def _straight(
    identifier: str, anchor_time: float, anchor: np.ndarray, velocity: np.ndarray
) -> Track:
    times = np.arange(anchor_time - 900.0, anchor_time + 900.0, 0.5)
    positions = anchor[None, :] + (times - anchor_time)[:, None] * velocity[None, :]

    return Track(identifier, None, times, positions)


def _scenario() -> list[Aircraft]:
    start = _START.timestamp()
    east, north = _TRUTH[0], _TRUTH[1]

    return [
        Aircraft(
            "400a01",
            "LOG24",
            _straight(
                "400a01",
                start + 100,
                np.array([east, north + 1_500, 600.0]),
                np.array([-75.0, 0.0, -1.0]),
            ),
            fundamental_hz=85.0,
            harmonics=6,
            audible=True,
        ),
        Aircraft(
            "400b02",
            "EZY61",
            _straight(
                "400b02",
                start + 210,
                np.array([east + 900, north, 450.0]),
                np.array([0.0, -70.0, 2.5]),
            ),
            fundamental_hz=410.0,
            harmonics=2,
            audible=True,
        ),
        Aircraft(
            "4ca0c3",
            "RYR9H",
            _straight(
                "4ca0c3",
                start + 150,
                np.array([-8_000.0, 6_000.0, 10_500.0]),
                np.array([160.0, 160.0, 0.0]),
            ),
            fundamental_hz=300.0,
            harmonics=1,
            audible=False,
        ),
    ]


def _grid_walk(seconds: int, rng: np.random.Generator) -> np.ndarray:
    values = np.empty(seconds)
    values[0] = 50.0

    for i in range(1, seconds):
        values[i] = values[i - 1] + 0.05 * (50.0 - values[i - 1]) + rng.normal(0, 0.004)

    return values


def _hum(grid_values: np.ndarray, offset_s: int, times: np.ndarray) -> np.ndarray:
    instantaneous = np.interp(
        times, np.arange(len(grid_values)) - offset_s, grid_values
    )
    phase = 2 * np.pi * np.cumsum(instantaneous) / _SAMPLE_RATE

    return 0.004 * np.sin(2 * phase) + 0.0015 * np.sin(3 * phase)


def _aircraft_audio(
    aircraft: Aircraft, times: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    coarse = np.arange(times[0], times[-1] + 0.05, 0.05)
    receiver = _TRUTH[None, :]
    factor, distance = doppler.propagate(
        aircraft.track, receiver, _START.timestamp() + coarse
    )
    factor = np.interp(times, coarse, factor[0])
    distance = np.interp(times, coarse, distance[0])
    phase = 2 * np.pi * np.cumsum(aircraft.fundamental_hz * factor) / _SAMPLE_RATE
    level = (40.0 if aircraft.fundamental_hz < 200 else 25.0) / distance
    tone = sum(np.sin(k * phase) / k for k in range(1, aircraft.harmonics + 1))
    sos = signal.butter(
        4, [100, 3_000], btype="bandpass", fs=_SAMPLE_RATE, output="sos"
    )
    rumble = signal.sosfilt(sos, rng.normal(0, 1, len(times)))

    return level * (0.5 * tone + 0.6 * rumble)


def _background(length: int, rng: np.random.Generator) -> np.ndarray:
    sos = signal.butter(2, 400, btype="lowpass", fs=_SAMPLE_RATE, output="sos")
    low = signal.sosfilt(sos, rng.normal(0, 1, length))

    return 0.02 * low / low.std() + 0.004 * rng.normal(0, 1, length)


def _flight_log(aircraft: list[Aircraft], rng: np.random.Generator) -> list[str]:
    lines: list[str] = []
    start = _START.timestamp()

    for poll in np.arange(start - 300, start + _DURATION_S + 300, 2.0):
        entries = []

        for plane in aircraft:
            seen = float(rng.uniform(0.0, 1.5))
            moment = poll - seen
            east, north, up = (
                np.interp(moment, plane.track.times, plane.track.positions[:, i])
                for i in range(3)
            )
            latitude, longitude = flights.to_geographic(_ORIGIN, east, north)
            entries.append(
                {
                    "hex": plane.identifier,
                    "flight": plane.callsign,
                    "lat": latitude,
                    "lon": longitude,
                    "alt_geom": round((up + _GROUND_HEIGHT_M) / _FEET_TO_M),
                    "seen_pos": round(seen, 1),
                },
            )

        response = {"now": int(poll * 1000), "ac": entries}
        lines.append(json.dumps({"poll_time": float(poll), "response": response}))

    return lines


async def _encode(wav_path: str, video_path: str, *, location: str) -> None:
    process = await asyncio.create_subprocess_exec(
        "ffmpeg",
        "-nostdin",
        "-v",
        "error",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "color=c=black:s=320x240:r=5",
        "-i",
        wav_path,
        "-shortest",
        "-c:v",
        "libx264",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-metadata",
        f"creation_time={_START.isoformat().replace('+00:00', 'Z')}",
        "-metadata",
        f"location={location}",
        video_path,
    )
    await process.wait()


def _write_text(path: str, text: str) -> None:
    with open(path, "w") as handle:
        handle.write(text)


async def create_scenario(directory: str, *, seed: int = 7) -> dict[str, object]:
    """Write a synthetic video, grid reference, flight log and ground truth."""
    os.makedirs(directory, exist_ok=True)
    rng = np.random.default_rng(seed)
    aircraft = _scenario()
    times = np.arange(_DURATION_S * _SAMPLE_RATE) / _SAMPLE_RATE
    walk = _grid_walk(7_200, rng)
    hum_offset = 3_600
    audio = _background(len(times), rng) + _hum(walk, hum_offset, times)

    for plane in aircraft:
        if plane.audible:
            audio += _aircraft_audio(plane, times, rng)

    reference = grid.GridSeries(
        start=_START - timedelta(seconds=hum_offset),
        step_s=1.0,
        values=walk,
    )
    await grid.write_csv(os.path.join(directory, "grid.csv"), reference)
    log = _flight_log(aircraft, rng)
    await asyncio.to_thread(
        _write_text, os.path.join(directory, "flights.jsonl"), "\n".join(log) + "\n"
    )
    latitude, longitude = flights.to_geographic(_ORIGIN, _TRUTH[0], _TRUTH[1])
    location = f"{latitude:+08.4f}{longitude:+09.4f}/"
    spliced = audio.copy()
    cut = 150 * _SAMPLE_RATE
    spliced[cut:] += (
        _hum(walk, hum_offset + 1_800, times)[cut:]
        - _hum(walk, hum_offset, times)[cut:]
    )

    for name, samples in (("clip", audio), ("spliced", spliced)):
        wav = os.path.join(directory, f"{name}.wav")
        peak = np.abs(samples).max()
        await asyncio.to_thread(
            wavfile.write, wav, _SAMPLE_RATE, (0.9 * samples / peak).astype(np.float32)
        )
        await _encode(wav, os.path.join(directory, f"{name}.mp4"), location=location)

    truth = {
        "start": _START.isoformat(),
        "latitude": latitude,
        "longitude": longitude,
        "origin": [_ORIGIN.latitude, _ORIGIN.longitude],
        "ground_height_m": _GROUND_HEIGHT_M,
        "splice_at_s": 150.0,
        "audible_aircraft": [a.callsign for a in aircraft if a.audible],
    }
    await asyncio.to_thread(
        _write_text, os.path.join(directory, "truth.json"), json.dumps(truth, indent=2)
    )

    return truth
