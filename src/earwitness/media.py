import asyncio
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from datetime import timezone

import numpy as np

from earwitness._common import ForensicError
from earwitness._common import OnSuccess

ANALYSIS_SAMPLE_RATE = 16_000

_ISO6709 = re.compile(r"([+-]\d+(?:\.\d+)?)([+-]\d+(?:\.\d+)?)")
_FILENAME_TIME = re.compile(r"(20\d{6})[_-]?(\d{6})")
_LOCATION_TAGS = (
    "com.apple.quicktime.location.iso6709",
    "location",
    "location-eng",
)


class MediaError(ForensicError):
    PROBE_FAILED = "probe_failed"
    DECODE_FAILED = "decode_failed"
    NO_AUDIO = "no_audio"


@dataclass(frozen=True, slots=True)
class Coordinates:
    latitude: float
    longitude: float


@dataclass(frozen=True, slots=True)
class Metadata:
    """creation_time is whatever the container says; phones and Google Photos
    often store the end of recording rather than the start. filename_time is
    parsed from names like 20240722_131415 and is local time, zone unknown.
    """

    creation_time: datetime | None
    location: Coordinates | None
    duration_s: float
    filename_time: datetime | None


@dataclass(frozen=True, slots=True)
class Recording:
    samples: np.ndarray
    sample_rate: int
    metadata: Metadata


async def _run(*args: str) -> bytes | None:
    process = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, _ = await process.communicate()

    if process.returncode != 0:
        return None

    return stdout


def _parse_time(value: str | None) -> datetime | None:
    if value is None:
        return None

    parsed = datetime.fromisoformat(value)

    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)

    return parsed.astimezone(timezone.utc)


def _parse_filename_time(path: str) -> datetime | None:
    found = _FILENAME_TIME.search(os.path.basename(path))

    if found is None:
        return None

    try:
        return datetime.strptime(found.group(1) + found.group(2), "%Y%m%d%H%M%S")
    # NOTE: strptime has no non-raising form; digits like 20241399 are invalid.
    except ValueError:
        return None


def _parse_location(tags: dict[str, str]) -> Coordinates | None:
    for key in _LOCATION_TAGS:
        value = tags.get(key)

        if value is None:
            continue

        found = _ISO6709.match(value)

        if found is not None:
            return Coordinates(float(found.group(1)), float(found.group(2)))

    return None


def _collect_tags(probe: dict[str, object]) -> dict[str, str]:
    tags: dict[str, str] = {}
    sections = [probe.get("format", {}), *probe.get("streams", [])]

    for section in sections:
        for key, value in section.get("tags", {}).items():
            tags.setdefault(key.lower(), value)

    return tags


async def probe(path: str) -> OnSuccess[Metadata]:
    output = await _run(
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        path,
    )

    if output is None:
        return MediaError.PROBE_FAILED

    data = json.loads(output)

    if not any(s.get("codec_type") == "audio" for s in data.get("streams", [])):
        return MediaError.NO_AUDIO

    tags = _collect_tags(data)
    duration = float(data.get("format", {}).get("duration", 0.0))

    return Metadata(
        creation_time=_parse_time(tags.get("creation_time")),
        location=_parse_location(tags),
        duration_s=duration,
        filename_time=_parse_filename_time(path),
    )


async def load(path: str) -> OnSuccess[Recording]:
    metadata = await probe(path)

    if isinstance(metadata, ForensicError):
        return metadata

    output = await _run(
        "ffmpeg",
        "-nostdin",
        "-v",
        "error",
        "-i",
        path,
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(ANALYSIS_SAMPLE_RATE),
        "-f",
        "f32le",
        "-",
    )

    if output is None or len(output) == 0:
        return MediaError.DECODE_FAILED

    samples = np.frombuffer(output, dtype=np.float32).astype(np.float64)

    return Recording(samples, ANALYSIS_SAMPLE_RATE, metadata)
