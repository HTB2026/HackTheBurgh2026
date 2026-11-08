"""Who made this file, when, where, and has it been through other software?

Reads container tags, the top-level MP4/QuickTime box layout and stream facts.
Phones write a recognisable layout; editors and social apps rewrite it.
"""

import asyncio
import math
import os
import re
import struct
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from typing import Any

import httpx

from earwitness.modules.base import CaseContext
from earwitness.modules.base import Reporter

_ISO6709 = re.compile(r"([+-]\d+(?:\.\d+)?)([+-]\d+(?:\.\d+)?)([+-]\d+(?:\.\d+)?)?")
_FILENAME_TIME = re.compile(r"(20\d{2})(\d{2})(\d{2})[_-]?(\d{2})(\d{2})(\d{2})")
_CONTAINER_BOXES = {
    b"moov",
    b"trak",
    b"mdia",
    b"minf",
    b"stbl",
    b"udta",
    b"edts",
    b"dinf",
}

# Software fingerprints in tags / handler names.
_SIGNATURES = [
    (re.compile(r"lavf|libavformat", re.I), "FFmpeg (libavformat)", "editor"),
    (re.compile(r"handbrake", re.I), "HandBrake", "editor"),
    (re.compile(r"premiere|adobe", re.I), "Adobe software", "editor"),
    (re.compile(r"capcut|bytedance", re.I), "CapCut / ByteDance", "editor"),
    (re.compile(r"davinci|resolve", re.I), "DaVinci Resolve", "editor"),
    (re.compile(r"imovie|final cut", re.I), "Apple iMovie / Final Cut", "editor"),
    (re.compile(r"whatsapp", re.I), "WhatsApp", "social"),
    (re.compile(r"telegram", re.I), "Telegram", "social"),
    (re.compile(r"google", re.I), "Google (Photos/Android)", "platform"),
    (re.compile(r"core media", re.I), "Apple Core Media (camera)", "camera"),
]


def _boxes(
    path: str, start: int = 0, end: int | None = None, depth: int = 0
) -> list[dict[str, Any]]:
    """Walk the ISO-BMFF box tree (top level and the usual containers)."""
    out: list[dict[str, Any]] = []
    end = end if end is not None else os.path.getsize(path)

    with open(path, "rb") as handle:
        offset = start

        while offset + 8 <= end and len(out) < 400:
            handle.seek(offset)
            header = handle.read(16)

            if len(header) < 8:
                break

            size, kind = struct.unpack(">I4s", header[:8])
            header_len = 8

            if size == 1 and len(header) >= 16:
                size = struct.unpack(">Q", header[8:16])[0]
                header_len = 16
            elif size == 0:
                size = end - offset

            if size < header_len or not re.fullmatch(rb"[ -~]{4}", kind):
                break

            box = {
                "type": kind.decode("latin-1"),
                "offset": offset,
                "size": size,
                "depth": depth,
            }
            out.append(box)

            if kind in _CONTAINER_BOXES and depth < 4:
                out.extend(_boxes(path, offset + header_len, offset + size, depth + 1))

            offset += size

    return out


def _parse_time(value: str | None) -> datetime | None:
    if not value:
        return None

    text = value.strip().replace("Z", "+00:00")

    # QuickTime writes "+0000" without a colon.
    text = re.sub(r"([+-]\d{2})(\d{2})$", r"\1:\2", text)

    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None

    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _solar_terms(when: datetime) -> tuple[float, float, float]:
    """(declination in radians, equation of time in minutes, UTC hour) at ``when``."""
    utc = when.astimezone(timezone.utc)
    day = utc.timetuple().tm_yday
    hour = utc.hour + utc.minute / 60 + utc.second / 3600
    gamma = 2 * math.pi / 365 * (day - 1 + (hour - 12) / 24)
    equation = 229.18 * (
        0.000075 + 0.001868 * math.cos(gamma) - 0.032077 * math.sin(gamma)
        - 0.014615 * math.cos(2 * gamma) - 0.040849 * math.sin(2 * gamma)
    )  # fmt: skip
    declination = (
        0.006918 - 0.399912 * math.cos(gamma) + 0.070257 * math.sin(gamma)
        - 0.006758 * math.cos(2 * gamma) + 0.000907 * math.sin(2 * gamma)
        - 0.002697 * math.cos(3 * gamma) + 0.00148 * math.sin(3 * gamma)
    )  # fmt: skip

    return declination, equation, hour


def subsolar_point(when: datetime) -> tuple[float, float]:
    """Where on Earth the sun is straight overhead at ``when`` (lat, lon in degrees)."""
    declination, equation, hour = _solar_terms(when)
    longitude = (720 - hour * 60 - equation) / 4

    return math.degrees(declination), (longitude + 180) % 360 - 180


def sun_position(
    when: datetime, latitude: float, longitude: float
) -> tuple[float, float]:
    """Solar elevation and azimuth in degrees (NOAA approximation, ~0.5°)."""
    declination, equation, hour = _solar_terms(when)
    true_solar = hour * 60 + equation + 4 * longitude
    hour_angle = math.radians(true_solar / 4 - 180)
    lat = math.radians(latitude)
    cos_zenith = math.sin(lat) * math.sin(declination) + math.cos(lat) * math.cos(
        declination
    ) * math.cos(hour_angle)
    zenith = math.acos(max(-1.0, min(1.0, cos_zenith)))
    elevation = 90 - math.degrees(zenith)
    azimuth = math.degrees(
        math.atan2(
            math.sin(hour_angle),
            math.cos(hour_angle) * math.sin(lat)
            - math.tan(declination) * math.cos(lat),
        )
    )

    return elevation, (azimuth + 180) % 360


async def _reverse_geocode(latitude: float, longitude: float) -> str | None:
    if os.environ.get("EARWITNESS_OFFLINE"):
        return None

    try:
        async with httpx.AsyncClient(timeout=6.0) as client:
            response = await client.get(
                "https://nominatim.openstreetmap.org/reverse",
                params={
                    "lat": latitude,
                    "lon": longitude,
                    "format": "jsonv2",
                    "zoom": 17,
                },
                headers={"User-Agent": "earwitness-forensics/0.1"},
            )

        if response.status_code != 200:
            return None

        return response.json().get("display_name")
    except httpx.HTTPError, ValueError:
        return None


class Provenance:
    name = "provenance"
    title = "PROVENANCE"
    description = "Container metadata, device, GPS, software fingerprints, box layout."

    async def run(self, ctx: CaseContext, out: Reporter) -> None:
        probe = await ctx.probe()
        fmt = probe.get("format", {})
        streams = probe.get("streams", [])
        tags: dict[str, str] = {}

        for section in [fmt, *streams]:
            for key, value in section.get("tags", {}).items():
                tags.setdefault(key.lower(), str(value))

        out.progress(0.2, "reading tags")
        out.artifact("probe", "json", "Raw container report (ffprobe)", data=probe)
        out.artifact(
            "tags",
            "table",
            "Container tags",
            data={"rows": [[k, v] for k, v in sorted(tags.items())]},
        )
        out.artifact(
            "streams",
            "table",
            "Streams",
            data={
                "rows": [
                    [
                        s.get("index"),
                        s.get("codec_type"),
                        s.get("codec_name"),
                        s.get("profile", ""),
                        f"{s.get('width', '')}x{s.get('height', '')}"
                        if s.get("width")
                        else s.get("sample_rate", ""),
                        s.get("bit_rate", ""),
                        s.get("duration", ""),
                        s.get("tags", {}).get("handler_name", ""),
                    ]
                    for s in streams
                ],
                "columns": [
                    "#",
                    "type",
                    "codec",
                    "profile",
                    "size/rate",
                    "bitrate",
                    "duration",
                    "handler",
                ],
            },
        )

        make = (
            tags.get("com.apple.quicktime.make")
            or tags.get("make")
            or tags.get("com.android.manufacturer")
        )
        model = (
            tags.get("com.apple.quicktime.model")
            or tags.get("model")
            or tags.get("com.android.model")
        )
        software = (
            tags.get("com.apple.quicktime.software")
            or tags.get("software")
            or tags.get("com.android.version")
        )
        published: dict[str, Any] = {"make": make, "model": model, "software": software}

        if make or model:
            out.finding(
                "info",
                f"Recorded on {' '.join(x for x in (make, model) if x)}",
                f"Device tags name the camera{f'; firmware/software {software}' if software else ''}.",
                tags=["device"],
                data={"make": make, "model": model, "software": software},
            )

        # ---- software fingerprints ---------------------------------------
        texts = " | ".join(
            [tags.get("encoder", ""), tags.get("handler_name", ""), tags.get("compatible_brands", ""),
             tags.get("major_brand", ""), *[s.get("tags", {}).get("handler_name", "") for s in streams],
             *[s.get("tags", {}).get("encoder", "") for s in streams]]
        )  # fmt: skip
        seen: set[str] = set()

        for pattern, label, role in _SIGNATURES:
            if pattern.search(texts) and label not in seen:
                seen.add(label)

                if role == "camera":
                    continue

                out.finding(
                    "warning" if role == "editor" else "notice",
                    f"Processed by {label}",
                    f"A tag or handler name carries the fingerprint of {label}. "
                    + (
                        "Camera-original files do not; the file was re-muxed or re-encoded."
                        if role != "platform"
                        else "Android's recorder and Google Photos exports both write it; "
                        "check the codec (VP9/AV1 means a Photos re-encode)."
                    ),
                    tags=["software", role],
                    data={"evidence": texts[:400]},
                )

        published["software_seen"] = sorted(seen)

        # ---- box layout -----------------------------------------------------
        out.progress(0.4, "walking box tree")
        boxes = await asyncio.to_thread(_boxes, ctx.path)
        top = [b["type"] for b in boxes if b["depth"] == 0]
        published["boxes"] = top

        if boxes:
            out.artifact(
                "boxes",
                "table",
                "Box (atom) layout",
                data={
                    "columns": ["depth", "type", "offset", "size"],
                    "rows": [
                        [b["depth"], b["type"], b["offset"], b["size"]]
                        for b in boxes[:200]
                    ],
                },
            )

            if "moov" in top and "mdat" in top:
                moov_first = top.index("moov") < top.index("mdat")
                apple = bool(tags.get("com.apple.quicktime.make"))

                if moov_first and apple:
                    out.finding(
                        "warning",
                        "Box layout rewritten",
                        "The moov index sits before the media data. iPhone camera originals "
                        "write it at the end; something re-saved this file (export, upload, editor).",
                        tags=["container"],
                        data={"top_level": top},
                    )
                elif not moov_first and not any(
                    role == "editor" for _, label, role in _SIGNATURES if label in seen
                ):
                    out.finding(
                        "info",
                        "Camera-style box layout",
                        "Index (moov) follows the media (mdat), as a recorder that writes "
                        "while filming leaves it. Consistent with an original file.",
                        tags=["container"],
                        data={"top_level": top},
                    )
                else:
                    out.finding(
                        "notice",
                        "Streaming-optimised layout",
                        "moov precedes mdat ('faststart'). Typical of web uploads, messaging "
                        "apps and editors; camera originals usually differ.",
                        tags=["container"],
                        data={"top_level": top},
                    )

        # ---- time ---------------------------------------------------------------
        out.progress(0.6, "reconciling clocks")
        duration = float(fmt.get("duration") or 0)
        apple_date_raw = tags.get("com.apple.quicktime.creationdate")
        apple_date = _parse_time(apple_date_raw)
        creation = _parse_time(tags.get("creation_time")) or apple_date
        filename = ctx.options.get("original_name") or os.path.basename(ctx.path)
        found = _FILENAME_TIME.search(filename)
        file_local = None

        if found:
            try:
                file_local = datetime(*(int(g) for g in found.groups()))
            except ValueError:
                file_local = None

        times: dict[str, Any] = {
            "creation_time_utc": creation,
            "apple_creationdate": apple_date_raw,
            "filename_local": file_local,
            "duration_s": duration,
        }

        if apple_date_raw and (
            offset := re.search(r"([+-]\d{2}):?(\d{2})$", apple_date_raw)
        ):
            hours = int(offset.group(1))
            times["utc_offset_h"] = hours + math.copysign(
                int(offset.group(2)) / 60, hours
            )

        if creation is not None:
            out.finding(
                "info",
                f"Container clock: {creation:%Y-%m-%d %H:%M:%S} UTC",
                "creation_time from the container. Phones often stamp the END of the "
                f"recording; if so it began at {(creation - timedelta(seconds=duration)):%H:%M:%S} UTC.",
                tags=["time"],
                data=times,
            )

        if creation is not None and file_local is not None:
            naive = creation.replace(tzinfo=None)
            best = None

            for anchor_name, anchor in (
                ("start", naive - timedelta(seconds=duration)),
                ("end", naive),
            ):
                delta_h = (file_local - anchor).total_seconds() / 3600
                rounded = round(delta_h * 4) / 4
                residual = abs(delta_h - rounded) * 3600

                if best is None or residual < best[2]:
                    best = (anchor_name, rounded, residual)

            anchor_name, offset_h, residual = best

            if residual <= 5 and abs(offset_h) <= 14:
                times["utc_offset_h"] = offset_h
                times["creation_time_marks"] = (
                    "end" if anchor_name == "start" else "start"
                )
                out.finding(
                    "notice",
                    f"Local clock was UTC{offset_h:+g}",
                    f"Filename time {file_local:%H:%M:%S} (local) and container time agree "
                    f"to {residual:.0f} s at a UTC{offset_h:+g} offset. The filename stamps the "
                    f"recording {'start' if anchor_name == 'start' else 'end'}, and the container "
                    f"stamps the {'end' if anchor_name == 'start' else 'start'}.",
                    tags=["time", "timezone"],
                    data={"offset_h": offset_h, "residual_s": residual},
                )
            else:
                out.finding(
                    "warning",
                    "Filename and container clocks disagree",
                    f"Filename says {file_local} local, container {creation} UTC, and no whole "
                    "time-zone offset reconciles them. The file was renamed, re-exported, or a clock was set.",
                    tags=["time", "inconsistency"],
                    data={"residual_s": residual, "offset_h": offset_h},
                )

        if creation is None and not make:
            out.finding(
                "notice",
                "Metadata stripped",
                "No creation time and no device tags. Social platforms and messaging apps "
                "strip these; camera originals keep them.",
                tags=["metadata"],
            )

        # Apps and downloaders stamp the moment they save a copy. A camera-less
        # file whose clock reads "just now" is dated by the copy, not the filming.
        times["copy_stamp"] = False

        if creation is not None and not (make or model):
            age_h = (datetime.now(timezone.utc) - creation).total_seconds() / 3600

            if -1 <= age_h <= 72:
                times["copy_stamp"] = True
                out.finding(
                    "warning",
                    "Timestamp is when this copy was made",
                    f"The file claims {creation:%Y-%m-%d %H:%M} UTC, only {max(age_h, 0):.1f} h before it "
                    "was examined, and it carries no camera make or model. Apps, messengers and "
                    "downloaders write the moment they save a copy, so this is not when it was filmed.",
                    tags=["time", "metadata"],
                    data={"age_hours": age_h},
                )

        published["times"] = times
        start_guess = None

        if creation is not None and not times["copy_stamp"]:
            start_guess = (
                creation - timedelta(seconds=duration)
                if times.get("creation_time_marks", "end") == "end"
                else creation
            )

        published["start_guess"] = start_guess
        # Which end of the recording the container stamps is device-dependent.
        published["start_candidates"] = (
            []
            if creation is None
            else [creation - timedelta(seconds=duration), creation]
        )

        # ---- streams ------------------------------------------------------------
        durations = {
            s.get("codec_type"): float(s["duration"])
            for s in streams
            if s.get("duration") not in (None, "N/A")
            and s.get("codec_type") in ("audio", "video")
        }

        if "audio" in durations and "video" in durations:
            gap = durations["audio"] - durations["video"]

            if abs(gap) > 0.25:
                out.finding(
                    "notice",
                    f"Audio and video lengths differ by {abs(gap):.2f} s",
                    f"Audio runs {durations['audio']:.2f} s and video {durations['video']:.2f} s. "
                    "Cameras keep them within a frame or two. Editing or re-muxing can leave a gap.",
                    tags=["streams"],
                    data=durations,
                )

        # ---- place ---------------------------------------------------------------
        out.progress(0.8, "locating")
        location_text = (
            tags.get("com.apple.quicktime.location.iso6709")
            or tags.get("location")
            or tags.get("location-eng")
        )
        accuracy = tags.get("com.apple.quicktime.location.accuracy.horizontal")
        published["location"] = None

        if location_text and (match := _ISO6709.match(location_text)):
            latitude, longitude = float(match.group(1)), float(match.group(2))
            altitude = float(match.group(3)) if match.group(3) else None
            accuracy_m = float(accuracy) if accuracy else None
            place = await _reverse_geocode(latitude, longitude)
            published["location"] = {
                "lat": latitude, "lon": longitude, "alt_m": altitude,
                "accuracy_m": accuracy_m, "place": place,
            }  # fmt: skip
            out.finding(
                "critical",
                f"GPS fix: {latitude:.5f}, {longitude:.5f}",
                (f"{place}. " if place else "")
                + "The file carries the phone's location"
                + (f" (±{accuracy_m:.0f} m)" if accuracy_m else "")
                + (f", altitude {altitude:.0f} m" if altitude is not None else "")
                + ".",
                tags=["location", "gps"],
                data=published["location"],
            )
            out.artifact(
                "map",
                "map",
                "Where",
                data={
                    "points": [
                        {"lat": latitude, "lon": longitude, "label": "GPS (metadata)",
                         "accuracy_m": accuracy_m, "kind": "gps"}
                    ]
                },
            )  # fmt: skip

            if start_guess is not None:
                elevation, azimuth = sun_position(start_guess, latitude, longitude)
                published["sun"] = {"elevation": elevation, "azimuth": azimuth}
                phase = (
                    "daylight" if elevation > 6 else
                    "golden hour / civil twilight" if elevation > -6 else
                    "nautical twilight" if elevation > -12 else "night"
                )  # fmt: skip
                out.finding(
                    "info",
                    f"Sun {elevation:+.0f}° elevation, azimuth {azimuth:.0f}° ({phase})",
                    "Computed from the claimed time and GPS fix. Scene lighting should match; "
                    "the LIGHT module checks this against the footage.",
                    tags=["sun", "time", "location"],
                    data=published["sun"] | {"phase": phase},
                )

        ctx.publish(self.name, published)
        out.progress(1.0)
