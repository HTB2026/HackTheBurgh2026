"""Where was this filmed? Propose, then verify against the real world.

The way an OSINT analyst works, automated:

1. Gather clues: several varied frames, zoomed edge and corner crops from
   geo-sleuth (https://github.com/Oldcircle/geo-sleuth), OCR text, spoken
   language and the transcript, what an AI listener heard (accents, sirens,
   announcements), and which power grid the lights run on.
2. Propose: a reasoning model lists candidate places, each with coordinates
   and, where there is one, a named landmark.
3. Verify: for every candidate, fetch the landmark's reference photo from
   Wikipedia and the satellite image of its coordinates (Esri World Imagery),
   and have a second pass compare them with the frames. A candidate becomes
   the answer only if it survives that comparison.

The model never sees the file's GPS or metadata. When the file does carry
GPS, the error in km is reported, which keeps the whole thing honest.

Use only on footage you are entitled to analyse; never to locate people who
have not agreed to be found. geo-sleuth is optional (EARWITNESS_GEOSLEUTH);
An AI key (OPENAI_API_KEY or GEMINI_API_KEY) is required for the reasoning steps.
"""

import asyncio
import base64
import glob
import json
import math
import os
from datetime import datetime
from datetime import timezone
from typing import Any

import cv2
import httpx
import numpy as np

from earwitness import ai
from earwitness.modules import geo_deep
from earwitness.modules.base import CaseContext
from earwitness.modules.base import Reporter
from earwitness.modules.base import Skip
from earwitness.modules.base import run_tool

_FRAMES = 6
_INTAKE_FRAMES = 2
_CANDIDATES = 4
_VERIFIED = 0.6
_USER_AGENT = "Earwitness/0.2 (https://earwitness.ussr.pl; hackathon research demo)"


def _has_location(lat: Any, lon: Any, radius_km: Any) -> bool:
    """A continent-wide estimate does not identify a location."""
    return (
        all(
            isinstance(v, (int, float)) and math.isfinite(v)
            for v in (lat, lon, radius_km)
        )
        and abs(lat) <= 90
        and abs(lon) <= 180
        and 0 <= radius_km < 2500
    )


_PROPOSE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["scene", "region_reasoning", "candidates"],
    "properties": {
        "scene": {
            "type": "string",
            "description": "Indoor/outdoor and what kind of place, one sentence.",
        },
        "region_reasoning": {
            "type": "string",
            "description": "Step by step: which clues narrow it to which country/region.",
        },
        "candidates": {
            "type": "array",
            "minItems": 3,
            "maxItems": 5,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "place",
                    "landmark",
                    "latitude",
                    "longitude",
                    "radius_km",
                    "probability",
                    "evidence",
                ],
                "properties": {
                    "place": {
                        "type": "string",
                        "description": "Most specific place: venue/street, city, country.",
                    },
                    "landmark": {
                        "type": "string",
                        "description": "Exact name of a visible landmark or building that would have a Wikipedia article (e.g. 'London Eye', 'McEwan Hall'), or '' if none.",
                    },
                    "latitude": {"type": "number"},
                    "longitude": {"type": "number"},
                    "radius_km": {
                        "type": "number",
                        "description": "Honest ~80% radius.",
                    },
                    "probability": {
                        "type": "number",
                        "description": "0..1, all candidates summing to <= 1.",
                    },
                    "evidence": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
    },
}

_VERIFY_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "verdicts",
        "best",
        "place",
        "latitude",
        "longitude",
        "radius_km",
        "confidence",
        "explanation",
    ],
    "properties": {
        "verdicts": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["candidate", "match", "matches", "mismatches"],
                "properties": {
                    "candidate": {"type": "integer"},
                    "match": {
                        "type": "number",
                        "description": "0..1: how well the reference photo / satellite view matches the frames.",
                    },
                    "matches": {"type": "array", "items": {"type": "string"}},
                    "mismatches": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
        "best": {
            "type": "integer",
            "description": "Index of the best candidate, or -1 if none is convincing.",
        },
        "place": {"type": "string"},
        "latitude": {"type": "number"},
        "longitude": {"type": "number"},
        "radius_km": {"type": "number"},
        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
        "explanation": {
            "type": "string",
            "description": "Two or three sentences a non-expert can follow.",
        },
    },
}

_PROPOSE = (
    "You are a world-class open-source geolocation analyst (top GeoGuessr / Bellingcat level). "
    "From the frames, zoomed crops and the non-visual clues, propose where this was filmed. Use "
    "every concrete clue: driving side, road markings, kerbs, bollards, street furniture, sign "
    "typography and language, plates, vehicles and liveries, architecture, building materials, "
    "vegetation, terrain, sky and sun, shop brands, plugs, the language and accent of voices, "
    "siren patterns, the mains frequency of the lighting. First reason about the country and "
    "region, then the city, then the exact spot. Give 3-5 candidates, most likely first, in at "
    "least two DIFFERENT cities, because look-alikes are common (many cities have domed halls, "
    "classical facades, Ferris wheels, waterfront skylines). Each needs coordinates, an honest "
    "radius, and a named landmark if one is visible. Indoor scenes usually allow only a wide "
    "guess: say so with a large radius. Never invent text you cannot read. Accents in a crowd "
    "are weak evidence: tourists and students travel."
)

_VERIFY = (
    "You are a sceptical verifier. The proposer is often wrong, and look-alike places are common. "
    "You see the evidence frames, then each candidate with a reference photo of its named "
    "landmark (from Wikipedia) and/or a satellite view of its coordinates, in random order. "
    "Compare specific, distinctive features: the exact shape of the dome or roof (round, "
    "octagonal, lantern), window rhythm, columns, materials and colour, neighbouring buildings, "
    "bridge type, skyline, coastline, road layout. Generic similarity (a dome, a classical "
    "building, a city at night) is NOT a match: a score above 0.7 needs at least three specific "
    "matching features and no contradicting one. Score each candidate 0..1, list matches and "
    "mismatches, then give the final answer. If no candidate is convincing, set best to -1, "
    "confidence low, and a wide radius around the most plausible region."
)


def _haversine_km(a: tuple[float, float], b: tuple[float, float]) -> float:
    lat1, lon1, lat2, lon2 = map(math.radians, (*a, *b))
    h = (
        math.sin((lat2 - lat1) / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    )

    return 2 * 6371.0 * math.asin(math.sqrt(h))


def _image_part(path: str, max_side: int = 1024) -> dict[str, str] | None:
    image = cv2.imread(path)

    if image is None:
        return None

    scale = max_side / max(image.shape[:2])

    if scale < 1:
        image = cv2.resize(
            image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
        )

    _, jpeg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])

    return {
        "type": "input_image",
        "image_url": "data:image/jpeg;base64,"
        + base64.b64encode(jpeg.tobytes()).decode(),
    }


async def _wikipedia_photo(
    client: httpx.AsyncClient, name: str, path: str
) -> dict[str, str] | None:
    """The lead image of the best-matching English Wikipedia article."""
    try:
        search = await client.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "list": "search",
                "srsearch": name,
                "srlimit": 1,
                "format": "json",
            },
        )
        hits = search.json().get("query", {}).get("search", [])

        if not hits:
            return None

        title = hits[0]["title"]
        summary = await client.get(
            f"https://en.wikipedia.org/api/rest_v1/page/summary/{title.replace(' ', '_')}"
        )
        data = summary.json()
        source = (data.get("thumbnail") or data.get("originalimage") or {}).get(
            "source"
        )

        if not source:
            return None

        image = await client.get(source)

        if image.status_code != 200 or len(image.content) > 4_000_000:
            return None

        with open(path, "wb") as handle:
            handle.write(image.content)

        return {
            "title": title,
            "url": data.get("content_urls", {}).get("desktop", {}).get("page", ""),
        }
    except httpx.HTTPError, ValueError, KeyError:
        return None


async def _satellite(
    client: httpx.AsyncClient, lat: float, lon: float, span_km: float, path: str
) -> bool:
    dlat = span_km / 111.0 / 2
    dlon = dlat / max(0.2, math.cos(math.radians(lat)))

    try:
        response = await client.get(
            "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/export",
            params={
                "bbox": f"{lon - dlon},{lat - dlat},{lon + dlon},{lat + dlat}",
                "bboxSR": 4326,
                "imageSR": 3857,
                "size": "640,640",
                "format": "jpg",
                "f": "image",
            },  # fmt: skip
        )
    except httpx.HTTPError:
        return False

    if response.status_code != 200 or not response.headers.get(
        "content-type", ""
    ).startswith("image"):
        return False

    with open(path, "wb") as handle:
        handle.write(response.content)

    return True


async def _pick_frames(
    ctx: CaseContext, duration: float, count: int = _FRAMES
) -> list[tuple[float, np.ndarray]]:
    """The sharpest frame from each of ``count`` equal stretches of the clip."""
    chosen = []
    edges = np.linspace(0.2, max(0.4, duration - 0.2), count + 1)

    for a, b in zip(edges[:-1], edges[1:]):
        best = None

        for t in np.linspace(a, b, 3):
            frame = await ctx.frame_at(float(t), width=1280)

            if frame is None:
                continue

            sharp = _sharpness(frame)

            if best is None or sharp > best[0]:
                best = (sharp, float(t), frame)

        if best:
            chosen.append((best[1], best[2]))

    return chosen


def _sharpness(rgb: np.ndarray) -> float:
    return float(cv2.Laplacian(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var())


_QUARTERS = ["top-left", "top-right", "bottom-left", "bottom-right"]


def _quarters(image: np.ndarray, max_side: int = 1024) -> list[np.ndarray]:
    """The four quarters of a picture, so small signs and plates stay legible."""
    h, w = image.shape[:2]
    tiles = []

    for row in range(2):
        for col in range(2):
            tile = image[
                row * h // 2 : (row + 1) * h // 2, col * w // 2 : (col + 1) * w // 2
            ]
            scale = min(1.0, max_side / max(tile.shape[:2]))
            tiles.append(
                cv2.resize(tile, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                if scale < 1
                else tile
            )

    return tiles


def _trusted_time(
    provenance: dict[str, Any], enf: dict[str, Any], still: bool
) -> tuple[datetime | None, str | None]:
    """A UTC capture time good enough for the sun and the weather, and where it came from."""
    if enf.get("confident") and enf.get("dated_start"):
        when, source = enf["dated_start"], "the power-grid dating"
    elif provenance.get("start_guess") and (not still or provenance.get("tz_known")):
        # A photo's clock without a UTC offset is local time: hours off, useless for the sun.
        when, source = provenance["start_guess"], "the file's own timestamp"
    else:
        return None, None

    if isinstance(when, str):
        when = datetime.fromisoformat(when)

    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)

    return when, source


class Geo:
    name = "geo"
    title = "GEOLOCATION"
    description = "Where it was filmed, from pixels and sound alone: candidates proposed, then verified against Wikipedia photos and satellite imagery."
    # No up-front dependencies: geo is the longest check, so it starts at once
    # (frames, intake) and waits for the clue modules only right before it asks
    # the model, giving up its worker slot meanwhile (ctx.waiting).
    depends: list[str] = []

    async def run(self, ctx: CaseContext, out: Reporter) -> None:
        provider = ai.for_case(ctx.options)

        if provider is None:
            raise Skip(ai.MISSING_KEY)

        info = await ctx.video_info()
        still = ctx.kind == "image"

        if info is None and not still:
            raise Skip("no video stream")

        deep = ctx.effort == "high"
        # Quick runs reason briefly; Standard and Deep think hard.
        reasoning = ctx.by_effort(low="low", medium="high")
        out.progress(0.05, "choosing frames")

        if still:
            frames = [(0.0, ctx.image_scaled(1280))] + [
                (0.0, tile) for tile in _quarters(ctx.image())
            ]
        else:
            frames = await _pick_frames(
                ctx, info.duration_s, ctx.by_effort(low=3, medium=_FRAMES, high=10)
            )

        if not frames:
            raise Skip("no decodable frames")

        # Each unit is a caption and an image, so passes can take different subsets.
        frame_units: list[list[dict[str, str]]] = []
        crop_units: list[list[dict[str, str]]] = []
        frame_paths = []
        labels = ["the whole picture"] + [f"{q} quarter" for q in _QUARTERS]

        for number, (time_s, frame) in enumerate(frames):
            path = out.path(f"frame{number}.jpg")
            cv2.imwrite(
                path,
                cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 92],
            )
            frame_paths.append((time_s, path))

            if still:
                out.artifact(
                    f"frame{number}",
                    "image",
                    f"View analysed: {labels[number]}",
                    file=path,
                )
            else:
                out.artifact(
                    f"frame{number}", "image", f"Frame analysed @ {time_s:.1f}s",
                    file=path, time_s=time_s,
                )  # fmt: skip
            frame_units.append(
                [
                    {
                        "type": "input_text",
                        "text": f"Image {number + 1} ({labels[number]}):"
                        if still
                        else f"Frame {number + 1} (at {time_s:.1f} s):",
                    },
                    _image_part(path),
                ]
            )

        if deep and not still:
            # Quarter crops of the two sharpest frames: small details at full size.
            for time_s, frame in sorted(frames, key=lambda f: -_sharpness(f[1]))[:2]:
                for name, tile in zip(_QUARTERS, _quarters(frame, 768)):
                    path = out.path(f"quarter_{time_s:.1f}_{name}.jpg")
                    cv2.imwrite(
                        path,
                        cv2.cvtColor(tile, cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, 90],
                    )
                    crop_units.append(
                        [{"type": "input_text", "text": f"{name} quarter of the frame at {time_s:.1f}s, enlarged:"},
                         _image_part(path, 768)]
                    )  # fmt: skip

        # ---- geo-sleuth intake: zoomed edges/corners + OCR -------------------
        ocr_lines: list[str] = []
        skill = os.environ.get("EARWITNESS_GEOSLEUTH")
        script = os.path.join(os.path.expanduser(skill or ""), "scripts", "intake.py")
        intake_frames = ctx.by_effort(low=0, medium=_INTAKE_FRAMES, high=4)

        if skill and os.path.isfile(script) and intake_frames:
            if still:
                full_path = out.path("full.jpg")
                cv2.imwrite(
                    full_path,
                    cv2.cvtColor(ctx.image_scaled(4000), cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 94],
                )
                sharpest = [(0.0, full_path)]
            else:
                sharpest = sorted(
                    frame_paths,
                    key=lambda p: (
                        -cv2.Laplacian(
                            cv2.imread(p[1], cv2.IMREAD_GRAYSCALE), cv2.CV_64F
                        ).var()
                    ),
                )[:intake_frames]

            for number, (time_s, path) in enumerate(sharpest):
                out.progress(
                    0.1 + 0.15 * number / len(sharpest),
                    f"geo-sleuth intake @ {time_s:.1f}s",
                )
                target = out.path(f"intake{number}")
                await run_tool(
                    "uv", "run", script, path, "--out-dir", target, "--no-rev"
                )
                summary = os.path.join(target, "intake.json")

                if not os.path.isfile(summary):
                    continue

                with open(summary) as handle:
                    ocr_lines += [
                        o["text"]
                        for o in json.load(handle).get("ocr", [])
                        if o.get("conf", 0) >= 0.3
                    ]

                with open(
                    os.path.join(target, "intake.md"),
                    encoding="utf-8",
                    errors="replace",
                ) as handle:
                    out.artifact(
                        f"intake{number}",
                        "text",
                        f"geo-sleuth intake @ {time_s:.1f}s",
                        data={"text": handle.read()[:20_000]},
                    )

                for crop in sorted(glob.glob(os.path.join(target, "edges", "*.jpg")))[
                    :8
                ]:
                    part = _image_part(crop, 640)

                    if part:
                        crop_units.append(
                            [
                                {
                                    "type": "input_text",
                                    "text": f"Zoomed crop {os.path.basename(crop)} of frame at {time_s:.1f}s:",
                                },
                                part,
                            ]
                        )

        evidence = [part for unit in frame_units + crop_units for part in unit]

        # ---- non-visual clues (never the file's metadata or GPS) --------------
        out.progress(0.3, "collecting clues from sound, text and light")
        async with ctx.waiting():
            speech = await ctx.wait_for("transcript") or {}
            ocr = await ctx.wait_for("ocr") or {}
            heard = await ctx.wait_for("sounds") or {}
            light = await ctx.wait_for("light") or {}

            if deep:
                # The capture time (for the sun and the weather) and what the
                # frames look like. GPS is fetched but never shown to the model.
                provenance = await ctx.wait_for("provenance") or {}
                enf = await ctx.wait_for("enf") or {}
                vision = await ctx.wait_for("vision") or {}
        ocr_lines += list(ocr.get("lines", []))[:30]
        clues = [
            "Text read in the frames (OCR): "
            + (", ".join(dict.fromkeys(ocr_lines)) or "none")
        ]

        if speech.get("language_clue"):
            clues.append(
                f"Voices speaking {speech['language_clue']} are audible (words unclear)."
            )
        elif speech.get("segments"):
            clues.append(
                "Transcript: " + " ".join(s["text"] for s in speech["segments"])[:500]
            )

        if heard:
            clues.append(
                "An AI listener heard: "
                + json.dumps(
                    {
                        k: heard.get(k)
                        for k in (
                            "environment",
                            "languages",
                            "accents",
                            "location_clues",
                            "time_clues",
                            "people_estimate",
                            "wildlife",
                        )
                        if heard.get(k)
                    },
                    ensure_ascii=False,
                )
            )

        if light.get("grid_hz"):
            clues.append(
                f"The lighting flickers at {2 * light['grid_hz']} Hz: the mains grid is {light['grid_hz']} Hz."
            )

        # ---- Deep: inventory, time, sun and stars ----------------------------------
        inventory: dict[str, Any] = {}
        zones: list[dict[str, Any]] = []
        notes: list[str] = []  # what each Deep method did, for the record
        when, when_source = None, None

        if deep:
            out.progress(0.33, "taking an inventory of every clue in the frames")
            inventory = await provider.structured(
                ctx.workdir, instructions=geo_deep.INVENTORY, content=evidence,
                schema=geo_deep.INVENTORY_SCHEMA, name="geolocation_inventory", effort="high",
            )  # fmt: skip
            listed = geo_deep.inventory_text(inventory)
            out.artifact(
                "inventory", "text", "Clue inventory: what a geolocation expert reads in the frames",
                data={"text": listed},
            )  # fmt: skip
            clues.append("Clue inventory from a first look at the frames:\n" + listed)
            when, when_source = _trusted_time(provenance, enf, still)

            if still and provenance.get("utc_offset_h") is not None:
                clues.append(
                    f"The camera's clock was set to UTC{provenance['utc_offset_h']:+g} (from the file; "
                    "travellers do not always change it)."
                )

            seen_sun = inventory.get("sun_elevation_deg")

            if (
                when is not None
                and inventory.get("sun_visible")
                and isinstance(seen_sun, (int, float))
                and 0 < seen_sun < 90
            ):
                ring = geo_deep.sun_ring(when, float(seen_sun))
                zones.append(ring)
                clues.append(
                    f"The sun stood about {seen_sun:.0f}° above the horizon at {when:%Y-%m-%d %H:%M} UTC "
                    f"(time from {when_source}). That puts the camera on a ring {ring['radius_km']:,.0f} km "
                    f"(±{ring['width_km'] / 2:,.0f} km) from ({ring['lat']:.1f}, {ring['lon']:.1f}), where the sun was overhead."
                )
                notes.append(
                    f"Sun: {ring['label']} ({inventory.get('sun_basis', '')})."
                )
            elif inventory.get("sun_visible"):
                notes.append(
                    "Sun: visible, but "
                    + (
                        "its height could not be judged."
                        if when
                        else "no trustworthy capture time, so it cannot place the camera."
                    )
                )

            if inventory.get("stars_visible"):
                out.progress(0.36, "identifying the stars in the sky")
                # Full resolution for photos: faint stars vanish when scaled down.
                # The first frame, because the inventory's horizon refers to it.
                night = ctx.image_scaled(3000) if still else frames[0][1]
                solved = await asyncio.to_thread(geo_deep.solve_stars, night)

                if solved:
                    star_note = (
                        f"Stars: identified offline ({solved['matches']} stars matched, field centre "
                        f"RA {solved['ra']:.1f}°, Dec {solved['dec']:+.1f}°)"
                    )
                    left, right = (
                        inventory.get("horizon_left"),
                        inventory.get("horizon_right"),
                    )
                    fix = None

                    if all(
                        isinstance(v, (int, float)) and 0 < v < 1 for v in (left, right)
                    ):
                        gray = cv2.cvtColor(night, cv2.COLOR_RGB2GRAY)
                        horizon = geo_deep.refine_horizon(
                            gray, float(left), float(right)
                        )
                        fix = geo_deep.zenith_from_stars(
                            solved, night.shape[1], night.shape[0], horizon
                        )

                    if fix is not None and fix["field_altitude"] > -5:
                        found = geo_deep.star_fix_zones(fix, when)
                        zones += found
                        lat = fix["latitude"]
                        clues.append(
                            f"Measured from the stars and the horizon (no metadata involved): the celestial pole stands "
                            f"{abs(lat):.0f}° {'above' if lat >= 0 else 'below'} the horizon, so the camera was at about "
                            f"{abs(lat):.0f}°{'N' if lat >= 0 else 'S'} latitude (±4°). Only propose places at that latitude."
                            + (
                                f" With the capture time: {found[1]['label']}."
                                if len(found) > 1
                                else ""
                            )
                        )
                        notes.append(
                            f"{star_note}; with the horizon, the zenith points at Dec {lat:+.1f}°, "
                            f"so the latitude is about {lat:.0f}° (±4°). "
                            + "; ".join(z["label"] for z in found)
                            + "."
                        )
                    else:
                        # No usable horizon: the declination alone bounds the latitude.
                        tilt = inventory.get("camera_tilt_deg")
                        altitude = (
                            float(tilt) + solved["above_centre_deg"]
                            if isinstance(tilt, (int, float))
                            else None
                        )
                        found = geo_deep.star_zones(
                            solved["ra"], solved["dec"], altitude, when
                        )
                        zones += found
                        clues.append(
                            f"The star field was identified: {'; '.join(z['label'] for z in found)}."
                        )
                        notes.append(
                            f"{star_note}; no clear horizon, so only a broad band: "
                            + "; ".join(z["label"] for z in found)
                            + "."
                        )
                else:
                    notes.append(
                        "Stars: visible, but the star field could not be identified "
                        "(too few sharp stars, or clouds, glare or a long exposure smeared them)."
                    )

        clue_text = "\n".join(clues)
        out.artifact(
            "clues",
            "text",
            "Non-visual clues given to the analyst",
            data={"text": clue_text},
        )

        # ---- propose ---------------------------------------------------------------
        out.progress(0.4, "proposing candidate places")

        async def propose(units: list[list[dict[str, str]]]) -> dict[str, Any]:
            return await provider.structured(
                ctx.workdir,
                instructions=_PROPOSE,
                content=[
                    {"type": "input_text", "text": clue_text},
                    *(p for u in units for p in u),
                ],
                schema=_PROPOSE_SCHEMA,
                name="geolocation_candidates",
                effort=reasoning,
            )

        if deep:
            # Three independent looks at different frames; where they agree is evidence.
            out.progress(0.4, "three independent analysts propose places")
            subsets = [
                frame_units + crop_units,
                frame_units[0::2] + crop_units[1::2],
                frame_units[1::2] + crop_units[0::2],
            ]
            runs = await asyncio.gather(
                *(propose(s) for s in subsets if s), return_exceptions=True
            )
            proposals = [r for r in runs if isinstance(r, dict)]

            if not proposals:
                raise next(r for r in runs if isinstance(r, BaseException))

            proposal = proposals[0]
            candidates = geo_deep.cluster([p["candidates"] for p in proposals], keep=5)
        else:
            proposal = await propose(frame_units + crop_units)
            candidates = proposal["candidates"][
                : ctx.by_effort(low=3, medium=_CANDIDATES)
            ]

        # ---- verify against the real world ------------------------------------------
        out.progress(0.6, "fetching reference photos and satellite views")
        rows = None

        if deep and inventory.get("horizon") in (
            "natural_hills",
            "mountains",
            "natural_flat",
        ):
            for _, frame in frames:
                rows = geo_deep.frame_skyline(frame)

                if rows is not None:
                    aspect = frame.shape[1] / frame.shape[0]
                    break

        seen_weather = (
            f"{vision.get('weather', '')} {vision.get('scene', '')}" if deep else ""
        )

        async with httpx.AsyncClient(
            timeout=20, follow_redirects=True, headers={"User-Agent": _USER_AGENT}
        ) as client:
            overpass = asyncio.Semaphore(2)

            async def gather(index: int, candidate: dict[str, Any]) -> dict[str, Any]:
                item: dict[str, Any] = {
                    "index": index,
                    "candidate": candidate,
                    "ground": [],
                    "checks": [],
                }
                lat, lon = float(candidate["latitude"]), float(candidate["longitude"])
                radius = float(candidate["radius_km"])

                if candidate.get("landmark"):
                    photo = out.path(f"ref{index}_wiki.jpg")
                    wiki = await _wikipedia_photo(client, candidate["landmark"], photo)

                    if wiki:
                        item["wiki"] = wiki | {"path": photo}

                spans = (
                    [1.0, 4.0] if deep and radius <= 50
                    else [min(3.0, max(0.6, radius * 2))] if radius <= 25 else []
                )  # fmt: skip

                for n, span in enumerate(spans):
                    sat = out.path(f"ref{index}_sat{'' if n == 0 else n + 1}.jpg")

                    if await _satellite(client, lat, lon, span, sat):
                        item.setdefault("satellites", []).append(
                            {"path": sat, "span_km": span}
                        )

                if not deep:
                    return item

                ground = await geo_deep.commons_photos(
                    client, lat, lon, int(max(1000, min(5000, radius * 1000)))
                )
                street = await geo_deep.mapillary_photos(client, lat, lon)

                for n, photo in enumerate([*ground, *street]):
                    kind = "ground" if n < len(ground) else "street"
                    path = out.path(f"ref{index}_{kind}{n}.jpg")

                    if await geo_deep.download(client, photo["thumb"], path):
                        item["ground"].append(photo | {"path": path, "kind": kind})

                async with overpass:
                    found = await geo_deep.osm_features(
                        client, lat, lon, list(inventory.get("osm_features") or [])
                    )

                if found:
                    item["checks"] += geo_deep.osm_checks(found)

                if rows is not None:
                    sky = await geo_deep.skyline_check(
                        client,
                        rows,
                        aspect,
                        lat,
                        lon,
                        hilly=inventory.get("horizon")
                        in ("natural_hills", "mountains"),
                    )

                    if sky:
                        item["skyline"] = {
                            k: v for k, v in sky.items() if k not in ("ok", "text")
                        }
                        item["checks"].append({"ok": sky["ok"], "text": sky["text"]})

                if when is not None:
                    sun = geo_deep.sun_check(when, lat, lon, inventory)

                    if sun:
                        item["checks"].append(sun)

                    weather = await geo_deep.weather_check(lat, lon, when, seen_weather)

                    if weather:
                        item["checks"].append(weather)

                for zone in zones:
                    if zone["kind"] == "band":
                        inside = zone["lat_min"] <= lat <= zone["lat_max"]
                        item["checks"].append(
                            {
                                "ok": inside,
                                "text": f"{'inside' if inside else 'outside'} the band: {zone['label']}",
                            }
                        )
                    elif zone["kind"] == "ring":
                        inside = geo_deep.in_ring((lat, lon), zone)
                        item["checks"].append(
                            {
                                "ok": inside,
                                "text": f"{'on' if inside else 'off'} the ring: {zone['label']}",
                            }
                        )

                return item

            async def assess(
                cands: list[dict[str, Any]], base: int
            ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
                refs = list(
                    await asyncio.gather(
                        *(gather(base + i, c) for i, c in enumerate(cands))
                    )
                )
                content: list[dict[str, str]] = [
                    p for u in frame_units[: 5 if still else 4] for p in u
                ]
                # A fixed shuffled order, without the proposer's odds, so the verifier
                # cannot simply rubber-stamp the favourite.
                order = sorted(
                    refs,
                    key=lambda r: (hash(r["candidate"]["place"]) ^ hash(ctx.id)) % 9973,
                )

                for item in order:
                    c = item["candidate"]
                    content.append(
                        {
                            "type": "input_text",
                            "text": f"Candidate {item['index']}: {c['place']} ({c['latitude']:.5f}, {c['longitude']:.5f}), landmark '{c['landmark'] or 'none'}'.",
                        }
                    )

                    if "wiki" in item and (
                        part := _image_part(item["wiki"]["path"], 768)
                    ):
                        content += [
                            {
                                "type": "input_text",
                                "text": f"Wikipedia photo of '{item['wiki']['title']}':",
                            },
                            part,
                        ]
                        out.artifact(
                            f"ref{item['index']}_wiki", "image",
                            f"Reference: Wikipedia photo of {item['wiki']['title']} (candidate {item['index'] + 1})",
                            file=item["wiki"]["path"],
                        )  # fmt: skip

                    for n, sat in enumerate(item.get("satellites", [])):
                        if part := _image_part(sat["path"], 640):
                            content += [
                                {"type": "input_text", "text": f"Satellite view, {sat['span_km']:.1f} km across, centred on the candidate:"},
                                part,
                            ]  # fmt: skip
                            out.artifact(
                                f"ref{item['index']}_sat{'' if n == 0 else n + 1}", "image",
                                f"Satellite at candidate {item['index'] + 1}, {sat['span_km']:.0f} km across: {c['place'][:60]}",
                                file=sat["path"],
                            )  # fmt: skip

                    for n, photo in enumerate(item["ground"]):
                        if part := _image_part(photo["path"], 768):
                            where = (
                                "Wikimedia Commons"
                                if photo["kind"] == "ground"
                                else "Mapillary street-level"
                            )
                            content += [
                                {"type": "input_text", "text": f"{where} photo taken {photo['dist_m']} m from the candidate:"},
                                part,
                            ]  # fmt: skip
                            out.artifact(
                                f"ref{item['index']}_{photo['kind']}{n}", "image",
                                f"{where} photo {photo['dist_m']:,} m from candidate {item['index'] + 1}",
                                file=photo["path"], data={"url": photo.get("url", "")},
                            )  # fmt: skip

                    if item["checks"]:
                        marks = {True: "PASS", False: "FAIL", None: "unclear"}
                        content.append(
                            {"type": "input_text", "text": "Automatic checks for this candidate: "
                             + "; ".join(f"{marks[k['ok']]}: {k['text']}" for k in item["checks"])}
                        )  # fmt: skip

                if not any(
                    "wiki" in r or r.get("satellites") or r["ground"] for r in refs
                ):
                    top = cands[0]
                    return refs, {
                        "verdicts": [], "best": -1, "place": top["place"], "latitude": top["latitude"],
                        "longitude": top["longitude"], "radius_km": max(float(top["radius_km"]), 50.0),
                        "confidence": "low", "explanation": "No landmark photo or satellite view could be checked, so the best guess stays unverified.",
                    }  # fmt: skip

                out.progress(
                    0.75 if base == 0 else 0.9,
                    "verifying candidates against the real world",
                )
                checked = await provider.structured(
                    ctx.workdir, instructions=_VERIFY, content=content,
                    schema=_VERIFY_SCHEMA, name="geolocation_verification", effort=reasoning,
                )  # fmt: skip

                return refs, checked

            references, verdict = await assess(candidates, 0)
            scores = {v["candidate"]: v for v in verdict["verdicts"]}

            # ---- Deep: one coarse-to-fine round when the answer is only regional ----
            def confirmed(v: dict[str, Any], s: dict[int, Any]) -> bool:
                return (
                    v["best"] >= 0 and s.get(v["best"], {}).get("match", 0) >= _VERIFIED
                )

            if (
                deep
                and not confirmed(verdict, scores)
                and _has_location(
                    verdict["latitude"], verdict["longitude"], verdict["radius_km"]
                )
                and float(verdict["radius_km"]) <= 150
            ):
                out.progress(0.82, f"zooming in on {verdict['place'][:40]}")
                lat, lon = float(verdict["latitude"]), float(verdict["longitude"])
                region: list[dict[str, str]] = []

                for span in (40.0, 12.0):
                    path = out.path(f"region_{span:.0f}km.jpg")

                    if await _satellite(client, lat, lon, span, path) and (
                        part := _image_part(path, 768)
                    ):
                        region += [
                            {
                                "type": "input_text",
                                "text": f"Satellite view of the region, {span:.0f} km across:",
                            },
                            part,
                        ]

                summary = "\n".join(
                    f"Earlier candidate {r['index']}: {r['candidate']['place']}; checks: "
                    + ("; ".join(k["text"] for k in r["checks"]) or "none")
                    for r in references
                )
                try:
                    refined = await provider.structured(
                        ctx.workdir, instructions=geo_deep.REFINE,
                        content=[
                            {"type": "input_text", "text": f"{clue_text}\n\nRegion: around {verdict['place']} "
                             f"({lat:.4f}, {lon:.4f}), radius {float(verdict['radius_km']):.0f} km.\n{summary}"},
                            *region, *(p for u in frame_units[:4] for p in u),
                        ],
                        schema=_PROPOSE_SCHEMA, name="geolocation_refine", effort="high",
                    )  # fmt: skip
                except Exception as error:  # noqa: BLE001 - the first answer stands
                    notes.append(f"Refinement failed: {error}")
                    refined = None

                if refined:
                    spots = [
                        c
                        | {
                            "radius_km": min(3.0, float(c["radius_km"])),
                            "refined": True,
                        }
                        for c in refined["candidates"][:5]
                    ]
                    more, second = await assess(spots, len(candidates))
                    second_scores = {v["candidate"]: v for v in second["verdicts"]}
                    candidates = candidates + spots
                    references = references + more
                    scores |= second_scores
                    notes.append(
                        f"Refinement: {len(spots)} exact spots near {verdict['place']} checked."
                    )

                    if confirmed(second, second_scores):
                        verdict = second

        # Deterministic guard: two or more failed automatic checks rule a candidate out.
        for item in references:
            failed = [k["text"] for k in item["checks"] if k["ok"] is False]
            score = scores.get(item["index"])

            if len(failed) >= 2 and score and score["match"] > 0.55:
                score["match"] = 0.55
                score["mismatches"] = [
                    *score["mismatches"],
                    "automatic checks failed: " + "; ".join(failed),
                ]

        best = verdict["best"]
        ranked = sorted((float(v["match"]) for v in scores.values()), reverse=True)
        margin = 0.0
        cluster_km = 0.0

        if 0 <= best < len(candidates) and best in scores:
            anchor = (candidates[best]["latitude"], candidates[best]["longitude"])
            near, rivals = [], []

            for index, candidate in enumerate(candidates):
                if index == best or index not in scores:
                    continue

                apart = _haversine_km(
                    anchor, (candidate["latitude"], candidate["longitude"])
                )
                (near if apart <= 3.0 else rivals).append(
                    (apart, scores[index]["match"])
                )

            # Neighbouring candidates support the answer; only distant ones compete.
            margin = scores[best]["match"] - max((m for _, m in rivals), default=0.0)
            cluster_km = max((a for a, m in near if m >= 0.4), default=0.0)
        verified = (
            best >= 0
            and scores.get(best, {}).get("match", 0) >= _VERIFIED
            and margin >= 0.2
        )

        if verified and margin < 0.35 and verdict["confidence"] == "high":
            verdict["confidence"] = "medium"

        if verified:
            verdict["radius_km"] = max(float(verdict["radius_km"]), cluster_km, 0.2)
        else:
            verdict["confidence"] = "low"
            verdict["radius_km"] = max(float(verdict["radius_km"]), 50.0)

        located = _has_location(
            verdict["latitude"], verdict["longitude"], verdict["radius_km"]
        )
        verified = verified and located

        # ---- report -----------------------------------------------------------------------
        if not deep:
            async with ctx.waiting():
                provenance = await ctx.wait_for("provenance") or {}

        gps = provenance.get("location")
        error_km = None
        points = [
            {"lat": c["latitude"], "lon": c["longitude"], "label": f"Lead {i + 1}: {c['place']}",
             "accuracy_m": float(c["radius_km"]) * 1000, "kind": "ai-candidate"}
            for i, c in enumerate(candidates) if _has_location(c["latitude"], c["longitude"], c["radius_km"])
        ]  # fmt: skip
        if located:
            points.append({
                "lat": verdict["latitude"], "lon": verdict["longitude"],
                "label": f"{'Verified' if verified else 'Best guess'}: {verdict['place']}",
                "accuracy_m": float(verdict["radius_km"]) * 1000, "kind": "ai",
            })  # fmt: skip

        if gps:
            if located:
                error_km = _haversine_km(
                    (gps["lat"], gps["lon"]),
                    (verdict["latitude"], verdict["longitude"]),
                )
            points.append(
                {
                    "lat": gps["lat"],
                    "lon": gps["lon"],
                    "label": "GPS (metadata)",
                    "accuracy_m": gps.get("accuracy_m"),
                    "kind": "gps",
                }
            )

        out.artifact(
            "map",
            "map",
            "Geolocation: leads, verified answer and GPS",
            data={"points": points, **({"zones": zones} if zones else {})},
        )
        checks = {r["index"]: r["checks"] for r in references if r["checks"]}

        def proposed(c: dict[str, Any]) -> str:
            if c.get("refined"):
                return "refined spot"

            text = f"{c['probability'] * 100:.0f}%"

            return text + (
                f" ({c['votes']}/{c['passes']} passes)" if c.get("passes") else ""
            )

        out.artifact(
            "leads",
            "table",
            "Candidate places and how they held up",
            data={
                "columns": [
                    "#",
                    "place",
                    "landmark",
                    "proposed",
                    "match",
                    "matches",
                    "mismatches",
                ]
                + (["checks"] if checks else []),
                "rows": [
                    [
                        i + 1,
                        c["place"],
                        c["landmark"] or "-",
                        proposed(c),
                        f"{scores[i]['match'] * 100:.0f}%"
                        if i in scores
                        else "not checked",
                        "; ".join(scores[i]["matches"]) if i in scores else "",
                        "; ".join(scores[i]["mismatches"]) if i in scores else "",
                    ]
                    + (
                        [
                            "; ".join(
                                (
                                    "✓ "
                                    if k["ok"]
                                    else "✗ "
                                    if k["ok"] is False
                                    else "? "
                                )
                                + k["text"]
                                for k in checks.get(i, [])
                            )
                        ]
                        if checks
                        else []
                    )
                    for i, c in enumerate(candidates)
                ],
            },  # fmt: skip
        )

        for item in references:
            if item.get("skyline"):
                sky = item["skyline"]
                out.artifact(
                    f"skyline{item['index']}", "series",
                    f"Terrain horizon at candidate {item['index'] + 1} ({sky['score'] * 100:.0f}% skyline match)",
                    data={"x": list(range(360)), "series": [{"name": "horizon from the elevation model", "y": sky["profile"]}],
                          "x_unit": "° heading", "y_unit": "°", "marks": {"best match": sky["heading"]}},
                )  # fmt: skip

        if notes:
            out.artifact(
                "deep",
                "text",
                "Deep geolocation: what each extra method did",
                data={"text": "\n".join(notes)},
            )

        detail = (
            verdict["explanation"]
            + " Region reasoning: "
            + proposal["region_reasoning"]
        )

        if verified:
            detail += f" Checked against reference imagery: {scores[best]['match'] * 100:.0f}% match ({'; '.join(scores[best]['matches'][:4])})."
        else:
            detail += (
                " No candidate clearly beat the others against real-world reference photos "
                f"(best {ranked[0] * 100:.0f}%, {margin * 100:.0f} points ahead of the nearest rival "
                "elsewhere), so this stays a lead."
                if ranked
                else " Nothing could be checked against real-world reference photos, so this stays a lead."
            )

        if deep and 0 <= best < len(candidates):
            lead = candidates[best]

            if lead.get("passes"):
                detail += f" {lead['votes']} of {lead['passes']} independent passes proposed this place."

            passed = [k["text"] for k in checks.get(best, []) if k["ok"]]
            failed = [k["text"] for k in checks.get(best, []) if k["ok"] is False]

            if passed or failed:
                detail += (
                    " Automatic checks there: "
                    + "; ".join(
                        [
                            *(f"passed: {t}" for t in passed),
                            *(f"failed: {t}" for t in failed),
                        ]
                    )
                    + "."
                )

        if error_km is not None:
            detail += f" The file's GPS (never shown to the model) is {error_km:,.1f} km away."

        guess = {
            "place": verdict["place"], "region": "", "country": "",
            "latitude": verdict["latitude"] if located else None, "longitude": verdict["longitude"] if located else None,
            "radius_km": verdict["radius_km"], "confidence": verdict["confidence"], "location_available": located,
            "verified": verified, "match": scores.get(best, {}).get("match") if best >= 0 else None,
            "candidates": candidates, "error_km": error_km, "model": proposal.get("_model"),
            "effort": ctx.effort, "checks": checks, "zones": zones,
            # Kept for the case brief: verification replaces the old second-opinion check.
            "corroboration": {"agree": verified, "second_guess": "reference imagery did not match" if not verified else verdict["place"], "apart_km": None},
        }  # fmt: skip
        out.finding(
            "notice",
            (
                f"{'Located' if verified else 'Pixels suggest'}: {verdict['place']} (±{verdict['radius_km']:.0f} km, {verdict['confidence']} confidence)"
                if located
                else "Location unknown"
            ),
            detail,
            time_s=None if still else frames[0][0],
            kind="inferred",
            tags=["location", "osint"],
            data=guess,
        )
        ctx.publish(self.name, guess)
        out.progress(1.0)
