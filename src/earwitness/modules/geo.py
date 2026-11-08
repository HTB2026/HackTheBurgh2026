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
from typing import Any

import cv2
import httpx
import numpy as np

from earwitness import ai
from earwitness.modules.base import CaseContext
from earwitness.modules.base import Reporter
from earwitness.modules.base import Skip
from earwitness.modules.base import run_tool

_FRAMES = 6
_INTAKE_FRAMES = 2
_CANDIDATES = 4
_VERIFIED = 0.6
_USER_AGENT = "earwitness-forensics/0.2 (hackathon research demo)"

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
    ctx: CaseContext, duration: float
) -> list[tuple[float, np.ndarray]]:
    """The sharpest frame from each of _FRAMES equal stretches of the clip."""
    chosen = []
    edges = np.linspace(0.2, max(0.4, duration - 0.2), _FRAMES + 1)

    for a, b in zip(edges[:-1], edges[1:]):
        best = None

        for t in np.linspace(a, b, 3):
            frame = await ctx.frame_at(float(t), width=1280)

            if frame is None:
                continue

            sharp = float(
                cv2.Laplacian(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var()
            )

            if best is None or sharp > best[0]:
                best = (sharp, float(t), frame)

        if best:
            chosen.append((best[1], best[2]))

    return chosen


class Geo:
    name = "geo"
    title = "GEOLOCATION"
    description = "Where it was filmed, from pixels and sound alone: candidates proposed, then verified against Wikipedia photos and satellite imagery."
    depends = ["provenance", "transcript", "ocr", "vision", "sounds", "light"]

    async def run(self, ctx: CaseContext, out: Reporter) -> None:
        provider = ai.for_case(ctx.options)

        if provider is None:
            raise Skip(ai.MISSING_KEY)

        info = await ctx.video_info()
        still = ctx.kind == "image"

        if info is None and not still:
            raise Skip("no video stream")

        out.progress(0.05, "choosing frames")

        if still:
            # The whole photograph, at the resolution the model takes, plus the
            # four quarters so that small signs and plates stay legible.
            image = ctx.image()
            h, w = image.shape[:2]
            frames = [(0.0, ctx.image_scaled(1280))]

            for row in range(2):
                for col in range(2):
                    tile = image[
                        row * h // 2 : (row + 1) * h // 2,
                        col * w // 2 : (col + 1) * w // 2,
                    ]
                    scale = min(1.0, 1024 / max(tile.shape[:2]))
                    frames.append(
                        (
                            0.0,
                            cv2.resize(
                                tile,
                                None,
                                fx=scale,
                                fy=scale,
                                interpolation=cv2.INTER_AREA,
                            )
                            if scale < 1
                            else tile,
                        )
                    )
        else:
            frames = await _pick_frames(ctx, info.duration_s)

        if not frames:
            raise Skip("no decodable frames")

        evidence: list[dict[str, str]] = []
        frame_paths = []
        labels = [
            "the whole picture",
            "top-left quarter",
            "top-right quarter",
            "bottom-left quarter",
            "bottom-right quarter",
        ]

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
            evidence += [
                {
                    "type": "input_text",
                    "text": f"Image {number + 1} ({labels[number]}):"
                    if still
                    else f"Frame {number + 1} (at {time_s:.1f} s):",
                },
                _image_part(path),
            ]

        # ---- geo-sleuth intake: zoomed edges/corners + OCR -------------------
        ocr_lines: list[str] = []
        skill = os.environ.get("EARWITNESS_GEOSLEUTH")
        script = os.path.join(os.path.expanduser(skill or ""), "scripts", "intake.py")

        if skill and os.path.isfile(script):
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
                )[:_INTAKE_FRAMES]

            for number, (time_s, path) in enumerate(sharpest):
                out.progress(
                    0.1 + 0.15 * number / _INTAKE_FRAMES,
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
                        evidence += [
                            {
                                "type": "input_text",
                                "text": f"Zoomed crop {os.path.basename(crop)} of frame at {time_s:.1f}s:",
                            },
                            part,
                        ]

        # ---- non-visual clues (never the file's metadata or GPS) --------------
        out.progress(0.3, "collecting clues from sound, text and light")
        speech = await ctx.wait_for("transcript") or {}
        ocr = await ctx.wait_for("ocr") or {}
        heard = await ctx.wait_for("sounds") or {}
        light = await ctx.wait_for("light") or {}
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
                        )
                    },
                    ensure_ascii=False,
                )
            )

        if light.get("grid_hz"):
            clues.append(
                f"The lighting flickers at {2 * light['grid_hz']} Hz: the mains grid is {light['grid_hz']} Hz."
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
        proposal = await provider.structured(
            ctx.workdir,
            instructions=_PROPOSE,
            content=[{"type": "input_text", "text": clue_text}, *evidence],
            schema=_PROPOSE_SCHEMA,
            name="geolocation_candidates",
            effort="high",
        )
        candidates = proposal["candidates"][:_CANDIDATES]

        # ---- verify against the real world ------------------------------------------
        out.progress(0.6, "fetching reference photos and satellite views")
        references: list[dict[str, Any]] = []

        async with httpx.AsyncClient(
            timeout=20, follow_redirects=True, headers={"User-Agent": _USER_AGENT}
        ) as client:

            async def gather(index: int, candidate: dict[str, Any]) -> dict[str, Any]:
                item: dict[str, Any] = {"index": index, "candidate": candidate}

                if candidate.get("landmark"):
                    photo = out.path(f"ref{index}_wiki.jpg")
                    wiki = await _wikipedia_photo(client, candidate["landmark"], photo)

                    if wiki:
                        item["wiki"] = wiki | {"path": photo}

                if float(candidate["radius_km"]) <= 25:
                    sat = out.path(f"ref{index}_sat.jpg")
                    span = min(3.0, max(0.6, float(candidate["radius_km"]) * 2))

                    if await _satellite(
                        client, candidate["latitude"], candidate["longitude"], span, sat
                    ):
                        item["satellite"] = {"path": sat, "span_km": span}

                return item

            references = await asyncio.gather(
                *(gather(i, c) for i, c in enumerate(candidates))
            )

        verify_content: list[dict[str, str]] = list(
            evidence[: 2 * min(5 if still else 4, len(frame_paths))]
        )

        # A fixed shuffled order, without the proposer's odds, so the verifier
        # cannot simply rubber-stamp the favourite.
        order = sorted(
            references,
            key=lambda r: (hash(r["candidate"]["place"]) ^ hash(ctx.id)) % 9973,
        )

        for item in order:
            c = item["candidate"]
            verify_content.append(
                {
                    "type": "input_text",
                    "text": f"Candidate {item['index']}: {c['place']} ({c['latitude']:.5f}, {c['longitude']:.5f}), landmark '{c['landmark'] or 'none'}'.",
                }
            )

            if "wiki" in item and (part := _image_part(item["wiki"]["path"], 768)):
                verify_content += [
                    {
                        "type": "input_text",
                        "text": f"Wikipedia photo of '{item['wiki']['title']}':",
                    },
                    part,
                ]
                out.artifact(
                    f"ref{item['index']}_wiki",
                    "image",
                    f"Reference: Wikipedia photo of {item['wiki']['title']} (candidate {item['index'] + 1})",
                    file=item["wiki"]["path"],
                )

            if "satellite" in item and (
                part := _image_part(item["satellite"]["path"], 640)
            ):
                verify_content += [
                    {
                        "type": "input_text",
                        "text": f"Satellite view, {item['satellite']['span_km']:.1f} km across, centred on the candidate:",
                    },
                    part,
                ]
                out.artifact(
                    f"ref{item['index']}_sat",
                    "image",
                    f"Satellite at candidate {item['index'] + 1}: {c['place'][:60]}",
                    file=item["satellite"]["path"],
                )

        has_references = any("wiki" in r or "satellite" in r for r in references)
        out.progress(0.75, "verifying candidates against the real world")

        if has_references:
            verdict = await provider.structured(
                ctx.workdir, instructions=_VERIFY, content=verify_content,
                schema=_VERIFY_SCHEMA, name="geolocation_verification", effort="high",
            )  # fmt: skip
        else:
            top = candidates[0]
            verdict = {
                "verdicts": [], "best": -1, "place": top["place"], "latitude": top["latitude"],
                "longitude": top["longitude"], "radius_km": max(float(top["radius_km"]), 50.0),
                "confidence": "low", "explanation": "No landmark photo or satellite view could be checked, so the best guess stays unverified.",
            }  # fmt: skip

        scores = {v["candidate"]: v for v in verdict["verdicts"]}
        best = verdict["best"]
        ranked = sorted((float(v["match"]) for v in verdict["verdicts"]), reverse=True)
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

        # ---- report -----------------------------------------------------------------------
        provenance = await ctx.wait_for("provenance") or {}
        gps = provenance.get("location")
        error_km = None
        points = [
            {"lat": c["latitude"], "lon": c["longitude"], "label": f"Lead {i + 1}: {c['place']}",
             "accuracy_m": float(c["radius_km"]) * 1000, "kind": "ai-candidate"}
            for i, c in enumerate(candidates)
        ]  # fmt: skip
        points.append({
            "lat": verdict["latitude"], "lon": verdict["longitude"],
            "label": f"{'Verified' if verified else 'Best guess'}: {verdict['place']}",
            "accuracy_m": float(verdict["radius_km"]) * 1000, "kind": "ai",
        })  # fmt: skip

        if gps:
            error_km = _haversine_km(
                (gps["lat"], gps["lon"]), (verdict["latitude"], verdict["longitude"])
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
            data={"points": points},
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
                ],
                "rows": [
                    [
                        i + 1,
                        c["place"],
                        c["landmark"] or "-",
                        f"{c['probability'] * 100:.0f}%",
                        f"{scores[i]['match'] * 100:.0f}%"
                        if i in scores
                        else "not checked",
                        "; ".join(scores[i]["matches"]) if i in scores else "",
                        "; ".join(scores[i]["mismatches"]) if i in scores else "",
                    ]
                    for i, c in enumerate(candidates)
                ],
            },  # fmt: skip
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

        if error_km is not None:
            detail += f" The file's GPS (never shown to the model) is {error_km:,.1f} km away."

        guess = {
            "place": verdict["place"], "region": "", "country": "",
            "latitude": verdict["latitude"], "longitude": verdict["longitude"],
            "radius_km": verdict["radius_km"], "confidence": verdict["confidence"],
            "verified": verified, "match": scores.get(best, {}).get("match") if best >= 0 else None,
            "candidates": candidates, "error_km": error_km, "model": proposal.get("_model"),
            # Kept for the case brief: verification replaces the old second-opinion check.
            "corroboration": {"agree": verified, "second_guess": "reference imagery did not match" if not verified else verdict["place"], "apart_km": None},
        }  # fmt: skip
        out.finding(
            "notice",
            f"{'Located' if verified else 'Pixels suggest'}: {verdict['place']} (±{verdict['radius_km']:.0f} km, {verdict['confidence']} confidence)",
            detail,
            time_s=None if still else frames[0][0],
            kind="inferred",
            tags=["location", "osint"],
            data=guess,
        )
        ctx.publish(self.name, guess)
        out.progress(1.0)
