"""What an analyst would notice in the frames (OpenAI vision, optional).

Keyframes go to a vision model with a strict JSON schema: setting, likely
region and its visual evidence, time of day, notable observations, and any
visual signs of manipulation. Everything it says is labelled AI-inferred.
"""

import base64

import cv2
import numpy as np

from earwitness import openai_client
from earwitness.modules.base import CaseContext
from earwitness.modules.base import Reporter
from earwitness.modules.base import Skip

_FRAMES = 6

_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "scene",
        "setting",
        "location",
        "time_of_day",
        "weather",
        "observations",
        "manipulation_signs",
    ],
    "properties": {
        "scene": {
            "type": "string",
            "description": "One or two sentences: what the footage shows.",
        },
        "setting": {
            "type": "string",
            "enum": ["indoor", "outdoor", "vehicle", "mixed", "unknown"],
        },
        "location": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "guess",
                "confidence",
                "evidence",
                "latitude",
                "longitude",
                "radius_km",
            ],
            "properties": {
                "latitude": {
                    "type": "number",
                    "description": "Best-estimate latitude (0 if unknown).",
                },
                "longitude": {
                    "type": "number",
                    "description": "Best-estimate longitude (0 if unknown).",
                },
                "radius_km": {
                    "type": "number",
                    "description": "Honest ~80% radius; 20000 if the place is unknown.",
                },
                "guess": {
                    "type": "string",
                    "description": "Most specific defensible place (country/city/venue), or 'unknown'.",
                },
                "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
                "evidence": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Visual clues: signage language, plugs, plates, architecture, vegetation, road markings…",
                },
            },
        },
        "time_of_day": {
            "type": "object",
            "additionalProperties": False,
            "required": ["guess", "evidence"],
            "properties": {"guess": {"type": "string"}, "evidence": {"type": "string"}},
        },
        "weather": {"type": "string"},
        "observations": {
            "type": "array",
            "maxItems": 8,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["frame", "category", "description", "significance"],
                "properties": {
                    "frame": {"type": "integer"},
                    "category": {
                        "type": "string",
                        "enum": [
                            "text",
                            "landmark",
                            "vehicle",
                            "person",
                            "object",
                            "screen",
                            "reflection",
                            "anomaly",
                        ],
                    },
                    "description": {"type": "string"},
                    "significance": {
                        "type": "string",
                        "description": "Why an investigator would care.",
                    },
                },
            },
        },
        "manipulation_signs": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["frame", "description"],
                "properties": {
                    "frame": {"type": "integer"},
                    "description": {"type": "string"},
                },
            },
        },
    },
}

_INSTRUCTIONS = (
    "You are a careful open-source-intelligence and video forensics analyst. You receive numbered "
    "keyframes from one video. Report only what is visible; never invent text you cannot read. Prefer "
    "'unknown' over guessing. Location clues: signage language and fonts, plug sockets, licence plates, "
    "road markings and driving side, architecture, vegetation, shop brands, weather and sun. Look for "
    "reflections (windows, glasses, screens) that reveal what is off-camera. Note manipulation signs only "
    "if concrete (warped background near a face, inconsistent shadows, mismatched lighting, cloned "
    "texture, impossible geometry)."
)


class Vision:
    name = "vision"
    title = "AI VISUAL ANALYST"
    description = "Keyframes read by a vision model: place, time of day, notable details, manipulation signs (optional)."
    depends = ["timeline"]

    async def run(self, ctx: CaseContext, out: Reporter) -> None:
        if not openai_client.enabled():
            raise Skip("set OPENAI_API_KEY to enable")

        info = await ctx.video_info()

        if info is None:
            raise Skip("no video stream")

        timeline = await ctx.wait_for("timeline") or {}
        times = list(np.linspace(0.3, max(0.3, info.duration_s - 0.3), _FRAMES))

        for cut in (timeline.get("cuts") or [])[:3]:
            times.append(cut + 0.3)

        times = sorted({round(t, 2) for t in times})[: _FRAMES + 2]
        content = [
            {
                "type": "input_text",
                "text": f"{len(times)} keyframes follow, numbered from 1 in time order.",
            }
        ]
        out.progress(0.1, "grabbing keyframes")

        for number, time_s in enumerate(times, start=1):
            frame = await ctx.frame_at(time_s, width=min(1024, info.width))

            if frame is None:
                continue

            _, jpeg = cv2.imencode(
                ".jpg",
                cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 82],
            )
            content.append(
                {"type": "input_text", "text": f"Frame {number} at {time_s:.1f} s:"}
            )
            content.append(
                {
                    "type": "input_image",
                    "image_url": "data:image/jpeg;base64,"
                    + base64.b64encode(jpeg.tobytes()).decode(),
                }
            )

        out.progress(0.3, "consulting vision model")
        result = await openai_client.structured(
            ctx.workdir,
            instructions=_INSTRUCTIONS,
            content=content,
            schema=_SCHEMA,
            name="visual_report",
        )
        frame_time = {i: t for i, t in enumerate(times, start=1)}
        model = result.get("_model", "model")
        out.artifact(
            "report",
            "text",
            f"Visual analyst ({model})",
            data={"text": result["scene"]},
        )
        out.finding(
            "info",
            f"Scene: {result['setting']}",
            result["scene"]
            + (
                f" Weather: {result['weather']}."
                if result.get("weather") and result["weather"].lower() != "unknown"
                else ""
            ),
            kind="inferred",
            tags=["scene"],
        )
        location = result["location"]

        if location["guess"].lower() != "unknown":
            out.finding(
                "notice",
                f"Looks like: {location['guess']} ({location['confidence']} confidence)",
                "Visual clues: " + "; ".join(location["evidence"][:6]) + ".",
                kind="inferred",
                tags=["location"],
                data=location,
            )

        if result["time_of_day"]["guess"].lower() != "unknown":
            out.finding(
                "info",
                f"Time of day: {result['time_of_day']['guess']}",
                result["time_of_day"]["evidence"],
                kind="inferred",
                tags=["time"],
            )

        for item in result["observations"][:5]:
            out.finding(
                "info",
                f"{item['category'].capitalize()}: {item['description'][:80]}",
                f"{item['description']} {item['significance']}",
                time_s=frame_time.get(item["frame"]),
                kind="inferred",
                tags=["observation", item["category"]],
            )

        for sign in result["manipulation_signs"][:5]:
            out.finding(
                "warning",
                "Possible visual manipulation",
                sign["description"],
                time_s=frame_time.get(sign["frame"]),
                kind="inferred",
                tags=["tamper"],
            )

        ctx.publish(self.name, result)
        out.progress(1.0)
