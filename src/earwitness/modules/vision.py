"""What an analyst would notice in the frames (AI vision, optional).

Keyframes go to a vision model with a strict JSON schema: setting, likely
region and its visual evidence, time of day, notable observations, and any
visual signs of manipulation. Everything it says is labelled AI-inferred.
"""

import base64

import cv2
import numpy as np

from earwitness import ai
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
        "synthetic": {
            "type": "object",
            "additionalProperties": False,
            "required": ["likelihood", "evidence"],
            "properties": {
                "likelihood": {
                    "type": "string",
                    "enum": [
                        "very unlikely",
                        "unlikely",
                        "possible",
                        "likely",
                        "very likely",
                    ],
                    "description": "How likely the picture was generated or heavily altered by an AI model.",
                },
                "evidence": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Concrete visual reasons: hands, text, reflections, textures, geometry, lighting, over-smooth skin.",
                },
            },
        },
    },
}
_SCHEMA["required"].append("synthetic")

_INSTRUCTIONS = (
    "You are a careful open-source-intelligence and video forensics analyst. You receive numbered "
    "keyframes from one video. Report only what is visible; never invent text you cannot read. Prefer "
    "'unknown' over guessing. Location clues: signage language and fonts, plug sockets, licence plates, "
    "road markings and driving side, architecture, vegetation, shop brands, weather and sun. Look for "
    "reflections (windows, glasses, screens) that reveal what is off-camera. Note manipulation signs only "
    "if concrete (warped background near a face, inconsistent shadows, mismatched lighting, cloned "
    "texture, impossible geometry). For 'synthetic', judge whether an AI model generated or heavily "
    "altered the picture from visual evidence only (malformed hands or text, impossible reflections, "
    "waxy skin, melted backgrounds, inconsistent perspective); say 'unlikely' when nothing concrete suggests it."
)

_STILL_INSTRUCTIONS = _INSTRUCTIONS.replace(
    "You receive numbered keyframes from one video.",
    "You receive one photograph: image 1 is the whole picture, images 2-5 are its four quarters at higher resolution (top-left, top-right, bottom-left, bottom-right).",
)


def _fit(rgb: np.ndarray, max_side: int) -> np.ndarray:
    scale = max_side / max(rgb.shape[:2])

    return (
        rgb
        if scale >= 1
        else cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    )


class Vision:
    name = "vision"
    title = "AI VISUAL ANALYST"
    description = "Keyframes read by a vision model: place, time of day, notable details, manipulation signs (optional)."
    depends = ["timeline"]

    async def run(self, ctx: CaseContext, out: Reporter) -> None:
        provider = ai.for_case(ctx.options)

        if provider is None:
            raise Skip(ai.MISSING_KEY)

        info = await ctx.video_info()
        still = ctx.kind == "image"

        if info is None and not still:
            raise Skip("no video stream")

        if still:
            image = ctx.image()
            h, w = image.shape[:2]
            frames = [("the whole picture", ctx.image_scaled(1536))]

            for row in range(2):
                for col in range(2):
                    tile = image[
                        row * h // 2 : (row + 1) * h // 2,
                        col * w // 2 : (col + 1) * w // 2,
                    ]
                    frames.append(
                        (
                            f"{'top' if row == 0 else 'bottom'}-{'left' if col == 0 else 'right'} quarter",
                            _fit(tile, 1024),
                        )
                    )

            times = [0.0] * len(frames)
            content = [
                {
                    "type": "input_text",
                    "text": "One photograph follows: the whole picture, then its four quarters at higher resolution.",
                }
            ]
        else:
            timeline = await ctx.wait_for("timeline") or {}
            times = list(np.linspace(0.3, max(0.3, info.duration_s - 0.3), _FRAMES))

            for cut in (timeline.get("cuts") or [])[:3]:
                times.append(cut + 0.3)

            times = sorted({round(t, 2) for t in times})[: _FRAMES + 2]
            frames = []
            content = [
                {
                    "type": "input_text",
                    "text": f"{len(times)} keyframes follow, numbered from 1 in time order.",
                }
            ]

        out.progress(0.1, "grabbing keyframes")

        for number, time_s in enumerate(times, start=1):
            if still:
                label, frame = frames[number - 1]
            else:
                label = f"at {time_s:.1f} s"
                frame = await ctx.frame_at(time_s, width=min(1024, info.width))

            if frame is None:
                continue

            _, jpeg = cv2.imencode(
                ".jpg",
                cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 82],
            )
            content.append(
                {
                    "type": "input_text",
                    "text": f"{'Image' if still else 'Frame'} {number} {label}:",
                }
            )
            content.append(
                {
                    "type": "input_image",
                    "image_url": "data:image/jpeg;base64,"
                    + base64.b64encode(jpeg.tobytes()).decode(),
                }
            )

        out.progress(0.3, "consulting vision model")
        result = await provider.structured(
            ctx.workdir,
            instructions=_STILL_INSTRUCTIONS if still else _INSTRUCTIONS,
            content=content,
            schema=_SCHEMA,
            name="visual_report",
        )
        frame_time = {i: (None if still else t) for i, t in enumerate(times, start=1)}
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

        synthetic = result.get("synthetic") or {}

        if synthetic.get("likelihood") in ("possible", "likely", "very likely"):
            out.finding(
                "warning" if synthetic["likelihood"] != "possible" else "notice",
                f"AI eyewitness: AI generation {synthetic['likelihood']}",
                "Judging from the picture alone: "
                + "; ".join(synthetic.get("evidence", [])[:5])
                + ". This is a model's opinion about another model's work; weigh it with the file-level signals.",
                kind="inferred",
                tags=["synthetic", "ai-opinion"],
                data=synthetic,
            )
        elif synthetic.get("likelihood"):
            out.finding(
                "info",
                f"AI eyewitness: AI generation {synthetic['likelihood']}",
                "Nothing in the picture itself suggested a generated image"
                + (
                    ": " + "; ".join(synthetic.get("evidence", [])[:3])
                    if synthetic.get("evidence")
                    else ""
                )
                + ".",
                kind="inferred",
                tags=["synthetic", "ai-opinion"],
                data=synthetic,
            )

        ctx.publish(self.name, result)
        out.progress(1.0)
