"""Gemini provider: the same calls as ``earwitness.ai.openai``, via google-genai.

The key comes from GEMINI_API_KEY (or GOOGLE_API_KEY). Answers are cached per
case in ``gemini-cache/``. Modules build content in the neutral format
(``input_text`` / ``input_image`` with a data URL) and the same JSON schemas
they use for OpenAI; this module converts both.
"""

import asyncio
import base64
import hashlib
import json
import os
from typing import Any

# Tried in order until one answers; the first that works is remembered.
_MODELS = [
    m
    for m in [
        os.environ.get("EARWITNESS_GEMINI_MODEL"),
        "gemini-3.8-flash",
        "gemini-3.7-flash",
        "gemini-3.5-flash",
    ]
    if m
]
# The transcript's cross-check pass must use a different model from the primary one.
_CHECK_MODELS = [
    m
    for m in [
        os.environ.get("EARWITNESS_GEMINI_CHECK_MODEL"),
        "gemini-3.7-flash",
        "gemini-3.5-flash",
    ]
    if m
]
_working_model: str | None = None


def _key() -> str | None:
    return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")


def enabled() -> bool:
    return bool(_key())


def _client():
    from google import genai

    return genai.Client(api_key=_key())


def _cache(workdir: str, key: str) -> str:
    directory = os.path.join(workdir, "gemini-cache")
    os.makedirs(directory, exist_ok=True)

    return os.path.join(
        directory, hashlib.sha1(key.encode()).hexdigest()[:20] + ".json"
    )


def _cached(path: str) -> dict[str, Any] | None:
    if not os.path.isfile(path):
        return None

    with open(path) as handle:
        return json.load(handle)


def _save(path: str, result: dict[str, Any]) -> dict[str, Any]:
    with open(path, "w") as handle:
        json.dump(result, handle, default=str)

    return result


def schema_for_gemini(schema: Any) -> Any:
    """An OpenAI strict-mode JSON schema in the form Gemini accepts.

    Type lists (``["string", "null"]``) become ``anyOf``; everything else the
    modules use (objects, arrays, enums, min/maxItems, additionalProperties)
    is passed through unchanged.
    """
    if isinstance(schema, list):
        return [schema_for_gemini(item) for item in schema]

    if not isinstance(schema, dict):
        return schema

    out = {key: schema_for_gemini(value) for key, value in schema.items()}

    if isinstance(out.get("type"), list):
        kinds = out.pop("type")
        out["anyOf"] = [{"type": kind} for kind in kinds]

    return out


def parts_for_gemini(content: list[dict[str, Any]]) -> list[Any]:
    """Neutral content parts (input_text / input_image) as Gemini parts."""
    from google.genai import types

    parts = []

    for part in content:
        kind = part.get("type")

        if kind == "input_text":
            parts.append(types.Part.from_text(text=part["text"]))
        elif kind == "input_image":
            url = part["image_url"]

            if not url.startswith("data:"):
                raise ValueError("Gemini images must be data: URLs")

            header, data = url.split(",", 1)
            mime = header[len("data:") :].split(";")[0] or "image/jpeg"
            parts.append(
                types.Part.from_bytes(data=base64.b64decode(data), mime_type=mime)
            )
        else:
            raise ValueError(f"unsupported content part for Gemini: {kind}")

    return parts


_BUSY = (429, 500, 502, 503, 504)
_BACKOFF_S = (2.0, 6.0)


def _fatal(error: Exception) -> bool:
    """A bad or unauthorised key fails the same way on every model: stop."""
    return getattr(error, "code", None) in (401, 403)


def _retry_after_s(error: Exception) -> float | None:
    """The wait Google asks for on a 429 (RetryInfo.retryDelay, e.g. "27603s")."""
    details = getattr(error, "details", None)

    if not isinstance(details, dict):
        return None

    for item in (details.get("error") or {}).get("details") or []:
        delay = str(item.get("retryDelay") or "")

        if delay.endswith("s"):
            try:
                return float(delay[:-1])
            except ValueError:
                return None

    return None


def _quota_spent(error: Exception) -> bool:
    """A 429 that only clears much later (a daily quota): retrying now is pointless."""
    if getattr(error, "code", None) != 429:
        return False

    wait = _retry_after_s(error)

    return (wait is not None and wait > 60) or "PerDay" in str(error)


def _explain(error: Exception | None) -> str:
    if error is not None and _quota_spent(error):
        wait = _retry_after_s(error)
        hours = f" It resets in about {wait / 3600:.0f} h." if wait else ""

        return (
            "Gemini quota used up for today on every model (the free tier allows "
            f"about 20 requests per model per day; one case uses 5-10).{hours} "
            "Rerun this case with OpenAI, or add billing to the Gemini key."
        )

    return f"no Gemini model available: {error}"


async def _generate(
    models: list[str],
    *,
    instructions: str,
    parts: list[Any],
    schema: dict[str, Any] | None,
) -> tuple[str, str]:
    """Text of the first model that answers, and that model's name."""
    from google.genai import errors
    from google.genai import types

    config = types.GenerateContentConfig(
        system_instruction=instructions,
        response_mime_type="application/json",
        response_json_schema=schema_for_gemini(schema) if schema else None,
    )
    client = _client()
    last_error: Exception | None = None

    for model in models:
        # A busy model (429/5xx, "high demand") is retried after a pause, then
        # the next model is tried; an unknown or unsupported model, or one whose
        # daily quota is spent, moves on at once.
        for pause in (0.0, *_BACKOFF_S):
            if pause:
                await asyncio.sleep(pause)

            try:
                response = await client.aio.models.generate_content(
                    model=model,
                    contents=[types.Content(role="user", parts=parts)],
                    config=config,
                )
            except errors.APIError as error:
                if _fatal(error):
                    raise

                last_error = error

                if getattr(error, "code", None) in _BUSY and not _quota_spent(error):
                    continue

                break

            return response.text or "", model

    raise RuntimeError(_explain(last_error))


def _models() -> list[str]:
    """The last model that worked first, then the rest (a model can run out of quota)."""
    if not _working_model:
        return _MODELS

    return [_working_model, *(m for m in _MODELS if m != _working_model)]


async def structured(
    workdir: str,
    *,
    instructions: str,
    content: list[dict[str, Any]],
    schema: dict[str, Any],
    name: str,
    effort: str | None = None,
) -> dict[str, Any]:
    """JSON-schema-constrained response from the first model that works.

    ``effort`` is accepted for parity with OpenAI; Gemini's Pro models decide
    how long to think on their own.
    """
    global _working_model

    path = _cache(
        workdir,
        json.dumps([instructions, content, schema, effort], sort_keys=True, default=str),
    )

    if (hit := _cached(path)) is not None:
        return hit

    text, model = await _generate(
        _models(),
        instructions=instructions,
        parts=parts_for_gemini(content),
        schema=schema,
    )
    _working_model = model
    result = json.loads(text)
    result["_model"] = model

    return _save(path, result)


_TRANSCRIPT_SCHEMA = {
    "type": "object",
    "properties": {
        "segments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "start": {"type": "number", "description": "seconds from the start"},
                    "end": {"type": "number", "description": "seconds from the start"},
                    "text": {"type": "string"},
                },
                "required": ["start", "end", "text"],
            },
        },
    },
    "required": ["segments"],
}
_TRANSCRIPT_INSTRUCTIONS = (
    "Transcribe the speech in this recording verbatim, in the language spoken (do not "
    "translate). Split it into segments of one sentence or phrase with start and end "
    "times in seconds. Leave out music, noise and silence. If nobody speaks, return no "
    "segments. Never invent words you cannot hear."
)


def _audio_part(data: bytes, mime: str) -> Any:
    from google.genai import types

    return types.Part.from_bytes(data=data, mime_type=mime)


def _mime(path: str) -> str:
    return "audio/mpeg" if path.lower().endswith(".mp3") else "audio/wav"


async def transcribe(
    workdir: str, audio_path: str, pass_: str = "primary"
) -> dict[str, Any]:
    """Speech to text with timed segments, in the shape whisper-1 returns."""
    with open(audio_path, "rb") as handle:
        audio = handle.read()

    path = _cache(workdir, f"transcribe:{pass_}:" + hashlib.sha1(audio).hexdigest())

    if (hit := _cached(path)) is not None:
        return hit

    models = _models()

    if pass_ == "check":
        models = [m for m in _CHECK_MODELS if m not in models[:1]]

    text, model = await _generate(
        models,
        instructions=_TRANSCRIPT_INSTRUCTIONS,
        parts=[_audio_part(audio, _mime(audio_path))],
        schema=_TRANSCRIPT_SCHEMA,
    )
    segments = [
        {
            "start": float(s.get("start", 0) or 0),
            "end": float(s.get("end", 0) or 0),
            "text": str(s.get("text", "")),
            "no_speech_prob": 0.0,
        }
        for s in json.loads(text).get("segments") or []
        if isinstance(s, dict)
    ]

    return _save(
        path,
        {
            "text": " ".join(s["text"].strip() for s in segments).strip(),
            "segments": segments,
            "model": model,
        },
    )


async def listen(workdir: str, wav_path: str, instructions: str) -> dict[str, Any]:
    """Ask Gemini about a recording; returns its JSON answer."""
    from google.genai import types

    with open(wav_path, "rb") as handle:
        audio = handle.read()

    path = _cache(workdir, "listen:" + instructions + hashlib.sha1(audio).hexdigest())

    if (hit := _cached(path)) is not None:
        return hit

    text, model = await _generate(
        _models(),
        instructions=instructions,
        parts=[
            types.Part.from_text(text="Analyse this recording. JSON only."),
            _audio_part(audio, _mime(wav_path)),
        ],
        schema=None,
    )
    start, end = text.find("{"), text.rfind("}")

    if start < 0 or end <= start:
        raise ValueError("model did not return JSON")

    result = json.loads(text[start : end + 1])
    result["_model"] = model

    return _save(path, result)
