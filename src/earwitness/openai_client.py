"""Thin OpenAI wrapper: key from the environment only, per-case caching, model fallback."""

import hashlib
import json
import os
from typing import Any

_VISION_MODELS = [
    m
    for m in [
        os.environ.get("EARWITNESS_OPENAI_MODEL"),
        "gpt-5.5",
        "gpt-5.4",
        "gpt-5",
        "gpt-4.1",
    ]
    if m
]
_working_model: str | None = None


def enabled() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY"))


def _client():
    from openai import AsyncOpenAI

    return AsyncOpenAI()


def _cache(workdir: str, key: str) -> str:
    directory = os.path.join(workdir, "openai-cache")
    os.makedirs(directory, exist_ok=True)

    return os.path.join(
        directory, hashlib.sha1(key.encode()).hexdigest()[:20] + ".json"
    )


async def structured(
    workdir: str,
    *,
    instructions: str,
    content: list[dict[str, Any]],
    schema: dict[str, Any],
    name: str,
    effort: str | None = None,
) -> dict[str, Any]:
    """JSON-schema-constrained response from the first model that works."""
    global _working_model
    from openai import APIStatusError
    from openai import NotFoundError

    key = json.dumps(
        [instructions, content, schema, effort], sort_keys=True, default=str
    )
    path = _cache(workdir, key)

    if os.path.isfile(path):
        with open(path) as handle:
            return json.load(handle)

    client = _client()
    models = [_working_model] if _working_model else _VISION_MODELS
    last_error: Exception | None = None

    for model in models:
        try:
            extra = {"reasoning": {"effort": effort}} if effort else {}
            response = await client.responses.create(
                **extra,
                model=model,
                instructions=instructions,
                input=[{"role": "user", "content": content}],
                text={
                    "format": {
                        "type": "json_schema",
                        "name": name,
                        "schema": schema,
                        "strict": True,
                    }
                },
            )
        except (NotFoundError, APIStatusError) as error:
            # Unknown model or no access: try the next one. Auth/quota errors repeat for all.
            last_error = error
            status = getattr(error, "status_code", None)

            if status in (401, 429):
                raise

            continue

        _working_model = model
        result = json.loads(response.output_text)
        result["_model"] = model

        with open(path, "w") as handle:
            json.dump(result, handle)

        return result

    raise RuntimeError(f"no OpenAI model available: {last_error}")


async def transcribe(
    workdir: str, audio_path: str, model: str = "whisper-1"
) -> dict[str, Any]:
    """Speech to text. whisper-1 gives timed segments and a language guess;
    newer models (gpt-4o-transcribe) give text only."""
    with open(audio_path, "rb") as handle:
        digest = hashlib.sha1(handle.read()).hexdigest()

    path = _cache(workdir, f"transcribe:{model}:" + digest)

    if os.path.isfile(path):
        with open(path) as handle:
            return json.load(handle)

    client = _client()

    with open(audio_path, "rb") as handle:
        if model == "whisper-1":
            response = await client.audio.transcriptions.create(
                model=model,
                file=handle,
                response_format="verbose_json",
                timestamp_granularities=["segment"],
            )
        else:
            response = await client.audio.transcriptions.create(
                model=model, file=handle
            )

    result = (
        response.model_dump() if hasattr(response, "model_dump") else dict(response)
    )

    with open(path, "w") as handle:
        json.dump(result, handle, default=str)

    return result


def _json_from(text: str) -> dict[str, Any]:
    start, end = text.find("{"), text.rfind("}")

    if start < 0 or end <= start:
        raise ValueError("model did not return JSON")

    return json.loads(text[start : end + 1])


async def listen(workdir: str, wav_path: str, instructions: str) -> dict[str, Any]:
    """Ask an audio-capable model about a recording; returns its JSON answer."""
    import base64

    with open(wav_path, "rb") as handle:
        audio = handle.read()

    path = _cache(workdir, "listen:" + instructions + hashlib.sha1(audio).hexdigest())

    if os.path.isfile(path):
        with open(path) as handle:
            return json.load(handle)

    client = _client()
    last: Exception | None = None

    for model in [
        m
        for m in (
            os.environ.get("EARWITNESS_AUDIO_MODEL"),
            "gpt-audio-1.5",
            "gpt-audio",
        )
        if m
    ]:
        try:
            response = await client.chat.completions.create(
                model=model,
                modalities=["text"],
                messages=[
                    {"role": "system", "content": instructions},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Analyse this recording. JSON only.",
                            },
                            {
                                "type": "input_audio",
                                "input_audio": {
                                    "data": base64.b64encode(audio).decode(),
                                    "format": "wav",
                                },
                            },
                        ],
                    },
                ],
            )
        except Exception as error:  # noqa: BLE001 - try the next model
            last = error

            continue

        result = _json_from(response.choices[0].message.content or "")
        result["_model"] = model

        with open(path, "w") as handle:
            json.dump(result, handle)

        return result

    raise RuntimeError(f"no audio model available: {last}")
