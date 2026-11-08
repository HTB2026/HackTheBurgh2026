"""Ollama provider: a self-hosted model (Gemma 4 by default) over Ollama's chat API.

Configured by EARWITNESS_OLLAMA_URL (e.g. http://10.8.96.3:11434) and
EARWITNESS_OLLAMA_MODEL (default "gemma4"). Gemma 4 takes text, images and
audio (Ollama passes audio in the message's ``images`` list), so every AI check
can run on it. Answers are constrained to the JSON schema when the server
supports it (Ollama's ``format``); Ollama's MLX runner on macOS does not, and
then the schema goes in the prompt and the reply is checked.

There is only one model here, so the transcript's cross-check pass reports
itself unavailable rather than pretending a second model agreed.

Answers are cached per case in ``ollama-cache/``.
"""

import asyncio
import base64
import hashlib
import json
import os
from typing import Any

import httpx

# Ollama's default context (2-4k tokens) is too small for photos plus long prompts.
_NUM_CTX = int(os.environ.get("EARWITNESS_OLLAMA_NUM_CTX", "32768"))
# Local models are slow, and several checks may queue on one GPU.
_TIMEOUT_S = float(os.environ.get("EARWITNESS_OLLAMA_TIMEOUT", "600"))


def _url() -> str | None:
    url = os.environ.get("EARWITNESS_OLLAMA_URL", "").strip().rstrip("/")

    return url or None


def model() -> str:
    return os.environ.get("EARWITNESS_OLLAMA_MODEL", "").strip() or "gemma4"


def enabled() -> bool:
    return _url() is not None


def _cache(workdir: str, key: str) -> str:
    directory = os.path.join(workdir, "ollama-cache")
    os.makedirs(directory, exist_ok=True)

    return os.path.join(
        directory, hashlib.sha1(key.encode()).hexdigest()[:20] + ".json"
    )


def message_for_ollama(content: list[dict[str, Any]]) -> dict[str, Any]:
    """Neutral content parts as one Ollama user message: text joined, images as base64."""
    texts: list[str] = []
    images: list[str] = []

    for part in content:
        kind = part.get("type")

        if kind == "input_text":
            texts.append(part["text"])
        elif kind == "input_image":
            url = part["image_url"]

            if not url.startswith("data:"):
                raise ValueError("Ollama images must be data: URLs")

            images.append(url.split(",", 1)[1])
            # Keep the order visible to the model: images are attached to the message.
            texts.append(f"[image {len(images)}]")
        else:
            raise ValueError(f"unsupported content part for Ollama: {kind}")

    message: dict[str, Any] = {"role": "user", "content": "\n\n".join(texts)}

    if images:
        message["images"] = images

    return message


class _NoStructuredOutput(RuntimeError):
    """The server cannot constrain output to a schema (e.g. Ollama's MLX runner on macOS)."""


# None: not known yet; False: this server rejected ``format``, so the schema goes in the prompt.
_constrained: bool | None = None


async def _chat(body: dict[str, Any]) -> dict[str, Any]:
    url = _url()

    if url is None:
        raise RuntimeError("EARWITNESS_OLLAMA_URL is not set")

    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT_S) as client:
            response = await client.post(f"{url}/api/chat", json=body)
    except httpx.HTTPError as error:
        raise RuntimeError(f"Ollama at {url} is not reachable: {error!r}") from error

    if response.status_code == 404 and "not found" in response.text:
        raise RuntimeError(
            f"model {body['model']!r} is not on the Ollama server; run "
            f"`ollama pull {body['model']}` there or set EARWITNESS_OLLAMA_MODEL"
        )

    if response.status_code == 501 and "structured output" in response.text:
        raise _NoStructuredOutput(response.text[:200])

    if response.status_code >= 400:
        raise RuntimeError(f"Ollama error {response.status_code}: {response.text[:300]}")

    return response.json()


def json_from(text: str) -> dict[str, Any]:
    """The JSON object in a model's reply, tolerating code fences or a stray sentence."""
    start, end = text.find("{"), text.rfind("}")

    if start < 0 or end <= start:
        raise ValueError(f"no JSON object in the reply: {text[:200]!r}")

    return json.loads(text[start : end + 1])


async def _json_answer(
    name_of_model: str,
    instructions: str,
    messages: list[dict[str, Any]],
    schema: dict[str, Any],
) -> dict[str, Any]:
    """Ask for JSON matching ``schema``: constrained by Ollama when the server can,
    otherwise with the schema spelled out in the prompt and the reply checked."""
    global _constrained

    base = {
        "model": name_of_model,
        "stream": False,
        "options": {"num_ctx": _NUM_CTX, "temperature": 0.2},
    }

    if _constrained is not False:
        try:
            reply = await _chat(
                base
                | {
                    "messages": [{"role": "system", "content": instructions}, *messages],
                    "format": schema,
                }
            )
            _constrained = True

            return json_from((reply.get("message") or {}).get("content") or "")
        except _NoStructuredOutput:
            _constrained = False

    prompt = (
        f"{instructions}\n\nAnswer with ONE JSON object and nothing else (no prose, no code "
        f"fences). It must match this JSON schema exactly, with every required field:\n"
        f"{json.dumps(schema, ensure_ascii=False)}"
    )
    reply = await _chat(
        base | {"messages": [{"role": "system", "content": prompt}, *messages]}
    )
    text = (reply.get("message") or {}).get("content") or ""

    try:
        result = json_from(text)
    except ValueError as error:
        raise RuntimeError(f"{name_of_model} did not return JSON: {error}") from error

    missing = [k for k in schema.get("required", []) if k not in result]

    if missing:
        raise RuntimeError(f"{name_of_model} left out required fields: {missing}")

    return result


async def structured(
    workdir: str,
    *,
    instructions: str,
    content: list[dict[str, Any]],
    schema: dict[str, Any],
    name: str,
    effort: str | None = None,
) -> dict[str, Any]:
    """JSON-schema-constrained answer. ``effort`` is accepted for parity and ignored."""
    name_of_model = model()
    path = _cache(
        workdir,
        json.dumps(
            [name_of_model, instructions, content, schema, effort],
            sort_keys=True,
            default=str,
        ),
    )

    if os.path.isfile(path):
        with open(path) as handle:
            return json.load(handle)

    result = await _json_answer(name_of_model, instructions, [message_for_ollama(content)], schema)
    result["_model"] = name_of_model

    with open(path, "w") as handle:
        json.dump(result, handle)

    return result


_TRANSCRIPT_SCHEMA = {
    "type": "object",
    "properties": {
        "segments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "start": {"type": "number"},
                    "end": {"type": "number"},
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
    "times in seconds from the start. Leave out music, noise and silence. If nobody "
    "speaks, return no segments. Never invent words you cannot hear."
)


def _audio_message(path: str, text: str) -> dict[str, Any]:
    with open(path, "rb") as handle:
        audio = base64.b64encode(handle.read()).decode()

    return {"role": "user", "content": text, "images": [audio]}


async def _as_wav(path: str) -> str:
    """Ollama only recognises WAV audio: convert anything else (16 kHz mono)."""
    if path.lower().endswith(".wav"):
        return path

    wav = os.path.splitext(path)[0] + ".ollama.wav"

    if not os.path.isfile(wav):
        process = await asyncio.create_subprocess_exec(
            "ffmpeg", "-nostdin", "-v", "error", "-y", "-i", path,
            "-ac", "1", "-ar", "16000", wav,
            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE,
        )  # fmt: skip
        _, error = await process.communicate()

        if process.returncode != 0:
            raise RuntimeError(f"could not convert audio to WAV: {error.decode()[:200]}")

    return wav


def _audio_key(path: str) -> str:
    with open(path, "rb") as handle:
        return hashlib.sha1(handle.read()).hexdigest()


async def transcribe(
    workdir: str, audio_path: str, pass_: str = "primary"
) -> dict[str, Any]:
    """Speech to text with timed segments, in the shape whisper-1 returns."""
    if pass_ == "check":
        raise RuntimeError("only one model on Ollama, so no independent second reading")

    name_of_model = model()
    path = _cache(workdir, f"transcribe:{name_of_model}:{_audio_key(audio_path)}")

    if os.path.isfile(path):
        with open(path) as handle:
            return json.load(handle)

    answer = await _json_answer(
        name_of_model,
        _TRANSCRIPT_INSTRUCTIONS,
        [_audio_message(await _as_wav(audio_path), "Transcribe this recording.")],
        _TRANSCRIPT_SCHEMA,
    )
    segments = [
        {
            "start": float(s.get("start", 0) or 0),
            "end": float(s.get("end", 0) or 0),
            "text": str(s.get("text", "")),
            "no_speech_prob": 0.0,
        }
        for s in answer.get("segments") or []
        if isinstance(s, dict)
    ]
    result = {
        "text": " ".join(s["text"].strip() for s in segments).strip(),
        "segments": segments,
        "model": name_of_model,
    }

    with open(path, "w") as handle:
        json.dump(result, handle)

    return result


async def listen(workdir: str, wav_path: str, instructions: str) -> dict[str, Any]:
    """Ask the model about a recording; returns its JSON answer."""
    name_of_model = model()
    path = _cache(workdir, f"listen:{name_of_model}:{instructions}{_audio_key(wav_path)}")

    if os.path.isfile(path):
        with open(path) as handle:
            return json.load(handle)

    reply = await _chat(
        {
            "model": name_of_model,
            "stream": False,
            "options": {"num_ctx": _NUM_CTX, "temperature": 0.2},
            "messages": [
                {"role": "system", "content": instructions},
                _audio_message(wav_path, "Analyse this recording. JSON only."),
            ],
        }
    )
    text = (reply.get("message") or {}).get("content") or ""

    try:
        result = json_from(text)
    except ValueError as error:
        raise RuntimeError(f"{name_of_model} did not return JSON: {error}") from error

    result["_model"] = name_of_model

    with open(path, "w") as handle:
        json.dump(result, handle)

    return result
