"""Which AI provider a case uses: OpenAI or Gemini, chosen per case.

Every provider module exposes the same functions:

- ``enabled() -> bool``: its API key is set.
- ``structured(workdir, *, instructions, content, schema, name, effort=None)``:
  a JSON-schema-constrained answer with ``_model`` added. ``content`` uses the
  neutral parts the modules build: ``{"type": "input_text", "text": ...}`` and
  ``{"type": "input_image", "image_url": "data:image/jpeg;base64,..."}``.
- ``transcribe(workdir, audio_path, pass_="primary" | "check")``:
  ``{"text", "segments": [{"start", "end", "text", "no_speech_prob"}], "model"}``.
  The two passes use different models so the transcript can be cross-checked.
- ``listen(workdir, wav_path, instructions)``: a JSON answer about a recording.

A case's choice lives in ``options["ai_provider"]``; without one (or if that
provider has no key) the server default applies: ``EARWITNESS_AI_PROVIDER`` if
available, otherwise the first provider with a key.
"""

import os
from types import ModuleType
from typing import Any

from earwitness.ai import gemini
from earwitness.ai import ollama
from earwitness.ai import openai

PROVIDERS: dict[str, ModuleType] = {"openai": openai, "gemini": gemini, "ollama": ollama}
LABELS = {"openai": "OpenAI", "gemini": "Gemini", "ollama": "Gemma"}
MISSING_KEY = "set OPENAI_API_KEY, GEMINI_API_KEY or EARWITNESS_OLLAMA_URL to enable"


def available() -> list[str]:
    return [name for name, provider in PROVIDERS.items() if provider.enabled()]


def default() -> str | None:
    names = available()
    wanted = os.environ.get("EARWITNESS_AI_PROVIDER", "").strip().lower()

    if wanted in names:
        return wanted

    return names[0] if names else None


def choice(options: dict[str, Any] | None) -> str | None:
    """The provider name a case will use, or None when no AI key is set."""
    wanted = (options or {}).get("ai_provider")

    return wanted if wanted in available() else default()


def for_case(options: dict[str, Any] | None) -> ModuleType | None:
    name = choice(options)

    return PROVIDERS[name] if name else None

