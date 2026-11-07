"""What is said, and in what language (OpenAI speech-to-text, cross-checked).

Speech models invent plausible sentences over noise, music and distant voices
("thanks for watching", a TV-news sign-off...). So two independent models
transcribe the audio: whisper-1 (timed segments) and gpt-4o-transcribe. When
they agree, the words are reported. When they disagree, the words are not
trustworthy, but if both heard the same language (writing system), that
language is still a real clue: voices speaking Chinese in the background say
something about where the recording was made.
"""

import difflib
import os
import re
import unicodedata

from earwitness import openai_client
from earwitness.modules.base import CaseContext
from earwitness.modules.base import Reporter
from earwitness.modules.base import Skip
from earwitness.modules.base import run_tool

# Phrases speech models are known to invent over music, noise or silence
# (learned from subtitled video). They are discarded, and the discard reported.
_HALLUCINATION = re.compile(
    r"thank(s| you)( so much)? for watching|like and subscribe|please subscribe|"
    r"subscribe to (my|the|our) channel|see you (in the )?next (video|time)|"
    r"subtitles by|amara\.org|多謝.{0,4}(觀看|收睇)|謝謝.{0,3}觀看|字幕|請不吝|"
    r"ご視聴ありがとうございました|untertitel|sous-titres",
    re.IGNORECASE,
)

# Writing systems, and the language each most plausibly means in speech.
_SCRIPTS = [
    ("han", "Chinese", lambda c: "一" <= c <= "鿿" or "㐀" <= c <= "䶿"),
    ("kana", "Japanese", lambda c: "぀" <= c <= "ヿ"),
    ("hangul", "Korean", lambda c: "가" <= c <= "힯" or "ᄀ" <= c <= "ᇿ"),
    (
        "cyrillic",
        "a Cyrillic-script language (Russian, Ukrainian, ...)",
        lambda c: "Ѐ" <= c <= "ӿ",
    ),
    (
        "arabic",
        "Arabic-script language (Arabic, Persian, Urdu, ...)",
        lambda c: "؀" <= c <= "ۿ",
    ),
    ("devanagari", "Hindi / Devanagari-script language", lambda c: "ऀ" <= c <= "ॿ"),
    ("thai", "Thai", lambda c: "฀" <= c <= "๿"),
    ("greek", "Greek", lambda c: "Ͱ" <= c <= "Ͽ"),
    ("hebrew", "Hebrew", lambda c: "֐" <= c <= "׿"),
    ("latin", None, lambda c: c.isascii() and c.isalpha() or "À" <= c <= "ɏ"),
]


def _script(text: str) -> tuple[str | None, str | None]:
    counts: dict[str, int] = {}

    for char in text:
        for name, _, test in _SCRIPTS:
            if test(char):
                counts[name] = counts.get(name, 0) + 1

                break

    if not counts:
        return None, None

    name = max(counts, key=counts.__getitem__)

    if name == "kana" or (name == "han" and counts.get("kana", 0) > 0):
        name = "kana"

    language = next(lang for n, lang, _ in _SCRIPTS if n == name)

    return name, language


_NUMBER_WORDS = {
    w: str(i)
    for i, w in enumerate(
        "zero one two three four five six seven eight nine ten eleven twelve thirteen "
        "fourteen fifteen sixteen seventeen eighteen nineteen twenty".split()
    )
} | {"thirty": "30", "forty": "40", "fifty": "50", "sixty": "60", "seventy": "70",
     "eighty": "80", "ninety": "90", "hundred": "100", "thousand": "1000"}  # fmt: skip


def _words(text: str) -> list[str]:
    text = unicodedata.normalize("NFKC", text).lower()
    tokens = re.findall(r"\w+", text)

    return [_NUMBER_WORDS.get(t, t) for t in tokens]


def _normalise(text: str) -> str:
    return "".join(_words(text))


def _agreement(a: str, b: str) -> float:
    """How far two readings say the same thing (0..1).

    Character similarity handles languages without spaces; word overlap of the
    shorter reading handles one model hearing more than the other ("BOOM!").
    Number words are folded to digits ("nine" == "9").
    """
    wa, wb = _words(a), _words(b)

    if not wa or not wb:
        return 0.0

    chars = difflib.SequenceMatcher(None, "".join(wa), "".join(wb)).ratio()
    shorter, longer = sorted((wa, wb), key=len)
    remaining = list(longer)
    hits = 0

    for word in shorter:
        if word in remaining:
            remaining.remove(word)
            hits += 1

    overlap = hits / len(shorter) if len(shorter) >= 3 else 0.0

    return max(chars, overlap)


class Transcript:
    name = "transcript"
    title = "TRANSCRIPT"
    description = "Speech, cross-checked by two speech models; spoken language as a clue (OpenAI, optional)."

    async def run(self, ctx: CaseContext, out: Reporter) -> None:
        if not openai_client.enabled():
            raise Skip("set OPENAI_API_KEY to enable")

        probe = await ctx.probe()

        if not any(s.get("codec_type") == "audio" for s in probe.get("streams", [])):
            raise Skip("no audio stream")

        out.progress(0.1, "extracting speech band")
        path = out.path("speech.mp3")

        # 16 kHz mono MP3 keeps uploads small (the API limit is 25 MB).
        if await run_tool(
            "ffmpeg", "-nostdin", "-v", "error", "-y", "-i", ctx.path, "-vn", "-ac", "1",
            "-ar", "16000", "-b:a", "48k", "-t", "1400", path,
        ) is None or not os.path.isfile(path):  # fmt: skip
            raise Skip("could not extract audio")

        out.progress(0.3, "transcribing with two models")
        whisper = await openai_client.transcribe(ctx.workdir, path, "whisper-1")

        try:
            second = await openai_client.transcribe(
                ctx.workdir, path, "gpt-4o-transcribe"
            )
            second_text = (second.get("text") or "").strip()
        except Exception as error:  # noqa: BLE001 - one model is still useful
            out.log(f"second speech model unavailable: {error}")
            second_text = None

        raw = [
            {
                "start": float(s["start"]),
                "end": float(s["end"]),
                "text": s["text"].strip(),
            }
            for s in whisper.get("segments") or []
            if s.get("text", "").strip()
            and float(s.get("no_speech_prob", 0) or 0) < 0.6
        ]
        whisper_text = " ".join(s["text"] for s in raw)
        kept = [s for s in raw if not _HALLUCINATION.search(s["text"])]
        discarded = [s["text"] for s in raw if _HALLUCINATION.search(s["text"])]
        kept_text = " ".join(s["text"] for s in kept)
        agreement = None if second_text is None else _agreement(kept_text, second_text)
        script_a, language_a = _script(whisper_text)
        script_b, language_b = _script(second_text or "")
        same_script = script_a is not None and script_a == script_b
        # Both models heard the same non-Latin script: a language clue even if the words differ.
        language = language_a if same_script else None
        reliable = bool(kept) and (second_text is None or (agreement or 0) >= 0.45)
        out.artifact(
            "readings",
            "table",
            "What each speech model heard",
            data={
                "columns": ["model", "text", "writing system"],
                "rows": [
                    ["whisper-1", whisper_text or "(nothing)", script_a or "-"],
                    [
                        "gpt-4o-transcribe",
                        second_text if second_text is not None else "(unavailable)",
                        script_b or "-",
                    ],
                    [
                        "agreement",
                        f"{agreement * 100:.0f}%" if agreement is not None else "-",
                        "",
                    ],
                ],
            },
        )

        if discarded:
            out.log(f"discarded {len(discarded)} invented phrase(s): {discarded}")

        published = {
            "segments": kept if reliable else [],
            "language": language or (whisper.get("language") if reliable else None),
            "language_clue": language,
            "agreement": agreement,
        }

        if reliable:
            out.artifact(
                "segments",
                "table",
                "Transcript",
                data={
                    "columns": ["start (s)", "end (s)", "text"],
                    "rows": [
                        [round(s["start"], 1), round(s["end"], 1), s["text"]]
                        for s in kept
                    ],
                },
            )
            words = sum(len(s["text"].split()) for s in kept)
            spoken = language or (whisper.get("language") or "").capitalize() or None
            out.finding(
                "notice",
                f"Speech: {words} words" + (f" in {spoken}" if spoken else ""),
                "First words: “"
                + kept_text[:220]
                + "…”"
                + (
                    f" Two speech models agree ({agreement * 100:.0f}%)."
                    if agreement is not None
                    else ""
                ),
                time_s=kept[0]["start"],
                kind="inferred",
                tags=["speech", "language"],
                data={"language": spoken, "agreement": agreement},
            )
            out.artifact("captions", "captions", "Subtitles", data={"segments": kept})
        elif language and language != "None":
            out.finding(
                "notice",
                f"Voices speaking {language}, words unclear",
                f"Both speech models heard {language} but wrote different sentences "
                f"(agreement {(agreement or 0) * 100:.0f}%), so the words cannot be trusted. "
                "The language itself is a clue to where this was recorded.",
                kind="inferred",
                tags=["speech", "language", "location"],
                data={
                    "language": language,
                    "readings": [whisper_text, second_text],
                    "agreement": agreement,
                },
            )
        else:
            out.finding(
                "info",
                "No intelligible speech",
                "No reliable words in the audio."
                + (
                    f" A speech model offered stock phrase(s) such as “{discarded[0]}”, "
                    "which these models invent over noise or music; discarded."
                    if discarded
                    else ""
                ),
                tags=["speech"],
                data={"discarded": discarded, "agreement": agreement},
            )

        ctx.publish(self.name, published)
        out.progress(1.0)
