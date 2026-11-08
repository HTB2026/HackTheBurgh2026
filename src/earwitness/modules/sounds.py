"""Sound sleuth: what an expert ear would pick out of the background.

An audio-capable model listens to the soundtrack and lists what it hears:
sirens (their pattern differs by country), announcements, birds, traffic,
church bells, languages and accents, crowd size, indoor or outdoor. Each item
is a clue to where and when the recording was made. Everything here is
AI-inferred and labelled so.
"""

import os

from earwitness import ai
from earwitness.modules.base import CaseContext
from earwitness.modules.base import Reporter
from earwitness.modules.base import Skip
from earwitness.modules.base import run_tool

_INSTRUCTIONS = (
    "You are a forensic audio analyst. Listen to the whole recording and report ONLY what you "
    "actually hear; never guess from nothing. Reply with JSON only, exactly this shape:\n"
    '{"environment": "one sentence: indoor/outdoor, size, setting",'
    ' "sounds": [{"label": "short name", "time_s": number, "detail": "what it is",'
    ' "clue": "what it suggests about place or time, or empty"}],'
    ' "languages": ["languages heard"], "accents": ["accents or dialects heard"],'
    ' "location_clues": ["each concrete clue: siren pattern, announcement text, vehicle type, birds, '
    'bells, call to prayer, ..."],'
    ' "time_clues": ["each concrete clue: dawn chorus, rush-hour traffic, New Year countdown, '
    'church bells striking N, ..."],'
    ' "people_estimate": "none / one / a few / a crowd of ~N",'
    ' "notable": "the single most revealing thing you hear, or empty"}\n'
    "Do not identify specific individuals. Do not speculate about anyone's ethnicity, religion, "
    "health or other sensitive traits."
)

# Deep runs listen harder for what places the countryside: wildlife has ranges.
_DEEP = (
    '\nAlso add the key "wildlife": ["each bird, insect, frog or farm animal you can name, '
    "with the species where you are confident (e.g. 'common wood pigeon', 'skylark', 'cicadas', "
    "'sheep'), and where it lives\"], and put rural clues (tractors, livestock, church bells, "
    "wind in trees, running water) in location_clues."
)


class Sounds:
    name = "sounds"
    title = "SOUND SLEUTH"
    description = "An AI listener names background sounds, languages and accents as clues to place and time (AI, optional)."

    async def run(self, ctx: CaseContext, out: Reporter) -> None:
        provider = ai.for_case(ctx.options)

        if provider is None:
            raise Skip(ai.MISSING_KEY)

        probe = await ctx.probe()

        if not any(s.get("codec_type") == "audio" for s in probe.get("streams", [])):
            raise Skip("no audio stream")

        out.progress(0.1, "preparing audio")
        path = out.path("listen.wav")

        if await run_tool(
            "ffmpeg", "-nostdin", "-v", "error", "-y", "-i", ctx.path, "-vn",
            "-ac", "1", "-ar", "16000", "-t", "180", path,
        ) is None or not os.path.isfile(path):  # fmt: skip
            raise Skip("could not extract audio")

        out.progress(0.3, "listening")
        heard = await provider.listen(
            ctx.workdir, path, _INSTRUCTIONS + ctx.by_effort(medium="", high=_DEEP)
        )
        sounds = [s for s in heard.get("sounds") or [] if isinstance(s, dict)]
        out.artifact(
            "heard",
            "table",
            f"What the AI listener heard ({heard.get('_model', 'audio model')})",
            data={
                "columns": ["at (s)", "sound", "detail", "clue"],
                "rows": [
                    [
                        s.get("time_s"),
                        s.get("label"),
                        s.get("detail"),
                        s.get("clue") or "",
                    ]
                    for s in sounds
                ],
            },
        )

        if heard.get("environment"):
            out.finding(
                "info",
                "Soundscape: " + str(heard["environment"])[:90],
                str(heard["environment"])
                + (
                    f" People: {heard['people_estimate']}."
                    if heard.get("people_estimate")
                    else ""
                ),
                kind="inferred",
                tags=["sound", "scene"],
                data={k: heard.get(k) for k in ("environment", "people_estimate")},
            )

        if heard.get("notable"):
            out.finding(
                "notice",
                "Most revealing sound: " + str(heard["notable"])[:80],
                str(heard["notable"]),
                kind="inferred",
                tags=["sound"],
            )

        for sound in sounds[:6]:
            if sound.get("clue"):
                out.finding(
                    "info",
                    f"Heard: {sound.get('label')}",
                    f"{sound.get('detail') or ''} Clue: {sound['clue']}",
                    time_s=float(sound["time_s"])
                    if isinstance(sound.get("time_s"), (int, float))
                    else None,
                    kind="inferred",
                    tags=["sound"],
                )

        accents = [a for a in heard.get("accents") or [] if a]
        languages = [a for a in heard.get("languages") or [] if a]

        if accents or languages:
            out.finding(
                "notice",
                "Voices: " + ", ".join((accents or languages)[:3]),
                "Languages heard: "
                + (", ".join(languages) or "none")
                + ". Accents: "
                + (", ".join(accents) or "none")
                + ".",
                kind="inferred",
                tags=["sound", "language", "location"],
                data={"languages": languages, "accents": accents},
            )

        if heard.get("time_clues"):
            out.finding(
                "notice",
                "Time clue in the sound: " + str(heard["time_clues"][0])[:70],
                "; ".join(map(str, heard["time_clues"])) + ".",
                kind="inferred",
                tags=["sound", "time"],
                data={"time_clues": heard["time_clues"]},
            )

        wildlife = [str(w) for w in heard.get("wildlife") or [] if w]

        if wildlife:
            out.finding(
                "info",
                "Wildlife heard: " + ", ".join(wildlife[:3])[:70],
                "; ".join(wildlife)
                + ". Species have ranges, so this narrows the region; birds are named by an AI listener and can be wrong.",
                kind="inferred",
                tags=["sound", "wildlife", "location"],
                data={"wildlife": wildlife},
            )

        ctx.publish(self.name, heard)
        out.progress(1.0)
