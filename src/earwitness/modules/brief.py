"""The case summary: WHEN / WHERE / WHO / INTEGRITY / HIDDEN.

Always produced locally from the measured findings with explicit rules, so the
verdict is reproducible. With an OpenAI key, a language model also writes a
short narrative brief that must cite finding IDs.
"""

import json
import re
from datetime import datetime
from datetime import timedelta
from typing import Any

from earwitness import openai_client
from earwitness.modules.base import CaseContext
from earwitness.modules.base import Finding
from earwitness.modules.base import Reporter

_EDIT_TAGS = {"splice", "cut", "inconsistency", "clone"}


def _with(findings: list[Finding], *tags: str, measured: bool = True) -> list[Finding]:
    return [
        f
        for f in findings
        if set(tags) & set(f.tags) and (not measured or f.kind == "measured")
    ]


def _clock(seconds: float | None) -> str:
    if seconds is None:
        return ""

    return f"{int(seconds // 60)}:{int(seconds % 60):02d}"


def _answer(key, question, answer, confidence, evidence, time_s=None):
    return {
        "key": key,
        "question": question,
        "answer": answer,
        "confidence": confidence,
        "evidence": [f.id for f in evidence],
        "time_s": time_s,
        "ai": any(f.kind == "inferred" for f in evidence),
    }


def _plain_answers(
    findings: list[Finding], local: dict[str, Any]
) -> list[dict[str, Any]]:
    """The case in plain English: one question per card, with a confidence word."""
    by_title = lambda *prefixes: [f for f in findings if f.title.startswith(prefixes)]  # noqa: E731
    answers = []

    # ---- edited? ----------------------------------------------------------
    edits = [
        f
        for f in findings
        if f.kind == "measured"
        and f.severity in ("warning", "critical")
        and ({"splice", "cut", "clone", "sync"} & set(f.tags))
    ]
    software = [f for f in findings if "editor" in f.tags]
    corroborated = by_title("Edit corroborated")
    first = min(
        (f for f in edits if f.time_s is not None), key=lambda f: f.time_s, default=None
    )

    contradicted_time = by_title("Metadata time contradicted")

    if corroborated:
        at = corroborated[0].time_s
        others = [
            f
            for f in edits
            if f.time_s is not None and at is not None and abs(f.time_s - at) > 1.5
        ]
        answer = f"Yes. Both the picture and the sound break at {_clock(at)}: two independent signs of a cut."

        if others:
            answer += (
                f" There {'are' if len(others) > 1 else 'is'} {len(others)} more suspicious "
                f"spot{'s' if len(others) > 1 else ''}, first at {_clock(min(f.time_s for f in others))}."
            )

        answers.append(
            _answer(
                "edited",
                "Has this video been edited?",
                answer,
                "strong",
                corroborated + edits[:3],
                at,
            )
        )
    elif edits:
        kinds = []

        for f in edits:
            label = (
                "the picture jumps between shots" if "cut" in f.tags
                else "part of the image was copied and pasted" if "clone" in f.tags
                else "the sound is out of step with the picture" if "sync" in f.tags
                else "the sound was cut and joined"
            )  # fmt: skip

            if label not in kinds:
                kinds.append(label)

        answer = (
            "Yes: "
            + "; ".join(kinds[:3])
            + (f" (first at {_clock(first.time_s)})" if first else "")
            + "."
        )
        strong = len({f.module for f in edits}) >= 2 or any(
            f.severity == "critical" for f in edits
        )
        answers.append(
            _answer(
                "edited",
                "Has this video been edited?",
                answer,
                "strong" if strong else "moderate",
                edits[:4],
                first.time_s if first else None,
            )
        )
    elif contradicted_time:
        answers.append(_answer(
            "edited", "Has this video been edited?",
            "The picture and sound were not cut, but the date stored in the file does not match "
            "when it was really filmed: the timestamp is wrong or was changed.",
            "strong", contradicted_time + software[:1],
        ))  # fmt: skip
    elif software:
        answers.append(_answer(
            "edited", "Has this video been edited?",
            "No cuts or splices found, but the file has been re-saved by other software, "
            "so it is not the camera's original.",
            "moderate", software[:2],
        ))  # fmt: skip
    else:
        checks = [
            f
            for f in findings
            if f.title.startswith(
                ("No hard cuts", "Steady", "Regular keyframe", "No cloned")
            )
        ]
        answers.append(_answer(
            "edited", "Has this video been edited?",
            "No signs of editing. The frames, sound and pixels are all consistent with one continuous recording.",
            "moderate" if len(checks) >= 2 else "weak", checks[:3],
        ))  # fmt: skip

    # ---- when? ----------------------------------------------------------------
    dated = by_title("ENF dates", "Lighting dates")
    contradicted = by_title("Metadata time contradicted")
    container = by_title("Container clock")

    if dated:
        moment = dated[0].title.split(" to ", 1)[-1]
        how = (
            "the flicker of the room's lights"
            if dated[0].title.startswith("Lighting")
            else "the hum of the power grid in the sound"
        )
        extra = " The time written in the file is wrong." if contradicted else ""
        answers.append(
            _answer(
                "when",
                "When was it filmed?",
                f"{moment}, worked out from {how}, matched against the national grid's records.{extra}",
                "strong",
                dated + contradicted,
            )
        )
    elif by_title("Timestamp is when this copy was made"):
        copy = by_title("Timestamp is when this copy was made")
        answers.append(
            _answer(
                "when",
                "When was it filmed?",
                "Unknown. The only date in the file is when this copy was saved or "
                "downloaded, not when it was filmed.",
                "unknown",
                copy + container,
            )
        )
    elif container:
        when = container[0].title.split(": ", 1)[-1]
        stamp = container[0].data.get("creation_time_utc")
        local = by_title("Local clock was UTC")

        if stamp:
            moment = datetime.fromisoformat(str(stamp))
            when = f"{moment:%-d %B %Y, %H:%M} UTC"

            if local:
                shifted = moment + timedelta(hours=float(local[0].data["offset_h"]))
                when = f"{shifted:%-d %B %Y, %H:%M} local time ({moment:%H:%M} UTC)"

        answers.append(
            _answer(
                "when",
                "When was it filmed?",
                f"The file says {when}. This comes from the file's own label, which can be changed, so it is not proven.",
                "weak",
                container + local,
            )
        )
    else:
        answers.append(
            _answer(
                "when",
                "When was it filmed?",
                "Not enough evidence. The file carries no date, and no time clues were found in the sound or light.",
                "unknown",
                [],
            )
        )

    # ---- where? ---------------------------------------------------------------
    gps = [f for f in findings if "gps" in f.tags]
    postcode = [
        f for f in findings if "text" in f.tags and "postcode" in f.title.lower()
    ]
    grid = [
        f
        for f in findings
        if f.title.startswith(("Lights flicker", "Mains hum locked"))
        and f.severity == "critical"
    ]
    geo_guess = [
        f
        for f in findings
        if f.module == "geo"
        and f.title.startswith(("Pixels say", "Located", "Pixels suggest"))
    ]
    ai_place = geo_guess or [
        f
        for f in findings
        if f.kind == "inferred"
        and "location" in f.tags
        and f.title.startswith("Looks like")
    ]

    resaved = bool(by_title("Timestamp is when this copy was made"))

    if gps and resaved:
        place = gps[0].data.get("place") or gps[0].title.split(": ", 1)[-1]
        answers.append(
            _answer(
                "where",
                "Where was it filmed?",
                f"{place}, from GPS saved in the file. This is a re-saved copy, so the GPS "
                "may mark where it was saved rather than filmed.",
                "moderate",
                gps,
            )
        )
    elif gps:
        place = gps[0].data.get("place") or gps[0].title.split(": ", 1)[-1]
        answers.append(
            _answer(
                "where",
                "Where was it filmed?",
                f"{place}, from the phone's GPS saved in the file.",
                "strong",
                gps,
            )
        )
    elif postcode:
        answers.append(
            _answer(
                "where",
                "Where was it filmed?",
                f"Near postcode {postcode[0].data.get('value')}, which is readable in the picture.",
                "moderate",
                postcode,
                postcode[0].time_s,
            )
        )
    elif ai_place:
        guess = ai_place[0]

        if guess.module == "geo":
            parts = (
                guess.data.get("place"),
                guess.data.get("region"),
                guess.data.get("country"),
            )
            place = ", ".join(
                str(p) for p in parts if p and not str(p).lower().startswith("unknown")
            )
            radius = float(guess.data.get("radius_km") or 0)
            near = f"within about {radius:.0f} km" if radius >= 1 else "very close"
            place = re.sub(
                r"^(best guess:|likely|probably|possibly)\s*", "", place, flags=re.I
            )
            answer = f"Probably {place} ({near}), judging only from what is visible in the picture."
            check = guess.data.get("corroboration") or {}

            if check.get("agree"):
                answer += (
                    " Verified: it matches reference photos and satellite imagery of the place."
                    if guess.data.get("verified")
                    else " A second, independent AI analysis agrees."
                )
            elif check:
                other = str(check.get("second_guess") or "no location")
                answer = (
                    f"Uncertain. One AI analysis suggests {place}; a second "
                    + (
                        "could not place it at all."
                        if other.lower().startswith(("unknown", "no location"))
                        else f"suggests {other}."
                    )
                    + " Treat this as a lead, not an answer."
                )
        else:
            answer = re.sub(
                r"\s*\((low|medium|high) confidence\)",
                "",
                guess.title.removeprefix("Looks like: "),
            )
            answer += ", judging from what is visible."

        confidence = {"high": "moderate", "medium": "moderate"}.get(
            guess.data.get("confidence"), "weak"
        )

        if not (guess.data.get("corroboration") or {}).get("agree", True):
            confidence = "weak"

        # A guess spanning a continent or more is not a location.
        if float(guess.data.get("radius_km") or 0) >= 2500:
            answer = (
                "Not enough evidence. Nothing in the picture narrows it down "
                "(the AI's best guess would span thousands of kilometres)."
            )
            confidence = "unknown"

        answers.append(
            _answer("where", "Where was it filmed?", answer, confidence, [guess])
        )
    elif grid:
        region = (
            "Europe, the UK, Africa, most of Asia or Australia"
            if "50 Hz" in grid[0].title
            else "the Americas or parts of East Asia"
        )
        answers.append(
            _answer(
                "where",
                "Where was it filmed?",
                f"Somewhere in {region}: the electricity there runs at the frequency we detected.",
                "weak",
                grid,
            )
        )
    else:
        answers.append(
            _answer(
                "where",
                "Where was it filmed?",
                "Not enough evidence. No location data or recognisable clues were found.",
                "unknown",
                [],
            )
        )

    # ---- device -----------------------------------------------------------------
    device = by_title("Recorded on")

    if device:
        answers.append(
            _answer(
                "device",
                "What recorded it?",
                device[0].title.removeprefix("Recorded on ") + ".",
                "strong",
                device,
            )
        )

    # ---- people -----------------------------------------------------------------
    pulses = [f for f in findings if f.module == "pulse" and "bpm" in f.title]

    if pulses:
        best = max(pulses, key=lambda f: f.data.get("snr_db", -99))
        reliable = best.data.get("snr_db", -99) > 0
        answers.append(_answer(
            "people", "Who is in it?",
            (f"At least {len(pulses)} face{'' if len(pulses) == 1 else 's'} clearly visible. "
             if any("face" in f.title for f in pulses) else "Visible skin, but no face. ")
            + (f"We can see a heartbeat of about {best.data['bpm']:.0f} beats per minute from tiny colour changes in the skin."
               if reliable else "Heart rate could not be read reliably (movement or compression)."),
            "moderate" if reliable else "weak", pulses[:3], best.time_s,
        ))  # fmt: skip

    # ---- hidden -----------------------------------------------------------------
    hidden_bits, hidden_f = [], []

    for f in findings:
        if "dtmf" in f.tags:
            hidden_bits.append(
                f"phone keypad tones spelling {f.title.split(': ', 1)[-1]}"
            )
        elif "ultrasonic" in f.tags:
            hidden_bits.append(
                f"a high-pitched tone too high for adults to hear ({f.data.get('hz', 0) / 1000:.1f} kHz)"
            )
        elif "doppler" in f.tags:
            hidden_bits.append(
                "the sound of something passing by (aircraft or vehicle)"
            )
        elif f.title.startswith("Lights flicker"):
            hidden_bits.append(
                "an invisible flicker from mains-powered lights, which reveals the electricity grid"
            )
        else:
            continue

        hidden_f.append(f)

    if hidden_bits:
        unique = list(dict.fromkeys(hidden_bits))
        first_hidden = next((f.time_s for f in hidden_f if f.time_s is not None), None)
        answers.append(
            _answer(
                "hidden",
                "Anything you can't see or hear?",
                "Yes: " + "; ".join(unique[:4]) + ".",
                "strong",
                hidden_f[:4],
                first_hidden,
            )
        )
    else:
        answers.append(
            _answer(
                "hidden",
                "Anything you can't see or hear?",
                "Nothing unusual hidden in the sound or picture.",
                "moderate",
                [],
            )
        )

    # ---- text in the picture ----------------------------------------------------
    readable = [
        f
        for f in findings
        if f.module == "ocr" and f.severity in ("warning", "critical")
    ]

    if readable:
        values = list(dict.fromkeys(f.title.split(": ", 1)[-1] for f in readable))
        answers.append(
            _answer(
                "text",
                "Any readable text or numbers?",
                "Yes: " + "; ".join(values[:5]) + ".",
                "moderate",
                readable[:5],
                readable[0].time_s,
            )
        )

    # ---- speech -----------------------------------------------------------------
    speech = [
        f for f in findings if f.module == "transcript" and f.title.startswith("Speech")
    ]
    voices = [
        f
        for f in findings
        if f.module == "transcript" and f.title.startswith("Voices speaking")
    ]

    if voices and not speech:
        answers.append(
            _answer(
                "speech",
                "What is said?",
                f"People can be heard speaking {voices[0].data.get('language')}, but the words "
                "cannot be made out reliably.",
                "weak",
                voices,
            )
        )

    if speech:
        answers.append(
            _answer(
                "speech",
                "What is said?",
                speech[0].detail.removeprefix("First words: "),
                "moderate",
                speech,
                speech[0].time_s,
            )
        )

    return answers


def _summary_plain(answers: list[dict[str, Any]]) -> str:
    edited = next(a for a in answers if a["key"] == "edited")
    when = next(
        (a for a in answers if a["key"] == "when" and a["confidence"] == "strong"), None
    )
    text = edited["answer"]

    if when:
        text += " Filmed " + when["answer"].split(",", 1)[0] + "."

    return text


def _local_brief(findings: list[Finding]) -> dict[str, Any]:
    serious = [
        f
        for f in findings
        if f.severity in ("warning", "critical") and f.kind == "measured"
    ]
    edits = [f for f in serious if _EDIT_TAGS & set(f.tags)]
    software = [f for f in serious if "editor" in f.tags]
    independent_modules = {f.module for f in edits}

    hard_cuts = [f for f in edits if "cut" in f.tags]

    if (
        len(independent_modules) >= 2
        or any(f.severity == "critical" for f in edits)
        or hard_cuts
    ):
        verdict = "EDITED"
    elif edits or software:
        verdict = "POSSIBLY EDITED"
    else:
        verdict = "NO SIGNS OF EDITING"

    bullets: list[str] = []
    when = _with(findings, "enf") and [
        f
        for f in _with(findings, "enf", "time")
        if f.title.startswith(("ENF dates", "Lighting dates"))
    ]
    container = [f for f in findings if f.title.startswith("Container clock")]

    if when:
        bullets.append(f"WHEN: {when[0].title} [{when[0].id}]")
    elif container:
        bullets.append(
            f"WHEN: {container[0].title}, from metadata only, not independently verified [{container[0].id}]"
        )
    else:
        bullets.append("WHEN: no time evidence in the file")

    gps = _with(findings, "gps")
    postcode = [f for f in _with(findings, "text") if "postcode" in f.title.lower()]
    grid = [
        f
        for f in findings
        if ("flicker" in f.tags or "enf" in f.tags)
        and "grid" in f.title.lower()
        and f.severity == "critical"
    ]
    place_bits = []

    if gps:
        place_bits.append(f"{gps[0].title} [{gps[0].id}]")

    if postcode:
        place_bits.append(f"{postcode[0].title} [{postcode[0].id}]")

    if grid:
        place_bits.append(f"{grid[0].title} [{grid[0].id}]")

    inferred_place = [
        f for f in findings if f.kind == "inferred" and "location" in f.tags
    ]

    if inferred_place:
        place_bits.append(f"AI: {inferred_place[0].title} [{inferred_place[0].id}]")

    bullets.append(
        "WHERE: " + ("; ".join(place_bits) if place_bits else "no location evidence")
    )
    device = _with(findings, "device")
    people = [f for f in _with(findings, "pulse") if "bpm" in f.title]

    if device:
        bullets.append(f"DEVICE: {device[0].title} [{device[0].id}]")

    if people:
        bullets.append(
            "PEOPLE: " + "; ".join(f"{f.title} [{f.id}]" for f in people[:3])
        )

    if edits or software:
        bullets.append(
            "INTEGRITY: "
            + "; ".join(f"{f.title} [{f.id}]" for f in (edits + software)[:6])
        )
    else:
        bullets.append(
            "INTEGRITY: frame clock, keyframes, audio continuity and pixels show no edit"
        )

    hidden = [
        f
        for f in findings
        if {"dtmf", "ultrasonic", "tone", "doppler"} & set(f.tags)
        and f.severity != "info"
    ]

    if hidden:
        bullets.append(
            "HIDDEN: " + "; ".join(f"{f.title} [{f.id}]" for f in hidden[:4])
        )

    # Corroboration: an edit seen by two independent senses is much stronger.
    cut_times = [
        f.time_s for f in edits if f.time_s is not None and f.module == "timeline"
    ]
    audio_times = [
        f.time_s
        for f in edits
        if f.time_s is not None and f.module in ("spectrum", "enf")
    ]
    agreements = [(v, a) for v in cut_times for a in audio_times if abs(v - a) <= 1.5]

    if agreements:
        v, a = agreements[0]
        bullets.append(
            f"CORROBORATED: picture cut at {v:.1f}s and audio break at {a:.1f}s coincide"
        )

    text = f"VERDICT: {verdict}.\n" + "\n".join(bullets)

    return {
        "verdict": verdict,
        "bullets": bullets,
        "text": text,
        "corroborated": bool(agreements),
        "corroborated_at": agreements[0][0] if agreements else None,
    }


_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["headline", "summary", "next_steps"],
    "properties": {
        "headline": {"type": "string", "description": "One line, max 90 characters."},
        "summary": {
            "type": "string",
            "description": "3-6 sentences. Cite finding ids in brackets like [F012].",
        },
        "next_steps": {"type": "array", "items": {"type": "string"}, "maxItems": 4},
    },
}


def _evidence(findings: list[Finding]) -> list[dict[str, Any]]:
    return [
        {"id": f.id, "module": f.module, "severity": f.severity, "kind": f.kind,
         "title": f.title, "detail": f.detail, "time_s": f.time_s}
        for f in findings
    ]  # fmt: skip


_DOSSIER_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["headline", "lines", "closing"],
    "properties": {
        "headline": {
            "type": "string",
            "description": "One chilling line, max 70 characters.",
        },
        "lines": {
            "type": "array",
            "minItems": 3,
            "maxItems": 10,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["text", "evidence", "certainty", "category"],
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Second person, present tense, max 160 characters.",
                    },
                    "evidence": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Finding ids it rests on.",
                    },
                    "certainty": {
                        "type": "string",
                        "enum": ["measured", "likely", "possible"],
                    },
                    "category": {
                        "type": "string",
                        "enum": [
                            "where",
                            "when",
                            "device",
                            "body",
                            "surroundings",
                            "integrity",
                            "hidden",
                        ],
                    },
                },
            },
        },
        "closing": {
            "type": "string",
            "description": "One line: what a stranger could do with this, max 140 characters.",
        },
    },
}

_DOSSIER = (
    "You write the reveal at the end of a privacy demonstration: 'what this video gives away "
    "about the person who filmed it'. Address the filmer as 'you', in short, cold, specific "
    "sentences, like a surveillance file read aloud (e.g. 'You were on the South Bank, facing the "
    "London Eye.' 'Your phone is an iPhone 17 Pro Max.' 'Your heart was beating at 72 beats per "
    "minute.'). Use ONLY the findings provided, cite their ids, and mark certainty honestly: "
    "'measured' for signal measurements and metadata, 'likely' for verified or corroborated AI "
    "inferences, 'possible' for uncorroborated AI guesses. Order from most unsettling to least. "
    "Hard rules: never identify or name any person; never speculate about ethnicity, religion, "
    "health conditions, sexuality, politics or other sensitive traits; no threats; do not "
    "exaggerate beyond the evidence. Skip anything trivial. The closing line names what a "
    "stranger could do with this information (e.g. find the spot, know when you were there)."
)


class Brief:
    name = "brief"
    title = "CASE BRIEF"
    description = "Verdict and summary across all modules (rule-based; plus AI narrative if enabled)."
    depends = ["*"]

    async def run(self, ctx: CaseContext, out: Reporter) -> None:
        findings = list(ctx.findings)
        local = _local_brief(findings)

        if local["corroborated"]:
            out.finding(
                "critical",
                "Edit corroborated by sound and picture",
                local["bullets"][-1].removeprefix("CORROBORATED: ").capitalize()
                + ". Two independent measurements agree on an edit point.",
                time_s=local["corroborated_at"],
                tags=["splice", "inconsistency"],
            )

        data: dict[str, Any] = dict(local)
        data["answers"] = _plain_answers(ctx.findings, local)
        data["summary_plain"] = _summary_plain(data["answers"])

        if openai_client.enabled() and findings:
            out.progress(0.4, "drafting narrative")
            evidence = [
                {"id": f.id, "module": f.module, "severity": f.severity, "kind": f.kind,
                 "title": f.title, "detail": f.detail, "time_s": f.time_s}
                for f in findings
            ]  # fmt: skip

            try:
                narrative = await openai_client.structured(
                    ctx.workdir,
                    instructions=(
                        "You write the summary of a video-forensics case file for a non-expert. Use ONLY "
                        "the findings given; cite their ids in brackets. Distinguish measured findings "
                        "from AI-inferred ones. Do not overstate: weak evidence is weak. The rule-based "
                        f"verdict is '{local['verdict']}'; agree with it unless the findings clearly contradict it."
                    ),
                    content=[{"type": "input_text", "text": str(evidence)}],
                    schema=_SCHEMA,
                    name="case_brief",
                )
                data["headline"] = narrative["headline"]
                data["narrative"] = narrative["summary"]
                data["next_steps"] = narrative["next_steps"]
                data["text"] = (
                    f"{narrative['headline']}\n\n{narrative['summary']}\n\n{local['text']}"
                )
                data["model"] = narrative.get("_model")
            except Exception as error:  # noqa: BLE001 - the brief must still appear
                out.log(f"AI narrative unavailable: {error}")

        out.artifact("brief", "brief", "Case brief", data=data)

        if openai_client.enabled() and findings:
            out.progress(0.7, "writing the dossier")

            try:
                dossier = await openai_client.structured(
                    ctx.workdir,
                    instructions=_DOSSIER,
                    content=[
                        {
                            "type": "input_text",
                            "text": json.dumps(_evidence(findings), ensure_ascii=False),
                        }
                    ],
                    schema=_DOSSIER_SCHEMA,
                    name="dossier",
                )
                known = {f.id for f in findings}
                dossier["lines"] = [
                    line | {"evidence": [e for e in line["evidence"] if e in known]}
                    for line in dossier["lines"]
                    if any(e in known for e in line["evidence"])
                ]
                bpm = next(
                    (
                        f.data.get("bpm")
                        for f in findings
                        if f.module == "pulse" and f.data.get("bpm")
                    ),
                    None,
                )
                dossier["heart_bpm"] = bpm
                out.artifact(
                    "dossier", "dossier", "What this video gives away", data=dossier
                )
            except Exception as error:  # noqa: BLE001 - optional
                out.log(f"dossier unavailable: {error}")

        ctx.publish(self.name, data)
        out.progress(1.0)
