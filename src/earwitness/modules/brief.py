"""The case summary: WHEN / WHERE / WHO / INTEGRITY / HIDDEN.

Always produced locally from the measured findings with explicit rules, so the
verdict is reproducible. With an AI key (OpenAI or Gemini), a language model also writes a
short narrative brief that must cite finding IDs.
"""

import json
import re
from datetime import datetime
from datetime import timedelta
from typing import Any

from earwitness import ai
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
        answer = f"Yes. Picture and sound both break at {_clock(at)}."

        if others:
            answer += (
                f" {len(others)} more suspicious spot{'s' if len(others) > 1 else ''}, "
                f"first at {_clock(min(f.time_s for f in others))}."
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
                "the picture jumps" if "cut" in f.tags
                else "part of the image was copied" if "clone" in f.tags
                else "the sound is out of sync" if "sync" in f.tags
                else "the sound was cut"
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
            "No cuts, but the date in the file is wrong or was changed.",
            "strong", contradicted_time + software[:1],
        ))  # fmt: skip
    elif software:
        answers.append(_answer(
            "edited", "Has this video been edited?",
            "No cuts, but other software re-saved it, so it is not the original.",
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
            "No signs of editing. Looks like one continuous recording.",
            "moderate" if len(checks) >= 2 else "weak", checks[:3],
        ))  # fmt: skip

    # ---- when? ----------------------------------------------------------------
    dated = by_title("ENF dates", "Lighting dates")
    contradicted = by_title("Metadata time contradicted")
    container = by_title("Container clock")

    if dated:
        moment = dated[0].title.split(" to ", 1)[-1]
        how = (
            "the light flicker"
            if dated[0].title.startswith("Lighting")
            else "the mains hum"
        )
        extra = " The date in the file is wrong." if contradicted else ""
        answers.append(
            _answer(
                "when",
                "When was it filmed?",
                f"{moment}, from {how} matched to power grid records.{extra}",
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
                "Unknown. The file only says when this copy was saved.",
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
                f"{when}, according to the file. Files can be edited, so this is not proof.",
                "weak",
                container + local,
            )
        )
    else:
        answers.append(
            _answer(
                "when",
                "When was it filmed?",
                "Unknown. No date in the file and no time clues in the sound or light.",
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
                f"{place}, from GPS in the file. This is a copy, so it may show where it was saved instead.",
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
                f"{place}, from the phone's GPS.",
                "strong",
                gps,
            )
        )
    elif postcode:
        answers.append(
            _answer(
                "where",
                "Where was it filmed?",
                f"Near postcode {postcode[0].data.get('value')}, visible in the picture.",
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
            answer = f"Probably {place} ({near}), going by what is visible."
            check = guess.data.get("corroboration") or {}

            if check.get("agree"):
                answer += (
                    " Matches reference and satellite photos."
                    if guess.data.get("verified")
                    else " A second AI agrees."
                )
            elif check:
                other = str(check.get("second_guess") or "no location")
                answer = (
                    f"Uncertain. One AI says {place}; another "
                    + (
                        "cannot place it."
                        if other.lower().startswith(("unknown", "no location"))
                        else f"says {other}."
                    )
                    + " Treat it as a lead only."
                )
        else:
            answer = re.sub(
                r"\s*\((low|medium|high) confidence\)",
                "",
                guess.title.removeprefix("Looks like: "),
            )
            answer += ", going by what is visible."

        confidence = {"high": "moderate", "medium": "moderate"}.get(
            guess.data.get("confidence"), "weak"
        )

        if not (guess.data.get("corroboration") or {}).get("agree", True):
            confidence = "weak"

        # A guess spanning a continent or more is not a location.
        if guess.data.get("location_available") is False or float(guess.data.get("radius_km") or 0) >= 2500:
            answer = "Unknown. Nothing in the picture narrows it down."
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
                f"Somewhere in {region}, going by the mains frequency.",
                "weak",
                grid,
            )
        )
    else:
        answers.append(
            _answer(
                "where",
                "Where was it filmed?",
                "Unknown. No location data or clues found.",
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
            (f"At least {len(pulses)} face{'' if len(pulses) == 1 else 's'} visible. "
             if any("face" in f.title for f in pulses) else "Skin visible, but no face. ")
            + (f"Heart rate about {best.data['bpm']:.0f} bpm, read from tiny skin colour changes."
               if reliable else "Heart rate could not be read (movement or compression)."),
            "moderate" if reliable else "weak", pulses[:3], best.time_s,
        ))  # fmt: skip

    # ---- hidden -----------------------------------------------------------------
    hidden_bits, hidden_f = [], []

    for f in findings:
        if "dtmf" in f.tags:
            hidden_bits.append(f"keypad tones: {f.title.split(': ', 1)[-1]}")
        elif "ultrasonic" in f.tags:
            hidden_bits.append(
                f"a tone too high for adults to hear ({f.data.get('hz', 0) / 1000:.1f} kHz)"
            )
        elif "doppler" in f.tags:
            hidden_bits.append("a passing aircraft or vehicle")
        elif f.title.startswith("Lights flicker"):
            hidden_bits.append("light flicker that reveals the power grid")
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
                "Nothing hidden in the sound or picture.",
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
                f"People speaking {voices[0].data.get('language')}, but the words are not clear.",
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


# ---------------------------------------------------------------- photographs

_ZONE_REGIONS = {
    -10: "Hawaii",
    -9: "Alaska",
    -8: "the US and Canadian west coast or Mexico's Pacific coast",
    -7: "the US Mountain states or western Mexico",
    -6: "the central US, Mexico or Central America",
    -5: "the US east coast, eastern Canada, Colombia or Peru",
    -4: "Chile, Venezuela or the eastern Caribbean",
    -3: "Brazil or Argentina",
    0: "the UK, Ireland, Portugal or West Africa",
    1: "western and central Europe or West Africa",
    2: "eastern Europe, Israel, Egypt or South Africa",
    3: "Turkey, Russia (Moscow), Saudi Arabia or East Africa",
    3.5: "Iran",
    4: "the UAE, Oman or the Caucasus",
    5: "Pakistan or Uzbekistan",
    5.5: "India or Sri Lanka",
    6: "Bangladesh or Kazakhstan",
    7: "Thailand, Vietnam or western Indonesia",
    8: "China, Hong Kong, Taiwan, Singapore, Malaysia, the Philippines or Western Australia",
    9: "Japan or Korea",
    9.5: "central Australia",
    10: "eastern Australia or Papua New Guinea",
    11: "eastern Australia in summer or the Solomon Islands",
    12: "New Zealand",
}


def _photo_answers(
    findings: list[Finding], local: dict[str, Any]
) -> list[dict[str, Any]]:
    """The photograph in plain English: one question per card."""
    by_title = lambda *prefixes: [f for f in findings if f.title.startswith(prefixes)]  # noqa: E731
    answers = []
    declared = [f for f in findings if "declared" in f.tags and "synthetic" in f.tags]
    splices = [
        f
        for f in findings
        if f.kind == "measured"
        and f.severity in ("warning", "critical")
        and {"splice", "clone"} & set(f.tags)
    ]
    thumbnail = by_title("Embedded thumbnail shows")
    editors = [
        f
        for f in findings
        if "editor" in f.tags and f.kind == "measured" and "synthetic" not in f.tags
    ]
    history = by_title("Edit history")
    modified = by_title("Modified ")
    resized = by_title("Resized or cropped")
    resaved = [
        f
        for f in findings
        if "resaved" in f.tags
        or f.title.startswith(("Metadata stripped", "Processed by"))
    ]
    double = by_title("Compressed twice")
    leads = [
        f
        for f in findings
        if f.severity == "notice"
        and ({"ela", "noise", "grid"} & set(f.tags))
        and f.kind == "measured"
    ]
    camera_style = by_title("Encoder fingerprint: camera-style")
    thumb_ok = by_title("Embedded thumbnail matches")
    maker_notes = by_title("Maker notes from")
    screenshot = by_title("This is a screenshot")

    # ---- edited? -------------------------------------------------------------------------------
    if declared:
        what = (
            declared[0].title.split(":", 1)[-1].strip()
            if ":" in declared[0].title
            else declared[0].title
        )
        answers.append(
            _answer(
                "edited",
                "Has this photo been edited?",
                f"Not a real photo: the file says an AI model made it ({what}).",
                "strong",
                declared[:2],
            )
        )
    elif splices:
        kinds = []

        for f in splices:
            label = ("part of the picture was copied" if "clone" in f.tags
                     else "the camera's preview shows a different picture" if "thumbnail" in f.tags
                     else "an area was pasted in" if "jpeg" in f.tags or "ela" in f.tags
                     else "an area has different grain" if "noise" in f.tags
                     else "an area compresses differently")  # fmt: skip

            if label not in kinds:
                kinds.append(label)

        strong = (
            len({f.module for f in splices}) >= 2
            or any(f.severity == "critical" for f in splices)
            or bool(thumbnail)
        )
        answers.append(
            _answer(
                "edited",
                "Has this photo been edited?",
                "Yes: " + "; ".join(kinds[:3]) + ".",
                "strong" if strong else "moderate",
                (thumbnail + splices)[:4],
            )
        )
    elif history or (editors and (modified or resized)):
        software = (
            editors[0].title.removeprefix("Processed by ")
            if editors
            else history[0]
            .data.get("history", [{}])[0]
            .get("softwareAgent", "editing software")
        )
        steps = (
            f" with {len(history[0].data.get('history', []))} recorded steps"
            if history
            else ""
        )
        when = (
            f", {modified[0].title.removeprefix('Modified ').split(' after')[0]} after it was taken"
            if modified
            else ""
        )
        answers.append(
            _answer(
                "edited",
                "Has this photo been edited?",
                f"Saved in {software}{steps}{when}. No pasting found, so probably a crop or filter.",
                "moderate",
                (editors + history + modified + resized)[:4],
            )
        )
    elif editors or resized or modified:
        f = (editors or resized or modified)[0]
        answers.append(
            _answer(
                "edited",
                "Has this photo been edited?",
                ("Resized or cropped. " if resized else "")
                + (
                    f"Re-saved by {editors[0].title.removeprefix('Processed by ')}. "
                    if editors
                    else ""
                )
                + ("Changed after it was taken. " if modified else "")
                + "No pasting found.",
                "moderate",
                [f] + [x for x in editors + resized + modified if x is not f][:3],
            )
        )
    elif double:
        answers.append(
            _answer(
                "edited",
                "Has this photo been edited?",
                "No changes found, but it was saved twice, so it is not the original file."
                + (
                    f" {len(leads)} area{'s' if len(leads) != 1 else ''} look{'s' if len(leads) == 1 else ''} different, but nothing confirms a paste; see the maps."
                    if leads
                    else ""
                ),
                "moderate",
                (double + leads)[:4],
            )
        )
    elif resaved and not screenshot:
        answers.append(
            _answer(
                "edited",
                "Has this photo been edited?",
                "No changes found, but an app or website re-saved it"
                + (
                    " and removed its camera details"
                    if any(f.title.startswith("Metadata stripped") for f in resaved)
                    else ""
                )
                + ", so it is not the original.",
                "moderate",
                resaved[:3],
            )
        )
    elif screenshot:
        answers.append(
            _answer(
                "edited",
                "Has this photo been edited?",
                "A screenshot, not a camera photo. No sign it was changed.",
                "moderate",
                screenshot[:1],
            )
        )
    else:
        checks = (
            camera_style
            + thumb_ok
            + maker_notes
            + by_title("No pasted", "No ghost", "No cloned")
        )
        answers.append(
            _answer(
                "edited",
                "Has this photo been edited?",
                "No signs of editing. Looks like one photo, saved once."
                + (
                    f" {len(leads)} area{'s' if len(leads) != 1 else ''} look{'s' if len(leads) == 1 else ''} slightly different, probably texture; see the maps."
                    if leads
                    else ""
                ),
                "strong"
                if len(checks) >= 3
                else "moderate"
                if len(checks) >= 1
                else "weak",
                (checks + leads)[:4],
            )
        )

    # ---- made by AI? -----------------------------------------------------------------
    signals = by_title("2 weak", "3 weak", "4 weak", "5 weak", "One weak")
    opinion = [f for f in findings if "ai-opinion" in f.tags]
    likely = [f for f in opinion if f.severity in ("warning", "notice")]
    none = by_title("No signs of AI generation")

    if declared:
        answers.append(
            _answer(
                "synthetic",
                "Was it made by AI?",
                "Yes. " + declared[0].detail.split(". ")[0] + ".",
                "strong",
                declared[:2],
            )
        )
    elif likely and signals:
        answers.append(
            _answer(
                "synthetic",
                "Was it made by AI?",
                "Possibly. "
                + likely[0]
                .detail.split(". ")[0]
                .removeprefix("Judging from the picture alone: ")
                + f"; and the file shows {signals[0].title.lower()}.",
                "moderate",
                likely[:1] + signals[:1],
            )
        )
    elif likely:
        answers.append(
            _answer(
                "synthetic",
                "Was it made by AI?",
                "Possibly, from its look alone: "
                + likely[0]
                .detail.split(". ")[0]
                .removeprefix("Judging from the picture alone: ")
                + ". The file has no AI label.",
                "weak",
                likely[:1],
            )
        )
    elif signals and signals[0].severity == "warning":
        answers.append(
            _answer(
                "synthetic",
                "Was it made by AI?",
                "Some weak signs, no proof: "
                + signals[0]
                .detail.split(". ")[0]
                .removeprefix("Nothing in the file declares a generator, but ")
                + ".",
                "weak",
                signals[:1],
            )
        )
    else:
        answers.append(
            _answer(
                "synthetic",
                "Was it made by AI?",
                ("No signs. " if none else "Nothing suggests it. ")
                + "No AI labels, and the pixels look like a real camera's.",
                "moderate" if none and (camera_style or maker_notes) else "weak",
                (none + opinion)[:2],
            )
        )

    # ---- when? -----------------------------------------------------------------------
    taken = by_title("Taken 20", "Taken 19")
    zone = by_title("Local clock was UTC")
    disagree = by_title(
        "Camera clock and GPS clock disagree",
        "Camera clock ",
        "File name and camera clock disagree",
    )
    copy = by_title("Timestamp is when this copy was made")
    weather_bad = by_title("Weather contradicts")

    if taken:
        when = taken[0].data.get("date_time_original") or taken[0].data.get(
            "xmp_create_date"
        )
        moment = datetime.fromisoformat(str(when)) if when else None
        tz_known = "UTC" in taken[0].title and "local clock" not in taken[0].title
        text = (
            f"{moment:%A %-d %B %Y at %H:%M}"
            if moment
            else taken[0].title.removeprefix("Taken ")
        )
        text += (
            " local time" if tz_known or zone else " (camera clock, time zone unknown)"
        )
        corroborated = bool(zone) or bool(maker_notes) and tz_known
        extra = ""

        if zone:
            extra = " Confirmed by the GPS clock."
        elif modified:
            extra = f" Changed {modified[0].title.removeprefix('Modified ').split(' after')[0]} later."

        if disagree:
            extra += " Another clock in the file disagrees, so be careful."

        if weather_bad:
            extra += " The weather then does not match the picture."

        confidence = (
            "weak"
            if disagree or weather_bad
            else "strong"
            if corroborated
            else "moderate"
        )
        answers.append(
            _answer(
                "when",
                "When was it taken?",
                f"{text}.{extra}",
                confidence,
                (taken + zone + disagree + modified + weather_bad)[:4],
            )
        )
    elif copy:
        answers.append(
            _answer(
                "when",
                "When was it taken?",
                "Unknown. The file only says when this copy was saved.",
                "unknown",
                copy[:1],
            )
        )
    else:
        answers.append(
            _answer(
                "when",
                "When was it taken?",
                "Unknown. The file has no date.",
                "unknown",
                by_title("Metadata stripped")[:1],
            )
        )

    # ---- where? ----------------------------------------------------------------------------------------
    gps = [f for f in findings if "gps" in f.tags]
    caption_place = by_title("Place written in the caption")
    postcode = [
        f for f in findings if "text" in f.tags and "postcode" in f.title.lower()
    ]
    geo_guess = [
        f
        for f in findings
        if f.module == "geo"
        and f.title.startswith(("Pixels say", "Located", "Pixels suggest"))
    ]
    looks = [
        f
        for f in findings
        if f.kind == "inferred"
        and "location" in f.tags
        and f.title.startswith("Looks like")
    ]

    if gps:
        place = gps[0].data.get("place") or gps[0].title.split(": ", 1)[-1]
        heading = gps[0].data.get("heading_deg")
        facing = (
            f", facing {_compass(heading)}" if isinstance(heading, (int, float)) else ""
        )
        copy_note = (
            " This is a copy, so the GPS may have been added later." if copy else ""
        )
        answers.append(
            _answer(
                "where",
                "Where was it taken?",
                f"{place}{facing}, from the phone's GPS.{copy_note}",
                "moderate" if copy else "strong",
                gps[:1],
            )
        )
    elif geo_guess:
        guess = geo_guess[0]
        place = str(guess.data.get("place") or "").strip()
        radius = float(guess.data.get("radius_km") or 0)
        verified = bool(guess.data.get("verified"))

        if guess.data.get("location_available") is False or radius >= 2500:
            answers.append(
                _answer(
                    "where",
                    "Where was it taken?",
                    "Unknown. Nothing in the picture narrows it down.",
                    "unknown",
                    [guess],
                )
            )
        else:
            answers.append(
                _answer(
                    "where",
                    "Where was it taken?",
                    f"{place} (within about {radius:.0f} km). Checked against reference and satellite photos."
                    if verified
                    else f"Probably {place} (within about {radius:.0f} km), going by what is visible.",
                    "moderate" if verified else "weak",
                    [guess],
                )
            )
    elif caption_place:
        answers.append(
            _answer(
                "where",
                "Where was it taken?",
                caption_place[0].title.split(": ", 1)[-1]
                + ", according to the file's caption.",
                "moderate",
                caption_place[:1],
            )
        )
    elif postcode:
        answers.append(
            _answer(
                "where",
                "Where was it taken?",
                f"Near postcode {postcode[0].data.get('value')}, visible in the picture.",
                "moderate",
                postcode[:1],
            )
        )
    elif looks:
        answer = re.sub(
            r"\s*\((low|medium|high) confidence\)",
            "",
            looks[0].title.removeprefix("Looks like: "),
        )
        answers.append(
            _answer(
                "where",
                "Where was it taken?",
                f"{answer}, going by what is visible.",
                "weak",
                looks[:1],
            )
        )
    elif zone:
        offset = float(zone[0].data.get("offset_h", 0))
        region = _ZONE_REGIONS.get(offset) or _ZONE_REGIONS.get(round(offset))
        answers.append(
            _answer(
                "where",
                "Where was it taken?",
                f"No GPS. The clock was set to UTC{offset:+g}"
                + (f", which suggests {region}" if region else "")
                + ".",
                "weak",
                zone[:1],
            )
        )
    else:
        answers.append(
            _answer(
                "where",
                "Where was it taken?",
                "Unknown. No GPS and no clues in the picture.",
                "unknown",
                [],
            )
        )

    # ---- device ----------------------------------------------------------------------------------------------
    device = by_title("Taken on ")

    if device:
        lens = device[0].data.get("lens")
        exposure = device[0].data.get("exposure")
        answers.append(
            _answer(
                "device",
                "What took it?",
                device[0].title.removeprefix("Taken on ")
                + (
                    f" ({lens})"
                    if lens and lens.lower() not in device[0].title.lower()
                    else ""
                )
                + (f": {exposure}" if exposure else "")
                + ".",
                "strong" if maker_notes else "moderate",
                (device + maker_notes)[:2],
            )
        )
    elif screenshot:
        answers.append(
            _answer(
                "device",
                "What took it?",
                "A screen capture, not a camera.",
                "moderate",
                screenshot[:1],
            )
        )

    # ---- people -----------------------------------------------------------------------------------------------
    faces = [
        f
        for f in findings
        if f.module == "faces" and "face" in f.tags and "No faces" not in f.title
    ]
    names = [f for f in faces if "identity" in f.tags]
    counted = [f for f in faces if "person" in f.tags]

    if names:
        answers.append(
            _answer(
                "people",
                "Who is in it?",
                f"The file names the people in it: {', '.join(names[0].data.get('names', [])[:3])}.",
                "strong",
                (names + counted)[:2],
            )
        )
    elif counted:
        recognisable = int(counted[0].data.get("recognisable") or 0)
        total = len(counted[0].data.get("faces") or [])
        answers.append(
            _answer(
                "people",
                "Who is in it?",
                f"{total} face{'s' if total != 1 else ''}, {recognisable} clear enough to recognise."
                if recognisable
                else f"{total} face{'s' if total != 1 else ''}, none clear enough to recognise.",
                "moderate",
                counted[:1],
            )
        )

    # ---- hidden ------------------------------------------------------------------------------------------------
    hidden = [
        f
        for f in findings
        if {"hidden", "steganography", "identity"} & set(f.tags)
        and f.severity in ("warning", "critical")
        and "camera" not in f.tags
    ]

    if hidden:
        bits = list(dict.fromkeys(f.title[0].lower() + f.title[1:] for f in hidden))
        answers.append(
            _answer(
                "hidden",
                "Anything hidden in the file?",
                "Yes: " + "; ".join(bits[:4]) + ".",
                "strong",
                hidden[:4],
            )
        )
    else:
        answers.append(
            _answer(
                "hidden",
                "Anything hidden in the file?",
                "Nothing hidden: no extra data, secret message or personal details.",
                "moderate",
                by_title("Nothing hidden")[:1],
            )
        )

    # ---- text --------------------------------------------------------------------------------------------------
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
            )
        )

    return answers


def _compass(deg: float) -> str:
    names = [
        "north",
        "north-east",
        "east",
        "south-east",
        "south",
        "south-west",
        "west",
        "north-west",
    ]

    return names[int((deg % 360) / 45 + 0.5) % 8]


def _local_brief_photo(findings: list[Finding]) -> dict[str, Any]:
    serious = [
        f
        for f in findings
        if f.severity in ("warning", "critical") and f.kind == "measured"
    ]
    declared = [f for f in findings if "declared" in f.tags and "synthetic" in f.tags]
    splices = [f for f in serious if {"splice", "clone"} & set(f.tags)]
    # Re-saves (double compression) say the file is not the original, not that
    # the picture changed; they do not move the verdict.
    edits_soft = [
        f
        for f in serious
        if {"editor", "history", "modified", "resized"} & set(f.tags)
        and "synthetic" not in f.tags
    ]
    synthetic_signals = [f for f in serious if "signals" in f.tags]

    if declared:
        verdict = "AI-GENERATED"
    elif splices:
        verdict = "EDITED"
    elif edits_soft or synthetic_signals:
        verdict = "POSSIBLY EDITED"
    else:
        verdict = "NO SIGNS OF EDITING"

    bullets: list[str] = []
    taken = [
        f
        for f in findings
        if f.title.startswith("Taken 20") or f.title.startswith("Taken 19")
    ]
    zone = [f for f in findings if f.title.startswith("Local clock was UTC")]

    if taken:
        bullets.append(
            f"WHEN: {taken[0].title} [{taken[0].id}]"
            + (
                f", confirmed by the GPS clock [{zone[0].id}]"
                if zone
                else ", from the camera's own labels"
            )
        )
    else:
        bullets.append("WHEN: no capture date in the file")

    gps = _with(findings, "gps")
    geo = [f for f in findings if f.module == "geo" and f.kind == "inferred"]
    place_bits = [f"{f.title} [{f.id}]" for f in gps[:1]] + [
        f"AI: {f.title} [{f.id}]" for f in geo[:1]
    ]
    bullets.append(
        "WHERE: " + ("; ".join(place_bits) if place_bits else "no location evidence")
    )
    device = [f for f in findings if f.title.startswith("Taken on ")]

    if device:
        bullets.append(f"DEVICE: {device[0].title} [{device[0].id}]")

    people = [f for f in findings if f.module == "faces" and "person" in f.tags]

    if people:
        bullets.append(f"PEOPLE: {people[0].title} [{people[0].id}]")

    integrity = declared + splices + edits_soft

    if integrity:
        bullets.append(
            "INTEGRITY: " + "; ".join(f"{f.title} [{f.id}]" for f in integrity[:6])
        )
    else:
        bullets.append(
            "INTEGRITY: metadata, compression tables, embedded thumbnail, grain and pixels show no edit"
        )

    hidden = [
        f
        for f in findings
        if {"hidden", "steganography", "identity"} & set(f.tags)
        and f.severity != "info"
    ]

    if hidden:
        bullets.append(
            "HIDDEN: " + "; ".join(f"{f.title} [{f.id}]" for f in hidden[:4])
        )

    # Two independent traces on the same region make the paste claim strong.
    modules = {f.module for f in splices}

    if len(modules) >= 2 or any(
        "corroborated_by" in f.data and f.data["corroborated_by"] for f in splices
    ):
        bullets.append(
            "CORROBORATED: two independent pixel traces mark the same region"
        )

    return {
        "verdict": verdict,
        "bullets": bullets,
        "text": f"VERDICT: {verdict}.\n" + "\n".join(bullets),
        "corroborated": len(modules) >= 2,
        "corroborated_at": None,
    }


def _summary_plain(answers: list[dict[str, Any]]) -> str:
    edited = next(a for a in answers if a["key"] == "edited")
    when = next(
        (a for a in answers if a["key"] == "when" and a["confidence"] == "strong"), None
    )
    text = edited["answer"]

    if when:
        verb = "Taken" if "taken" in when["question"] else "Filmed"
        text += f" {verb} " + when["answer"].split(",", 1)[0].rstrip(".") + "."

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
        still = ctx.kind == "image"
        noun = "photo" if still else "video"
        local = _local_brief_photo(findings) if still else _local_brief(findings)

        if local["corroborated"] and not still:
            out.finding(
                "critical",
                "Edit corroborated by sound and picture",
                local["bullets"][-1].removeprefix("CORROBORATED: ").capitalize()
                + ". Two independent measurements agree on an edit point.",
                time_s=local["corroborated_at"],
                tags=["splice", "inconsistency"],
            )

        data: dict[str, Any] = dict(local)
        data["kind"] = ctx.kind
        data["answers"] = (
            _photo_answers(ctx.findings, local)
            if still
            else _plain_answers(ctx.findings, local)
        )
        data["summary_plain"] = _summary_plain(data["answers"])

        provider = ai.for_case(ctx.options)

        if provider is not None and findings:
            out.progress(0.4, "drafting narrative")
            evidence = [
                {"id": f.id, "module": f.module, "severity": f.severity, "kind": f.kind,
                 "title": f.title, "detail": f.detail, "time_s": f.time_s}
                for f in findings
            ]  # fmt: skip

            try:
                narrative = await provider.structured(
                    ctx.workdir,
                    instructions=(
                        f"You write the summary of a {noun}-forensics case file for a non-expert. Use ONLY "
                        "the findings given; cite their ids in brackets. Distinguish measured findings "
                        "from AI-inferred ones. Do not overstate: weak evidence is weak. The rule-based "
                        f"verdict is '{local['verdict']}'; agree with it unless the findings clearly contradict it."
                    ),
                    content=[{"type": "input_text", "text": str(evidence)}],
                    schema=_SCHEMA,
                    name="case_brief",
                    effort=ctx.by_effort(low="low"),
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

        if provider is not None and findings:
            out.progress(0.7, "writing the dossier")

            try:
                dossier = await provider.structured(
                    ctx.workdir,
                    instructions=_DOSSIER.replace("this video", f"this {noun}").replace(
                        "filmed it", "took it" if still else "filmed it"
                    ),
                    content=[
                        {
                            "type": "input_text",
                            "text": json.dumps(_evidence(findings), ensure_ascii=False),
                        }
                    ],
                    schema=_DOSSIER_SCHEMA,
                    name="dossier",
                    effort=ctx.by_effort(low="low"),
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
                    "dossier", "dossier", f"What this {noun} gives away", data=dossier
                )
            except Exception as error:  # noqa: BLE001 - optional
                out.log(f"dossier unavailable: {error}")

        ctx.publish(self.name, data)
        out.progress(1.0)
