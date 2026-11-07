"""Text in the picture (local OCR with Tesseract).

Signs, screens, plates, receipts and clocks in the background say where and
when something was filmed. Each sampled frame is contrast-enhanced and
upscaled before recognition; the recognised words are then checked for
patterns that matter: UK postcodes, number plates, phone numbers, web
addresses, dates and clock times.
"""

import asyncio
import csv
import io
import re
import shutil
from typing import Any

import cv2
import numpy as np

from earwitness.modules.base import CaseContext
from earwitness.modules.base import Reporter
from earwitness.modules.base import Skip
from earwitness.modules.base import run_tool

_FRAMES = 14
_MIN_CONF = 75

_PATTERNS = [
    (
        "postcode",
        re.compile(r"\b([A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2})\b"),
        "UK postcode",
        "location",
    ),
    (
        "plate",
        re.compile(r"\b([A-Z]{2}\d{2}\s?[A-Z]{3})\b"),
        "UK number plate",
        "vehicle",
    ),
    (
        "phone",
        # UK-style (0xx / +44) or international (+cc) numbers, 10-13 digits.
        re.compile(
            r"(?<![\d])((?:\+\d{1,3}[\s-]?|0)\d{2,4}[\s-]?\d{3,4}[\s-]?\d{3,4})(?![\d])"
        ),
        "phone number",
        "contact",
    ),
    (
        "url",
        re.compile(
            r"\b((?:https?://|www\.)[\w.-]+\.[a-z]{2,}\S*|[\w-]+\.(?:com|co\.uk|org|net|io|ac\.uk|gov\.uk|ed\.ac\.uk))\b",
            re.I,
        ),
        "web address",
        "contact",
    ),
    ("email", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b"), "email address", "contact"),
    (
        "date",
        re.compile(r"\b(\d{1,2}[/.-]\d{1,2}[/.-](?:\d{2}|\d{4})|\d{4}-\d{2}-\d{2})\b"),
        "date",
        "time",
    ),
    (
        "clock",
        re.compile(r"\b([01]?\d|2[0-3]):[0-5]\d(?::[0-5]\d)?\b"),
        "clock time",
        "time",
    ),
]


def _still_views(ctx: CaseContext) -> list[tuple[str, np.ndarray]]:
    image = ctx.image()

    if image is None:
        return []

    views = [("the whole picture", ctx.image_scaled(2000))]
    h, w = image.shape[:2]

    if max(h, w) > 2400:
        for row in range(2):
            for col in range(2):
                tile = image[
                    row * h // 2 : (row + 1) * h // 2, col * w // 2 : (col + 1) * w // 2
                ]

                if tile.shape[1] > 2000:
                    scale = 2000 / tile.shape[1]
                    tile = cv2.resize(
                        tile, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
                    )

                views.append(
                    (
                        f"{'top' if row == 0 else 'bottom'}-{'left' if col == 0 else 'right'} quarter",
                        tile,
                    )
                )

    return views


def _prepare(frame: np.ndarray, zoom: float = 1.6) -> bytes:
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    if zoom != 1.0:
        gray = cv2.resize(gray, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_CUBIC)
    gray = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(gray)
    _, encoded = cv2.imencode(".png", gray)

    return encoded.tobytes()


def _words(tsv: str) -> list[dict[str, Any]]:
    rows = list(
        csv.DictReader(io.StringIO(tsv), delimiter="\t", quoting=csv.QUOTE_NONE)
    )
    words = []

    for row in rows:
        text = (row.get("text") or "").strip()

        try:
            conf = float(row.get("conf") or -1)
        except ValueError:
            continue

        if conf < _MIN_CONF or len(text) < 2 or not re.search(r"[A-Za-z0-9]", text):
            continue

        # Reject OCR confetti: mostly punctuation or one repeated character.
        if (
            sum(c.isalnum() for c in text) < 0.6 * len(text)
            or len(set(text.lower())) == 1
        ):
            continue

        words.append(
            {
                "text": text,
                "conf": conf,
                "line": (row.get("block_num"), row.get("par_num"), row.get("line_num")),
                "box": [
                    int(row["left"]),
                    int(row["top"]),
                    int(row["width"]),
                    int(row["height"]),
                ],
            }
        )

    return words


def _lines(words: list[dict[str, Any]]) -> list[str]:
    grouped: dict[Any, list[str]] = {}

    for word in words:
        grouped.setdefault(word["line"], []).append(word["text"])

    return [" ".join(parts) for parts in grouped.values() if len(" ".join(parts)) >= 3]


class Ocr:
    name = "ocr"
    title = "TEXT IN FRAME (OCR)"
    description = "Reads signs, screens and plates with Tesseract; flags postcodes, plates, phones, URLs, dates."
    depends = ["timeline"]

    async def run(self, ctx: CaseContext, out: Reporter) -> None:
        if shutil.which("tesseract") is None:
            raise Skip("tesseract not installed")

        info = await ctx.video_info()
        still = ctx.kind == "image"

        if info is None and not still:
            raise Skip("no video stream")

        if still:
            # One picture at full resolution: the whole view, then quadrants at
            # native pixels so small signs and plates are not lost to downscaling.
            views = _still_views(ctx)
            times = [0.0] * len(views)
        else:
            timeline = await ctx.wait_for("timeline") or {}
            times = list(np.linspace(0.3, max(0.3, info.duration_s - 0.3), _FRAMES))
            times += [c + 0.3 for c in (timeline.get("cuts") or [])[:6]]
            times = sorted({round(t, 2) for t in times})
            views = []

        seen: dict[str, dict[str, Any]] = {}
        width = min(1600, info.width) if info else None

        for number, time_s in enumerate(times):
            if still:
                label, frame = views[number]
                out.progress(number / len(times), f"reading {label}")
            else:
                out.progress(number / len(times), f"reading frame @ {time_s:.1f}s")
                frame = await ctx.frame_at(time_s, width=width)

            if frame is None:
                continue

            image = await asyncio.to_thread(
                _prepare, frame, 1.0 if still and frame.shape[1] >= 1400 else 1.6
            )
            found_lines: list[str] = []

            # Sparse mode catches scattered signs; automatic layout catches blocks
            # such as plates and labels that sparse mode skips.
            for psm in ("11", "3") if not (still and number > 0) else ("11",):
                tsv = await run_tool(
                    "tesseract",
                    "stdin",
                    "stdout",
                    "--psm",
                    psm,
                    "-l",
                    "eng",
                    "tsv",
                    stdin=image,
                )

                if tsv:
                    found_lines += _lines(_words(tsv.decode("utf-8", "replace")))

            for line in found_lines:
                key = re.sub(r"\W+", "", line.lower())

                if len(key) < 3:
                    continue

                entry = seen.setdefault(key, {"text": line, "times": [], "frames": 0})
                if time_s not in entry["times"]:
                    entry["times"].append(time_s)
                    entry["frames"] += 1

        lines = sorted(seen.values(), key=lambda e: (-e["frames"], e["times"][0]))
        out.artifact(
            "text",
            "table",
            "Recognised text",
            data={
                "columns": ["view", "views", "text"]
                if still
                else ["first seen (s)", "frames", "text"],
                "rows": [
                    [0 if still else e["times"][0], e["frames"], e["text"]]
                    for e in lines[:200]
                ],
            },
        )
        hits = []
        joined = "\n".join(e["text"] for e in lines)

        for key, pattern, label, tag in _PATTERNS:
            for match in pattern.finditer(
                joined if key not in ("postcode", "plate") else joined.upper()
            ):
                value = match.group(0).strip()
                entry = next(
                    (
                        e
                        for e in lines
                        if value.lower().replace(" ", "")
                        in e["text"].lower().replace(" ", "")
                    ),
                    None,
                )
                hits.append(
                    {
                        "kind": key,
                        "value": value,
                        "label": label,
                        "tag": tag,
                        "time_s": entry["times"][0] if entry else None,
                    }
                )

        unique = {(h["kind"], h["value"]): h for h in hits}

        for hit in list(unique.values())[:12]:
            out.finding(
                "critical" if hit["kind"] in ("postcode", "plate") else "warning",
                f"{hit['label'][0].upper() + hit['label'][1:]} in frame: {hit['value']}",
                f"OCR reads a {hit['label']} in the picture. "
                + {
                    "location": "A postcode places the scene to a street.",
                    "vehicle": "A plate identifies a vehicle at the scene.",
                    "contact": "Contact details tie the scene to a business or person.",
                    "time": "Visible dates and clocks can confirm or contradict the claimed time.",
                }[hit["tag"]],
                time_s=None if still else hit["time_s"],
                tags=["text", hit["tag"]],
                data=hit,
            )

        persistent = [
            e for e in lines if e["frames"] >= 2 or len(e["text"]) >= 8 or still
        ][:8]

        if persistent:
            out.finding(
                "notice",
                f"{len(lines)} text fragments readable in frame",
                "Most legible: "
                + "; ".join(f'"{e["text"]}"' for e in persistent[:5])
                + ".",
                time_s=None if still else persistent[0]["times"][0],
                tags=["text"],
                data={"lines": [e["text"] for e in persistent]},
            )
        else:
            out.finding(
                "info",
                "No legible text in frame",
                "OCR found no readable text in the sampled frames"
                + (f" beyond {len(lines)} short fragments." if lines else "."),
                tags=["text"],
            )

        ctx.publish(
            self.name,
            {"lines": [e["text"] for e in lines[:80]], "hits": list(unique.values())},
        )
        out.progress(1.0)
