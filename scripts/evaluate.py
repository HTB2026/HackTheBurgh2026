"""Run the full suite over a set of real videos and tabulate the results.

    uv run python scripts/evaluate.py OUT_DIR VIDEO [VIDEO ...] [--truth truth.json]

truth.json (optional) maps file names to known facts, e.g.
    {"IMG_1261.MOV": {"edited": false, "place": "Edinburgh"}}
Every video without an entry is assumed unedited. The script writes
OUT_DIR/results.json and prints a table: verdict, edit-type warnings (false
positives on unedited footage), module errors, the plain answers and timings.
"""

import argparse
import asyncio
import json
import math
import os
import shutil
import sys
import time

from dotenv import load_dotenv

from earwitness.server.jobs import CaseStore

EDIT_TAGS = {"splice", "cut", "clone", "sync", "inconsistency", "duplicate", "timing"}


def _haversine_km(a: tuple[float, float], b: tuple[float, float]) -> float:
    lat1, lon1, lat2, lon2 = map(math.radians, (*a, *b))
    h = (
        math.sin((lat2 - lat1) / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    )

    return 2 * 6371.0 * math.asin(math.sqrt(h))


async def run_one(store: CaseStore, path: str) -> dict:
    case = store.create(path, {"original_name": os.path.basename(path)})
    shutil.copy(path, case.evidence_path)
    started = time.time()
    store.start(case)
    await case.task
    events = case.events
    findings = [e["finding"] for e in events if e["type"] == "finding"]
    modules = {e["module"]: e for e in events if e["type"] == "module"}
    brief = next(
        (
            e["artifact"]["data"]
            for e in events
            if e["type"] == "artifact" and e["artifact"]["id"] == "brief:brief"
        ),
        {},
    )

    return {
        "file": os.path.basename(path),
        "case": case.id,
        "seconds": round(time.time() - started, 1),
        "verdict": brief.get("verdict"),
        "summary": brief.get("summary_plain"),
        "answers": brief.get("answers", []),
        "edit_flags": [
            {
                "module": f["module"],
                "severity": f["severity"],
                "title": f["title"],
                "time_s": f["time_s"],
            }
            for f in findings
            if f["severity"] in ("warning", "critical")
            and EDIT_TAGS & set(f["tags"])
            and f["kind"] == "measured"
        ],
        "geo_guess": next(
            (
                {
                    "lat": f["data"].get("latitude"),
                    "lon": f["data"].get("longitude"),
                    "radius_km": f["data"].get("radius_km"),
                    "title": f["title"],
                }
                for f in findings
                if f["module"] == "geo"
                and f["title"].startswith(("Pixels say", "Located", "Pixels suggest"))
            ),
            None,
        ),
        "errors": {
            m: e.get("message") for m, e in modules.items() if e["status"] == "error"
        },
        "skipped": {
            m: e.get("message") for m, e in modules.items() if e["status"] == "skipped"
        },
        "findings": [
            {
                "module": f["module"],
                "severity": f["severity"],
                "title": f["title"],
                "kind": f["kind"],
            }
            for f in findings
        ],
    }


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("out")
    parser.add_argument("videos", nargs="+")
    parser.add_argument("--truth")
    args = parser.parse_args()
    load_dotenv(
        os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"
        )
    )
    truth = json.load(open(args.truth)) if args.truth else {}
    store = CaseStore(os.path.join(args.out, "cases"))
    results = []

    for path in args.videos:
        print(f"... {os.path.basename(path)}", file=sys.stderr, flush=True)
        result = await run_one(store, path)
        result["truth"] = truth.get(result["file"], {"edited": False})
        results.append(result)

    false_alarms = 0

    for r in results:
        expected_edit = r["truth"].get("edited", False)
        flagged = bool(r["edit_flags"]) or r["verdict"] == "EDITED"
        wrong = flagged != expected_edit
        false_alarms += wrong and not expected_edit
        print(
            f"\n=== {r['file']}  [{r['seconds']}s]  verdict={r['verdict']}  {'WRONG' if wrong else 'ok'}"
        )

        for flag in r["edit_flags"]:
            print(
                f"   EDIT FLAG {flag['severity']:8} {flag['module']:10} {flag['title']}"
            )

        guess, known = r.get("geo_guess"), r["truth"]

        if guess and guess.get("lat") is not None and "lat" in known:
            km = _haversine_km(
                (known["lat"], known["lon"]), (guess["lat"], guess["lon"])
            )
            r["geo_error_km"] = km
            inside = km <= (guess.get("radius_km") or 0)
            print(
                f"   GEO {km:,.1f} km from truth ({known['place']}); "
                f"stated ±{guess.get('radius_km') or 0:.0f} km -> {'inside' if inside else 'OUTSIDE'}"
            )

        for module, message in r["errors"].items():
            print(f"   ERROR {module}: {message}")

        for answer in r["answers"]:
            print(
                f"   [{answer['confidence']:8}] {answer['question']} {answer['answer']}"
            )

    with open(os.path.join(args.out, "results.json"), "w") as handle:
        json.dump(results, handle, indent=2, default=str)

    print(
        f"\n{len(results)} videos, {false_alarms} false edit alarms, "
        f"{sum(bool(r['errors']) for r in results)} with module errors"
    )


if __name__ == "__main__":
    asyncio.run(main())
