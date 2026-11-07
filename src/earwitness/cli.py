import argparse
import asyncio
import json
from datetime import datetime
from datetime import timezone

from earwitness import adsb
from earwitness import analysis
from earwitness import grid
from earwitness import hum
from earwitness import logging
from earwitness import media
from earwitness import report
from earwitness import synth
from earwitness._common import ForensicError
from earwitness.flights import Origin

logger = logging.get_logger(__name__)

_FEEDS = {
    "airplanes.live": "https://api.airplanes.live",
    "adsb.lol": "https://api.adsb.lol",
}


def _origin(text: str) -> Origin:
    latitude, longitude = (float(part) for part in text.split(","))

    return Origin(latitude, longitude)


def _time(text: str) -> datetime:
    parsed = datetime.fromisoformat(text)

    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="earwitness")
    commands = parser.add_subparsers(dest="command", required=True)

    analyse = commands.add_parser("analyse", help="Date and locate a recording.")
    analyse.add_argument("file")
    analyse.add_argument("--grid", help="Grid frequency CSV (NESO or own reference).")
    analyse.add_argument("--flights", help="Flight log JSONL from log-flights.")
    analyse.add_argument(
        "--origin", type=_origin, help="LAT,LON the flight log is centred on."
    )
    analyse.add_argument(
        "--ground-height", type=float, default=100.0, help="Metres, GNSS datum."
    )
    analyse.add_argument(
        "--start", type=_time, help="Known recording start (ISO, UTC)."
    )
    analyse.add_argument(
        "--half-width", type=float, default=15_000.0, help="Search half-width (m)."
    )
    analyse.add_argument(
        "--out", default="earwitness-out", help="Directory for figures."
    )

    reference = commands.add_parser(
        "reference", help="Turn a long hum recording into a grid CSV."
    )
    reference.add_argument("file")
    reference.add_argument("--start", type=_time, required=True)
    reference.add_argument("--out", required=True)

    log = commands.add_parser("log-flights", help="Record live aircraft positions.")
    log.add_argument("--origin", type=_origin, required=True)
    log.add_argument("--radius-nm", type=int, default=25)
    log.add_argument("--interval", type=float, default=2.0)
    log.add_argument("--source", choices=sorted(_FEEDS), default="airplanes.live")
    log.add_argument("--out", default="flights.jsonl")

    serve = commands.add_parser("serve", help="Run the forensic suite web UI.")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)

    demo = commands.add_parser("demo", help="Write the demo pack of planted clips.")
    demo.add_argument("--out", default="demo")

    database = commands.add_parser("db", help="Manage the Tiger Data database.")
    database.add_argument(
        "action",
        choices=["migrate", "status", "import-cache"],
        help="migrate: apply the schema; status: what is stored; "
        "import-cache: load old ~/.cache/earwitness/grid/*.npz windows.",
    )

    synthesise = commands.add_parser(
        "synthesise", help="Write a synthetic test scenario."
    )
    synthesise.add_argument("--out", default="synthetic")

    return parser


async def _analyse(args: argparse.Namespace) -> None:
    options = analysis.Options(
        grid_path=args.grid,
        flight_path=args.flights,
        origin=args.origin,
        ground_height_m=args.ground_height,
        start_override=args.start,
        half_width_m=args.half_width,
    )
    result = await analysis.analyse(args.file, options)

    if isinstance(result, ForensicError):
        print(json.dumps({"file": args.file, "error": str(result)}))

        return

    summary = report.to_dict(result)
    summary["figures"] = report.plot(result, args.out)
    print(json.dumps(summary, indent=2))


async def _reference(args: argparse.Namespace) -> None:
    recording = await media.load(args.file)

    if isinstance(recording, ForensicError):
        print(json.dumps({"error": str(recording)}))

        return

    profile = hum.detect(recording.samples, recording.sample_rate)

    if isinstance(profile, ForensicError):
        print(json.dumps({"error": str(profile)}))

        return

    trace = hum.extract(recording.samples, recording.sample_rate, profile)
    await grid.write_csv(args.out, grid.from_trace(trace, args.start))
    print(
        json.dumps(
            {"written": args.out, "points": len(trace.values), "snr_db": profile.snr_db}
        )
    )


async def _log_flights(args: argparse.Namespace) -> None:
    logging.initialise_logging()
    feed = adsb.default(base_url=_FEEDS[args.source])
    logger.info(
        "Recording aircraft positions.", extra={"out": args.out, "source": args.source}
    )
    await adsb.record(
        feed,
        args.out,
        args.origin,
        radius_nm=args.radius_nm,
        interval_s=args.interval,
    )


async def _db(args: argparse.Namespace) -> None:
    from earwitness import db

    match args.action:
        case "migrate":
            applied = await db.migrate()
            print(json.dumps({"schema": db.schema(), "applied": applied}))
        case "status":
            await db.migrate()
            print(json.dumps(db.to_json(await db.status()), indent=2))
        case "import-cache":
            await db.migrate()
            print(json.dumps(await _import_cache(), indent=2))


async def _import_cache() -> dict[str, int]:
    """Move grid windows from the old .npz file cache into the hypertable."""
    import glob
    import os

    import numpy as np

    from earwitness import db

    imported = rows = 0

    for path in sorted(glob.glob(os.path.expanduser("~/.cache/earwitness/grid/*.npz"))):
        data = np.load(path)
        times, values = [float(t) for t in data["t"]], [float(v) for v in data["v"]]

        if not times:
            continue

        rows += await db.store_grid(
            str(data["source"]),
            times,
            values,
            datetime.fromtimestamp(min(times), tz=timezone.utc),
            datetime.fromtimestamp(max(times), tz=timezone.utc),
        )
        imported += 1

    return {"windows": imported, "rows_stored": rows}


async def _main(args: argparse.Namespace) -> None:
    match args.command:
        case "analyse":
            await _analyse(args)
        case "reference":
            await _reference(args)
        case "log-flights":
            await _log_flights(args)
        case "db":
            await _db(args)
        case "demo":
            from earwitness import demo

            print(json.dumps(await demo.build_pack(args.out), indent=2))
        case "synthesise":
            print(json.dumps(await synth.create_scenario(args.out), indent=2))
        case _:
            print(json.dumps({"error": "unknown_command"}))


def main() -> None:
    args = _parser().parse_args()

    if args.command == "serve":
        import uvicorn

        uvicorn.run(
            "earwitness.server.app:app",
            host=args.host,
            port=args.port,
            timeout_graceful_shutdown=2,
        )

        return

    asyncio.run(_main(args))
