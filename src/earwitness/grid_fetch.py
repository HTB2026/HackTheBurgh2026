"""Fetch GB grid frequency around a moment, without downloading whole months.

Sources, in order of preference:
- NESO historic system frequency: 1 s resolution, monthly CSVs published after
  the month ends. The CSV rows are ~27 bytes, so we seek with HTTP Range requests
  straight to the hours we need.
- Elexon Insights FREQ: 15 s resolution, near real time.

Everything is cached under ~/.cache/earwitness/grid.
"""

import asyncio
import hashlib
import json
import os
from datetime import datetime
from datetime import timedelta
from datetime import timezone

import httpx
import numpy as np

from earwitness.grid import GridSeries

_NESO_PACKAGE = (
    "https://api.neso.energy/api/3/action/package_show?id=system-frequency-data"
)
_ELEXON = "https://data.elexon.co.uk/bmrs/api/v1/system/frequency"
_ROW_BYTES = 27
_CACHE = os.path.expanduser("~/.cache/earwitness/grid")
_MONTHS = [
    "january", "february", "march", "april", "may", "june", "july",
    "august", "september", "october", "november", "december",
]  # fmt: skip


def _cache_path(*parts: object) -> str:
    os.makedirs(_CACHE, exist_ok=True)
    key = hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()[:16]

    return os.path.join(_CACHE, key)


async def _neso_resource(
    client: httpx.AsyncClient, year: int, month: int
) -> str | None:
    index_path = _cache_path("neso-index")

    if os.path.isfile(index_path) and os.path.getmtime(index_path) > (
        datetime.now().timestamp() - 86_400
    ):
        with open(index_path) as handle:
            index = json.load(handle)
    else:
        response = await client.get(_NESO_PACKAGE)
        response.raise_for_status()
        index = {
            r["name"].strip().lower(): r["url"]
            for r in response.json()["result"]["resources"]
            if r.get("url", "").endswith(".csv")
        }

        with open(index_path, "w") as handle:
            json.dump(index, handle)

    wanted = f"{_MONTHS[month - 1]} {year}"

    for name, url in index.items():
        if name.startswith(wanted):
            return url

    suffix = f"-{year}-{month}.csv"

    return next((u for u in index.values() if u.endswith(suffix)), None)


def _parse_rows(text: str) -> tuple[list[float], list[float]]:
    times, values = [], []

    for line in text.splitlines():
        parts = line.strip().split(",")

        if len(parts) < 2 or not parts[0][:1].isdigit():
            continue

        try:
            moment = datetime.fromisoformat(parts[0]).replace(tzinfo=timezone.utc)
            times.append(moment.timestamp())
            values.append(float(parts[1]))
        except ValueError:
            continue

    return times, values


async def _range(client: httpx.AsyncClient, url: str, start: int, end: int) -> str:
    response = await client.get(url, headers={"Range": f"bytes={max(0, start)}-{end}"})
    response.raise_for_status()

    return response.text


async def _neso_window(
    client: httpx.AsyncClient,
    url: str,
    month_start: datetime,
    start: datetime,
    end: datetime,
) -> tuple[list[float], list[float]]:
    # Seek: estimate byte offset from seconds into the month, then correct.
    target = start.timestamp()
    offset = int((target - month_start.timestamp()) * _ROW_BYTES)

    for _ in range(5):
        text = await _range(client, url, offset, offset + 4_096)
        times, _ = _parse_rows(text.split("\n", 1)[-1])

        if not times:
            offset = max(0, offset - 200_000)

            continue

        error_s = times[0] - target

        if -120 <= error_s <= 0:
            break

        offset -= int((error_s + 60) * _ROW_BYTES)
        offset = max(0, offset)

    span = int((end.timestamp() - start.timestamp() + 180) * _ROW_BYTES * 1.05)
    text = await _range(client, url, offset, offset + span)

    return _parse_rows(text.split("\n", 1)[-1] if offset > 0 else text)


async def _elexon_window(
    client: httpx.AsyncClient, start: datetime, end: datetime
) -> tuple[list[float], list[float]]:
    times, values = [], []
    cursor = start

    while cursor < end:
        chunk_end = min(end, cursor + timedelta(hours=24))
        response = await client.get(
            _ELEXON,
            params={
                "from": cursor.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "to": chunk_end.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "format": "json",
            },
        )
        response.raise_for_status()

        for row in response.json().get("data", []):
            moment = datetime.fromisoformat(
                row["measurementTime"].replace("Z", "+00:00")
            )
            times.append(moment.timestamp())
            values.append(float(row["frequency"]))

        cursor = chunk_end

    return times, values


def _to_series(times: list[float], values: list[float]) -> GridSeries | None:
    if len(times) < 10:
        return None

    order = np.argsort(times)
    t = np.asarray(times)[order]
    v = np.asarray(values)[order]
    keep = np.concatenate(([True], np.diff(t) > 0))
    t, v = t[keep], v[keep]
    step = float(np.median(np.diff(t)))
    regular = np.arange(t[0], t[-1] + step / 2, step)

    return GridSeries(
        start=datetime.fromtimestamp(t[0], tz=timezone.utc),
        step_s=step,
        values=np.interp(regular, t, v),
    )


async def fetch(start: datetime, end: datetime) -> tuple[GridSeries | None, str]:
    """Grid frequency covering [start, end]; returns (series, source description)."""
    start = start.astimezone(timezone.utc)
    end = end.astimezone(timezone.utc)
    cache = _cache_path("window", start.isoformat(), end.isoformat())

    if os.path.isfile(cache + ".npz"):
        data = np.load(cache + ".npz")
        source = str(data["source"])

        return _to_series(list(data["t"]), list(data["v"])), source

    times: list[float] = []
    values: list[float] = []
    source = "none"

    async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
        try:
            # NESO months can straddle the window; fetch each month part.
            cursor = start

            while cursor < end:
                month_start = cursor.replace(
                    day=1, hour=0, minute=0, second=0, microsecond=0
                )
                next_month = (month_start + timedelta(days=32)).replace(day=1)
                part_end = min(end, next_month)
                url = await _neso_resource(client, cursor.year, cursor.month)

                if url is None:
                    times, values = [], []

                    break

                t, v = await _neso_window(client, url, month_start, cursor, part_end)
                times += t
                values += v
                cursor = part_end

            if times:
                source = "NESO historic frequency (1 s)"
        except httpx.HTTPError, ValueError, KeyError:
            times, values = [], []

        if not times:
            try:
                times, values = await _elexon_window(client, start, end)
                source = "Elexon FREQ (15 s)"
            except httpx.HTTPError, ValueError, KeyError:
                times, values = [], []

    inside = [
        (t, v)
        for t, v in zip(times, values)
        if start.timestamp() - 60 <= t <= end.timestamp() + 60
    ]

    if inside:
        t_arr, v_arr = zip(*inside)
        np.savez(
            cache + ".npz",
            t=np.array(t_arr),
            v=np.array(v_arr),
            source=np.array(source),
        )

        return _to_series(list(t_arr), list(v_arr)), source

    return None, source


if __name__ == "__main__":
    import sys

    async def _main() -> None:
        moment = datetime.fromisoformat(sys.argv[1])
        series, source = await fetch(
            moment - timedelta(hours=1), moment + timedelta(hours=1)
        )
        print(
            source,
            None
            if series is None
            else (series.start, series.step_s, len(series.values), series.values[:5]),
        )

    asyncio.run(_main())
