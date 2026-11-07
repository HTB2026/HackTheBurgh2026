"""Tiger Data (TimescaleDB) storage: the grid-frequency record and the case index.

The database is required. Its URL comes from TIMESCALE_SERVICE_URL (read from
the repo's .env); tables live in the schema named by EARWITNESS_DB_SCHEMA
(default ``earwitness``), which lets tests run in a throwaway schema.

Long-lived processes (the server) open a connection pool for their event loop
with ``open_pool``; anything else (CLI, demo, tests) gets a one-shot connection.
"""

import asyncio
import hashlib
import json
import os
import re
from collections.abc import AsyncIterator
from collections.abc import Iterable
from contextlib import asynccontextmanager
from datetime import datetime
from datetime import timezone
from typing import Any

import psycopg
from dotenv import load_dotenv
from psycopg import sql
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_SCHEMA_SQL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "schema.sql")
_IDENT = re.compile(r"^[a-z_][a-z0-9_]{0,62}$")

load_dotenv(os.path.join(_ROOT, ".env"))

_pools: dict[asyncio.AbstractEventLoop, AsyncConnectionPool] = {}


class DatabaseNotConfigured(RuntimeError):
    def __init__(self) -> None:
        super().__init__(
            "TIMESCALE_SERVICE_URL is not set. Earwitness needs its Tiger Data "
            "database: put the service URL in .env (see .env.example)."
        )


def url() -> str:
    value = os.environ.get("TIMESCALE_SERVICE_URL", "").strip()

    if not value:
        raise DatabaseNotConfigured()

    return value


def schema() -> str:
    name = os.environ.get("EARWITNESS_DB_SCHEMA", "earwitness").strip()

    if not _IDENT.match(name):
        raise ValueError(f"EARWITNESS_DB_SCHEMA {name!r} is not a plain identifier")

    return name


async def _configure(conn: psycopg.AsyncConnection) -> None:
    await conn.execute(
        sql.SQL("SET search_path TO {}, public").format(sql.Identifier(schema()))
    )
    await conn.commit()


async def open_pool() -> AsyncConnectionPool:
    """Open (once) a pool bound to the running event loop."""
    loop = asyncio.get_running_loop()

    if loop not in _pools:
        pool = AsyncConnectionPool(
            url(), min_size=1, max_size=5, configure=_configure, open=False
        )
        await pool.open(wait=True, timeout=30)
        _pools[loop] = pool

    return _pools[loop]


async def close_pool() -> None:
    pool = _pools.pop(asyncio.get_running_loop(), None)

    if pool is not None:
        await pool.close()


@asynccontextmanager
async def connection() -> AsyncIterator[psycopg.AsyncConnection]:
    pool = _pools.get(asyncio.get_running_loop())

    if pool is not None:
        async with pool.connection() as conn:
            yield conn

        return

    async with await psycopg.AsyncConnection.connect(url()) as conn:
        await _configure(conn)
        yield conn


def _statements(text: str) -> list[str]:
    lines = [line for line in text.splitlines() if not line.lstrip().startswith("--")]

    return [s.strip() for s in "\n".join(lines).split(";") if s.strip()]


async def migrate() -> bool:
    """Apply schema.sql if it changed since the last run; True when applied."""
    with open(_SCHEMA_SQL) as handle:
        text = handle.read()

    digest = hashlib.sha256(text.encode()).hexdigest()
    name = sql.Identifier(schema())

    # Continuous aggregates cannot be created inside a transaction.
    async with await psycopg.AsyncConnection.connect(url(), autocommit=True) as conn:
        await conn.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(name))
        await conn.execute(sql.SQL("SET search_path TO {}, public").format(name))
        await conn.execute(
            "CREATE TABLE IF NOT EXISTS schema_version ("
            " digest TEXT PRIMARY KEY, applied_at TIMESTAMPTZ NOT NULL DEFAULT now())"
        )
        seen = await (
            await conn.execute(
                "SELECT 1 FROM schema_version WHERE digest = %s", [digest]
            )
        ).fetchone()

        if seen:
            return False

        for statement in _statements(text):
            await conn.execute(statement)

        await conn.execute("INSERT INTO schema_version (digest) VALUES (%s)", [digest])

    return True


async def drop_schema() -> None:
    async with await psycopg.AsyncConnection.connect(url(), autocommit=True) as conn:
        await conn.execute(
            sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(sql.Identifier(schema()))
        )


# ---- grid frequency --------------------------------------------------------


async def grid_window(
    start: datetime, end: datetime, sources: Iterable[str]
) -> tuple[str, list[float], list[float]] | None:
    """Stored samples for the first source (in preference order) covering the window."""
    async with connection() as conn:
        for source in sources:
            covered = await (
                await conn.execute(
                    "SELECT 1 FROM grid_coverage"
                    " WHERE source = %s AND start_ts <= %s AND end_ts >= %s LIMIT 1",
                    [source, start, end],
                )
            ).fetchone()

            if not covered:
                continue

            rows = await (
                await conn.execute(
                    "SELECT extract(epoch FROM time)::float8, hz FROM grid_frequency"
                    " WHERE source = %s"
                    " AND time BETWEEN %s - interval '60 s' AND %s + interval '60 s'"
                    " ORDER BY time",
                    [source, start, end],
                )
            ).fetchall()

            if rows:
                return source, [r[0] for r in rows], [r[1] for r in rows]

    return None


async def store_grid(
    source: str,
    times: list[float],
    values: list[float],
    start: datetime,
    end: datetime,
) -> int:
    """Bulk-load samples and record [start, end] as covered; returns rows stored."""
    async with connection() as conn:
        async with conn.transaction():
            await conn.execute(
                "CREATE TEMP TABLE grid_load (time TIMESTAMPTZ, source TEXT,"
                " hz DOUBLE PRECISION) ON COMMIT DROP"
            )

            async with conn.cursor().copy(
                "COPY grid_load (time, source, hz) FROM STDIN"
            ) as copy:
                for t, v in zip(times, values):
                    await copy.write_row(
                        (datetime.fromtimestamp(t, tz=timezone.utc), source, v)
                    )

            cursor = await conn.execute(
                "INSERT INTO grid_frequency (time, source, hz)"
                " SELECT DISTINCT ON (time) time, source, hz FROM grid_load"
                " ORDER BY time ON CONFLICT DO NOTHING"
            )
            stored = cursor.rowcount
            await conn.execute(
                "INSERT INTO grid_coverage (source, start_ts, end_ts) VALUES (%s, %s, %s)",
                [source, start, end],
            )

    return stored


async def grid_minutes(start: datetime, end: datetime) -> list[dict[str, Any]]:
    async with connection() as conn:
        cursor = await conn.cursor(row_factory=dict_row).execute(
            "SELECT bucket, source, avg_hz, min_hz, max_hz, samples"
            " FROM grid_frequency_1m WHERE bucket BETWEEN %s AND %s"
            " ORDER BY source, bucket",
            [start, end],
        )

        return await cursor.fetchall()


# ---- case index ------------------------------------------------------------


def _dated_start(findings: list[dict[str, Any]]) -> str | None:
    """When the grid (hum or lamp flicker) dated the recording, if it did."""
    for finding in findings:
        data = finding.get("data") or {}

        if (
            finding.get("module") == "enf"
            and data.get("start")
            and data.get("confident", True)
        ):
            return str(data["start"])

    return None


async def index_case(case: Any) -> None:
    """Upsert one case's summary and replace its findings (reruns stay correct)."""
    summary = case.summary()
    custody = summary.get("case") or {}
    events = [e for e in case.events if e["type"] == "finding"]
    findings = [e["finding"] for e in events]

    async with connection() as conn:
        async with conn.transaction():
            await conn.execute(
                "INSERT INTO cases (id, filename, created, status, sha256, duration_s,"
                " verdict, summary_plain, dated_start, options, updated_at)"
                " VALUES (%s, %s, to_timestamp(%s), %s, %s, %s, %s, %s, %s, %s, now())"
                " ON CONFLICT (id) DO UPDATE SET filename = EXCLUDED.filename,"
                " status = EXCLUDED.status, sha256 = EXCLUDED.sha256,"
                " duration_s = EXCLUDED.duration_s, verdict = EXCLUDED.verdict,"
                " summary_plain = EXCLUDED.summary_plain,"
                " dated_start = EXCLUDED.dated_start, options = EXCLUDED.options,"
                " updated_at = now()",
                [
                    case.id,
                    case.filename,
                    case.created,
                    case.status,
                    custody.get("sha256"),
                    custody.get("duration_s"),
                    summary.get("verdict"),
                    summary.get("summary_plain"),
                    _dated_start(findings),
                    Jsonb(case.options),
                ],
            )
            await conn.execute("DELETE FROM findings WHERE case_id = %s", [case.id])

            if events:
                async with conn.cursor() as cursor:
                    await cursor.executemany(
                        "INSERT INTO findings (time, case_id, finding_id, module,"
                        " severity, title, detail, tags, data)"
                        " VALUES (to_timestamp(%s), %s, %s, %s, %s, %s, %s, %s, %s)"
                        " ON CONFLICT DO NOTHING",
                        [
                            (
                                e["ts"],
                                case.id,
                                f["id"],
                                f["module"],
                                f["severity"],
                                f["title"],
                                f.get("detail"),
                                list(f.get("tags") or []),
                                Jsonb(f.get("data") or {}),
                            )
                            for e, f in zip(events, findings)
                        ],
                    )


async def delete_case(case_id: str) -> None:
    async with connection() as conn:
        async with conn.transaction():
            await conn.execute("DELETE FROM findings WHERE case_id = %s", [case_id])
            await conn.execute("DELETE FROM cases WHERE id = %s", [case_id])


async def search_findings(
    *,
    since: datetime | None = None,
    until: datetime | None = None,
    severity: str | None = None,
    module: str | None = None,
    text: str | None = None,
    limit: int = 200,
) -> list[dict[str, Any]]:
    clauses, params = [sql.SQL("TRUE")], []

    for column, op, value in (
        ("f.time", ">=", since),
        ("f.time", "<=", until),
        ("f.severity", "=", severity),
        ("f.module", "=", module),
    ):
        if value is not None:
            clauses.append(sql.SQL(f"{column} {op} %s"))
            params.append(value)

    if text:
        clauses.append(sql.SQL("(f.title ILIKE %s OR f.detail ILIKE %s)"))
        params += [f"%{text}%", f"%{text}%"]

    query = sql.SQL(
        "SELECT f.time, f.case_id, f.finding_id, f.module, f.severity, f.title,"
        " f.detail, f.tags, f.data, c.filename, c.verdict, c.dated_start"
        " FROM findings f JOIN cases c ON c.id = f.case_id"
        " WHERE {} ORDER BY f.time DESC LIMIT %s"
    ).format(sql.SQL(" AND ").join(clauses))

    async with connection() as conn:
        cursor = await conn.cursor(row_factory=dict_row).execute(
            query, [*params, limit]
        )

        return await cursor.fetchall()


async def search_cases(
    *, dated_from: datetime | None = None, dated_to: datetime | None = None
) -> list[dict[str, Any]]:
    """Cases the grid dated into [dated_from, dated_to]."""
    async with connection() as conn:
        cursor = await conn.cursor(row_factory=dict_row).execute(
            "SELECT id, filename, created, status, verdict, dated_start FROM cases"
            " WHERE dated_start IS NOT NULL"
            " AND (%(lo)s::timestamptz IS NULL OR dated_start >= %(lo)s)"
            " AND (%(hi)s::timestamptz IS NULL OR dated_start <= %(hi)s)"
            " ORDER BY dated_start",
            {"lo": dated_from, "hi": dated_to},
        )

        return await cursor.fetchall()


# ---- health and status -----------------------------------------------------


async def health() -> dict[str, Any]:
    try:
        async with connection() as conn:
            row = await (
                await conn.execute(
                    "SELECT extversion FROM pg_extension WHERE extname = 'timescaledb'"
                )
            ).fetchone()

        return {"ok": True, "timescaledb": row[0] if row else None, "schema": schema()}
    except (psycopg.Error, DatabaseNotConfigured, OSError) as error:
        return {"ok": False, "error": type(error).__name__}


async def status() -> dict[str, Any]:
    async with connection() as conn:

        async def one(query: str) -> Any:
            cursor = conn.cursor(row_factory=dict_row)

            return await (await cursor.execute(query)).fetchall()

        return {
            "schema": schema(),
            "timescaledb": (
                await one(
                    "SELECT extversion FROM pg_extension WHERE extname = 'timescaledb'"
                )
            )[0]["extversion"],
            "hypertables": await one(
                "SELECT hypertable_name, num_chunks,"
                " hypertable_size(format('%I.%I', hypertable_schema,"
                " hypertable_name)::regclass) AS bytes"
                " FROM timescaledb_information.hypertables"
                " WHERE hypertable_schema = current_schema()"
            ),
            "grid": await one(
                "SELECT source, count(*) AS samples, min(time) AS first,"
                " max(time) AS last FROM grid_frequency GROUP BY source"
            ),
            "coverage": await one(
                "SELECT source, start_ts, end_ts, fetched_at FROM grid_coverage"
                " ORDER BY start_ts"
            ),
            "cases": (await one("SELECT count(*) AS n FROM cases"))[0]["n"],
            "findings": (await one("SELECT count(*) AS n FROM findings"))[0]["n"],
        }


def to_json(value: Any) -> Any:
    """Make DB rows JSON-ready (datetimes to ISO strings)."""
    return json.loads(json.dumps(value, default=lambda v: v.isoformat()))
