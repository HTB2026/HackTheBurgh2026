"""Every test run gets its own schema on the Tiger Data service, dropped after."""

import asyncio
import os
import uuid

import pytest
from psycopg import sql

os.environ["EARWITNESS_DB_SCHEMA"] = f"test_{uuid.uuid4().hex[:10]}"

from earwitness import db  # noqa: E402 - must see the test schema name


async def _seed_grid() -> None:
    """Reuse the grid record already stored in the main schema (saves NESO trips)."""
    main = os.environ.get("EARWITNESS_GRID_SEED_SCHEMA", "earwitness")

    async with db.connection() as conn:
        present = await (
            await conn.execute(
                "SELECT to_regclass(%s) IS NOT NULL", [f"{main}.grid_coverage"]
            )
        ).fetchone()

        if not present[0]:
            return

        await conn.execute(
            sql.SQL(
                "INSERT INTO grid_frequency SELECT * FROM {}.grid_frequency"
                " ON CONFLICT DO NOTHING"
            ).format(sql.Identifier(main))
        )
        await conn.execute(
            sql.SQL("INSERT INTO grid_coverage SELECT * FROM {}.grid_coverage").format(
                sql.Identifier(main)
            )
        )


@pytest.fixture(scope="session", autouse=True)
def database():
    asyncio.run(db.migrate())
    asyncio.run(_seed_grid())

    yield db.schema()

    asyncio.run(db.drop_schema())
