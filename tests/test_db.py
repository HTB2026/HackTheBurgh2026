"""The Tiger Data store: grid round-trip and the case index."""

import asyncio
import shutil
from datetime import datetime
from datetime import timedelta
from datetime import timezone

import httpx
import pytest

from earwitness import db
from earwitness import demo as fixtures
from earwitness import grid_fetch
from earwitness.server.jobs import CaseStore


def test_grid_fetch_stores_then_serves_from_database(monkeypatch):
    start = datetime(2001, 6, 3, 12, 0, tzinfo=timezone.utc)
    end = start + timedelta(minutes=20)
    calls = {"neso": 0}

    async def fake_resource(client, year, month):
        return "https://example.invalid/neso.csv"

    async def fake_window(client, url, month_start, lo, hi):
        calls["neso"] += 1
        times = [lo.timestamp() + s for s in range(int((hi - lo).total_seconds()) + 1)]

        return times, [50.0 + 0.01 * ((s % 60) - 30) / 30 for s in range(len(times))]

    monkeypatch.setattr(grid_fetch, "_neso_resource", fake_resource)
    monkeypatch.setattr(grid_fetch, "_neso_window", fake_window)
    first, source = asyncio.run(grid_fetch.fetch(start, end))

    assert source == grid_fetch._NESO
    assert first is not None and calls["neso"] == 1

    # Any HTTP now would fail: the second fetch must come from the database.
    async def no_network(*args, **kwargs):
        raise httpx.ConnectError("offline")

    monkeypatch.setattr(grid_fetch, "_neso_window", no_network)
    monkeypatch.setattr(grid_fetch, "_elexon_window", no_network)
    inner, source = asyncio.run(
        grid_fetch.fetch(start + timedelta(minutes=5), end - timedelta(minutes=5))
    )

    assert source == grid_fetch._NESO
    assert inner is not None
    assert inner.step_s == pytest.approx(1.0)
    assert abs((inner.start - (start + timedelta(minutes=4))).total_seconds()) <= 1
    assert inner.values.min() >= 49.99 and inner.values.max() <= 50.01


def test_partial_window_is_not_marked_covered(monkeypatch):
    start = datetime(2001, 7, 1, 0, 0, tzinfo=timezone.utc)
    end = start + timedelta(hours=1)
    calls = {"n": 0}

    async def fake_resource(client, year, month):
        return "https://example.invalid/neso.csv"

    async def half_window(client, url, month_start, lo, hi):
        calls["n"] += 1

        return [lo.timestamp() + s for s in range(1800)], [50.0] * 1800

    monkeypatch.setattr(grid_fetch, "_neso_resource", fake_resource)
    monkeypatch.setattr(grid_fetch, "_neso_window", half_window)
    asyncio.run(grid_fetch.fetch(start, end))
    asyncio.run(grid_fetch.fetch(start, end))

    # Only half the hour arrived, so the full hour is asked for again.
    assert calls["n"] == 2


@pytest.fixture(scope="module")
def planted(tmp_path_factory):
    return fixtures.planted_video(str(tmp_path_factory.mktemp("v") / "planted.mp4"))


def test_case_index_tracks_runs_reruns_and_deletes(tmp_path, planted):
    async def go():
        store = CaseStore(str(tmp_path / "cases"))
        case = store.create(planted, {"original_name": "planted.mp4"})
        shutil.copy(planted, case.evidence_path)
        store.start(case, only=["provenance", "spectrum"])
        await case.task
        first = await db.search_findings(text="", limit=1000)
        mine = [r for r in first if r["case_id"] == case.id]

        # A rerun replaces the findings instead of piling up duplicates.
        case.events.clear()
        store.start(case, only=["spectrum"])
        await case.task
        rerun = [
            r for r in await db.search_findings(limit=1000) if r["case_id"] == case.id
        ]
        critical = await db.search_findings(severity="critical", limit=1000)
        future = await db.search_findings(
            since=datetime.now(timezone.utc) + timedelta(days=1)
        )
        store.delete(case.id)
        await db.delete_case(case.id)
        after = await db.search_findings(limit=1000)

        return case, mine, rerun, critical, future, after

    case, mine, rerun, critical, future, after = asyncio.run(go())

    assert mine, "findings were not indexed"
    assert {r["filename"] for r in mine} == {"planted.mp4"}
    assert {r["module"] for r in rerun} == {"spectrum"}
    assert len({r["finding_id"] for r in rerun}) == len(rerun)
    assert all(r["severity"] == "critical" for r in critical)
    assert future == []
    assert all(r["case_id"] != case.id for r in after)
