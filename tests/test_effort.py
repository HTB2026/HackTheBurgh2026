"""Effort levels: recorded per case, changeable on rerun, and gating modules."""

import asyncio

import cv2
import numpy as np
import pytest
from test_auth import ALICE  # noqa: F401 - fixtures below are used by name
from test_auth import bearer
from test_auth import client  # noqa: F401
from test_auth import owned_case
from test_auth import profiles  # noqa: F401
from test_auth import token

from earwitness.modules import registry
from earwitness.modules.base import CaseContext
from earwitness.server import app as server
from earwitness.server import jobs


def test_bad_effort_is_refused_before_a_case_exists(client):  # noqa: F811
    before = len(server.store.cases)
    response = client.post(
        "/api/cases?effort=extreme",
        files={"file": ("clip.mp4", b"not a video")},
        headers={"Authorization": f"Bearer {token()}"},
    )

    assert response.status_code == 400
    assert len(server.store.cases) == before


def test_rerun_changes_and_records_the_effort(client, profiles, monkeypatch):  # noqa: F811
    case = owned_case(ALICE)
    restarted = []
    monkeypatch.setattr(
        server.store, "restart", lambda c, only=None: restarted.append(c.id)
    )

    assert (
        client.post(
            f"/api/cases/{case.id}/rerun?effort=fast", headers=bearer(ALICE)
        ).status_code
        == 400
    )

    response = client.post(
        f"/api/cases/{case.id}/rerun?effort=high", headers=bearer(ALICE)
    )

    assert response.status_code == 200
    assert response.json()["effort"] == "high"
    assert case.options["effort"] == "high"
    assert (
        case.history[-1]["action"] == "reran" and case.history[-1]["effort"] == "high"
    )
    assert restarted == [case.id]

    # Saved to case.json, so it survives a restart of the server.
    open(case.evidence_path, "wb").close()
    assert (
        type(server.store)(server.store.root).cases[case.id].options["effort"] == "high"
    )


def test_cases_without_an_effort_are_standard(client, profiles):  # noqa: F811
    case = owned_case(ALICE)

    assert case.summary()["effort"] == "medium"


@pytest.mark.parametrize(
    "options, effort, picked",
    [
        ({}, "medium", 6),
        ({"effort": "low"}, "low", 3),
        ({"effort": "high"}, "high", 10),
        ({"effort": "silly"}, "medium", 6),
    ],
)
def test_by_effort(options, effort, picked):
    ctx = CaseContext("t", "/nonexistent.mp4", "/tmp", options)

    assert ctx.effort == effort
    assert ctx.by_effort(low=3, medium=6, high=10) == picked
    # A level left out falls back to the medium value.
    assert ctx.by_effort(low="low") == ("low" if effort == "low" else None)


class _DeepOnly:
    name = "deep_only"
    title = "DEEP"
    description = "runs only at high effort"
    min_effort = "high"

    async def run(self, ctx, out):
        ctx.publish(self.name, "ran")


class _Waiter:
    name = "waiter"
    title = "WAITER"
    description = "waits on deep_only"

    async def run(self, ctx, out):
        out.finding(
            "info", f"deep_only said {await ctx.wait_for('deep_only', 5)!r}", ""
        )


@pytest.mark.parametrize(
    "effort, said, status", [("medium", "None", "skipped"), ("high", "'ran'", "done")]
)
def test_min_effort_skips_without_blocking_others(
    tmp_path, monkeypatch, effort, said, status
):
    monkeypatch.setattr(
        registry, "all_modules", lambda kind="video": [_DeepOnly(), _Waiter()]
    )
    store = jobs.CaseStore(str(tmp_path))
    case = store.create("still.png", {"effort": effort})
    cv2.imwrite(case.evidence_path, np.full((32, 48, 3), 128, np.uint8))

    asyncio.run(asyncio.wait_for(jobs.run_case(case), 60))

    statuses = {e["module"]: e["status"] for e in case.events if e["type"] == "module"}
    assert statuses["deep_only"] == status
    assert statuses["waiter"] == "done"
    assert [e["finding"]["title"] for e in case.events if e["type"] == "finding"] == [
        f"deep_only said {said}"
    ]
    assert (
        next(e for e in case.events if e["type"] == "case")["case"]["effort"] == effort
    )
