"""HTTP + WebSocket front door for the forensic suite."""

import asyncio
import os
import shutil
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime
from datetime import timezone
from typing import Any

from fastapi import Depends
from fastapi import FastAPI
from fastapi import HTTPException
from fastapi import UploadFile
from fastapi import WebSocket
from fastapi import WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.responses import HTMLResponse
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from earwitness import db
from earwitness.modules import registry
from earwitness.server import auth
from earwitness.server import enhance
from earwitness.server import report_html
from earwitness.server.jobs import CaseStore

_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
_WEB = os.path.join(_ROOT, "web", "dist")

# earwitness.db has loaded .env by now.
store = CaseStore(os.environ.get("EARWITNESS_CASES", os.path.join(_ROOT, "cases")))


@asynccontextmanager
async def _lifespan(_: FastAPI) -> AsyncIterator[None]:
    # The database is required: fail at startup, not on the first case.
    db.url()
    await db.migrate()
    await db.open_pool()
    await store.sync_index()

    try:
        yield
    finally:
        await db.close_pool()


app = FastAPI(title="earwitness", lifespan=_lifespan)


def _when(text: str | None, name: str) -> datetime | None:
    if not text:
        return None

    try:
        moment = datetime.fromisoformat(text)
    except ValueError:
        raise HTTPException(400, f"{name} must be an ISO date/time")

    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


class PathCase(BaseModel):
    path: str
    options: dict[str, Any] = {}


class EnhanceRequest(BaseModel):
    time_s: float
    x: float
    y: float
    w: float
    h: float
    frames: int = 9


def _case(case_id: str):
    case = store.cases.get(case_id)

    if case is None:
        raise HTTPException(404, "no such case")

    return case


@app.get("/api/health")
async def health() -> dict[str, Any]:
    return {
        "ok": True,
        "openai": bool(os.environ.get("OPENAI_API_KEY")),
        "database": await db.health(),
        "auth": auth.public_config(),
        "modules": [
            {"name": m.name, "title": m.title, "description": m.description}
            for m in registry.all_modules()
        ],
    }


@app.get("/api/cases")
async def list_cases() -> list[dict[str, Any]]:
    return sorted(
        (c.summary() for c in store.cases.values()), key=lambda s: -s["created"]
    )


@app.post("/api/cases")
async def upload(
    file: UploadFile,
    approx_time: str | None = None,
    search_hours: float | None = None,
    user: auth.User = Depends(auth.require_user),
) -> dict[str, Any]:
    options: dict[str, Any] = {"original_name": file.filename, "uploaded_by": user.sub}

    if approx_time:
        options["approx_time"] = approx_time

    if search_hours:
        options["search_hours"] = search_hours

    case = store.create(file.filename or "evidence.mp4", options)

    with open(case.evidence_path, "wb") as handle:
        await asyncio.to_thread(shutil.copyfileobj, file.file, handle, 1 << 20)

    store.start(case)

    return case.summary()


@app.post("/api/cases/from-path")
async def from_path(
    request: PathCase, user: auth.User = Depends(auth.require_user)
) -> dict[str, Any]:
    """Open a file already on this machine (no upload; handy for big files)."""
    path = os.path.expanduser(request.path)

    if not os.path.isfile(path):
        raise HTTPException(400, "file not found")

    options = {
        "original_name": os.path.basename(path),
        **request.options,
        "uploaded_by": user.sub,
    }
    # A grid record saved next to the evidence is used automatically.
    sibling = os.path.join(os.path.dirname(path), "grid.csv")

    if "grid_path" not in options and os.path.isfile(sibling):
        options["grid_path"] = sibling

    case = store.create(path, options)
    await asyncio.to_thread(shutil.copyfile, path, case.evidence_path)
    store.start(case)

    return case.summary()


@app.get("/api/cases/{case_id}")
async def get_case(case_id: str) -> dict[str, Any]:
    case = _case(case_id)

    return {"summary": case.summary(), "events": case.events}


@app.delete("/api/cases/{case_id}")
async def delete_case(
    case_id: str, _: auth.User = Depends(auth.require_user)
) -> dict[str, Any]:
    deleted = store.delete(case_id)
    await db.delete_case(case_id)

    return {"deleted": deleted}


@app.get("/api/search")
async def search(
    since: str | None = None,
    until: str | None = None,
    severity: str | None = None,
    module: str | None = None,
    q: str | None = None,
    limit: int = 200,
) -> list[dict[str, Any]]:
    """Findings across every case, newest first."""
    rows = await db.search_findings(
        since=_when(since, "since"),
        until=_when(until, "until"),
        severity=severity,
        module=module,
        text=q,
        limit=max(1, min(limit, 1000)),
    )

    return db.to_json(rows)


@app.get("/api/search/dated")
async def search_dated(
    start: str | None = None, end: str | None = None
) -> list[dict[str, Any]]:
    """Cases the grid record dated into [start, end]."""
    return db.to_json(
        await db.search_cases(
            dated_from=_when(start, "start"), dated_to=_when(end, "end")
        )
    )


@app.get("/api/grid")
async def grid_minutes(start: str, end: str) -> list[dict[str, Any]]:
    """Stored GB grid frequency as one-minute buckets (continuous aggregate)."""
    lo, hi = _when(start, "start"), _when(end, "end")

    if lo is None or hi is None or hi <= lo or (hi - lo).days > 31:
        raise HTTPException(400, "need start < end, at most 31 days apart")

    return db.to_json(await db.grid_minutes(lo, hi))


@app.post("/api/cases/{case_id}/rerun")
async def rerun(
    case_id: str,
    modules: list[str] | None = None,
    _: auth.User = Depends(auth.require_user),
) -> dict[str, Any]:
    case = _case(case_id)

    if case.task and not case.task.done():
        raise HTTPException(409, "already running")

    case.events.clear()

    for queue in list(case.listeners):
        queue.put_nowait({"type": "reset", "seq": -1})

    events = os.path.join(case.workdir, "events.jsonl")

    if os.path.isfile(events):
        os.remove(events)

    case._log = None
    case.status = "queued"
    store.start(case, only=modules)

    return case.summary()


@app.get("/api/cases/{case_id}/files/{path:path}")
async def case_file(case_id: str, path: str) -> FileResponse:
    case = _case(case_id)
    full = os.path.realpath(os.path.join(case.workdir, path))

    if not full.startswith(
        os.path.realpath(case.workdir) + os.sep
    ) or not os.path.isfile(full):
        raise HTTPException(404, "no such file")

    return FileResponse(full)


@app.post("/api/cases/{case_id}/enhance")
async def enhance_region(case_id: str, request: EnhanceRequest) -> JSONResponse:
    case = _case(case_id)
    result = await enhance.enhance(
        case,
        request.time_s,
        (request.x, request.y, request.w, request.h),
        request.frames,
    )

    if result is None:
        raise HTTPException(422, "could not enhance this region")

    return JSONResponse(result)


@app.get("/api/cases/{case_id}/report", response_class=HTMLResponse)
async def report(case_id: str) -> HTMLResponse:
    return HTMLResponse(report_html.render(_case(case_id)))


@app.websocket("/ws/cases/{case_id}")
async def stream(socket: WebSocket, case_id: str) -> None:
    case = store.cases.get(case_id)
    await socket.accept()

    if case is None:
        await socket.send_json({"type": "error", "message": "no such case"})
        await socket.close()

        return

    queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    # Register before replaying so no event falls between the two.
    case.listeners.add(queue)

    try:
        replay = list(case.events)

        for event in replay:
            await socket.send_json(event)

        seen = len(replay)

        while True:
            event = await queue.get()

            if event["type"] == "reset":
                seen = 0
                await socket.send_json(event)

                continue

            if event["seq"] < seen:
                continue

            await socket.send_json(event)
    except WebSocketDisconnect:
        pass
    finally:
        case.listeners.discard(queue)


if os.path.isdir(_WEB):
    app.mount(
        "/assets", StaticFiles(directory=os.path.join(_WEB, "assets")), name="assets"
    )

    @app.get("/{path:path}")
    async def spa(path: str) -> FileResponse:
        candidate = os.path.join(_WEB, path)

        if path and os.path.isfile(candidate):
            return FileResponse(candidate)

        return FileResponse(os.path.join(_WEB, "index.html"))
