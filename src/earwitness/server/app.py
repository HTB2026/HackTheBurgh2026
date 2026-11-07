"""HTTP + WebSocket front door for the forensic suite."""

import asyncio
import os
import shutil
from typing import Any

from dotenv import load_dotenv
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

from earwitness.modules import registry
from earwitness.server import enhance
from earwitness.server import report_html
from earwitness.server.jobs import CaseStore

_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
_WEB = os.path.join(_ROOT, "web", "dist")

load_dotenv(os.path.join(_ROOT, ".env"))

app = FastAPI(title="earwitness")
store = CaseStore(os.environ.get("EARWITNESS_CASES", os.path.join(_ROOT, "cases")))


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
    file: UploadFile, approx_time: str | None = None, search_hours: float | None = None
) -> dict[str, Any]:
    options: dict[str, Any] = {"original_name": file.filename}

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
async def from_path(request: PathCase) -> dict[str, Any]:
    """Open a file already on this machine (no upload; handy for big files)."""
    path = os.path.expanduser(request.path)

    if not os.path.isfile(path):
        raise HTTPException(400, "file not found")

    options = {"original_name": os.path.basename(path), **request.options}
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
async def delete_case(case_id: str) -> dict[str, Any]:
    return {"deleted": store.delete(case_id)}


@app.post("/api/cases/{case_id}/rerun")
async def rerun(case_id: str, modules: list[str] | None = None) -> dict[str, Any]:
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
