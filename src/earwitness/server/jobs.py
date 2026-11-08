"""Runs every module on one piece of evidence and fans events out to listeners."""

import asyncio
import hashlib
import json
import os
import shutil
import time
import traceback
import uuid
from typing import Any

import psycopg

from earwitness import ai
from earwitness import db
from earwitness.modules import registry
from earwitness.modules.base import CaseContext
from earwitness.modules.base import Reporter
from earwitness.modules.base import Skip
from earwitness.modules.base import run_tool
from earwitness.server import auth


class Case:
    def __init__(
        self, case_id: str, workdir: str, filename: str, options: dict[str, Any]
    ):
        self.id = case_id
        self.workdir = workdir
        self.filename = filename
        self.options = options
        self.created = time.time()
        self.events: list[dict[str, Any]] = []
        self.listeners: set[asyncio.Queue[dict[str, Any]]] = set()
        self.status = "queued"
        self.ctx: CaseContext | None = None
        self.task: asyncio.Task[None] | None = None
        # Logged right after the `case` event of the next run (e.g. why it restarted).
        self.note: str | None = None
        # Who uploaded it ({id, name, picture}, public) and every hand that
        # touched it since. Kept in case.json: events.jsonl is wiped by a rerun.
        self.owner: dict[str, Any] | None = None
        self.history: list[dict[str, Any]] = []
        self._log = None

    @property
    def evidence_path(self) -> str:
        return os.path.join(
            self.workdir, "evidence" + os.path.splitext(self.filename)[1].lower()
        )

    def emit(self, event: dict[str, Any]) -> None:
        event = {"seq": len(self.events), "ts": round(time.time(), 3), **event}
        # Encode before keeping it: an event that cannot be sent would otherwise
        # stay in memory and break every page load and WebSocket of the case.
        # Anything still not JSON-able becomes text; memory then matches disk.
        line = json.dumps(event, default=str)
        event = json.loads(line)
        self.events.append(event)

        if self._log is None:
            self._log = open(os.path.join(self.workdir, "events.jsonl"), "a")

        self._log.write(line + "\n")
        self._log.flush()

        for queue in list(self.listeners):
            queue.put_nowait(event)

    def summary(self) -> dict[str, Any]:
        case_event = next((e for e in self.events if e["type"] == "case"), None)
        findings = [e["finding"] for e in self.events if e["type"] == "finding"]
        brief = next(
            (
                e["artifact"]["data"]
                for e in reversed(self.events)
                if e["type"] == "artifact" and e["artifact"]["id"] == "brief:brief"
            ),
            {},
        )
        where = next(
            (a for a in brief.get("answers") or [] if a.get("key") == "where"), None
        )
        located = where is not None and where.get("confidence") != "unknown"

        return {
            "id": self.id,
            "filename": self.filename,
            "created": self.created,
            "status": self.status,
            "kind": (case_event["case"].get("kind") if case_event else None) or "video",
            "findings": len(findings),
            "warnings": sum(f["severity"] in ("warning", "critical") for f in findings),
            "case": case_event["case"] if case_event else None,
            "verdict": brief.get("verdict"),
            "summary_plain": brief.get("summary_plain"),
            "place": where["answer"] if located else None,
            "place_confidence": where["confidence"] if located else None,
            "ai_provider": self.options.get("ai_provider"),
            "owner": self.owner,
            "history": self.history,
        }

    def record(self, action: str, by: dict[str, Any] | None, **extra: Any) -> None:
        """Add a line to the handling record (call ``CaseStore.save_meta`` after)."""
        self.history.append(
            {"action": action, "at": round(time.time(), 3), "by": by, **extra}
        )


class CaseStore:
    def __init__(self, root: str):
        self.root = root
        self.cases: dict[str, Case] = {}
        os.makedirs(root, exist_ok=True)
        self._load_existing()

    def _load_existing(self) -> None:
        for case_id in sorted(os.listdir(self.root)):
            workdir = os.path.join(self.root, case_id)
            meta_path = os.path.join(workdir, "case.json")

            if not os.path.isfile(meta_path):
                continue

            with open(meta_path) as handle:
                meta = json.load(handle)

            case = Case(case_id, workdir, meta["filename"], meta.get("options", {}))

            if not os.path.isfile(case.evidence_path):
                # The upload was cut off (the file is still `.part`, or was never
                # written): the sender is gone, so there is nothing to resume.
                shutil.rmtree(workdir, ignore_errors=True)
                continue

            case.created = meta.get("created", case.created)
            case.owner = meta.get("owner")

            if case.owner is None and case.options.get("uploaded_by"):
                # Uploaded before names were recorded: the owner is known, not named.
                case.owner = {
                    "id": auth.public_id(case.options["uploaded_by"]),
                    "name": None,
                    "picture": None,
                }

            case.history = meta.get("history", [])
            events_path = os.path.join(workdir, "events.jsonl")

            if os.path.isfile(events_path):
                with open(events_path) as handle:
                    case.events = [json.loads(line) for line in handle if line.strip()]

            done = any(e["type"] == "done" for e in case.events)
            case.status = "done" if done else "interrupted"
            self.cases[case_id] = case

    def create(self, filename: str, options: dict[str, Any] | None = None) -> Case:
        case_id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        workdir = os.path.join(self.root, case_id)
        os.makedirs(workdir)
        case = Case(case_id, workdir, os.path.basename(filename), options or {})
        self.save_meta(case)
        self.cases[case_id] = case

        return case

    def save_meta(self, case: Case) -> None:
        """Write case.json (filename, creation time, options) for restarts."""
        with open(os.path.join(case.workdir, "case.json"), "w") as handle:
            json.dump(
                {
                    "filename": case.filename,
                    "created": case.created,
                    "options": case.options,
                    "owner": case.owner,
                    "history": case.history,
                },
                handle,
            )

    def delete(self, case_id: str) -> bool:
        case = self.cases.pop(case_id, None)

        if case is None:
            return False

        if case.task and not case.task.done():
            case.task.cancel()

        shutil.rmtree(case.workdir, ignore_errors=True)

        return True

    async def sync_index(self) -> None:
        """Backfill the database index with every case on disk."""
        for case in list(self.cases.values()):
            await db.index_case(case)

    def start(self, case: Case, *, only: list[str] | None = None) -> None:
        case.task = asyncio.create_task(run_case(case, only=only))

    def restart(self, case: Case, *, only: list[str] | None = None) -> None:
        """Forget a case's events and run its analysis again from the start."""
        case.events.clear()

        for queue in list(case.listeners):
            queue.put_nowait({"type": "reset", "seq": -1})

        if case._log is not None:
            case._log.close()
            case._log = None

        events = os.path.join(case.workdir, "events.jsonl")

        if os.path.isfile(events):
            os.remove(events)

        case.status = "queued"
        self.start(case, only=only)

    async def resume_interrupted(self) -> None:
        """Rerun, one at a time, every case whose analysis a restart cut short.

        Module results live in memory while a case runs, so a case cannot pick
        up where it stopped; it starts over. AI answers are cached in the case
        folder, so a rerun does not pay for them twice.
        """
        waiting = [c for c in self.cases.values() if c.status == "interrupted"]

        for case in waiting:
            case.status = "queued"

        for case in waiting:
            # Deleted, or rerun by hand, while it waited its turn.
            if self.cases.get(case.id) is not case or case.task is not None:
                continue

            case.note = "The server restarted during the analysis, so it was run again from the start."
            self.restart(case)

            # wait() neither raises the case's error nor its cancellation (a
            # delete), so one broken case does not stop the rest.
            await asyncio.wait([case.task])


def _image_proxy(case: Case, ctx: CaseContext) -> str | None:
    """A browser-viewable, upright JPEG of the photo (HEIC/TIFF/16-bit need one) plus a thumbnail."""
    from earwitness import photo

    image = ctx.image()

    if image is None:
        return None

    out = os.path.join(case.workdir, "proxy.jpg")
    photo.save_jpeg(out, photo.fit(image, 3000), 92)
    photo.save_jpeg(
        os.path.join(case.workdir, "thumb.jpg"), photo.thumbnail_of(image, 480), 85
    )

    return out


async def _make_proxy(case: Case) -> str | None:
    """An H.264 copy every browser can play (phones record HEVC/VP9)."""
    out = os.path.join(case.workdir, "proxy.mp4")

    if os.path.isfile(out):
        return out

    result = await run_tool(
        "ffmpeg", "-nostdin", "-v", "error", "-y", "-i", case.evidence_path,
        "-map", "0:v:0?", "-map", "0:a:0?",
        "-vf", "scale='min(1280,iw)':-2",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", out,
    )  # fmt: skip

    return out if result is not None and os.path.isfile(out) else None


async def _thumbnail(case: Case) -> None:
    await run_tool(
        "ffmpeg", "-nostdin", "-v", "error", "-y", "-ss", "0.5", "-i", case.evidence_path,
        "-frames:v", "1", "-vf", "scale=320:-2,format=yuvj420p", os.path.join(case.workdir, "thumb.jpg"),
    )  # fmt: skip


def _hash_file(path: str) -> tuple[str, str]:
    sha, md5 = hashlib.sha256(), hashlib.md5()

    with open(path, "rb") as handle:
        while chunk := handle.read(1 << 20):
            sha.update(chunk)
            md5.update(chunk)

    return sha.hexdigest(), md5.hexdigest()


def _fps(text: str | None) -> float | None:
    if not text or text in ("0/0", "N/A"):
        return None

    num, _, den = text.partition("/")

    return round(float(num) / float(den), 3) if den and float(den) else float(num)


def _num(value: Any) -> float | None:
    try:
        return float(value)
    except TypeError, ValueError:
        return None


async def _custody(case: "Case", probe: dict[str, Any]) -> dict[str, Any]:
    """Chain-of-custody facts: hashes, container, every stream, every tag."""
    sha256, md5 = await asyncio.to_thread(_hash_file, case.evidence_path)
    fmt = probe.get("format", {})
    tags: dict[str, str] = {}

    for key, value in fmt.get("tags", {}).items():
        tags[key] = str(value)

    for stream in probe.get("streams", []):
        for key, value in stream.get("tags", {}).items():
            tags.setdefault(f"stream{stream.get('index')}.{key}", str(value))

    streams = [
        {
            "index": s.get("index"),
            "type": s.get("codec_type"),
            "codec": s.get("codec_name"),
            "codec_long": s.get("codec_long_name"),
            "profile": s.get("profile"),
            "width": s.get("width"),
            "height": s.get("height"),
            "fps": _fps(s.get("avg_frame_rate"))
            if s.get("codec_type") == "video"
            else None,
            "pix_fmt": s.get("pix_fmt"),
            "color_range": s.get("color_range"),
            "color_space": s.get("color_space"),
            "bit_rate": _num(s.get("bit_rate")),
            "sample_rate": _num(s.get("sample_rate")),
            "channels": s.get("channels"),
            "channel_layout": s.get("channel_layout"),
            "language": s.get("tags", {}).get("language"),
            "handler": s.get("tags", {}).get("handler_name"),
            "duration_s": _num(s.get("duration")),
            "frames": _num(s.get("nb_frames")),
        }
        for s in probe.get("streams", [])
    ]

    container = {
        "format_name": fmt.get("format_name"),
        "format_long_name": fmt.get("format_long_name"),
        "bit_rate": _num(fmt.get("bit_rate")),
        "size_bytes": _num(fmt.get("size")),
        "duration_s": _num(fmt.get("duration")),
    }

    if not probe and case.ctx is not None and case.ctx.kind == "image":
        info = case.ctx.image_info()

        if info is not None:
            container = {
                "format_name": info.format.lower(),
                "format_long_name": {
                    "JPEG": "JPEG (Joint Photographic Experts Group)", "MPO": "JPEG multi-picture (MPO)",
                    "PNG": "PNG (Portable Network Graphics)", "HEIF": "HEIF / HEIC (ISO base media, HEVC-coded picture)",
                    "WEBP": "WebP", "TIFF": "TIFF", "GIF": "GIF", "BMP": "Windows bitmap", "AVIF": "AVIF",
                }.get(info.format, info.format),
                "bit_rate": None,
                "size_bytes": float(os.path.getsize(case.evidence_path)),
                "duration_s": None,
            }  # fmt: skip
            streams = [{
                "index": 0, "type": "image", "codec": info.format.lower(), "codec_long": container["format_long_name"],
                "profile": info.mode, "width": info.stored_width, "height": info.stored_height, "fps": None,
                "pix_fmt": f"{info.mode} {info.bits}-bit", "color_range": None, "color_space": None, "bit_rate": None,
                "sample_rate": None, "channels": None, "channel_layout": None, "language": None, "handler": None,
                "duration_s": None, "frames": info.frames,
            }]  # fmt: skip

    return {
        "sha256": sha256,
        "md5": md5,
        "ingested_at": case.created,
        "container": container,
        "streams": streams,
        "tags": tags,
    }


async def _index(case: Case) -> None:
    # The files are the record of the run; a database hiccup must not sink it.
    try:
        await db.index_case(case)
    except (psycopg.Error, OSError) as error:
        case.emit(
            {"type": "log", "module": "index",
             "message": f"database index not updated: {type(error).__name__}: {error}"}
        )  # fmt: skip


async def run_case(case: Case, *, only: list[str] | None = None) -> None:
    case.status = "running"
    # Mark the case running in the index without holding up the analysis.
    started_index = asyncio.create_task(_index(case))
    ctx = CaseContext(case.id, case.evidence_path, case.workdir, case.options)
    case.ctx = ctx
    kind = ctx.kind
    probe = await ctx.probe() if kind != "image" else {}
    info = await ctx.video_info()
    image_info = await asyncio.to_thread(ctx.image_info) if kind == "image" else None

    if kind == "image" and image_info is None:
        # Not decodable as a picture after all: treat it as a generic media file.
        kind = ctx.kind = "video"
        probe = await ctx.probe()
        info = await ctx.video_info()

    has_audio = any(s.get("codec_type") == "audio" for s in probe.get("streams", []))
    modules = [m for m in registry.all_modules(kind) if only is None or m.name in only]

    # Modules not scheduled in this run (or that exist only for another media
    # kind) count as finished with nothing to say, so anything waiting on them
    # carries on immediately.
    scheduled = {m.name for m in modules}

    for other_kind in registry.kinds():
        for other in registry.all_modules(other_kind):
            if other.name not in scheduled:
                ctx.publish(other.name, None)

    if kind == "image":
        proxy_task = asyncio.create_task(asyncio.to_thread(_image_proxy, case, ctx))
        await proxy_task
    else:
        proxy_task = asyncio.create_task(_make_proxy(case))
        await _thumbnail(case)

    case.emit(
        {
            "type": "case",
            "case": {
                "id": case.id,
                "filename": case.filename,
                "kind": kind,
                # The AI provider this run uses (None: no AI key on the server).
                "ai_provider": ai.choice(case.options),
                "duration_s": float(probe.get("format", {}).get("duration") or 0),
                "size_bytes": int(
                    probe.get("format", {}).get("size")
                    or os.path.getsize(case.evidence_path)
                ),
                "has_audio": has_audio,
                "video": None
                if info is None
                else {
                    "width": info.width,
                    "height": info.height,
                    "fps": round(info.fps, 3),
                    "codec": info.codec,
                    "frames": info.frame_count,
                },
                "image": None
                if image_info is None
                else {
                    "width": image_info.width,
                    "height": image_info.height,
                    "format": image_info.format,
                    "mode": image_info.mode,
                    "bits": image_info.bits,
                    "frames": image_info.frames,
                    "orientation": image_info.orientation,
                    "megapixels": round(image_info.width * image_info.height / 1e6, 1),
                },
                "thumb_url": f"/api/cases/{case.id}/files/thumb.jpg",
                "media_url": f"/api/cases/{case.id}/files/proxy.jpg"
                if kind == "image"
                else None,
                "original_url": f"/api/cases/{case.id}/files/{os.path.basename(case.evidence_path)}",
                **(await _custody(case, probe)),
                "owner": case.owner,
                "history": case.history,
                "modules": [
                    {"name": m.name, "title": m.title, "description": m.description}
                    for m in modules
                ],
            },
        }
    )

    if case.note:
        case.emit({"type": "log", "module": "core", "message": case.note})
        case.note = None

    for module in modules:
        case.emit(
            {"type": "module", "module": module.name, "status": "queued", "progress": 0}
        )

    # Heavy modules are numpy/OpenCV bound; a small limit keeps the UI lively.
    gate = asyncio.Semaphore(int(os.environ.get("EARWITNESS_PARALLEL", "4")))

    async def one(module: Any) -> None:
        reporter = Reporter(module.name, ctx, case.emit)
        depends = list(getattr(module, "depends", ()))

        if depends == ["*"]:
            depends = [m.name for m in modules if m.name != module.name]

        if depends:
            case.emit(
                {"type": "module", "module": module.name, "status": "waiting",
                 "progress": 0, "message": "waiting for " + ", ".join(depends)}
            )  # fmt: skip

        for dependency in depends:
            if any(m.name == dependency for m in modules):
                await ctx.wait_for(dependency)

        started = time.time()

        async with gate:
            case.emit(
                {
                    "type": "module",
                    "module": module.name,
                    "status": "running",
                    "progress": 0,
                }
            )

            try:
                await module.run(ctx, reporter)
            except Skip as reason:
                ctx.publish(module.name, None)
                case.emit(
                    {"type": "module", "module": module.name, "status": "skipped",
                     "progress": 1, "message": str(reason)}
                )  # fmt: skip

                return
            except asyncio.CancelledError:
                raise
            except Exception as error:  # noqa: BLE001 - one module must not sink the case
                ctx.publish(module.name, None)
                case.emit(
                    {"type": "module", "module": module.name, "status": "error",
                     "progress": 1, "message": f"{type(error).__name__}: {error}",
                     "trace": traceback.format_exc()[-2000:]}
                )  # fmt: skip

                return

        if module.name not in ctx._published:
            ctx.publish(module.name, None)

        case.emit(
            {"type": "module", "module": module.name, "status": "done", "progress": 1,
             "elapsed_s": round(time.time() - started, 2)}
        )  # fmt: skip

    async def proxy() -> None:
        path = await proxy_task
        name = "proxy.jpg" if kind == "image" else "proxy.mp4"
        case.emit(
            {
                "type": "media",
                "media_url": f"/api/cases/{case.id}/files/{name}" if path else None,
            }
        )

    await asyncio.gather(proxy(), *(one(m) for m in modules))
    case.status = "done"
    await started_index
    await _index(case)
    case.emit({"type": "done"})
