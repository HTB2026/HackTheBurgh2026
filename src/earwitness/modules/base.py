"""Shared machinery for analysis modules.

A module is an object with a ``name``, a ``title`` and an async ``run(ctx, out)``.
It reports through ``out`` (a Reporter): progress, findings and artifacts. The
case runner turns those calls into JSON events that the web UI streams.
"""

import asyncio
import itertools
import json
import os
import subprocess
import time
from collections.abc import Awaitable
from collections.abc import Callable
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Literal
from typing import Protocol

import numpy as np

from earwitness import media
from earwitness._common import ForensicError

Severity = Literal["info", "notice", "warning", "critical"]
Kind = Literal["measured", "inferred"]

_finding_ids = itertools.count(1)


@dataclass(slots=True)
class Finding:
    module: str
    severity: Severity
    title: str
    detail: str
    time_s: float | None = None
    time_end_s: float | None = None
    kind: Kind = "measured"
    tags: list[str] = field(default_factory=list)
    data: dict[str, Any] = field(default_factory=dict)
    id: str = ""


@dataclass(slots=True)
class Artifact:
    module: str
    name: str
    kind: str
    title: str
    url: str | None = None
    data: Any = None
    time_s: float | None = None
    id: str = ""


class Emit(Protocol):
    def __call__(self, event: dict[str, Any]) -> None: ...


class Reporter:
    """What a module uses to talk to the outside world."""

    def __init__(self, module: str, case: "CaseContext", emit: Emit) -> None:
        self._module = module
        self._case = case
        self._emit = emit
        self._last_progress = -1.0

    def progress(self, fraction: float, message: str = "") -> None:
        fraction = float(min(1.0, max(0.0, fraction)))

        if fraction - self._last_progress < 0.01 and not message:
            return

        self._last_progress = fraction
        self._emit(
            {
                "type": "module",
                "module": self._module,
                "status": "running",
                "progress": round(fraction, 3),
                "message": message,
            }
        )

    def log(self, message: str) -> None:
        self._emit({"type": "log", "module": self._module, "message": message})

    def finding(
        self,
        severity: Severity,
        title: str,
        detail: str,
        **extra: Any,
    ) -> Finding:
        finding = Finding(self._module, severity, title, detail, **extra)
        finding.id = f"F{next(_finding_ids):03d}"
        finding.data = _jsonable(finding.data)
        self._case.findings.append(finding)
        self._emit({"type": "finding", "finding": asdict(finding)})

        return finding

    def artifact(
        self,
        name: str,
        kind: str,
        title: str,
        *,
        data: Any = None,
        file: str | None = None,
        time_s: float | None = None,
    ) -> Artifact:
        url = None

        if file is not None:
            url = f"/api/cases/{self._case.id}/files/{os.path.relpath(file, self._case.workdir)}"

        artifact = Artifact(
            self._module,
            name,
            kind,
            title,
            url=url,
            data=_jsonable(data),
            time_s=time_s,
            id=f"{self._module}:{name}",
        )
        self._case.artifacts[artifact.id] = artifact
        self._emit({"type": "artifact", "artifact": asdict(artifact)})

        return artifact

    def path(self, name: str) -> str:
        directory = os.path.join(self._case.workdir, self._module)
        os.makedirs(directory, exist_ok=True)

        return os.path.join(directory, name)


class Module(Protocol):
    name: str
    title: str
    description: str

    async def run(self, ctx: "CaseContext", out: Reporter) -> None: ...


class Skip(Exception):
    """Raised by a module that has nothing to work on (e.g. no audio)."""


@dataclass(frozen=True, slots=True)
class VideoInfo:
    width: int
    height: int
    fps: float
    frame_count: int
    duration_s: float
    codec: str
    rotation: int


@dataclass(frozen=True, slots=True)
class Frames:
    """Decoded frames at reduced size, RGB uint8, shape (n, h, w, 3)."""

    pixels: np.ndarray
    times_s: np.ndarray
    fps: float


class CaseContext:
    """Everything modules share about one piece of evidence.

    Expensive decodes are done once and cached; modules can also publish
    intermediate results for others with ``publish`` / ``wait_for``.
    """

    def __init__(self, case_id: str, path: str, workdir: str, options: dict[str, Any]):
        self.id = case_id
        self.path = path
        self.workdir = workdir
        self.options = options
        self.findings: list[Finding] = []
        self.artifacts: dict[str, Artifact] = {}
        self._cache: dict[str, Any] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._published: dict[str, Any] = {}
        self._events: dict[str, asyncio.Event] = {}

    async def _cached[T](self, key: str, make: Callable[[], Awaitable[T]]) -> T:
        lock = self._locks.setdefault(key, asyncio.Lock())

        async with lock:
            if key not in self._cache:
                self._cache[key] = await make()

        return self._cache[key]

    # ---- shared results between modules ---------------------------------
    def publish(self, key: str, value: Any) -> None:
        self._published[key] = value
        self._events.setdefault(key, asyncio.Event()).set()

    async def wait_for(self, key: str, timeout: float = 600.0) -> Any:
        event = self._events.setdefault(key, asyncio.Event())

        try:
            await asyncio.wait_for(event.wait(), timeout)
        except TimeoutError:
            return None

        return self._published.get(key)

    # ---- probing and decoding -------------------------------------------
    async def probe(self) -> dict[str, Any]:
        async def make() -> dict[str, Any]:
            output = await run_tool(
                "ffprobe",
                "-v",
                "error",
                "-print_format",
                "json",
                "-show_format",
                "-show_streams",
                self.path,
            )

            return json.loads(output) if output else {}

        return await self._cached("probe", make)

    async def video_info(self) -> VideoInfo | None:
        async def make() -> VideoInfo | None:
            probe = await self.probe()
            stream = next(
                (
                    s
                    for s in probe.get("streams", [])
                    if s.get("codec_type") == "video"
                    and not s.get("disposition", {}).get("attached_pic")
                ),
                None,
            )

            if stream is None:
                return None

            fps = _rate(stream.get("avg_frame_rate")) or _rate(
                stream.get("r_frame_rate")
            )
            duration = float(
                stream.get("duration") or probe.get("format", {}).get("duration") or 0
            )
            frames = int(stream.get("nb_frames") or round(duration * (fps or 30)))
            rotation = 0

            for side in stream.get("side_data_list", []):
                if "rotation" in side:
                    rotation = int(side["rotation"])

            if "rotate" in stream.get("tags", {}):
                rotation = int(stream["tags"]["rotate"])

            width, height = int(stream["width"]), int(stream["height"])

            if abs(rotation) % 180 == 90:
                width, height = height, width

            return VideoInfo(
                width=width,
                height=height,
                fps=fps or 30.0,
                frame_count=frames,
                duration_s=duration,
                codec=stream.get("codec_name", "?"),
                rotation=rotation,
            )

        return await self._cached("video_info", make)

    async def audio(self) -> media.Recording | None:
        async def make() -> media.Recording | None:
            recording = await media.load(self.path)

            return None if isinstance(recording, ForensicError) else recording

        return await self._cached("audio", make)

    async def audio_native(self) -> tuple[np.ndarray, int] | None:
        """Mono audio at the file's own sample rate (for high-frequency work)."""

        async def make() -> tuple[np.ndarray, int] | None:
            probe = await self.probe()
            stream = next(
                (s for s in probe.get("streams", []) if s.get("codec_type") == "audio"),
                None,
            )

            if stream is None:
                return None

            rate = int(stream.get("sample_rate") or 44_100)
            output = await run_tool(
                "ffmpeg", "-nostdin", "-v", "error", "-i", self.path, "-vn",
                "-ac", "1", "-ar", str(rate), "-f", "f32le", "-",
            )  # fmt: skip

            if not output:
                return None

            return np.frombuffer(output, dtype=np.float32).astype(np.float64), rate

        return await self._cached("audio_native", make)

    async def frames(self, width: int, *, max_frames: int = 9_000) -> Frames | None:
        """Every frame (up to max_frames) scaled to ``width`` pixels wide."""

        async def make() -> Frames | None:
            info = await self.video_info()

            if info is None:
                return None

            height = max(2, int(round(width * info.height / info.width / 2)) * 2)
            args = ["ffmpeg", "-nostdin", "-v", "error", "-i", self.path]
            args += ["-frames:v", str(max_frames), "-vf", f"scale={width}:{height}"]
            args += [
                "-fps_mode",
                "passthrough",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-",
            ]
            output = await run_tool(*args)

            if not output:
                return None

            pixels = np.frombuffer(output, dtype=np.uint8)
            count = len(pixels) // (width * height * 3)
            pixels = pixels[: count * width * height * 3].reshape(
                count, height, width, 3
            )
            times = await self.frame_times()
            times = (
                times[:count] if len(times) >= count else np.arange(count) / info.fps
            )

            return Frames(pixels, times, info.fps)

        return await self._cached(f"frames:{width}:{max_frames}", make)

    async def frame_table(self) -> list[dict[str, Any]]:
        """Per-frame packet facts: pts, type, size, key flag."""

        async def make() -> list[dict[str, Any]]:
            output = await run_tool(
                "ffprobe", "-v", "error", "-select_streams", "v:0",
                "-show_entries",
                "packet=pts_time,dts_time,size,flags,duration_time",
                "-of", "json", self.path,
            )  # fmt: skip

            if not output:
                return []

            packets = json.loads(output).get("packets", [])
            rows = []

            for packet in packets:
                if packet.get("pts_time") in (None, "N/A"):
                    continue

                rows.append(
                    {
                        "pts": float(packet["pts_time"]),
                        "size": int(packet.get("size", 0)),
                        "key": "K" in packet.get("flags", ""),
                    }
                )

            rows.sort(key=lambda r: r["pts"])

            return rows

        return await self._cached("frame_table", make)

    async def frame_times(self) -> np.ndarray:
        table = await self.frame_table()

        if not table:
            return np.array([])

        times = np.array([row["pts"] for row in table])

        return times - times[0]

    async def frame_at(
        self, time_s: float, *, width: int | None = None
    ) -> np.ndarray | None:
        """One full-resolution (or scaled) RGB frame at ``time_s``."""
        info = await self.video_info()

        if info is None:
            return None

        out_width = width or info.width
        out_height = max(2, int(round(out_width * info.height / info.width / 2)) * 2)
        output = await run_tool(
            "ffmpeg", "-nostdin", "-v", "error", "-ss", f"{max(0.0, time_s):.3f}",
            "-i", self.path, "-frames:v", "1",
            "-vf", f"scale={out_width}:{out_height}",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
        )  # fmt: skip

        if not output or len(output) < out_width * out_height * 3:
            return None

        return np.frombuffer(output, dtype=np.uint8)[
            : out_width * out_height * 3
        ].reshape(out_height, out_width, 3)

    async def clip_frames(
        self, start_s: float, count: int, *, width: int | None = None
    ) -> np.ndarray | None:
        """``count`` consecutive RGB frames starting at ``start_s``."""
        info = await self.video_info()

        if info is None:
            return None

        out_width = width or info.width
        out_height = max(2, int(round(out_width * info.height / info.width / 2)) * 2)
        output = await run_tool(
            "ffmpeg", "-nostdin", "-v", "error", "-ss", f"{max(0.0, start_s):.3f}",
            "-i", self.path, "-frames:v", str(count),
            "-vf", f"scale={out_width}:{out_height}",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
        )  # fmt: skip

        if not output:
            return None

        size = out_width * out_height * 3
        n = len(output) // size

        return np.frombuffer(output[: n * size], dtype=np.uint8).reshape(
            n, out_height, out_width, 3
        )


def _rate(text: str | None) -> float | None:
    if not text or text in ("0/0", "N/A"):
        return None

    if "/" in text:
        numerator, denominator = text.split("/")

        return float(numerator) / float(denominator) if float(denominator) else None

    return float(text)


async def run_tool(*args: str, stdin: bytes | None = None) -> bytes | None:
    process = await asyncio.create_subprocess_exec(
        *args,
        stdin=subprocess.PIPE if stdin is not None else subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdout, _ = await process.communicate(stdin)

    if process.returncode != 0:
        return None

    return stdout


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]

    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())

    if isinstance(value, (np.floating, float)):
        f = float(value)

        return None if not np.isfinite(f) else round(f, 6)

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.bool_):
        return bool(value)

    if hasattr(value, "isoformat"):
        return value.isoformat()

    return value


def downsample_series(
    x: np.ndarray, ys: dict[str, np.ndarray], max_points: int = 2_000
) -> dict[str, Any]:
    """A compact {x, series} payload for the UI's charts."""
    x = np.asarray(x, dtype=float)
    step = max(1, int(np.ceil(len(x) / max_points)))

    return {
        "x": np.round(x[::step], 4),
        "series": [
            {"name": name, "y": np.round(np.asarray(y, dtype=float)[::step], 6)}
            for name, y in ys.items()
        ],
    }


def now() -> float:
    return time.time()
