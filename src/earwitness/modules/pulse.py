"""Heartbeats in the pixels (remote photoplethysmography).

Each heartbeat pushes blood into the skin and changes its colour by a fraction
of a percent: invisible to the eye, but measurable from the average colour of
a face over time. This module finds faces, tracks them, extracts the pulse
with the POS method (Wang et al., 2017) and renders an Eulerian colour
magnification so the pulse becomes visible.

The same face tracks also answer "who appears, and when": every face followed
for at least a second is listed with its time on screen. Faces are only
followed from frame to frame, never identified or compared with other cases.
"""

import asyncio
import os
import subprocess
from typing import Any

import cv2
import httpx
import numpy as np
from scipy import signal

from earwitness.modules.base import CaseContext
from earwitness.modules.base import Reporter
from earwitness.modules.base import Skip
from earwitness.modules.base import downsample_series

_MODEL_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
_MODEL = os.path.expanduser("~/.cache/earwitness/face_detection_yunet_2023mar.onnx")
_DETECT_WIDTH = 640
_ANALYSE_WIDTH = 480
_MAX_SECONDS = 120.0
_MIN_SECONDS = 6.0
_BPM_LOW, _BPM_HIGH = 42.0, 200.0


async def _model() -> str | None:
    if os.path.isfile(_MODEL) and os.path.getsize(_MODEL) > 100_000:
        return _MODEL

    os.makedirs(os.path.dirname(_MODEL), exist_ok=True)

    try:
        async with httpx.AsyncClient(timeout=60, follow_redirects=True) as client:
            response = await client.get(_MODEL_URL)
            response.raise_for_status()

        with open(_MODEL, "wb") as handle:
            handle.write(response.content)
    except httpx.HTTPError:
        return None

    return _MODEL


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[0] + a[2], b[0] + b[2]), min(a[1] + a[3], b[1] + b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)

    return inter / (a[2] * a[3] + b[2] * b[3] - inter + 1e-9)


def _read_frames(path: str, width: int, height: int, limit: int):
    process = subprocess.Popen(
        ["ffmpeg", "-nostdin", "-v", "error", "-i", path, "-frames:v", str(limit),
         "-vf", f"scale={width}:{height}", "-fps_mode", "passthrough",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        stdout=subprocess.PIPE,
    )  # fmt: skip
    size = width * height * 3

    try:
        while True:
            chunk = process.stdout.read(size)

            if len(chunk) < size:
                break

            yield np.frombuffer(chunk, dtype=np.uint8).reshape(height, width, 3)
    finally:
        process.stdout.close()
        process.wait()


def _skin_fallback(frame: np.ndarray) -> np.ndarray | None:
    """Largest skin-coloured blob, for footage with skin but no detectable face."""
    ycc = cv2.cvtColor(frame, cv2.COLOR_RGB2YCrCb)
    mask = cv2.inRange(ycc, (40, 135, 80), (240, 175, 125))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask)

    if count <= 1:
        return None

    index = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, w, h, area = stats[index]

    if area < 0.01 * frame.shape[0] * frame.shape[1]:
        return None

    return np.array([x, y, w, h], dtype=float)


def _track_faces(
    model: str, path: str, width: int, height: int, fps: float, limit: int
):
    detector = cv2.FaceDetectorYN.create(model, "", (width, height), 0.7)
    step = max(1, int(round(fps / 3)))  # detect 3x a second
    tracks: list[dict[str, Any]] = []
    first_frame = None

    for index, frame in enumerate(_read_frames(path, width, height, limit)):
        if first_frame is None:
            first_frame = frame

        if index % step:
            continue

        _, faces = detector.detect(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        for face in [] if faces is None else faces:
            box = face[:4].astype(float)
            best = max(tracks, key=lambda t: _iou(t["boxes"][-1], box), default=None)

            if (
                best is not None
                and _iou(best["boxes"][-1], box) > 0.3
                and best["frames"][-1] < index
            ):
                best["frames"].append(index)
                best["boxes"].append(box)
                track = best
            else:
                track = {"frames": [index], "boxes": [box]}
                tracks.append(track)

            area = box[2] * box[3]

            if area > track.get("crop_area", 0):
                x, y, w, h = box
                pad = 0.25 * w
                crop = frame[
                    int(max(0, y - pad)) : int(y + h + pad),
                    int(max(0, x - pad)) : int(x + w + pad),
                ]

                if crop.size:
                    track["crop_area"] = area
                    track["crop"] = crop.copy()
                    track["crop_frame"] = index

    return tracks, first_frame


def _appearances(tracks: list[dict[str, Any]], fps: float) -> list[dict[str, Any]]:
    """Face tracks -> face appearances on screen with their time intervals.

    A track broken by a turned head or a missed detection is rejoined when the next
    track starts within 2 s, near where it ended, at a similar size. Tracks seen for
    under a second (3 detections) are dropped as detector noise.
    """
    # The tracker rejoins a box seen in the same place any time later (after a cut it may be
    # someone else), so split tracks at gaps over 2 s first; only short gaps are bridged below.
    pieces = []

    for t in tracks:
        start = 0

        for i in range(1, len(t["frames"]) + 1):
            if i == len(t["frames"]) or t["frames"][i] - t["frames"][i - 1] > 2 * fps:
                piece = {"frames": t["frames"][start:i], "boxes": t["boxes"][start:i]}

                if t.get("crop_frame") in piece["frames"]:
                    piece["crop"], piece["crop_area"] = (
                        t.get("crop"),
                        t.get("crop_area", 0),
                    )

                pieces.append(piece)
                start = i

    ordered = sorted((t for t in pieces if t["frames"]), key=lambda t: t["frames"][0])
    merged: list[dict[str, Any]] = []

    for t in ordered:
        x, y, w, h = t["boxes"][0]
        join = None

        for m in merged:
            mx, my, mw, mh = m["boxes"][-1]
            gap = t["frames"][0] - m["frames"][-1]

            if (
                0 < gap <= 2 * fps
                and np.hypot(x + w / 2 - mx - mw / 2, y + h / 2 - my - mh / 2)
                <= 1.5 * max(w, mw)
                and 0.6 <= w / max(mw, 1e-6) <= 1.6
            ):
                join = m
                break

        if join is None:
            merged.append(
                {
                    "frames": list(t["frames"]),
                    "boxes": list(t["boxes"]),
                    "crop": t.get("crop"),
                    "crop_area": t.get("crop_area", 0),
                }
            )
        else:
            join["frames"] += t["frames"]
            join["boxes"] += t["boxes"]

            if t.get("crop_area", 0) > join["crop_area"]:
                join["crop"], join["crop_area"] = t.get("crop"), t["crop_area"]

    step = max(1, int(round(fps / 3)))
    people = []

    for m in merged:
        if len(m["frames"]) < 3:
            continue

        intervals: list[list[float]] = []

        for f in m["frames"]:
            t = f / fps

            if intervals and t - intervals[-1][1] <= 1.0 + step / fps:
                intervals[-1][1] = t
            else:
                intervals.append([t, t])

        for iv in intervals:
            iv[1] += step / fps  # a detection covers the frames until the next one

        people.append({
            "first_s": intervals[0][0],
            "last_s": intervals[-1][1],
            "seconds": sum(b - a for a, b in intervals),
            "intervals": intervals,
            "width_px": float(np.median([b[2] for b in m["boxes"]])),
            "crop": m["crop"],
        })  # fmt: skip

    people.sort(key=lambda p: p["first_s"])

    for i, p in enumerate(people):
        p["index"] = i + 1

    return people


def _clock(t: float) -> str:
    return f"{int(t // 60)}:{t % 60:04.1f}"


def _pos(rgb: np.ndarray, fps: float) -> np.ndarray:
    n = len(rgb)
    window = int(1.6 * fps)
    h = np.zeros(n)
    projection = np.array([[0, 1, -1], [-2, 1, 1]], dtype=float)

    for start in range(0, n - window + 1):
        block = rgb[start : start + window]
        normalised = block / (block.mean(axis=0) + 1e-9)
        s = projection @ normalised.T
        pulse = s[0] + (s[0].std() / (s[1].std() + 1e-9)) * s[1]
        h[start : start + window] += pulse - pulse.mean()

    return h


def _heart_rate(h: np.ndarray, fps: float) -> dict[str, Any]:
    sos = signal.butter(
        3, [_BPM_LOW / 60, _BPM_HIGH / 60], btype="band", fs=fps, output="sos"
    )
    filtered = signal.sosfiltfilt(sos, h)
    nperseg = min(len(filtered), int(fps * 12))
    freqs, power = signal.welch(
        filtered, fs=fps, nperseg=nperseg, nfft=max(4096, nperseg)
    )
    band = (freqs >= _BPM_LOW / 60) & (freqs <= _BPM_HIGH / 60)
    peak = freqs[band][np.argmax(power[band])]
    near = (np.abs(freqs - peak) <= 0.1) | (np.abs(freqs - 2 * peak) <= 0.2)
    snr = 10 * np.log10(power[band & near].sum() / (power[band & ~near].sum() + 1e-12))
    # Heart rate over time, from 8 s windows.
    times, rates = [], []
    window = int(8 * fps)

    for start in range(0, len(filtered) - window + 1, int(fps)):
        chunk = filtered[start : start + window]
        f, p = signal.periodogram(chunk * np.hanning(len(chunk)), fs=fps, nfft=8192)
        b = (f >= _BPM_LOW / 60) & (f <= _BPM_HIGH / 60)
        times.append((start + window / 2) / fps)
        rates.append(60 * f[b][np.argmax(p[b])])

    return {
        "bpm": float(peak * 60),
        "snr_db": float(snr),
        "waveform": filtered,
        "spectrum": (freqs[band] * 60, power[band]),
        "over_time": (np.array(times), np.array(rates)),
    }


def _magnify(
    path: str,
    track: dict[str, Any],
    scale: float,
    start: int,
    count: int,
    fps: float,
    bpm: float,
    out_path: str,
    src_size: tuple[int, int],
) -> bool:
    """Eulerian colour magnification of a face-following crop.

    Only chroma is amplified (brightness changes are mostly motion), and the gain
    is set so the pulse becomes a clearly visible but not garish tint.
    """
    width, height = src_size
    frames_idx = np.array(track["frames"], dtype=float)
    boxes = np.array(track["boxes"], dtype=float) / scale
    side = int(np.median(np.maximum(boxes[:, 2], boxes[:, 3])) * 1.8) // 2 * 2
    side = min(side, width, height)

    if side < 32:
        return False

    out = 320
    frames = []

    for index, frame in enumerate(_read_frames(path, width, height, start + count)):
        if index < start:
            continue

        x, y, w, h = (np.interp(index, frames_idx, boxes[:, k]) for k in range(4))
        x0 = int(np.clip(x + w / 2 - side / 2, 0, width - side))
        y0 = int(np.clip(y + h / 2 - side / 2, 0, height - side))
        crop = frame[y0 : y0 + side, x0 : x0 + side]
        frames.append(cv2.resize(crop, (out, out), interpolation=cv2.INTER_AREA))

    if len(frames) < fps * 3:
        return False

    video = np.stack(frames)
    ycc = np.stack([cv2.cvtColor(f, cv2.COLOR_RGB2YCrCb) for f in video]).astype(
        np.float32
    )
    small = np.stack([cv2.pyrDown(cv2.pyrDown(cv2.pyrDown(f))) for f in ycc])
    spectrum = np.fft.fft(small, axis=0)
    freqs = np.fft.fftfreq(len(small), 1 / fps)
    spectrum[np.abs(np.abs(freqs) - bpm / 60) > 0.2] = 0
    band = np.real(np.fft.ifft(spectrum, axis=0)).astype(np.float32)
    band[..., 0] = 0  # leave brightness alone
    gain = 6.0 / (float(np.percentile(np.abs(band[..., 1:]), 95)) + 1e-6)
    gain = float(np.clip(gain, 10, 150))
    amplified = []

    for original, colour, pulse in zip(video, ycc, band):
        up = pulse

        for _ in range(3):
            up = cv2.pyrUp(up)

        up = cv2.resize(up, (out, out))
        magnified = cv2.cvtColor(
            np.clip(colour + gain * up, 0, 255).astype(np.uint8), cv2.COLOR_YCrCb2RGB
        )
        amplified.append(np.concatenate([original, magnified], axis=1))

    data = np.stack(amplified)
    process = subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{out * 2}x{out}", "-r", f"{fps:.3f}", "-i", "-",
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
         "-movflags", "+faststart", out_path],
        input=data.tobytes(), check=False,
    )  # fmt: skip

    return process.returncode == 0


class Pulse:
    name = "pulse"
    title = "PULSE (rPPG)"
    description = "Heart rate of people on camera from invisible skin-colour changes; Eulerian magnification."
    # Slow and rarely decisive: left out of Quick runs.
    min_effort = "medium"

    async def run(self, ctx: CaseContext, out: Reporter) -> None:
        info = await ctx.video_info()

        if info is None:
            raise Skip("no video stream")

        model = await _model()

        if model is None:
            raise Skip("face model unavailable (offline?)")

        fps = info.fps
        limit = int(min(info.duration_s, _MAX_SECONDS) * fps)
        width = min(_DETECT_WIDTH, info.width)
        height = int(round(width * info.height / info.width / 2)) * 2
        out.progress(0.1, "detecting and tracking faces")
        tracks, first = await asyncio.to_thread(
            _track_faces, model, ctx.path, width, height, fps, limit
        )
        self._report_people(out, _appearances(tracks, fps), info, limit / fps)

        if info.duration_s < _MIN_SECONDS:
            out.log(
                f"clip shorter than {_MIN_SECONDS:.0f} s: too short to read a pulse"
            )
            ctx.publish(self.name, {"faces": 0})

            return

        tracks = [
            t
            for t in tracks
            if len(t["frames"]) * (fps / 3) >= _MIN_SECONDS * fps * 0.6
        ]
        tracks.sort(
            key=lambda t: (
                -len(t["frames"]) * np.median([b[2] * b[3] for b in t["boxes"]])
            )
        )
        source = "face"

        if not tracks and first is not None:
            box = _skin_fallback(first)

            if box is not None:
                tracks = [{"frames": [0, limit - 1], "boxes": [box, box]}]
                source = "skin region"

        if not tracks:
            out.finding(
                "info",
                "No faces to measure",
                "No face is visible long enough (6 s or more) to read a pulse from.",
                tags=["pulse", "face"],
            )
            ctx.publish(self.name, {"faces": 0})

            return

        tracks = tracks[:3]
        out.progress(0.35, f"sampling skin colour on {len(tracks)} {source}(s)")
        a_w = _ANALYSE_WIDTH if info.width >= _ANALYSE_WIDTH else info.width
        a_h = int(round(a_w * info.height / info.width / 2)) * 2

        def sample() -> list[tuple[np.ndarray, np.ndarray]]:
            per_track = []

            for track in tracks:
                frames = np.array(track["frames"])
                boxes = np.array(track["boxes"]) * (a_w / width)
                per_track.append((frames, boxes, [], []))

            for index, frame in enumerate(_read_frames(ctx.path, a_w, a_h, limit)):
                ycc = None

                for frames, boxes, values, indices in per_track:
                    if index < frames[0] or index > frames[-1]:
                        continue

                    box = np.array(
                        [np.interp(index, frames, boxes[:, k]) for k in range(4)]
                    )
                    x, y, w, h = box
                    # Forehead and cheeks: the central 60% width, upper 75% height.
                    x0, x1 = int(x + 0.2 * w), int(x + 0.8 * w)
                    y0, y1 = int(y + 0.05 * h), int(y + 0.8 * h)
                    roi = frame[max(0, y0) : y1, max(0, x0) : x1]

                    if roi.size == 0:
                        continue

                    if ycc is None:
                        ycc = cv2.cvtColor(frame, cv2.COLOR_RGB2YCrCb)

                    mask = (
                        cv2.inRange(
                            ycc[max(0, y0) : y1, max(0, x0) : x1],
                            (30, 133, 77),
                            (250, 178, 130),
                        )
                        > 0
                    )
                    pixels = (
                        roi[mask]
                        if mask.sum() > 0.2 * mask.size
                        else roi.reshape(-1, 3)
                    )
                    values.append(pixels.mean(axis=0))
                    indices.append(index)

            return [(np.array(i), np.array(v)) for _, _, v, i in per_track]

        samples = await asyncio.to_thread(sample)
        out.progress(0.6, "extracting pulse (POS)")
        published = {"faces": len(tracks), "people": []}
        best = None

        for number, ((indices, rgb), track) in enumerate(zip(samples, tracks), start=1):
            if len(rgb) < _MIN_SECONDS * fps:
                continue

            h = _pos(rgb.astype(float), fps)
            result = await asyncio.to_thread(_heart_rate, h, fps)
            t0 = indices[0] / fps
            label = f"{source} {number}"
            reliable = result["snr_db"] > 0.0
            published["people"].append(
                {"label": label, "bpm": result["bpm"], "snr_db": result["snr_db"]}
            )
            out.artifact(
                f"waveform_{number}",
                "series",
                f"Pulse signal, {label}",
                data=downsample_series(
                    t0 + np.arange(len(result["waveform"])) / fps,
                    {"pos": result["waveform"]},
                )
                | {"x_unit": "s"},
            )
            out.artifact(
                f"bpm_{number}",
                "series",
                f"Heart rate over time, {label} (bpm)",
                data=downsample_series(
                    t0 + result["over_time"][0], {"bpm": result["over_time"][1]}
                )
                | {"x_unit": "s"},
            )
            out.finding(
                "notice" if reliable else "info",
                f"{'Pulse' if reliable else 'Weak pulse'} {result['bpm']:.0f} bpm, {label}"
                if reliable
                else f"Possible pulse ~{result['bpm']:.0f} bpm, {label} (low confidence)",
                f"Skin colour on the {label} oscillates at {result['bpm'] / 60:.2f} Hz, a heart rate "
                f"of {result['bpm']:.0f} bpm (signal-to-noise {result['snr_db']:+.1f} dB). "
                + (
                    "Motion, compression and lighting can mimic a pulse; treat as indicative."
                    if not reliable
                    else "Invisible to the eye; recovered from sub-1% colour changes."
                ),
                time_s=float(t0),
                tags=["pulse", "face"],
                data={
                    "bpm": result["bpm"],
                    "snr_db": result["snr_db"],
                    "box": track["boxes"][0],
                },
            )

            if best is None or result["snr_db"] > best[0]["snr_db"]:
                best = (result, track, indices)

        if best is not None:
            out.progress(0.8, "rendering Eulerian magnification")
            result, track, indices = best
            count = int(min(len(indices), fps * 10))
            path = out.path("magnified.mp4")
            ok = await asyncio.to_thread(
                _magnify, ctx.path, track, width / info.width,
                int(indices[0]), count, fps, result["bpm"], path, (info.width, info.height),
            )  # fmt: skip

            if ok:
                out.artifact(
                    "magnified",
                    "video",
                    "Eulerian colour magnification (left: original, right: pulse colour amplified)",
                    file=path,
                )

        ctx.publish(self.name, published)
        out.progress(1.0)

    def _report_people(
        self, out: Reporter, people: list[dict[str, Any]], info: Any, seconds: float
    ) -> None:
        if not people:
            return

        for p in people[:12]:
            crop = p.pop("crop", None)

            if crop is not None and crop.size:
                scale = max(1.0, 160 / max(crop.shape[:2]))
                big = cv2.resize(
                    crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
                )
                path = out.path(f"person{p['index']}.jpg")
                cv2.imwrite(
                    path,
                    cv2.cvtColor(big, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 90],
                )
                out.artifact(f"person{p['index']}", "image", f"Appearance #{p['index']}: on screen from {_clock(p['first_s'])}",
                             file=path, data={"caption": f"Seen for {p['seconds']:.0f} s in total; largest view in the clip. Not identified.", "time_s": p["first_s"]})  # fmt: skip

        for p in people:
            p.pop("crop", None)

        n = len(people)
        # Faces on screen together are certainly different people: the most at once is a floor.
        events = sorted([(a, 1) for p in people for a, _ in p["intervals"]] + [(b, -1) for p in people for _, b in p["intervals"]], key=lambda e: (e[0], e[1]))  # fmt: skip
        together, most, when = 0, 0, people[0]["first_s"]

        for t, d in events:
            together += d

            if together > most:
                most, when = together, t

        rows = [[f"#{p['index']}", ", ".join(f"{_clock(a)}–{_clock(b)}" for a, b in p["intervals"][:6]) + (" …" if len(p["intervals"]) > 6 else ""),
                 f"{p['seconds']:.0f} s", f"{p['width_px'] * info.width / min(_DETECT_WIDTH, info.width):.0f} px"] for p in people[:30]]  # fmt: skip
        out.artifact(
            "people",
            "table",
            f"Face appearances and when ({n})",
            data={
                "columns": ["appearance", "on screen", "total", "face width"],
                "rows": rows,
            },
        )
        longest = max(people, key=lambda p: p["seconds"])
        out.finding(
            "info",
            (f"At least {most} different people on screen ({n} face appearances)" if most > 1
             else f"{n} face appearances on screen" if n > 1
             else f"1 face on screen, from {_clock(people[0]['first_s'])}"),
            (f"Faces were followed from frame to frame, three times a second, over the first {seconds:.0f} s. "
             + (f"Up to {most} faces are on screen together (at {_clock(when)}), so at least {most} different people appear. " if most > 1 else "")
             + (f"Appearance #{longest['index']} is the longest ({longest['seconds']:.0f} s). " if n > 1 else "")
             + "Nobody is identified, so the same person seen again after a cut, or after leaving the frame or turning away for more than two seconds, "
             "counts as a new appearance; faces in profile, very small or blurred may be missed."),
            time_s=float(people[0]["first_s"]),
            tags=["person", "appearance"],
            data={"count": n, "at_least": most, "people": people[:30]},
        )  # fmt: skip
