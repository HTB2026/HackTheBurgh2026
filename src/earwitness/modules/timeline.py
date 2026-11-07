"""Frame-level timeline integrity: timing gaps, duplicated frames, cuts, GOP anomalies.

Camera footage has a steady rhythm: frames arrive at the frame rate, keyframes
at a fixed interval, and consecutive frames always differ a little (sensor
noise). Edits break that rhythm in ways that are invisible when watching.
"""

import asyncio
from typing import Any

import numpy as np
from PIL import Image

from earwitness.modules.base import CaseContext
from earwitness.modules.base import Reporter
from earwitness.modules.base import Skip
from earwitness.modules.base import downsample_series

_THUMB_WIDTH = 160
_STRIP_TILES = 48
_CUT_Z = 8.0
_CUT_MIN_DIFF = 18.0
_DUP_MAX_DIFF = 0.15


def _robust_z(values: np.ndarray) -> np.ndarray:
    median = np.median(values)
    mad = np.median(np.abs(values - median)) * 1.4826 + 1e-9

    return (values - median) / mad


def _frame_differences(pixels: np.ndarray) -> np.ndarray:
    diffs = np.array(
        [
            np.abs(b.astype(np.int16) - a.astype(np.int16)).mean()
            for a, b in zip(pixels[:-1], pixels[1:])
        ]
    )

    return np.concatenate(([0.0], diffs))


def _histogram_distance(pixels: np.ndarray) -> np.ndarray:
    """Colour-histogram change between frames; robust to motion, sensitive to cuts."""
    quantised = (pixels // 32).astype(np.int32)
    codes = quantised[..., 0] * 64 + quantised[..., 1] * 8 + quantised[..., 2]
    hist = np.stack([np.bincount(c.ravel(), minlength=512) for c in codes]).astype(
        float
    )
    hist /= hist.sum(axis=1, keepdims=True)
    distance = 0.5 * np.abs(np.diff(hist, axis=0)).sum(axis=1)

    return np.concatenate(([0.0], distance))


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    runs = []
    start = None

    for index, value in enumerate([*mask, False]):
        if value and start is None:
            start = index
        elif not value and start is not None:
            runs.append((start, index))
            start = None

    return runs


def _filmstrip(pixels: np.ndarray, times: np.ndarray, path: str) -> dict[str, Any]:
    count = min(_STRIP_TILES, len(pixels))
    picks = np.linspace(0, len(pixels) - 1, count).astype(int)
    tiles = pixels[picks]
    h, w = tiles.shape[1:3]
    sheet = np.concatenate(list(tiles), axis=1)
    Image.fromarray(sheet).save(path, quality=80)

    return {
        "tile_width": w,
        "tile_height": h,
        "times": times[picks].tolist(),
        "count": count,
    }


def _periodicity(sizes: np.ndarray, gop: int | None) -> tuple[int, float] | None:
    """Strongest period (in frames) of P-frame size fluctuation, with its prominence."""
    if len(sizes) < 120:
        return None

    x = np.log(sizes + 1.0)
    x = x - np.convolve(x, np.ones(31) / 31, mode="same")
    spectrum = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    freqs = np.fft.rfftfreq(len(x))
    valid = (freqs > 1 / 120) & (freqs < 1 / 5)

    if not valid.any():
        return None

    index = np.flatnonzero(valid)[np.argmax(spectrum[valid])]
    prominence = float(spectrum[index] / (np.median(spectrum[valid]) + 1e-12))
    period = int(round(1 / freqs[index]))

    if gop and (period % gop == 0 or abs(period - gop) <= 1):
        return None

    return period, prominence


class Timeline:
    name = "timeline"
    title = "FRAME TIMELINE"
    description = "Frame timing, dropped/duplicated frames, scene cuts, keyframe rhythm, recompression."

    async def run(self, ctx: CaseContext, out: Reporter) -> None:
        info = await ctx.video_info()

        if info is None:
            raise Skip("no video stream")

        out.progress(0.05, "reading packet table")
        table = await ctx.frame_table()

        if len(table) < 10:
            raise Skip("too few frames")

        pts = np.array([r["pts"] for r in table])
        pts = pts - pts[0]
        sizes = np.array([r["size"] for r in table], dtype=float)
        keys = np.array([r["key"] for r in table])
        intervals = np.diff(pts)
        nominal = float(np.median(intervals))
        published: dict[str, Any] = {
            "cuts": [],
            "fps": 1 / nominal if nominal else info.fps,
        }

        # ---- timing --------------------------------------------------------
        gaps = np.flatnonzero(intervals > 2.5 * nominal)
        jitter = float(np.std(intervals) / nominal) if nominal else 0.0
        out.artifact(
            "intervals",
            "series",
            "Frame interval (ms)",
            data=downsample_series(pts[1:], {"interval_ms": intervals * 1000})
            | {"x_unit": "s", "y_unit": "ms"},
        )

        if len(gaps):
            worst = int(gaps[np.argmax(intervals[gaps])])
            missing = int(np.sum(np.round(intervals[gaps] / nominal) - 1))
            out.finding(
                "warning" if len(gaps) <= 5 else "notice",
                f"{len(gaps)} timing gap{'s' if len(gaps) > 1 else ''} (~{missing} frames missing)",
                f"Frames normally arrive every {nominal * 1000:.1f} ms. The largest gap, "
                f"{intervals[worst] * 1000:.0f} ms, is at {pts[worst]:.2f} s. Gaps come from dropped "
                "frames (heavy load, low light) or from cutting footage out.",
                time_s=float(pts[worst]),
                tags=["timing"],
                data={"gaps_s": pts[gaps][:50], "gap_ms": intervals[gaps][:50] * 1000},
            )
        elif jitter > 0.2:
            out.finding(
                "info",
                "Variable frame rate",
                f"Frame intervals vary by {jitter * 100:.0f}%. Phones do this in low light; "
                "it is not by itself a sign of editing.",
                tags=["timing"],
            )
        else:
            out.finding(
                "info",
                f"Steady {1 / nominal:.2f} fps frame clock",
                f"{len(pts)} frames with no timing gaps (jitter {jitter * 100:.1f}%).",
                tags=["timing"],
            )

        # ---- keyframe rhythm ----------------------------------------------
        key_index = np.flatnonzero(keys)
        gop = None

        if len(key_index) >= 3:
            spacing = np.diff(key_index)
            values, counts = np.unique(spacing, return_counts=True)
            gop = int(values[np.argmax(counts)])
            odd = [
                int(key_index[i + 1])
                for i, s in enumerate(spacing)
                if s < gop * 0.8 and i + 1 < len(spacing)  # last GOP may be short
            ]
            published["gop"] = gop
            out.artifact(
                "keyframes",
                "markers",
                "Keyframes",
                data={"times": pts[key_index], "gop": gop},
            )

            if odd:
                out.finding(
                    "warning",
                    f"{len(odd)} off-rhythm keyframe{'s' if len(odd) > 1 else ''}",
                    f"Keyframes normally come every {gop} frames ({gop * nominal:.2f} s). "
                    f"Early ones at {', '.join(f'{pts[i]:.2f}s' for i in odd[:6])}. Encoders force a "
                    "keyframe at a scene change or a splice point.",
                    time_s=float(pts[odd[0]]),
                    tags=["gop", "splice"],
                    data={"times": pts[odd]},
                )
            else:
                out.finding(
                    "info",
                    f"Regular keyframe rhythm (every {gop} frames)",
                    "No unscheduled keyframes. Consistent with one continuous encode.",
                    tags=["gop"],
                )

        bitrate_bins = np.arange(0, pts[-1] + 1.0, 0.5)
        index = np.clip(np.digitize(pts, bitrate_bins) - 1, 0, len(bitrate_bins) - 1)
        kbps = (
            np.bincount(index, weights=sizes, minlength=len(bitrate_bins))
            * 8
            / 0.5
            / 1000
        )
        out.artifact(
            "bitrate",
            "series",
            "Video bitrate (kbit/s)",
            data=downsample_series(bitrate_bins, {"kbps": kbps}) | {"x_unit": "s"},
        )

        period = _periodicity(sizes[~keys], gop)
        published["p_frame_period"] = period

        # ---- pixel-level -----------------------------------------------------
        out.progress(0.3, "decoding frames")
        frames = await ctx.frames(_THUMB_WIDTH)

        if frames is None or len(frames.pixels) < 3:
            ctx.publish(self.name, published)

            return

        out.progress(0.6, "differencing frames")
        times = frames.times_s
        diffs, hist = await asyncio.gather(
            asyncio.to_thread(_frame_differences, frames.pixels),
            asyncio.to_thread(_histogram_distance, frames.pixels),
        )
        published["motion"] = {"times": times, "energy": diffs}
        pixels = frames.pixels.astype(np.float32)
        luma = (
            0.299 * pixels[..., 0] + 0.587 * pixels[..., 1] + 0.114 * pixels[..., 2]
        ).mean(axis=(1, 2))
        warmth = pixels[..., 0].mean(axis=(1, 2)) / (
            pixels[..., 2].mean(axis=(1, 2)) + 1e-6
        )
        out.artifact(
            "light_curve",
            "series",
            "Brightness and colour balance over time",
            data=downsample_series(
                times, {"brightness_0_255": luma, "red_blue_ratio_x100": warmth * 100}
            )
            | {"x_unit": "s"},
        )
        strip = _filmstrip(frames.pixels, times, out.path("filmstrip.jpg"))
        out.artifact(
            "filmstrip",
            "filmstrip",
            "Filmstrip",
            data=strip,
            file=out.path("filmstrip.jpg"),
        )
        out.artifact(
            "difference",
            "series",
            "Frame-to-frame change",
            data=downsample_series(
                times, {"pixel_diff": diffs, "histogram_change": hist * 100}
            )
            | {"x_unit": "s"},
        )

        # Scene cuts: a lone spike in change between two stretches that are each
        # internally steady. Fast pans change a lot too, but over many frames.
        z = _robust_z(hist[1:])
        cut_index = []

        for i in np.flatnonzero((z > _CUT_Z) & (diffs[1:] > _CUT_MIN_DIFF)) + 1:
            before = np.median(diffs[max(1, i - 3) : i]) if i > 1 else 0.0
            after = np.median(diffs[i + 1 : i + 4]) if i + 1 < len(diffs) else 0.0

            # A flash (camera flash, lightning, a strobe) changes one frame and
            # then the picture returns: the frames either side of the flash match.
            def gap(a: int, b: int) -> float:
                if a < 0 or b >= len(frames.pixels):
                    return float("inf")

                return float(
                    np.abs(
                        frames.pixels[b].astype(np.int16)
                        - frames.pixels[a].astype(np.int16)
                    ).mean()
                )

            if min(gap(i - 1, i + 1), gap(i - 2, i)) < 0.35 * diffs[i]:
                continue

            if diffs[i] >= 3.0 * max(before, after, 1.0):
                cut_index.append(int(i))

        cuts = []

        for index_ in cut_index:
            if not cuts or index_ - cuts[-1] > 5:
                cuts.append(index_)

        cut_times = [float(times[i]) for i in cuts]
        published["cuts"] = cut_times

        for i in cuts[:20]:
            out.finding(
                "warning",
                f"Hard cut at {times[i]:.2f} s",
                f"The picture changes completely between two frames (histogram distance "
                f"{hist[i]:.2f}, {min(z[i - 1], 999):.0f}σ above normal). A single camera take cannot cut. "
                "This is an edit point or a camera restart.",
                time_s=float(times[i]),
                tags=["cut", "splice"],
                data={"histogram": hist[i], "pixel_diff": diffs[i]},
            )

        if not cuts:
            out.finding(
                "info",
                "No hard cuts",
                "Every frame flows from the one before: one continuous shot.",
                tags=["cut"],
            )

        # Duplicated frames: identical neighbours while the scene otherwise moves.
        moving = np.median(diffs[1:]) > 0.8
        dup = (diffs < _DUP_MAX_DIFF) & moving
        dup[0] = False
        all_runs = [(a, b) for a, b in _runs(dup) if b - a >= 1]
        # Recorders often repeat the first frames while the sensor starts up.
        startup = [(a, b) for a, b in all_runs if a <= 3]
        runs = [r for r in all_runs if r not in startup]

        if startup:
            out.finding(
                "info",
                f"Camera start-up: first {startup[0][1]} frames repeated",
                "The opening frames are identical while the recorder spins up, as phone "
                "camera apps leave them. Not an edit.",
                time_s=0.0,
                tags=["duplicate"],
            )

        if runs and moving:
            total = sum(b - a for a, b in runs)
            longest = max(runs, key=lambda r: r[1] - r[0])
            out.finding(
                "warning" if longest[1] - longest[0] >= 3 else "notice",
                f"{total} duplicated frame{'s' if total > 1 else ''}",
                f"Frames that are pixel-identical to the previous one while the scene is moving "
                f"({len(runs)} runs, longest {longest[1] - longest[0]} frames at {times[longest[0]]:.2f} s). "
                "A camera never repeats a frame exactly. Frame-rate conversion, freeze-frames "
                "and padding over removed footage do.",
                time_s=float(times[longest[0]]),
                tags=["duplicate", "splice"],
                data={"runs": [[float(times[a]), b - a] for a, b in runs[:100]]},
            )

        ctx.publish(self.name, published)
        out.progress(1.0)
