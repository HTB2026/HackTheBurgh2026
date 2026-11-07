"""Do sound and picture belong together?

Sharp sounds (a clap, a door, footsteps, plosives) coincide with sudden visual
change. Cross-correlating the audio onset envelope with frame-to-frame motion
energy measures the offset between the tracks. Cameras record them in lock
step; a consistent offset means the audio was replaced, re-dubbed or shifted.
"""

import numpy as np
from scipy import signal

from earwitness.modules.base import CaseContext
from earwitness.modules.base import Reporter
from earwitness.modules.base import Skip

_RATE = 100.0
_MAX_LAG_S = 1.0


def _best_lag(
    audio: np.ndarray, visual: np.ndarray
) -> tuple[float, float, np.ndarray, np.ndarray]:
    max_lag = int(_MAX_LAG_S * _RATE)
    correlation = signal.correlate(audio, visual, mode="full") / len(audio)
    lags = signal.correlation_lags(len(audio), len(visual), mode="full")
    keep = np.abs(lags) <= max_lag
    correlation, lags = correlation[keep], lags[keep]
    best = int(np.argmax(correlation))
    others = np.abs(lags - lags[best]) > 10
    z = float(
        (correlation[best] - correlation[others].mean())
        / (correlation[others].std() + 1e-9)
    )

    return float(lags[best] / _RATE * 1000), z, correlation, lags


def _prepare(times: np.ndarray, values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    x = np.interp(grid, times, values)
    # Keep only the changes (onsets), then standardise.
    x = np.maximum(0.0, np.diff(x, prepend=x[0])) if values is not None else x
    x = x - signal.medfilt(x, 51)
    x = np.maximum(x, 0)

    return (x - x.mean()) / (x.std() + 1e-9)


class AvSync:
    name = "avsync"
    title = "A/V SYNC"
    description = (
        "Offset between audio and picture from matching sound onsets to visual events."
    )
    depends = ["timeline", "spectrum"]

    async def run(self, ctx: CaseContext, out: Reporter) -> None:
        timeline = await ctx.wait_for("timeline") or {}
        spectrum = await ctx.wait_for("spectrum") or {}
        motion = timeline.get("motion")
        onsets = spectrum.get("onsets")

        if motion is None or onsets is None:
            raise Skip("needs both audio and video")

        end = min(float(np.max(motion["times"])), float(np.max(onsets["t"])))

        if end < 4:
            raise Skip("too short")

        grid = np.arange(0, end, 1 / _RATE)
        visual = _prepare(
            np.asarray(motion["times"]), np.asarray(motion["energy"], dtype=float), grid
        )
        audio = np.interp(grid, onsets["t"], onsets["v"])
        audio = audio - signal.medfilt(audio, 51)
        audio = (np.maximum(audio, 0) - np.maximum(audio, 0).mean()) / (
            np.maximum(audio, 0).std() + 1e-9
        )
        lag_ms, z, correlation, lags = _best_lag(audio, visual)
        # A real offset is the same throughout; a chance alignment is not.
        half = len(grid) // 2
        halves = [
            _best_lag(audio[:half], visual[:half]),
            _best_lag(audio[half:], visual[half:]),
        ]
        consistent = all(h[1] >= 4 and abs(h[0] - lag_ms) <= 40 for h in halves)
        out.artifact(
            "xcorr",
            "series",
            "Audio↔video cross-correlation (lag ms; positive = audio late)",
            data={
                "x": (lags / _RATE * 1000).round(1),
                "series": [{"name": "correlation", "y": correlation.round(5)}],
                "x_unit": "ms",
            },
        )
        published = {
            "lag_ms": lag_ms,
            "z": z,
            "halves_ms": [h[0] for h in halves],
            "consistent": consistent,
        }

        if z < 5 or (abs(lag_ms) > 50 and not consistent):
            out.finding(
                "info",
                "Too few shared audio-visual events to measure sync",
                "No sharp sound lines up clearly and consistently with a visual change, so sync "
                f"could not be tested (best match {lag_ms:+.0f} ms at {z:.1f}σ; the two halves of the "
                f"clip disagree: {halves[0][0]:+.0f} / {halves[1][0]:+.0f} ms).",
                tags=["sync"],
                data=published,
            )
        elif abs(lag_ms) <= 50:
            out.finding(
                "info",
                f"Sound and picture in sync ({lag_ms:+.0f} ms)",
                f"Sharp sounds line up with visual events to within a frame or two ({z:.0f}σ match). "
                "Consistent with audio recorded together with the picture.",
                tags=["sync"],
                data=published,
            )
        else:
            out.finding(
                "warning" if abs(lag_ms) < 200 else "critical",
                f"Audio {'lags' if lag_ms > 0 else 'leads'} picture by {abs(lag_ms):.0f} ms",
                f"Visual events and the sounds they make are consistently {abs(lag_ms):.0f} ms apart "
                f"({z:.0f}σ match). A camera records both in lock step. The soundtrack was shifted, "
                "replaced or re-dubbed.",
                tags=["sync", "splice"],
                data=published,
            )

        ctx.publish(self.name, published)
