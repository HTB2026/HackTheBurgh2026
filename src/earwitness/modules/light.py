"""The light in the scene: mains flicker, sensor readout, and the sun.

Lamps on AC mains flicker at twice the grid frequency (100 Hz in Europe/GB,
120 Hz in the Americas). Too fast for the eye, but a phone's rolling shutter
reads the sensor row by row, so the flicker becomes faint horizontal bands that
crawl through the picture. Sampled at the frame rate, the flicker aliases to a
predictable low frequency (100 Hz at 30 fps -> 10 Hz). Finding that alias tells
which grid powered the lights, how fast the sensor reads out, and gives a
visual electrical-network-frequency trace.
"""

import asyncio
from typing import Any

import numpy as np
from scipy import signal

from earwitness.modules.base import CaseContext
from earwitness.modules.base import Reporter
from earwitness.modules.base import Skip
from earwitness.modules.base import downsample_series
from earwitness.modules.base import run_tool

_ROWS = 360
_COLS = 48


def _alias(flicker_hz: float, fps: float) -> float:
    folded = flicker_hz % fps

    return min(folded, fps - folded)


async def _row_profiles(ctx: CaseContext, max_frames: int = 3_600) -> np.ndarray | None:
    output = await run_tool(
        "ffmpeg", "-nostdin", "-v", "error", "-i", ctx.path, "-frames:v", str(max_frames),
        "-vf", f"scale={_COLS}:{_ROWS},format=gray", "-fps_mode", "passthrough",
        "-f", "rawvideo", "-",
    )  # fmt: skip

    if not output:
        return None

    pixels = np.frombuffer(output, dtype=np.uint8)
    n = len(pixels) // (_ROWS * _COLS)

    return pixels[: n * _ROWS * _COLS].reshape(n, _ROWS, _COLS).astype(np.float32)


def _flicker_analysis(stack: np.ndarray, fps: float) -> dict[str, Any]:
    n = len(stack)
    rows = stack.mean(axis=2)  # (frames, rows)
    # Remove scene content: each row's slow trend over time, and each frame's
    # overall exposure. What survives is fast, row-dependent modulation.
    detrended = rows - signal.savgol_filter(rows, min(n - (1 - n % 2), 15), 2, axis=0)
    detrended -= detrended.mean(axis=1, keepdims=True)
    window = np.hanning(n)[:, None]
    spectrum = np.fft.rfft(detrended * window, axis=0)
    freqs = np.fft.rfftfreq(n, 1 / fps)
    power = (np.abs(spectrum) ** 2).mean(axis=1)
    results = {}

    for grid, flicker in ((50, 100.0), (60, 120.0)):
        alias = _alias(flicker, fps)
        results[grid] = {"alias_hz": alias, "snr_db": None, "coherence": 0.0}

        if alias < 0.4:
            continue  # bands stand still at this frame rate; no temporal signature

        band = np.abs(freqs - alias) <= max(0.15, 2 * fps / n)
        side = (
            (np.abs(freqs - alias) > 0.6)
            & (np.abs(freqs - alias) < 3.0)
            & (freqs > 0.3)
        )

        if not band.any() or not side.any():
            continue

        peak = int(np.flatnonzero(band)[np.argmax(power[band])])
        snr = 10 * np.log10(power[peak] / (np.median(power[side]) + 1e-12))
        # Rolling-shutter signature: phase advances smoothly down the frame.
        phase = np.unwrap(np.angle(spectrum[peak]))
        y = np.arange(len(phase))
        slope, intercept = np.polyfit(y, phase, 1)
        residual = phase - (slope * y + intercept)
        coherence = float(np.abs(np.mean(np.exp(1j * residual))))
        readout_ms = abs(slope * len(phase)) / (2 * np.pi * flicker) * 1000
        results[grid] = {
            "alias_hz": float(freqs[peak]),
            "expected_alias_hz": alias,
            "snr_db": float(snr),
            "coherence": coherence,
            "readout_ms": float(readout_ms),
        }

    results["spectrum"] = {"f": freqs, "p": 10 * np.log10(power + 1e-12)}

    return results


def _enf_trace(
    stack: np.ndarray, fps: float, flicker: float, alias: float
) -> tuple[np.ndarray, np.ndarray] | None:
    """Track the alias frequency over time -> implied mains frequency."""
    rows = stack.mean(axis=2)
    n = len(rows)
    detrended = rows - signal.savgol_filter(rows, min(n - (1 - n % 2), 15), 2, axis=0)
    # Collapse rows into one signal by projecting onto the band's phase pattern.
    spectrum = np.fft.rfft(detrended * np.hanning(n)[:, None], axis=0)
    freqs = np.fft.rfftfreq(n, 1 / fps)
    peak = int(np.argmin(np.abs(freqs - alias)))
    weights = np.conj(spectrum[peak]) / (np.abs(spectrum[peak]) + 1e-9)
    analytic = signal.hilbert(detrended, axis=0) @ weights
    window = int(fps * 8)

    if n < window * 2:
        return None

    times, values = [], []

    for start in range(0, n - window, int(fps)):
        chunk = analytic[start : start + window]
        zoom = np.fft.fft(chunk * np.hanning(window), 16 * window)
        zf = np.fft.fftfreq(16 * window, 1 / fps)
        near = np.abs(np.abs(zf) - alias) < 1.0
        best = zf[near][np.argmax(np.abs(zoom[near]))]
        times.append((start + window / 2) / fps)
        values.append(abs(best))

    measured_alias = np.array(values)
    folds = round((flicker - alias) / fps)
    sign = (
        1
        if abs(folds * fps + alias - flicker) < abs(folds * fps - alias - flicker)
        else -1
    )
    implied_flicker = folds * fps + sign * measured_alias

    return np.array(times), implied_flicker / 2


class Light:
    name = "light"
    title = "LIGHT & FLICKER"
    description = "Mains lighting flicker via rolling shutter (grid region, sensor readout), scene lighting vs sun."
    depends = ["provenance"]

    async def run(self, ctx: CaseContext, out: Reporter) -> None:
        info = await ctx.video_info()

        if info is None:
            raise Skip("no video stream")

        out.progress(0.1, "reading sensor rows")
        stack = await _row_profiles(ctx)

        if stack is None or len(stack) < info.fps * 3:
            raise Skip("clip too short for flicker analysis (needs 3 s)")

        times = await ctx.frame_times()
        fps = float(1 / np.median(np.diff(times))) if len(times) > 10 else info.fps
        out.progress(0.4, f"searching flicker aliases at {fps:.3f} fps")
        result = await asyncio.to_thread(_flicker_analysis, stack, fps)
        published: dict[str, Any] = {"fps": fps}
        spectrum = result.pop("spectrum")
        keep = spectrum["f"] > 0.3
        out.artifact(
            "flicker_spectrum",
            "series",
            "Row-brightness modulation spectrum",
            data=downsample_series(
                spectrum["f"][keep], {"power_db": spectrum["p"][keep]}
            )
            | {
                "x_unit": "Hz",
                "y_unit": "dB",
                "marks": {
                    f"{g} Hz grid alias": r["alias_hz"] for g, r in result.items()
                },
            },
        )
        candidates = [
            (grid, r)
            for grid, r in result.items()
            if r["snr_db"] is not None and r["snr_db"] > 10 and r["coherence"] > 0.6
        ]

        if candidates:
            grid, best = max(candidates, key=lambda item: item[1]["snr_db"])
            published.update({"grid_hz": grid, **best})
            region = (
                "GB / Europe / most of Asia, Africa, Australia"
                if grid == 50
                else "the Americas, parts of Japan, Korea, Taiwan, Saudi Arabia"
            )
            out.finding(
                "critical",
                f"Lights flicker at {2 * grid} Hz: {grid} Hz grid",
                f"Invisible banding from the rolling shutter beats at {best['alias_hz']:.2f} Hz, "
                f"where {2 * grid} Hz lamp flicker lands at {fps:.2f} fps ({best['snr_db']:.0f} dB above "
                f"the background, phase-coherent down the frame). The scene was lit by mains power from a "
                f"{grid} Hz grid: {region}.",
                tags=["flicker", "location", "enf"],
                data=published,
            )
            out.finding(
                "info",
                f"Sensor readout ≈ {best['readout_ms']:.1f} ms",
                "Measured from how fast the flicker band's phase changes down the frame. Each camera "
                "model has a characteristic rolling-shutter readout time, a weak fingerprint of the device.",
                tags=["device", "flicker"],
                data={"readout_ms": best["readout_ms"]},
            )

            trace = await asyncio.to_thread(
                _enf_trace, stack, fps, 2.0 * grid, best["expected_alias_hz"]
            )

            if trace is not None:
                published["enf_trace"] = {"t": trace[0], "f": trace[1]}
                out.artifact(
                    "light_enf",
                    "series",
                    "Grid frequency implied by flicker",
                    data=downsample_series(trace[0], {"mains_hz": trace[1]})
                    | {"x_unit": "s", "y_unit": "Hz"},
                )
        else:
            stills = [g for g, r in result.items() if r["alias_hz"] < 0.4]
            out.finding(
                "info",
                "No mains flicker in the picture",
                "No rolling-shutter flicker bands found. Daylight, battery or DC-driven LED lighting, "
                "or exposure locked to the flicker period (many phones do this)."
                + (
                    f" At {fps:.0f} fps a {stills[0]} Hz grid's flicker would stand still and cannot be seen this way."
                    if stills
                    else ""
                ),
                tags=["flicker"],
            )

        # ---- scene lighting vs the sun -------------------------------------
        out.progress(0.8, "scene illumination")
        frames = await ctx.frames(160)

        if frames is not None:
            sample = frames.pixels[:: max(1, len(frames.pixels) // 60)].astype(
                np.float32
            )
            luma = float(
                (
                    0.299 * sample[..., 0]
                    + 0.587 * sample[..., 1]
                    + 0.114 * sample[..., 2]
                ).mean()
            )
            rb = float(sample[..., 0].mean() / (sample[..., 2].mean() + 1e-6))
            published.update({"luma": luma, "red_blue": rb})
            provenance = await ctx.wait_for("provenance") or {}
            sun = provenance.get("sun")
            artificial = bool(candidates)
            warmth = (
                "warm (tungsten/LED)"
                if rb > 1.25
                else "cool (daylight/overcast/fluorescent)"
                if rb < 0.95
                else "neutral"
            )

            if sun is not None:
                elevation = sun["elevation"]
                inconsistent = (
                    elevation < -12 and luma > 110 and rb < 1.0 and not artificial
                )
                out.finding(
                    "warning" if inconsistent else "info",
                    "Lighting contradicts claimed time"
                    if inconsistent
                    else "Lighting vs claimed time",
                    f"Scene brightness {luma:.0f}/255, colour {warmth}, "
                    f"{'mains-powered lighting detected' if artificial else 'no mains flicker'}. "
                    f"At the claimed time and place the sun was {elevation:+.0f}°. "
                    + (
                        "A bright, cool, flicker-free scene looks like daylight, but the sun was well below the horizon."
                        if inconsistent
                        else "Nothing here contradicts the claimed time."
                    ),
                    tags=["sun", "time"] + (["inconsistency"] if inconsistent else []),
                    data={
                        "luma": luma,
                        "red_blue": rb,
                        "sun_elevation": elevation,
                        "artificial": artificial,
                    },
                )

        ctx.publish(self.name, published)
        out.progress(1.0)
