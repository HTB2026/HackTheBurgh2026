"""Electrical network frequency: date the recording and test its continuity.

Wraps the original earwitness engine (hum.py, grid.py). Mains hum leaks into
recordings near powered equipment; its frequency follows the grid's, which
wanders unpredictably and is logged to the second. Matching the two dates the
recording, and pieces from different times do not line up.
"""

import asyncio
import os
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from typing import Any

import numpy as np

from earwitness import grid
from earwitness import grid_fetch
from earwitness import hum
from earwitness._common import ForensicError
from earwitness.modules.base import CaseContext
from earwitness.modules.base import Reporter
from earwitness.modules.base import Skip
from earwitness.modules.base import downsample_series

_SEARCH_HOURS = 3.0


def _level_match(
    trace: hum.HumTrace, reference: grid.GridSeries
) -> dict[str, Any] | None:
    """Absolute-level match: RMS difference at every offset (short clips).

    Correlation ignores the mean, which wastes the strongest clue in a short
    clip: the grid sits at a particular frequency at a particular moment.
    """
    block = max(1, int(round(reference.step_s / trace.step_s)))
    usable = len(trace.values) // block * block

    if usable < block:
        return None

    query = trace.values[:usable].reshape(-1, block).mean(axis=1)
    weights = 1.0 - trace.masked[:usable].reshape(-1, block).mean(axis=1)
    n = len(query)

    if n < 3 or len(reference.values) < n + 2:
        return None

    windows = np.lib.stride_tricks.sliding_window_view(reference.values, n)
    rms = np.sqrt(
        (((windows - query) ** 2) * weights).sum(axis=1) / max(weights.sum(), 1e-9)
    )
    best = int(np.argmin(rms))
    masked = rms.copy()
    masked[max(0, best - n) : best + n] = np.inf
    runner = float(masked.min()) if np.isfinite(masked).any() else float("nan")
    first_centre = trace.first_s + (block - 1) * trace.step_s / 2
    start = reference.start + timedelta(seconds=best * reference.step_s - first_centre)

    return {
        "start": start,
        "rms_hz": float(rms[best]),
        "runner_up_rms_hz": runner,
        "rms_curve": rms,
        "offset_index": best,
        "query": query,
    }


class Enf:
    name = "enf"
    title = "MAINS HUM / ENF"
    description = (
        "Electrical network frequency in the audio: dating and splice detection."
    )
    depends = ["provenance", "light"]

    async def _reference(
        self, ctx: CaseContext, out: Reporter
    ) -> tuple[grid.GridSeries | None, str | None]:
        if path := ctx.options.get("grid_path"):
            loaded = await grid.load_csv(path)

            return (loaded if isinstance(loaded, grid.GridSeries) else None), (
                f"supplied grid file {os.path.basename(path)}"
            )

        provenance = await ctx.wait_for("provenance") or {}
        centre = provenance.get("start_guess")

        if isinstance(centre, str):
            centre = datetime.fromisoformat(centre)

        if centre is None:
            return None, None

        if centre.tzinfo is None:
            centre = centre.replace(tzinfo=timezone.utc)

        hours = _SEARCH_HOURS
        out.progress(0.6, "fetching GB grid record")

        return await grid_fetch.fetch(
            centre - timedelta(hours=hours), centre + timedelta(hours=hours)
        )

    async def _from_light(self, ctx: CaseContext, out: Reporter) -> bool:
        """Date the recording from lamp flicker when the audio has no hum."""
        light = await ctx.wait_for("light") or {}
        flicker = light.get("enf_trace")

        if flicker is None or light.get("grid_hz") != 50:
            return False

        times = np.asarray(flicker["t"], dtype=float)
        values = np.asarray(flicker["f"], dtype=float)

        if len(values) < 12:
            return False

        reference, source = await self._reference(ctx, out)

        if reference is None:
            out.finding(
                "notice",
                "Grid trace from the lights; no record to compare",
                "Lamp flicker gives a mains-frequency trace, but there is no time "
                "window to search. Supply an approximate date to date the footage.",
                tags=["enf", "flicker"],
            )

            return True

        profile = hum.HumProfile(50.0, 2, 0.0, {})
        trace = hum.HumTrace(
            values=values,
            masked=np.zeros(len(values), dtype=bool),
            first_s=float(times[0]),
            step_s=float(np.median(np.diff(times))),
            profile=profile,
        )
        level = _level_match(trace, reference)
        match = grid.match(trace, reference)

        if level is None:
            return False

        correlated = isinstance(match, grid.TimeMatch) and match.confident
        start = match.recording_start if correlated else level["start"]
        confident = correlated or (
            level["rms_hz"] < 0.008
            and level["runner_up_rms_hz"] > 1.8 * level["rms_hz"]
        )
        candidates = (await ctx.wait_for("provenance") or {}).get(
            "start_candidates"
        ) or []
        drift = min(
            ((start - c).total_seconds() for c in candidates), key=abs, default=None
        )
        grid_times = (
            reference.start.timestamp()
            + np.arange(len(reference.values)) * reference.step_s
            - start.timestamp()
        )
        inside = (grid_times >= times[0] - 20) & (grid_times <= times[-1] + 20)
        out.artifact(
            "light_vs_grid",
            "series",
            "Flicker-derived mains vs grid record",
            data={
                "x": np.round(times, 3),
                "series": [
                    {"name": "from lamp flicker", "y": np.round(values, 5)},
                    {
                        "name": f"grid ({source})",
                        "x": np.round(grid_times[inside], 3),
                        "y": np.round(reference.values[inside], 5),
                    },
                ],
                "x_unit": "s",
                "y_unit": "Hz",
            },
        )
        detail = (
            f"The flicker of the lights, tracked frame by frame, matches the GB grid "
            f"record to {level['rms_hz'] * 1000:.1f} mHz RMS (next best moment "
            f"{level['runner_up_rms_hz'] * 1000:.1f} mHz). Source: {source}."
        )

        if drift is not None:
            detail += f" Metadata differs by {drift:+.0f} s."

        out.finding(
            "critical" if confident else "notice",
            f"Lighting dates footage to {start:%Y-%m-%d %H:%M:%S} UTC"
            if confident
            else "Flicker trace: no convincing grid match",
            detail,
            tags=["enf", "flicker", "time"],
            data={"start": start, "confident": confident, "metadata_diff_s": drift},
        )
        if confident and drift is not None and abs(drift) > 120:
            out.finding(
                "critical",
                "Metadata time contradicted by the lights",
                f"The lamp flicker places the recording {abs(drift) / 60:.1f} minutes "
                f"{'later' if drift > 0 else 'earlier'} than the file's own timestamp. "
                "The metadata clock was wrong or the timestamp was altered.",
                tags=["enf", "time", "inconsistency"],
            )

        ctx.publish(
            self.name,
            {
                "present": True,
                "source": "light",
                "dated_start": start,
                "confident": confident,
            },
        )

        return True

    async def run(self, ctx: CaseContext, out: Reporter) -> None:
        out.progress(0.05, "decoding audio")
        recording = await ctx.audio()

        if recording is None:
            if not await self._from_light(ctx, out):
                raise Skip("no audio stream and no lighting flicker")

            return

        samples, rate = recording.samples, recording.sample_rate
        out.progress(0.2, "searching 50/60 Hz families")
        profile = await asyncio.to_thread(hum.detect, samples, rate)

        if isinstance(profile, ForensicError):
            reason = str(profile)
            out.finding(
                "info",
                "No mains hum in the audio",
                "Could not find a stable 50/60 Hz hum or harmonic"
                + (" (clip too short, under 16 s)" if reason == "too_short" else "")
                + ". Phone microphones filter low frequencies and many scenes have no "
                "powered equipment nearby. The LIGHT module looks for the grid in lamp flicker instead.",
                tags=["enf"],
                data={"reason": reason},
            )

            if not await self._from_light(ctx, out):
                ctx.publish(self.name, {"present": False})

            return

        out.artifact(
            "candidates",
            "bars",
            "Hum candidates (SNR dB)",
            data={
                "labels": list(profile.candidates),
                "values": list(profile.candidates.values()),
            },
        )
        out.progress(0.35, f"tracking {profile.nominal_hz:.0f} Hz x{profile.harmonic}")
        trace = await asyncio.to_thread(hum.extract, samples, rate, profile)

        if trace.masked.mean() > hum.MAX_MASKED_FRACTION:
            out.finding(
                "notice",
                "Hum present but mostly masked",
                f"A {profile.nominal_hz:.0f} Hz family hum is there (SNR {profile.snr_db} dB), but other "
                "sounds own its band most of the time, so it cannot be tracked reliably.",
                tags=["enf"],
            )
            ctx.publish(self.name, {"present": True, "usable": False})

            return

        offset_hz = float(np.median(trace.values)) - profile.nominal_hz

        if abs(offset_hz) > 0.2:
            out.finding(
                "info",
                f"Tone near {profile.nominal_hz:.0f} Hz is not mains hum",
                f"A steady tone sits at {np.median(trace.values):.3f} Hz, {offset_hz:+.3f} Hz from the "
                "nominal grid frequency. Power grids stay within about ±0.2 Hz, so this is a motor, fan "
                "or other machine, not the grid.",
                tags=["enf", "tone"],
                data={"median_hz": float(np.median(trace.values))},
            )
            ctx.publish(self.name, {"present": False})

            return

        family = (
            "GB/Europe/Asia/Africa (50 Hz)"
            if profile.nominal_hz == 50
            else "the Americas / parts of Asia (60 Hz)"
        )
        out.finding(
            "critical",
            f"Mains hum locked: {profile.nominal_hz:.0f} Hz grid",
            f"Hum at harmonic x{profile.harmonic} with {profile.snr_db} dB SNR. Recorded on a "
            f"{profile.nominal_hz:.0f} Hz grid: {family}. Mean {trace.values.mean():.4f} Hz.",
            tags=["enf", "location"],
            data={
                "nominal_hz": profile.nominal_hz,
                "harmonic": profile.harmonic,
                "snr_db": profile.snr_db,
            },
        )
        trace_times = trace.first_s + np.arange(len(trace.values)) * trace.step_s
        series: dict[str, Any] = {"hum_hz": trace.values}
        published: dict[str, Any] = {
            "present": True,
            "usable": True,
            "nominal_hz": profile.nominal_hz,
            "mean_hz": float(trace.values.mean()),
            "trace": {"t": trace_times, "f": trace.values},
        }

        out.progress(0.5, "phase continuity")
        splices = await asyncio.to_thread(hum.find_splices, samples, rate, trace)

        for splice in splices:
            out.finding(
                "critical",
                f"Hum phase jumps at {splice.time_s:.1f} s",
                f"The hum's phase steps by {np.degrees(splice.phase_step_rad):+.0f}°. A continuous "
                "recording keeps the phase locked to the grid; a jump means audio was cut here.",
                time_s=splice.time_s,
                tags=["splice", "enf"],
                data={"phase_step_rad": splice.phase_step_rad},
            )

        published["splices"] = [s.time_s for s in splices]

        # ---- reference --------------------------------------------------------
        reference = None
        source = None

        if path := ctx.options.get("grid_path"):
            loaded = await grid.load_csv(path)
            reference = loaded if isinstance(loaded, grid.GridSeries) else None
            source = f"supplied grid file {path}"
        elif profile.nominal_hz == 50.0:
            provenance = await ctx.wait_for("provenance") or {}
            centre = provenance.get("start_guess")

            if isinstance(centre, str):
                centre = datetime.fromisoformat(centre)

            if centre is not None:
                hours = _SEARCH_HOURS
                out.progress(0.6, "fetching GB grid record")
                reference, source = await grid_fetch.fetch(
                    centre - timedelta(hours=hours), centre + timedelta(hours=hours)
                )

        if reference is None:
            out.finding(
                "notice",
                "Hum captured; no grid record to compare",
                "Give an approximate date (or a grid CSV) to date this recording to the second."
                if profile.nominal_hz == 50
                else "Automatic grid records cover the GB grid only.",
                tags=["enf"],
            )
            out.artifact(
                "trace",
                "series",
                "Hum frequency",
                data=downsample_series(trace_times, series)
                | {"x_unit": "s", "y_unit": "Hz"},
            )
            ctx.publish(self.name, published)

            return

        out.progress(0.75, "matching against grid")
        published["grid_source"] = source
        duration = len(samples) / rate
        match = grid.match(trace, reference)
        level = _level_match(trace, reference)
        start = None
        confident = False

        if isinstance(match, grid.TimeMatch) and match.confident:
            start, confident = match.recording_start, True
            detail = (
                f"Hum wander correlates {match.correlation:.2f} with the grid "
                f"(next best elsewhere {match.runner_up:.2f})."
            )
        elif level is not None:
            start = level["start"]
            confident = (
                level["rms_hz"] < 0.006
                and level["runner_up_rms_hz"] > 1.8 * level["rms_hz"]
            )
            detail = (
                f"Hum level matches the grid to {level['rms_hz'] * 1000:.1f} mHz RMS "
                f"(next best elsewhere {level['runner_up_rms_hz'] * 1000:.1f} mHz)."
            )
        else:
            detail = "No usable match."

        if start is not None:
            grid_times = (
                reference.start.timestamp()
                + np.arange(len(reference.values)) * reference.step_s
                - start.timestamp()
            )
            inside = (grid_times >= -30) & (grid_times <= duration + 30)
            out.artifact(
                "trace",
                "series",
                "Hum vs grid frequency",
                data={
                    "x": np.round(trace_times, 3),
                    "series": [
                        {
                            "name": "hum (this recording)",
                            "y": np.round(trace.values, 5),
                        },
                        {
                            "name": "grid (" + (source or "reference") + ")",
                            "x": np.round(grid_times[inside], 3),
                            "y": np.round(reference.values[inside], 5),
                        },
                    ],
                    "x_unit": "s",
                    "y_unit": "Hz",
                },
            )

            if level is not None:
                curve_times = [
                    (
                        reference.start + timedelta(seconds=i * reference.step_s)
                    ).isoformat()
                    for i in range(
                        0,
                        len(level["rms_curve"]),
                        max(1, len(level["rms_curve"]) // 1500),
                    )
                ]
                out.artifact(
                    "search",
                    "series",
                    "Grid search: mismatch by candidate start time",
                    data={
                        "x_iso": curve_times,
                        "series": [
                            {
                                "name": "rms_hz",
                                "y": level["rms_curve"][
                                    :: max(1, len(level["rms_curve"]) // 1500)
                                ],
                            }
                        ],
                    },
                )

            candidates = (await ctx.wait_for("provenance") or {}).get(
                "start_candidates"
            ) or []
            drift = min(
                ((start - c).total_seconds() for c in candidates), key=abs, default=None
            )
            published.update({"dated_start": start, "confident": confident})

            if confident:
                out.finding(
                    "critical",
                    f"ENF dates recording to {start:%Y-%m-%d %H:%M:%S} UTC",
                    detail
                    + f" Source: {source}."
                    + (
                        f" Metadata agrees to {drift:+.0f} s."
                        if drift is not None and abs(drift) <= 120
                        else f" Metadata differs by {drift:+.0f} s."
                        if drift is not None
                        else ""
                    ),
                    tags=["enf", "time"],
                    data={"start": start, "metadata_diff_s": drift, "source": source},
                )
            else:
                out.finding(
                    "notice",
                    "No convincing grid match in the search window",
                    detail
                    + f" Source: {source}. Either the recording lies outside the window, "
                    "the hum is too weak, or the grid record does not cover it.",
                    tags=["enf", "time"],
                    data={"best_start": start, "source": source},
                )

            if confident and drift is not None and abs(drift) > 120:
                out.finding(
                    "critical",
                    "Metadata time contradicted by the grid",
                    f"The hum says the recording began {abs(drift) / 60:.1f} minutes "
                    f"{'after' if drift > 0 else 'before'} the time in the file's metadata.",
                    tags=["enf", "time", "inconsistency"],
                )

        segments = grid.match_segments(trace, reference) if duration >= 120 else []

        if segments:
            starts = sorted(
                (
                    (s.match.recording_start.timestamp(), s)
                    for s in segments
                    if s.match.correlation > 0.7
                ),
                key=lambda pair: pair[0],
            )
            clusters: list[list[tuple[float, Any]]] = []

            for moment, segment in starts:
                if clusters and moment - clusters[-1][-1][0] <= 2.0:
                    clusters[-1].append((moment, segment))
                else:
                    clusters.append([(moment, segment)])

            supported = sorted(
                (c for c in clusters if len(c) >= 2), key=len, reverse=True
            )
            centre = (
                supported[0][0][0]
                if supported
                else np.median([s.match.recording_start.timestamp() for s in segments])
            )
            out.artifact(
                "segments",
                "series",
                "Per-window dating (offset from main cluster, s)",
                data={
                    "x": [s.offset_s + s.duration_s / 2 for s in segments],
                    "series": [
                        {
                            "name": "offset_s",
                            "y": [
                                s.match.recording_start.timestamp() - centre
                                for s in segments
                            ],
                        }
                    ],
                },
            )

            if len(supported) >= 2:
                windows = [
                    [
                        min(seg.offset_s for _, seg in c),
                        datetime.fromtimestamp(c[0][0], tz=timezone.utc),
                    ]
                    for c in supported
                ]
                out.finding(
                    "critical",
                    f"Audio pieces date to {len(supported)} different times",
                    "Windows of this audio, dated independently against the grid, form separate "
                    "groups: "
                    + "; ".join(
                        f"from {w[0]:.0f} s -> {w[1]:%H:%M:%S}" for w in windows
                    )
                    + ". A single take dates consistently; this was assembled from recordings "
                    "made at different times.",
                    time_s=float(windows[1][0]),
                    tags=["splice", "enf"],
                    data={"groups": windows},
                )
            elif supported:
                out.finding(
                    "info",
                    "All windows date consistently",
                    f"{len(supported[0])} of {len(segments)} windows agree on one start time.",
                    tags=["enf"],
                )

        ctx.publish(self.name, published)
        out.progress(1.0)
