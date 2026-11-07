import asyncio
import csv
from dataclasses import dataclass
from datetime import datetime
from datetime import timedelta
from datetime import timezone

import numpy as np
from scipy import signal

from earwitness._common import ForensicError
from earwitness._common import OnSuccess
from earwitness.hum import HumTrace

_MIN_QUERY_POINTS = 12


class GridError(ForensicError):
    EMPTY = "empty"
    UNPARSEABLE = "unparseable"
    QUERY_TOO_SHORT = "query_too_short"
    REFERENCE_TOO_SHORT = "reference_too_short"


@dataclass(frozen=True, slots=True)
class GridSeries:
    """Grid frequency sampled every step_s seconds from start (UTC)."""

    start: datetime
    step_s: float
    values: np.ndarray


@dataclass(frozen=True, slots=True)
class TimeMatch:
    recording_start: datetime
    correlation: float
    z_score: float
    runner_up: float
    scores: np.ndarray

    @property
    def confident(self) -> bool:
        # NOTE: The z-score is reported but not used: neighbouring offsets of a
        # slowly varying series are correlated, which keeps z modest even for
        # clean matches. The gap to the best non-overlapping offset is the test.
        return self.correlation >= 0.8 and self.correlation - self.runner_up >= 0.2


def _parse_time(text: str) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(text.strip())
    # NOTE: datetime offers no non-raising parser.
    except ValueError:
        return None

    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)

    return parsed.astimezone(timezone.utc)


def _read_rows(path: str) -> list[list[str]]:
    with open(path, newline="") as handle:
        return list(csv.reader(handle))


def _regularise(times: np.ndarray, values: np.ndarray) -> tuple[float, np.ndarray]:
    step = float(np.median(np.diff(times)))
    grid = np.arange(times[0], times[-1] + step / 2, step)

    return step, np.interp(grid, times, values)


async def load_csv(path: str) -> OnSuccess[GridSeries]:
    """Load any CSV whose first two columns are an ISO time and a frequency.

    Works for NESO monthly files and for this tool's own reference output. Times
    without an offset are taken as UTC (NESO publishes in GMT).
    """
    rows = await asyncio.to_thread(_read_rows, path)
    times: list[float] = []
    values: list[float] = []

    for row in rows:
        if len(row) < 2:
            continue

        moment = _parse_time(row[0])

        if moment is None:
            continue

        times.append(moment.timestamp())
        values.append(float(row[1]))

    if len(times) < 2:
        return GridError.UNPARSEABLE if rows else GridError.EMPTY

    order = np.argsort(times)
    step, regular = _regularise(np.asarray(times)[order], np.asarray(values)[order])

    return GridSeries(
        start=datetime.fromtimestamp(times[order[0]], tz=timezone.utc),
        step_s=step,
        values=regular,
    )


def _write_rows(path: str, series: GridSeries) -> None:
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["time", "frequency"])

        for index, value in enumerate(series.values):
            moment = series.start + timedelta(seconds=index * series.step_s)
            writer.writerow([moment.isoformat(), f"{value:.5f}"])


async def write_csv(path: str, series: GridSeries) -> None:
    await asyncio.to_thread(_write_rows, path, series)


def from_trace(trace: HumTrace, recording_start: datetime) -> GridSeries:
    return GridSeries(
        start=recording_start + timedelta(seconds=trace.first_s),
        step_s=trace.step_s,
        values=trace.values,
    )


def _correlate(query: np.ndarray, reference: np.ndarray) -> np.ndarray:
    n = len(query)
    q = (query - query.mean()) / query.std()
    ref = reference - reference.mean()
    numerator = signal.fftconvolve(ref, q[::-1], mode="valid")
    c1 = np.concatenate(([0.0], np.cumsum(ref)))
    c2 = np.concatenate(([0.0], np.cumsum(ref * ref)))
    s1 = c1[n:] - c1[:-n]
    s2 = c2[n:] - c2[:-n]
    std = np.sqrt(np.maximum(s2 / n - (s1 / n) ** 2, 1e-18))

    return numerator / (n * std)


def match(trace: HumTrace, reference: GridSeries) -> OnSuccess[TimeMatch]:
    block = max(1, int(round(reference.step_s / trace.step_s)))
    usable = len(trace.values) // block * block
    query = trace.values[:usable].reshape(-1, block).mean(axis=1)

    if len(query) < _MIN_QUERY_POINTS:
        return GridError.QUERY_TOO_SHORT

    if len(reference.values) < len(query):
        return GridError.REFERENCE_TOO_SHORT

    scores = _correlate(query, reference.values)
    best = int(np.argmax(scores))
    masked = scores.copy()
    masked[max(0, best - len(query)) : best + len(query)] = -np.inf
    runner_up = float(masked.max()) if np.isfinite(masked).any() else float("nan")
    first_block_centre = trace.first_s + (block - 1) * trace.step_s / 2
    start = reference.start + timedelta(
        seconds=best * reference.step_s - first_block_centre,
    )

    return TimeMatch(
        recording_start=start,
        correlation=float(scores[best]),
        z_score=float((scores[best] - scores.mean()) / scores.std()),
        runner_up=runner_up,
        scores=scores,
    )


@dataclass(frozen=True, slots=True)
class SegmentMatch:
    offset_s: float
    duration_s: float
    match: TimeMatch


def match_segments(
    trace: HumTrace,
    reference: GridSeries,
    *,
    window_s: float = 90.0,
    hop_s: float = 30.0,
) -> list[SegmentMatch]:
    """Date overlapping windows independently; an edited clip dates inconsistently."""
    width = int(window_s / trace.step_s)
    hop = int(hop_s / trace.step_s)
    segments: list[SegmentMatch] = []

    for start in range(0, len(trace.values) - width + 1, hop):
        part = HumTrace(
            values=trace.values[start : start + width],
            masked=trace.masked[start : start + width],
            first_s=trace.first_s + start * trace.step_s,
            step_s=trace.step_s,
            profile=trace.profile,
        )
        found = match(part, reference)

        if isinstance(found, TimeMatch):
            segments.append(SegmentMatch(start * trace.step_s, window_s, found))

    return segments
