import math
from dataclasses import dataclass
from datetime import datetime
from datetime import timedelta

import numpy as np

from earwitness import doppler
from earwitness import flights
from earwitness import grid
from earwitness import hum
from earwitness import media
from earwitness import tones
from earwitness._common import ForensicError
from earwitness._common import OnSuccess
from earwitness.doppler import Assignment
from earwitness.doppler import Curve
from earwitness.doppler import Location
from earwitness.flights import Origin
from earwitness.grid import TimeMatch
from earwitness.hum import HumProfile
from earwitness.hum import HumTrace
from earwitness.hum import Splice
from earwitness.media import Recording
from earwitness.tones import Flyover
from earwitness.tones import Spectrogram

_PARALLEL_DEGREES = 30.0


@dataclass(frozen=True, slots=True)
class Options:
    grid_path: str | None
    flight_path: str | None
    origin: Origin | None
    ground_height_m: float
    start_override: datetime | None
    half_width_m: float


@dataclass(frozen=True, slots=True)
class Placement:
    location: Location
    latitude: float
    longitude: float
    assignments: list[Assignment]
    ambiguous: bool
    error_vs_metadata_m: float | None


@dataclass(frozen=True, slots=True)
class Analysis:
    path: str
    recording: Recording
    hum_profile: HumProfile | ForensicError
    hum_trace: HumTrace | None
    splices: list[Splice]
    reference: grid.GridSeries | None
    time_match: TimeMatch | ForensicError | None
    segments: list[grid.SegmentMatch]
    start: datetime | None
    start_source: str | None
    spectrogram: Spectrogram
    flyovers: list[Flyover]
    tracks: list[flights.Track]
    placement: Placement | ForensicError | None


def _choose_start(
    recording: Recording,
    match: TimeMatch | ForensicError | None,
    override: datetime | None,
) -> tuple[datetime | None, str | None]:
    if override is not None:
        return override, "override"

    if isinstance(match, TimeMatch) and match.confident:
        return match.recording_start, "hum"

    if recording.metadata.creation_time is not None:
        return recording.metadata.creation_time, "metadata"

    return None, None


def _heading(assignment: Assignment) -> float:
    middle = assignment.curve.times.mean()
    velocity = np.array(
        [
            np.interp(middle, *_velocity_series(assignment.track, axis))
            for axis in (0, 1)
        ],
    )

    return math.degrees(math.atan2(velocity[0], velocity[1])) % 180.0


def _velocity_series(track: flights.Track, axis: int) -> tuple[np.ndarray, np.ndarray]:
    return track.times, np.gradient(track.positions[:, axis], track.times)


def _is_ambiguous(assignments: list[Assignment]) -> bool:
    headings = [_heading(a) for a in assignments]

    for first in headings:
        for second in headings:
            difference = abs(first - second)

            if min(difference, 180.0 - difference) >= _PARALLEL_DEGREES:
                return False

    return True


async def _place(
    recording: Recording,
    flyovers: list[Flyover],
    start: datetime,
    flight_path: str,
    origin: Origin,
    options: Options,
) -> OnSuccess[tuple[list[flights.Track], Placement | None]]:
    tracks = await flights.load_log(
        flight_path,
        origin,
        ground_height_m=options.ground_height_m,
    )

    if isinstance(tracks, ForensicError):
        return tracks

    base = start.timestamp()
    curves = [
        Curve(base + times, frequencies)
        for times, frequencies in (tones.resample(f) for f in flyovers)
    ]
    solution = doppler.solve(curves, tracks, half_width=options.half_width_m)

    if solution is None:
        return tracks, None

    assignments, location = solution
    latitude, longitude = flights.to_geographic(
        origin,
        location.east,
        location.north,
    )
    error = None

    if (truth := recording.metadata.location) is not None:
        east, north = flights.to_local(origin, truth.latitude, truth.longitude)
        error = math.hypot(east - location.east, north - location.north)

    return tracks, Placement(
        location=location,
        latitude=latitude,
        longitude=longitude,
        assignments=assignments,
        ambiguous=_is_ambiguous(assignments),
        error_vs_metadata_m=error,
    )


async def analyse(path: str, options: Options) -> OnSuccess[Analysis]:
    recording = await media.load(path)

    if isinstance(recording, ForensicError):
        return recording

    profile = hum.detect(recording.samples, recording.sample_rate)
    trace = None
    splices: list[Splice] = []
    reference = None
    match: TimeMatch | ForensicError | None = None

    if isinstance(profile, HumProfile):
        extracted = hum.extract(recording.samples, recording.sample_rate, profile)

        if extracted.masked.mean() > hum.MAX_MASKED_FRACTION:
            profile = hum.HumError.NOT_PRESENT
        else:
            trace = extracted
            splices = hum.find_splices(recording.samples, recording.sample_rate, trace)

    if options.grid_path is not None:
        loaded = await grid.load_csv(options.grid_path)
        reference = loaded if isinstance(loaded, grid.GridSeries) else None
        match = loaded if isinstance(loaded, ForensicError) else None

    segments: list[grid.SegmentMatch] = []

    if reference is not None and trace is not None and trace.profile.nominal_hz == 50.0:
        match = grid.match(trace, reference)
        segments = grid.match_segments(trace, reference)

    start, source = _choose_start(recording, match, options.start_override)
    spectrogram = tones.compute_spectrogram(recording.samples, recording.sample_rate)
    flyovers = tones.find_flyovers(spectrogram)
    tracks: list[flights.Track] = []
    placement: Placement | ForensicError | None = None

    flight_path, origin = options.flight_path, options.origin

    if (
        flight_path is not None
        and origin is not None
        and start is not None
        and flyovers
    ):
        placed = await _place(recording, flyovers, start, flight_path, origin, options)

        if isinstance(placed, ForensicError):
            placement = placed
        else:
            tracks, placement = placed

    return Analysis(
        path=path,
        recording=recording,
        hum_profile=profile,
        hum_trace=trace,
        splices=splices,
        reference=reference,
        time_match=match,
        segments=segments,
        start=start,
        start_source=source,
        spectrogram=spectrogram,
        flyovers=flyovers,
        tracks=tracks,
        placement=placement,
    )


def recording_end(analysis: Analysis) -> datetime | None:
    if analysis.start is None:
        return None

    duration = len(analysis.recording.samples) / analysis.recording.sample_rate

    return analysis.start + timedelta(seconds=duration)
