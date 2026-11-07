from dataclasses import dataclass

import numpy as np

from earwitness.flights import Track

SPEED_OF_SOUND = 337.0

_EMISSION_ITERATIONS = 6
_GRID_POINTS = 81
_REFINE_STEPS = 4
_REFINE_FACTOR = 6.0
# Largest accepted RMS misfit in log-frequency (about 1.5% in pitch).
_MAX_RMS = 0.015


@dataclass(frozen=True, slots=True)
class Curve:
    """Observed tone frequency at arrival times (UNIX seconds)."""

    times: np.ndarray
    frequencies: np.ndarray


@dataclass(frozen=True, slots=True)
class Assignment:
    curve: Curve
    track: Track
    fit: float


@dataclass(frozen=True, slots=True)
class Location:
    east: float
    north: float
    cost: float
    map_east: np.ndarray
    map_north: np.ndarray
    cost_map: np.ndarray


def _position_and_velocity(
    track: Track, times: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    flat = times.ravel()
    velocity_samples = np.gradient(track.positions, track.times, axis=0)
    position = np.stack(
        [np.interp(flat, track.times, track.positions[:, i]) for i in range(3)],
        axis=-1,
    )
    velocity = np.stack(
        [np.interp(flat, track.times, velocity_samples[:, i]) for i in range(3)],
        axis=-1,
    )

    return position.reshape(*times.shape, 3), velocity.reshape(*times.shape, 3)


def propagate(
    track: Track,
    receivers: np.ndarray,
    arrival_times: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Doppler factor f_obs / f0 and emission distance, shaped (receivers, times)."""
    arrival = np.broadcast_to(arrival_times, (len(receivers), len(arrival_times)))
    emission = arrival.copy()

    for _ in range(_EMISSION_ITERATIONS):
        position, _ = _position_and_velocity(track, emission)
        distance = np.linalg.norm(position - receivers[:, None, :], axis=-1)
        emission = arrival - distance / SPEED_OF_SOUND

    position, velocity = _position_and_velocity(track, emission)
    offset = position - receivers[:, None, :]
    distance = np.linalg.norm(offset, axis=-1)
    radial = np.einsum("gtk,gtk->gt", offset, velocity) / distance

    return SPEED_OF_SOUND / (SPEED_OF_SOUND + radial), distance


def _residual_variance(curve: Curve, track: Track, receivers: np.ndarray) -> np.ndarray:
    factor, _ = propagate(track, receivers, curve.times)
    residual = np.log(curve.frequencies)[None, :] - np.log(factor)

    return residual.var(axis=1)


def _grid(
    centre: tuple[float, float], half_width: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    axis = np.linspace(-half_width, half_width, _GRID_POINTS)
    east, north = np.meshgrid(centre[0] + axis, centre[1] + axis)
    receivers = np.stack([east.ravel(), north.ravel(), np.zeros(east.size)], axis=-1)

    return east, north, receivers


def _covers(track: Track, curve: Curve, *, lead_s: float) -> bool:
    return bool(
        track.times[0] <= curve.times[0] - lead_s
        and track.times[-1] >= curve.times[-1],
    )


def _closest_horizontal(track: Track, curve: Curve) -> float:
    window = (track.times >= curve.times[0] - 60) & (track.times <= curve.times[-1])

    if not window.any():
        return float("inf")

    return float(np.linalg.norm(track.positions[window, :2], axis=1).min())


def _candidates(curve: Curve, tracks: list[Track], *, half_width: float) -> list[Track]:
    return [
        track
        for track in tracks
        if _covers(track, curve, lead_s=30.0)
        and _closest_horizontal(track, curve) <= half_width + 5_000
    ]


def _squared_error(curve: Curve, track: Track, receivers: np.ndarray) -> np.ndarray:
    """Sum of squared log-frequency residuals, so long, steep curves weigh most."""
    return _residual_variance(curve, track, receivers) * len(curve.times)


def _refine(
    assignments: list[Assignment], centre: tuple[float, float], width: float
) -> tuple[tuple[float, float], float]:
    best_cost = float("inf")

    for _ in range(_REFINE_STEPS):
        _, _, local = _grid(centre, width)
        values = sum(_squared_error(a.curve, a.track, local) for a in assignments)
        index = int(np.argmin(values))
        centre = (float(local[index, 0]), float(local[index, 1]))
        best_cost = float(values[index])
        width /= _REFINE_FACTOR

    return centre, best_cost


def solve(
    curves: list[Curve],
    tracks: list[Track],
    *,
    half_width: float,
) -> tuple[list[Assignment], Location] | None:
    """Jointly choose the receiver position and which aircraft made each curve.

    Any single curve can be fitted by almost any aircraft from some position, so
    curves are only assigned at the position where all of them fit at once:
    cost(r) = sum over curves of the best-fitting aircraft's residual at r.
    """
    map_east, map_north, receivers = _grid((0.0, 0.0), half_width)
    usable: list[tuple[Curve, list[Track], np.ndarray]] = []

    for curve in curves:
        candidates = _candidates(curve, tracks, half_width=half_width)

        if not candidates:
            continue

        residuals = np.stack([_squared_error(curve, t, receivers) for t in candidates])
        usable.append((curve, candidates, residuals))

    if not usable:
        return None

    joint = sum(residuals.min(axis=0) for _, _, residuals in usable)
    best = int(np.argmin(joint))
    assignments: list[Assignment] = []

    for curve, candidates, residuals in usable:
        choice = int(residuals[:, best].argmin())
        variance = float(residuals[choice, best]) / len(curve.times)

        if variance**0.5 > _MAX_RMS:
            continue

        fit = 1 - variance / float(np.log(curve.frequencies).var())
        assignments.append(Assignment(curve, candidates[choice], fit))

    if not assignments:
        return None

    centre, cost = _refine(
        assignments,
        (float(receivers[best, 0]), float(receivers[best, 1])),
        half_width / _REFINE_FACTOR,
    )
    location = Location(
        east=centre[0],
        north=centre[1],
        cost=cost,
        map_east=map_east,
        map_north=map_north,
        cost_map=joint.reshape(map_east.shape),
    )

    return assignments, location
