import os
from datetime import datetime
from datetime import timedelta
from datetime import timezone

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from earwitness._common import ForensicError
from earwitness.analysis import Analysis
from earwitness.analysis import Placement
from earwitness.grid import TimeMatch

_CLUSTER_TOLERANCE_S = 2.0
_MIN_CLUSTER_WINDOWS = 2


def _cluster_starts(starts: list[float]) -> list[tuple[float, int]]:
    clusters: list[list[float]] = []

    for start in sorted(starts):
        if clusters and start - clusters[-1][-1] <= _CLUSTER_TOLERANCE_S:
            clusters[-1].append(start)
        else:
            clusters.append([start])

    ranked = sorted(clusters, key=len, reverse=True)

    return [(float(np.median(c)), len(c)) for c in ranked]


def _hum_section(result: Analysis) -> dict[str, object]:
    profile = result.hum_profile

    if isinstance(profile, ForensicError):
        return {"present": False, "reason": str(profile)}

    return {
        "present": True,
        "nominal_hz": profile.nominal_hz,
        "harmonic": profile.harmonic,
        "snr_db": profile.snr_db,
        "grid_family": "50 Hz (GB, Europe, most of Asia/Africa)"
        if profile.nominal_hz == 50.0
        else "60 Hz (Americas, parts of Asia)",
        "candidates_snr_db": profile.candidates,
        "masked_percent": None
        if result.hum_trace is None
        else round(100 * float(result.hum_trace.masked.mean()), 1),
    }


def _time_section(result: Analysis) -> dict[str, object]:
    section: dict[str, object] = {
        "source": result.start_source,
        "start": result.start.isoformat() if result.start else None,
    }
    match = result.time_match

    if isinstance(match, TimeMatch):
        section["hum_match"] = {
            "recording_start": match.recording_start.isoformat(),
            "correlation": round(match.correlation, 3),
            "z_score": round(match.z_score, 1),
            "runner_up": round(match.runner_up, 3),
            "confident": match.confident,
        }
    elif isinstance(match, ForensicError):
        section["hum_match"] = {"error": str(match)}

    section["segments"] = [
        {
            "from_s": s.offset_s,
            "to_s": s.offset_s + s.duration_s,
            "dated_start": s.match.recording_start.isoformat(),
            "correlation": round(s.match.correlation, 3),
        }
        for s in result.segments
    ]
    clusters = _cluster_starts(
        [s.match.recording_start.timestamp() for s in result.segments]
    )
    supported = [c for c in clusters if c[1] >= _MIN_CLUSTER_WINDOWS]
    section["dated_clusters"] = [
        {"start": datetime.fromtimestamp(t, tz=timezone.utc).isoformat(), "windows": n}
        for t, n in clusters
    ]
    # Heuristic: an unedited clip's windows agree on one start time (single
    # windows can still be wrong); two well-supported times suggest an edit.
    section["consistent"] = None if not supported else len(supported) == 1

    return section


def _placement_section(
    placement: Placement | ForensicError | None,
) -> dict[str, object] | None:
    if placement is None:
        return None

    if isinstance(placement, ForensicError):
        return {"error": str(placement)}

    return {
        "latitude": round(placement.latitude, 6),
        "longitude": round(placement.longitude, 6),
        "ambiguous_mirror": placement.ambiguous,
        "error_vs_metadata_m": None
        if placement.error_vs_metadata_m is None
        else round(placement.error_vs_metadata_m, 1),
        "aircraft": [
            {
                "track": a.track.identifier,
                "callsign": a.track.callsign,
                "fit": round(a.fit, 4),
                "span_unix": [
                    round(float(a.curve.times[0]), 1),
                    round(float(a.curve.times[-1]), 1),
                ],
                "start_hz": round(float(a.curve.frequencies[0]), 1),
            }
            for a in placement.assignments
        ],
    }


def to_dict(result: Analysis) -> dict[str, object]:
    metadata = result.recording.metadata

    return {
        "file": result.path,
        "duration_s": round(
            len(result.recording.samples) / result.recording.sample_rate, 2
        ),
        "metadata": {
            "creation_time": metadata.creation_time.isoformat()
            if metadata.creation_time
            else None,
            "start_if_creation_time_is_end": None
            if metadata.creation_time is None
            else (metadata.creation_time - timedelta(seconds=metadata.duration_s))
            .replace(microsecond=0)
            .isoformat(),
            "filename_time_local": metadata.filename_time.isoformat()
            if metadata.filename_time
            else None,
            "location": None
            if metadata.location is None
            else [metadata.location.latitude, metadata.location.longitude],
        },
        "hum": _hum_section(result),
        "splices": [
            {"time_s": round(s.time_s, 1), "phase_step_rad": round(s.phase_step_rad, 2)}
            for s in result.splices
        ],
        "time": _time_section(result),
        "flyovers": [
            {
                "start_s": round(f.span[0], 1),
                "end_s": round(f.span[1], 1),
                "start_hz": round(float(f.frequencies[0]), 1),
                "end_hz": round(float(f.frequencies[-1]), 1),
                "drop_percent": round(100 * (1 - np.exp(-f.drop)), 1),
                "salience_db": round(f.salience_db, 1),
            }
            for f in result.flyovers
        ],
        "location": _placement_section(result.placement),
    }


def _plot_hum(result: Analysis, path: str) -> None:
    trace = result.hum_trace

    if trace is None:
        return

    figure, axis = plt.subplots(figsize=(10, 3.5))
    seconds = trace.first_s + np.arange(len(trace.values)) * trace.step_s
    axis.plot(seconds, trace.values, label="Recording", linewidth=1.2)
    match = result.time_match
    reference = result.reference

    if isinstance(match, TimeMatch) and reference is not None:
        offset = (match.recording_start - reference.start).total_seconds()
        index = np.arange(len(reference.values))
        reference_seconds = index * reference.step_s - offset
        shown = (reference_seconds >= seconds[0]) & (reference_seconds <= seconds[-1])
        axis.plot(
            reference_seconds[shown],
            reference.values[shown],
            label=f"Grid reference (r={match.correlation:.2f})",
            linewidth=1.0,
            alpha=0.8,
        )

    for splice in result.splices:
        axis.axvline(splice.time_s, color="red", linestyle="--", linewidth=0.8)

    axis.set_xlabel("Seconds into recording")
    axis.set_ylabel("Grid frequency (Hz)")
    axis.legend(loc="upper right")
    figure.tight_layout()
    figure.savefig(path, dpi=130)
    plt.close(figure)


def _plot_spectrogram(result: Analysis, path: str) -> None:
    spectrogram = result.spectrogram
    figure, axis = plt.subplots(figsize=(10, 4))
    axis.pcolormesh(
        spectrogram.times_s,
        spectrogram.frequencies,
        spectrogram.whitened_db,
        shading="auto",
        cmap="magma",
        vmin=0,
        vmax=20,
    )

    for flyover in result.flyovers:
        axis.plot(flyover.times_s, flyover.frequencies, color="cyan", linewidth=1.2)

    axis.set_yscale("log")
    axis.set_xlabel("Seconds into recording")
    axis.set_ylabel("Frequency (Hz)")
    figure.tight_layout()
    figure.savefig(path, dpi=130)
    plt.close(figure)


def _plot_location(result: Analysis, path: str) -> None:
    placement = result.placement

    if not isinstance(placement, Placement):
        return

    location = placement.location
    figure, axis = plt.subplots(figsize=(7, 7))
    axis.contourf(
        location.map_east / 1000,
        location.map_north / 1000,
        np.log10(location.cost_map),
        levels=30,
        cmap="viridis_r",
    )
    assigned = {a.track.identifier for a in placement.assignments}

    for track in result.tracks:
        highlighted = track.identifier in assigned
        axis.plot(
            track.positions[:, 0] / 1000,
            track.positions[:, 1] / 1000,
            color="white" if highlighted else "grey",
            linewidth=1.5 if highlighted else 0.4,
            alpha=1.0 if highlighted else 0.5,
        )

    axis.plot(
        location.east / 1000,
        location.north / 1000,
        "r*",
        markersize=16,
        label="Estimate",
    )
    axis.set_xlim(location.map_east.min() / 1000, location.map_east.max() / 1000)
    axis.set_ylim(location.map_north.min() / 1000, location.map_north.max() / 1000)
    axis.set_xlabel("East of origin (km)")
    axis.set_ylabel("North of origin (km)")
    axis.set_aspect("equal")
    axis.legend(loc="upper right")
    figure.tight_layout()
    figure.savefig(path, dpi=130)
    plt.close(figure)


def plot(result: Analysis, directory: str) -> list[str]:
    os.makedirs(directory, exist_ok=True)
    paths = {
        "hum.png": _plot_hum,
        "spectrogram.png": _plot_spectrogram,
        "location.png": _plot_location,
    }
    written: list[str] = []

    for name, draw in paths.items():
        target = os.path.join(directory, name)

        if os.path.exists(target):
            os.remove(target)

        draw(result, target)

        if os.path.exists(target):
            written.append(target)

    return written
