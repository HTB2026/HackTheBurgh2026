from dataclasses import dataclass

import numpy as np
from scipy import signal

_WINDOW = 8_192
_HOP = 2_000
_LOW_HZ = 60.0
_HIGH_HZ = 2_000.0
_LOG_BINS = 480
_SEARCH_WINDOW_S = 60.0
_SEARCH_HOP_S = 15.0
_MAX_DOWN_BINS = 2
_MAX_UP_BINS = 1
_JUMP_PENALTY_DB = 1.5
_ACTIVE_DB = 6.0
_MAX_MEAN_STEP_BINS = 0.5
_MIN_ACTIVE_S = 10.0
_MIN_DROP = 0.02
_MIN_SALIENCE_DB = 6.0
_MAX_OVERLAP = 0.1


@dataclass(frozen=True, slots=True)
class Spectrogram:
    times_s: np.ndarray
    frequencies: np.ndarray
    whitened_db: np.ndarray


@dataclass(frozen=True, slots=True)
class Flyover:
    """A tone whose pitch falls over time, as a passing aircraft's does."""

    times_s: np.ndarray
    frequencies: np.ndarray
    salience_db: float
    drop: float

    @property
    def span(self) -> tuple[float, float]:
        return float(self.times_s[0]), float(self.times_s[-1])


def _log_resample(
    linear_freqs: np.ndarray, db: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    targets = np.geomspace(_LOW_HZ, _HIGH_HZ, _LOG_BINS)
    upper = np.clip(np.searchsorted(linear_freqs, targets), 1, len(linear_freqs) - 1)
    lower = upper - 1
    span = linear_freqs[upper] - linear_freqs[lower]
    weight = (targets - linear_freqs[lower]) / span

    return targets, db[lower] * (1 - weight[:, None]) + db[upper] * weight[:, None]


def compute_spectrogram(samples: np.ndarray, sample_rate: int) -> Spectrogram:
    window = signal.windows.hann(_WINDOW)
    transform = signal.ShortTimeFFT(window, _HOP, fs=sample_rate, scale_to="magnitude")
    magnitude = np.abs(transform.stft(samples))
    times = transform.t(len(samples))
    inside = (times >= 0) & (times <= len(samples) / sample_rate)
    db = 20 * np.log10(magnitude[:, inside] + 1e-12)
    freqs, log_db = _log_resample(transform.f, db)
    # Removing each bin's typical level drops stationary tones (hum, fans);
    # removing each frame's typical level drops broadband bursts.
    whitened = log_db - np.median(log_db, axis=1, keepdims=True)
    whitened -= np.median(whitened, axis=0, keepdims=True)

    return Spectrogram(times[inside], freqs, np.clip(whitened, -10.0, 30.0))


def _viterbi(emission: np.ndarray) -> np.ndarray:
    frames, bins = emission.shape
    score = emission[0].copy()
    back = np.zeros((frames, bins), dtype=np.int8)

    for t in range(1, frames):
        best = np.full(bins, -np.inf)
        choice = np.zeros(bins, dtype=np.int8)

        for delta in range(-_MAX_DOWN_BINS, _MAX_UP_BINS + 1):
            candidate = np.full(bins, -np.inf)

            if delta >= 0:
                candidate[delta:] = score[: bins - delta]
            else:
                candidate[:delta] = score[-delta:]

            candidate -= _JUMP_PENALTY_DB * abs(delta)
            better = candidate > best
            best[better] = candidate[better]
            choice[better] = delta

        score = best + emission[t]
        back[t] = choice

    path = np.empty(frames, dtype=np.int64)
    path[-1] = int(np.argmax(score))

    for t in range(frames - 1, 0, -1):
        path[t - 1] = path[t] - back[t, path[t]]

    return path


def _longest_run(mask: np.ndarray) -> tuple[int, int]:
    best = (0, 0)
    start = None

    for index, value in enumerate([*mask, False]):
        if value and start is None:
            start = index
        elif not value and start is not None:
            if index - start > best[1] - best[0]:
                best = (start, index)

            start = None

    return best


def _refine(emission: np.ndarray, path: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    rows = np.arange(len(path))
    inner = np.clip(path, 1, emission.shape[1] - 2)
    a = emission[rows, inner - 1]
    b = emission[rows, inner]
    c = emission[rows, inner + 1]
    denominator = a - 2 * b + c
    offset = np.where(
        denominator < 0,
        0.5 * (a - c) / np.where(denominator < 0, denominator, -1.0),
        0.0,
    )
    ratio = freqs[1] / freqs[0]

    return freqs[inner] * ratio ** np.clip(offset, -0.5, 0.5)


def _trace_window(spectrogram: Spectrogram, start: int, stop: int) -> Flyover | None:
    emission = spectrogram.whitened_db[:, start:stop].T
    path = _viterbi(emission)
    along = emission[np.arange(len(path)), path]
    frame_s = float(np.median(np.diff(spectrogram.times_s)))
    kernel = np.ones(8) / 8
    smooth = np.convolve(along, kernel, mode="same")
    # A real tone moves slowly across bins; a path through noise hops about.
    steps = np.abs(np.diff(path, prepend=path[0]))
    steady = np.convolve(steps, kernel, mode="same") <= _MAX_MEAN_STEP_BINS
    run_start, run_stop = _longest_run((smooth >= _ACTIVE_DB) & steady)

    if (run_stop - run_start) * frame_s < _MIN_ACTIVE_S:
        return None

    frequencies = _refine(emission, path, spectrogram.frequencies)[run_start:run_stop]
    head = np.median(np.log(frequencies[:16]))
    tail = np.median(np.log(frequencies[-16:]))
    salience = float(along[run_start:run_stop].mean())

    if head - tail < _MIN_DROP or salience < _MIN_SALIENCE_DB:
        return None

    return Flyover(
        times_s=spectrogram.times_s[start:stop][run_start:run_stop],
        frequencies=frequencies,
        salience_db=salience,
        drop=float(head - tail),
    )


def _overlap(a: Flyover, b: Flyover) -> float:
    (a0, a1), (b0, b1) = a.span, b.span
    shared = max(0.0, min(a1, b1) - max(a0, b0))

    return shared / min(a1 - a0, b1 - b0)


def find_flyovers(spectrogram: Spectrogram) -> list[Flyover]:
    times = spectrogram.times_s
    frame_s = float(np.median(np.diff(times)))
    width = int(_SEARCH_WINDOW_S / frame_s)
    hop = int(_SEARCH_HOP_S / frame_s)
    starts = range(0, max(1, len(times) - width + 1), hop)
    found = [
        flyover
        for start in starts
        if (flyover := _trace_window(spectrogram, start, start + width)) is not None
    ]
    kept: list[Flyover] = []

    for flyover in sorted(found, key=lambda f: f.salience_db, reverse=True):
        if all(_overlap(flyover, other) < _MAX_OVERLAP for other in kept):
            kept.append(flyover)

    return sorted(kept, key=lambda f: f.span[0])


def resample(flyover: Flyover, *, step_s: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Average log-frequency into step_s bins to keep the location search cheap."""
    edges = np.arange(flyover.times_s[0], flyover.times_s[-1] + step_s, step_s)
    index = np.digitize(flyover.times_s, edges) - 1
    log_f = np.log(flyover.frequencies)
    occupied = np.unique(index)
    times = np.array([flyover.times_s[index == i].mean() for i in occupied])
    values = np.array([np.exp(log_f[index == i].mean()) for i in occupied])

    return times, values
