from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from scipy import signal

from earwitness._common import ForensicError
from earwitness._common import OnSuccess

_HUM_SAMPLE_RATE = 1_000
_NOMINALS = (50.0, 60.0)
_HARMONICS = (1, 2, 3, 4)
_PRESENT_SNR_DB = 6.0
# Real hum stands out at one grid frequency; broadband noise scores both alike.
_FAMILY_MARGIN_DB = 3.0
# Above this share of masked frames the trace is mostly not hum.
MAX_MASKED_FRACTION = 0.5
_FRAME_S = 8.0
_HOP_S = 1.0
_ZOOM_POINTS = 2_048
_CHUNK_FRAMES = 256
_STABLE_RESULTANT = 0.9
_SPLICE_WINDOW = 100
_MASK_LOUDNESS_RATIO = 4.0
_MASK_DEVIATION_HZ = 0.2
_MASK_MEDIAN_S = 31
_MASK_MARGIN_S = 3


class HumError(ForensicError):
    TOO_SHORT = "too_short"
    NOT_PRESENT = "not_present"


@dataclass(frozen=True, slots=True)
class HumProfile:
    nominal_hz: float
    harmonic: int
    snr_db: float
    candidates: dict[str, float]


@dataclass(frozen=True, slots=True)
class HumTrace:
    """Fundamental frequency estimates, one per step_s, the first at first_s."""

    values: np.ndarray
    masked: np.ndarray
    first_s: float
    step_s: float
    profile: HumProfile


@dataclass(frozen=True, slots=True)
class Splice:
    time_s: float
    phase_step_rad: float


def _downsample(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    return signal.resample_poly(samples, _HUM_SAMPLE_RATE, sample_rate)


def _band_snr_db(freqs: np.ndarray, psd: np.ndarray, centre: float, k: int) -> float:
    peak_band = np.abs(freqs - centre) <= 0.6 * k
    offset = np.abs(freqs - centre)
    noise_band = (offset >= 1.5 * k) & (offset <= 5.0 * k)

    return float(10 * np.log10(psd[peak_band].max() / np.median(psd[noise_band])))


def detect(samples: np.ndarray, sample_rate: int) -> OnSuccess[HumProfile]:
    x = _downsample(samples, sample_rate)

    if len(x) < _FRAME_S * _HUM_SAMPLE_RATE * 2:
        return HumError.TOO_SHORT

    segment = min(16_384, len(x))
    freqs, psd = signal.welch(x, fs=_HUM_SAMPLE_RATE, nperseg=segment)
    candidates: dict[str, float] = {}

    for nominal in _NOMINALS:
        for k in _HARMONICS:
            snr = _band_snr_db(freqs, psd, nominal * k, k)
            candidates[f"{nominal:.0f}x{k}"] = round(snr, 1)

    best = max(candidates, key=candidates.__getitem__)
    nominal_text, harmonic_text = best.split("x")
    profile = HumProfile(
        nominal_hz=float(nominal_text),
        harmonic=int(harmonic_text),
        snr_db=candidates[best],
        candidates=candidates,
    )

    other = max(
        snr
        for name, snr in candidates.items()
        if not name.startswith(nominal_text + "x")
    )

    if profile.snr_db < _PRESENT_SNR_DB or profile.snr_db - other < _FAMILY_MARGIN_DB:
        return HumError.NOT_PRESENT

    return profile


def _parabolic_offset(spectrum: np.ndarray, peak: np.ndarray) -> np.ndarray:
    rows = np.arange(len(peak))
    inner = np.clip(peak, 1, spectrum.shape[1] - 2)
    a = np.log(spectrum[rows, inner - 1] + 1e-20)
    b = np.log(spectrum[rows, inner] + 1e-20)
    c = np.log(spectrum[rows, inner + 1] + 1e-20)
    denominator = a - 2 * b + c
    safe = np.where(denominator == 0, 1.0, denominator)

    return inner + np.where(denominator == 0, 0.0, 0.5 * (a - c) / safe)


def extract(samples: np.ndarray, sample_rate: int, profile: HumProfile) -> HumTrace:
    x = _downsample(samples, sample_rate)
    k = profile.harmonic
    centre = profile.nominal_hz * k
    sos = signal.butter(
        4,
        [centre - 1.0 * k, centre + 1.0 * k],
        btype="bandpass",
        fs=_HUM_SAMPLE_RATE,
        output="sos",
    )
    x = signal.sosfiltfilt(sos, x)
    frame = int(_FRAME_S * _HUM_SAMPLE_RATE)
    hop = int(_HOP_S * _HUM_SAMPLE_RATE)
    frames = np.lib.stride_tricks.sliding_window_view(x, frame)[::hop]
    low, high = centre - 0.6 * k, centre + 0.6 * k
    zoom = signal.ZoomFFT(frame, [low, high], m=_ZOOM_POINTS, fs=_HUM_SAMPLE_RATE)
    window = signal.windows.hann(frame)
    bin_width = (high - low) / _ZOOM_POINTS
    estimates: list[np.ndarray] = []
    loudness: list[np.ndarray] = []

    for start in range(0, len(frames), _CHUNK_FRAMES):
        spectrum = np.abs(zoom(frames[start : start + _CHUNK_FRAMES] * window))
        peak = spectrum.argmax(axis=1)
        position = _parabolic_offset(spectrum, peak)
        estimates.append((low + position * bin_width) / k)
        loudness.append(spectrum[np.arange(len(peak)), peak])

    values = np.concatenate(estimates)
    masked = _mask(values, np.concatenate(loudness))
    filled = values.copy()

    if masked.any() and not masked.all():
        index = np.arange(len(values))
        filled[masked] = np.interp(index[masked], index[~masked], values[~masked])

    return HumTrace(
        values=filled,
        masked=masked,
        first_s=_FRAME_S / 2,
        step_s=_HOP_S,
        profile=profile,
    )


def _mask(values: np.ndarray, loudness: np.ndarray) -> np.ndarray:
    """Flag frames where something other than the hum owns the band.

    Grid events change the hum's frequency, not its loudness, so a peak far
    louder or quieter than usual means another tone (an aircraft, a motor) or
    masking noise. Wild jumps from the local median are flagged as well.
    """
    typical = np.median(loudness)
    ratio = loudness / typical
    local = ndimage.median_filter(values, size=_MASK_MEDIAN_S, mode="nearest")
    flagged = (
        (ratio > _MASK_LOUDNESS_RATIO)
        | (ratio < 1 / _MASK_LOUDNESS_RATIO)
        | (np.abs(values - local) > _MASK_DEVIATION_HZ)
    )

    return ndimage.binary_dilation(flagged, iterations=_MASK_MARGIN_S)


def _wrap(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2 * np.pi) - np.pi


def find_splices(
    samples: np.ndarray,
    sample_rate: int,
    trace: HumTrace,
    *,
    min_step_rad: float = 0.6,
    robust_z: float = 5.0,
) -> list[Splice]:
    """Find sudden steps in the hum's phase, which an unedited recording lacks.

    The hum's phase is predicted from the frequency trace; what is left drifts
    slowly unless the audio was cut. A cut whose phase step happens to be under
    min_step_rad (roughly one cut in five) is not detected.
    """
    k = trace.profile.harmonic
    centre = trace.profile.nominal_hz * k
    x = _downsample(samples, sample_rate)
    sos = signal.butter(
        4,
        [centre - 1.0 * k, centre + 1.0 * k],
        btype="bandpass",
        fs=_HUM_SAMPLE_RATE,
        output="sos",
    )
    phase = np.unwrap(np.angle(signal.hilbert(signal.sosfiltfilt(sos, x))))
    seconds = np.arange(len(x)) / _HUM_SAMPLE_RATE
    trace_seconds = trace.first_s + np.arange(len(trace.values)) * trace.step_s
    expected = (
        2 * np.pi * np.cumsum(k * np.interp(seconds, trace_seconds, trace.values))
    )
    residual = (phase - expected / _HUM_SAMPLE_RATE)[:: _HUM_SAMPLE_RATE // 100]
    half = _SPLICE_WINDOW

    if len(residual) <= 4 * half:
        return []

    # Circular means over one-second windows either side of each instant. The
    # resultant length is near 1 only where the hum dominates the band, so
    # windows where something else masks it are not trusted.
    phasor = np.concatenate(([0j], np.cumsum(np.exp(1j * residual))))
    window_means = (phasor[half:] - phasor[:-half]) / half
    # A half-second gap either side skips the band-pass filter's smearing of a cut.
    gap = half // 2
    before = window_means[: -(half + 2 * gap)]
    after = window_means[half + 2 * gap :]
    steps = np.angle(after * np.conj(before))
    trusted = (np.abs(before) >= _STABLE_RESULTANT) & (
        np.abs(after) >= _STABLE_RESULTANT
    )
    deviation = np.median(np.abs(steps[trusted])) * 1.4826 if trusted.any() else 0.0
    threshold = max(min_step_rad, robust_z * deviation)
    # Ignore the filter's start-up and run-out at the edges of the recording.
    edge = 3 * half
    flagged = trusted & (np.abs(steps) > threshold)
    centre_s = (np.arange(len(steps)) + half + gap) / 100
    trace_index = np.clip(
        np.round((centre_s - trace.first_s) / trace.step_s).astype(int),
        0,
        len(trace.masked) - 1,
    )
    flagged &= ~trace.masked[trace_index]
    flagged[:edge] = False
    flagged[-edge:] = False
    candidates = np.flatnonzero(flagged)
    splices: list[Splice] = []
    previous = -2 * half

    for index in candidates:
        splice = Splice((index + half + gap) / 100, float(steps[index]))

        if index - previous > half:
            splices.append(splice)
        elif abs(splice.phase_step_rad) > abs(splices[-1].phase_step_rad):
            splices[-1] = splice

        previous = index

    return splices
