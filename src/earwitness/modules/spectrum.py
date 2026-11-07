"""What the audio holds that the ear skips over.

Spectrogram for the UI, plus detectors for things a listener would never
notice: steady tones (machinery, electronics, ultrasonic beacons), keypad
(DTMF) tones, inserted digital silence, abrupt changes of room tone, the
bandwidth left by earlier lossy encodes, and the falling pitch of passing
vehicles and aircraft.
"""

import asyncio
from typing import Any

import matplotlib
import numpy as np
from PIL import Image
from scipy import ndimage
from scipy import signal

from earwitness import tones
from earwitness.modules.base import CaseContext
from earwitness.modules.base import Reporter
from earwitness.modules.base import Skip
from earwitness.modules.base import downsample_series

_DTMF_LOW = (697, 770, 852, 941)
_DTMF_HIGH = (1209, 1336, 1477, 1633)
_DTMF_KEYS = ["123A", "456B", "789C", "*0#D"]
_LUT = (matplotlib.colormaps["magma"](np.linspace(0, 1, 256))[:, :3] * 255).astype(
    np.uint8
)


def _render(db: np.ndarray, path: str, *, floor: float = 70.0) -> tuple[float, float]:
    """Save a spectrogram (freq x time, dB) as a PNG, low frequencies at the bottom."""
    top = float(np.percentile(db, 99.7))
    low = top - floor
    scaled = np.clip((db - low) / floor, 0, 1)
    image = _LUT[(scaled * 255).astype(np.uint8)][::-1]
    Image.fromarray(image).save(path, optimize=True)

    return low, top


def _stft_image(
    x: np.ndarray,
    rate: int,
    *,
    max_columns: int,
    nperseg: int,
    f_max: float | None,
    rows: int,
):
    hop = max(nperseg // 4, int(np.ceil(len(x) / max_columns)))
    freqs, times, z = signal.stft(
        x, fs=rate, nperseg=nperseg, noverlap=nperseg - hop, boundary=None, padded=False
    )
    power = 20 * np.log10(np.abs(z) + 1e-10)

    if f_max is not None:
        keep = freqs <= f_max
        freqs, power = freqs[keep], power[keep]

    if len(freqs) > rows:
        edges = np.linspace(0, len(freqs), rows + 1).astype(int)
        power = np.stack(
            [power[a:b].max(axis=0) for a, b in zip(edges[:-1], edges[1:])]
        )
        freqs = np.array([freqs[a] for a in edges[:-1]])

    return freqs, times, power


def _bandwidth(x: np.ndarray, rate: int) -> float | None:
    freqs, psd = signal.welch(x, fs=rate, nperseg=4096)
    db = 10 * np.log10(psd + 1e-20)
    smooth = ndimage.uniform_filter1d(db, 9)
    loud = np.percentile(smooth[(freqs > 300) & (freqs < 4000)], 50)

    # Cutoff: the last frequency within 50 dB of the speech band, followed by a cliff.
    above = np.flatnonzero(smooth > loud - 50)

    if not len(above):
        return None

    cutoff_index = above[-1]

    if cutoff_index >= len(freqs) - 8:
        return None

    tail = smooth[cutoff_index + 4 : cutoff_index + 40]

    if len(tail) and smooth[cutoff_index] - tail.mean() < 15:
        return None

    return float(freqs[cutoff_index])


def _tones(x: np.ndarray, rate: int) -> list[dict[str, Any]]:
    nperseg = int(2 ** np.ceil(np.log2(rate / 2)))  # ~2 Hz bins
    freqs, times, z = signal.stft(x, fs=rate, nperseg=nperseg, noverlap=nperseg // 2)
    power = np.abs(z) ** 2 + 1e-20
    db = 10 * np.log10(power)
    background = ndimage.median_filter(db, size=(41, 1), mode="nearest")
    excess = db - background
    persistent = (excess > 10).mean(axis=1)
    strength = np.median(excess, axis=1)
    found = []
    peaks, _ = signal.find_peaks(
        np.where(persistent > 0.6, strength, 0), height=8, distance=5
    )

    for index in peaks:
        f = float(freqs[index])

        if f < 40:
            continue

        found.append(
            {
                "hz": f,
                "excess_db": float(strength[index]),
                "presence": float(persistent[index]),
            }
        )

    return sorted(found, key=lambda t: -t["excess_db"])[:12]


def _classify_tone(hz: float) -> str:
    for nominal in (50, 60):
        k = hz / nominal

        if abs(k - round(k)) * nominal < 1.0 and round(k) <= 12:
            return f"mains harmonic ({nominal} Hz x{round(k)})"

    if 15_600 <= hz <= 15_800:
        return "line-scan whine (CRT / old TV, 15.7 kHz)"

    if hz >= 17_500:
        return "near-ultrasonic tone: tracking beacon, pest repeller, sensor or device"

    if hz >= 8_000:
        return "high-pitched electronic whine (switching power supply, coil)"

    if hz < 300:
        return "low drone (motor, fan, compressor, engine)"

    return "steady tone (machinery, electronics, alarm)"


def _digital_silence(x: np.ndarray, rate: int) -> list[tuple[float, float]]:
    zero = np.abs(x) < 1e-7
    labels, count = ndimage.label(zero)

    if count == 0:
        return []

    runs = ndimage.find_objects(labels)
    out = []
    edge = int(0.25 * rate)

    for run in runs:
        start, stop = run[0].start, run[0].stop

        if stop - start >= 0.02 * rate and start > edge and stop < len(x) - edge:
            out.append((start / rate, (stop - start) / rate))

    return out


def _floor_steps(
    x: np.ndarray, rate: int
) -> tuple[np.ndarray, np.ndarray, list[dict[str, float]]]:
    frame = int(0.05 * rate)
    n = len(x) // frame
    rms = np.sqrt((x[: n * frame].reshape(n, frame) ** 2).mean(axis=1) + 1e-14)
    db = 20 * np.log10(rms)
    per_s = 20
    seconds = n // per_s

    if seconds < 8:
        return np.array([]), np.array([]), []

    floor = np.array(
        [np.percentile(db[i * per_s : (i + 1) * per_s], 10) for i in range(seconds)]
    )
    steps = []
    side = 4

    def spread(values: np.ndarray) -> float:
        return float(np.subtract(*np.percentile(values, [75, 25])))

    for i in range(side, seconds - side + 1):
        before, after = floor[i - side : i], floor[i : i + side]
        jump = np.median(after) - np.median(before)

        if abs(jump) > 8 and spread(before) < 4 and spread(after) < 4:
            if not steps or i - steps[-1]["index"] > side:
                steps.append({"index": i, "time_s": float(i), "jump_db": float(jump)})
            elif abs(jump) > abs(steps[-1]["jump_db"]):
                steps[-1] = {"index": i, "time_s": float(i), "jump_db": float(jump)}

    # Fine check at 0.1 s: a splice switches between two steady backgrounds in
    # an instant. A loud event starting (fireworks, applause, an engine) is
    # preceded by bursts and followed by a fluctuating level, so reject those.
    times = np.arange(n) * 0.05
    confirmed = []

    for step in steps:
        centres = np.arange(step["time_s"] - 1.5, step["time_s"] + 1.5, 0.1)
        fine = np.array(
            [
                np.percentile(db[(times >= c - 0.25) & (times < c + 0.25)], 10)
                if ((times >= c - 0.25) & (times < c + 0.25)).any()
                else np.nan
                for c in centres
            ]
        )
        jumps = np.abs(np.diff(fine))
        at = (
            float(centres[int(np.nanargmax(jumps)) + 1])
            if np.isfinite(jumps).any()
            else step["time_s"]
        )
        side_before = db[(times >= at - 2.2) & (times < at - 0.3)]
        side_after = db[(times >= at + 0.3) & (times < at + 2.2)]

        def steady(values: np.ndarray) -> bool:
            if len(values) < 10:
                return False

            windows = [
                np.percentile(values[k : k + 10], 10)
                for k in range(0, len(values) - 9, 2)
            ]

            return float(np.ptp(windows)) < 4.0

        if steady(side_before) and steady(side_after):
            confirmed.append(step | {"time_s": round(at, 2)})

    return np.arange(seconds) + 0.5, floor, confirmed


def _goertzel(frames: np.ndarray, rate: int, freq: float) -> np.ndarray:
    n = frames.shape[1]
    k = 2 * np.cos(2 * np.pi * freq / rate)
    s1 = np.zeros(len(frames))
    s2 = np.zeros(len(frames))

    for i in range(n):
        s0 = frames[:, i] + k * s1 - s2
        s2, s1 = s1, s0

    return s1**2 + s2**2 - k * s1 * s2


def _dtmf(x: np.ndarray, rate: int) -> list[dict[str, Any]]:
    if rate != 8_000:
        x = signal.resample_poly(x, 8_000, rate)
        rate = 8_000

    frame = 320  # 40 ms
    n = len(x) // frame

    if n < 3:
        return []

    frames = x[: n * frame].reshape(n, frame) * np.hanning(frame)
    total = (frames**2).sum(axis=1) * frame + 1e-12
    low = np.stack([_goertzel(frames, rate, f) for f in _DTMF_LOW], axis=1)
    high = np.stack([_goertzel(frames, rate, f) for f in _DTMF_HIGH], axis=1)
    keys = []

    for i in range(n):
        lo, hi = np.argsort(low[i])[::-1], np.argsort(high[i])[::-1]
        a, b = low[i, lo[0]], high[i, hi[0]]
        ok = (
            a > 8 * low[i, lo[1]]
            and b > 8 * high[i, hi[1]]
            and (a + b) / total[i] > 0.12
            and 0.1 < a / b < 10
        )
        keys.append(_DTMF_KEYS[lo[0]][hi[0]] if ok else None)

    found = []
    i = 0

    while i < n:
        if keys[i] is None:
            i += 1

            continue

        j = i

        while j < n and keys[j] == keys[i]:
            j += 1

        if j - i >= 2:
            found.append(
                {
                    "key": keys[i],
                    "time_s": i * frame / rate,
                    "duration_s": (j - i) * frame / rate,
                }
            )

        i = j

    return found


def _onset_envelope(x: np.ndarray, rate: int) -> tuple[np.ndarray, np.ndarray]:
    """Spectral flux at 100 Hz: how much new sound starts at each instant."""
    hop = rate // 100
    freqs, times, z = signal.stft(x, fs=rate, nperseg=hop * 4, noverlap=hop * 3)
    mag = np.log1p(np.abs(z) * 100)
    flux = np.maximum(0, np.diff(mag, axis=1)).sum(axis=0)

    return times[1:], flux


class Spectrum:
    name = "spectrum"
    title = "SPECTRAL ANALYSIS"
    description = "Spectrogram, hidden tones, ultrasonics, DTMF, digital silence, room-tone breaks, Doppler sweeps."

    async def run(self, ctx: CaseContext, out: Reporter) -> None:
        out.progress(0.05, "decoding audio at native rate")
        native = await ctx.audio_native()

        if native is None:
            raise Skip("no audio stream")

        x, rate = native
        duration = len(x) / rate
        published: dict[str, Any] = {}

        if duration < 0.5 or np.abs(x).max() < 1e-6:
            out.finding(
                "notice",
                "Silent audio track",
                "The audio track carries no signal.",
                tags=["audio"],
            )
            ctx.publish(self.name, published)

            return

        out.progress(0.15, "rendering spectrograms")
        freqs, times, power = await asyncio.to_thread(
            _stft_image, x, rate, max_columns=2400, nperseg=2048, f_max=None, rows=512
        )
        low_db, top_db = _render(power, out.path("spectrogram.png"))
        out.artifact(
            "spectrogram",
            "spectrogram",
            "Full-band spectrogram",
            file=out.path("spectrogram.png"),
            data={
                "t0": 0.0,
                "t1": duration,
                "f0": 0.0,
                "f1": rate / 2,
                "scale": "linear",
                "db": [low_db, top_db],
            },
        )
        recording = await ctx.audio()

        if recording is not None:
            f2, t2, p2 = await asyncio.to_thread(
                _stft_image, recording.samples, recording.sample_rate,
                max_columns=2400, nperseg=8192, f_max=1_000, rows=400,
            )  # fmt: skip
            _render(p2, out.path("spectrogram_low.png"), floor=60)
            out.artifact(
                "spectrogram_low",
                "spectrogram",
                "Low band 0–1 kHz (hum, engines, motors)",
                file=out.path("spectrogram_low.png"),
                data={
                    "t0": 0.0,
                    "t1": duration,
                    "f0": 0.0,
                    "f1": 1000.0,
                    "scale": "linear",
                },
            )

        # Waveform envelope for the timeline.
        bins = min(2000, len(x))
        edges = np.linspace(0, len(x), bins + 1).astype(int)
        peak = np.array(
            [
                np.abs(x[a:b]).max() if b > a else 0
                for a, b in zip(edges[:-1], edges[1:])
            ]
        )
        out.artifact(
            "waveform",
            "waveform",
            "Waveform",
            data={"t1": duration, "peak": np.round(peak, 4)},
        )

        hop = max(1, rate // 10)
        count = len(x) // hop
        loud = 20 * np.log10(
            np.sqrt((x[: count * hop].reshape(count, hop) ** 2).mean(axis=1)) + 1e-7
        )
        out.artifact(
            "loudness",
            "series",
            "Loudness over time (dBFS, 0.1 s steps)",
            data=downsample_series(np.arange(count) * hop / rate, {"loudness_db": loud})
            | {"x_unit": "s", "y_unit": "dBFS"},
        )
        out.progress(0.35, "bandwidth")
        cutoff = await asyncio.to_thread(_bandwidth, x, rate)
        published["cutoff_hz"] = cutoff

        if cutoff is not None and cutoff < rate / 2 * 0.9:
            label = (
                "narrowband: telephone, VoIP or a voice-note codec" if cutoff < 8_500 else
                "low-bitrate lossy encode, typical of messaging-app or social re-encodes" if cutoff < 15_500 else
                "standard lossy encoder low-pass"
            )  # fmt: skip
            out.finding(
                "notice" if cutoff < 15_500 else "info",
                f"Audio band-limited at {cutoff / 1000:.1f} kHz",
                f"The file is stored at {rate} Hz, so it could hold up to {rate / 2000:.1f} kHz, "
                f"but nothing exists above {cutoff / 1000:.1f} kHz: {label}. "
                "A cliff below the container's limit is left by an earlier, lower-quality encode.",
                tags=["audio", "compression"],
                data={"cutoff_hz": cutoff, "rate": rate},
            )

        out.progress(0.45, "hunting steady tones")
        found = await asyncio.to_thread(_tones, x, rate)
        published["tones"] = found

        for tone in found[:6]:
            kind = _classify_tone(tone["hz"])
            ultrasonic = tone["hz"] >= 17_500
            out.finding(
                "warning" if ultrasonic else "notice",
                f"Hidden tone at {tone['hz']:,.0f} Hz",
                f"{kind.capitalize()}. Present {tone['presence'] * 100:.0f}% of the time, "
                f"{tone['excess_db']:.0f} dB above the surrounding noise"
                + (" and above the range most adults can hear." if ultrasonic else "."),
                tags=["tone", "audio"] + (["ultrasonic"] if ultrasonic else []),
                data=tone | {"kind": kind},
            )

        out.progress(0.55, "digital silence")
        silences = await asyncio.to_thread(_digital_silence, x, rate)

        if silences:
            out.finding(
                "critical",
                f"{len(silences)} stretch{'es' if len(silences) > 1 else ''} of perfect digital silence",
                "Exact-zero samples in the middle of the recording, longest "
                f"{max(s[1] for s in silences) * 1000:.0f} ms. A microphone always picks up some noise. "
                "Perfect silence is pasted in by editing software (muted words, gaps between joined clips).",
                time_s=silences[0][0],
                tags=["splice", "audio"],
                data={"runs": silences[:50]},
            )

        out.progress(0.65, "room tone continuity")
        floor_t, floor_db, steps = await asyncio.to_thread(_floor_steps, x, rate)
        published["floor_steps"] = steps

        if len(floor_t):
            out.artifact(
                "noise_floor",
                "series",
                "Background noise floor (dBFS)",
                data=downsample_series(floor_t, {"floor_db": floor_db})
                | {"x_unit": "s", "y_unit": "dB"},
            )

        for step in steps[:8]:
            out.finding(
                "warning",
                f"Room tone jumps {step['jump_db']:+.0f} dB at {step['time_s']:.0f} s",
                "The background noise is steady, then switches to a different steady level "
                "within a second. Rooms rarely do that; joined recordings do. Check for a video "
                "cut at the same moment.",
                time_s=step["time_s"],
                tags=["splice", "audio"],
                data=step,
            )

        out.progress(0.75, "keypad tones")
        keys = await asyncio.to_thread(_dtmf, x, rate)

        if keys:
            digits = "".join(k["key"] for k in keys)
            out.finding(
                "critical",
                f"Keypad tones decoded: {digits}",
                f"{len(keys)} DTMF tone pairs, the sounds a phone keypad makes, decoded from the audio. "
                "These can reveal a dialled number or a door/alarm code.",
                time_s=keys[0]["time_s"],
                tags=["dtmf", "audio"],
                data={"keys": keys},
            )

        clipped = float(np.mean(np.abs(x) > 0.995))

        if clipped > 0.001:
            out.finding(
                "info",
                f"Clipping on {clipped * 100:.2f}% of samples",
                "The signal hit the recorder's ceiling: very loud sounds, or gain applied afterwards.",
                tags=["audio"],
            )

        out.progress(0.85, "Doppler sweeps")

        if recording is not None and duration > 30:
            spectrogram = await asyncio.to_thread(
                tones.compute_spectrogram, recording.samples, recording.sample_rate
            )
            flyovers = await asyncio.to_thread(tones.find_flyovers, spectrogram)
            published["flyovers"] = [
                {"start_s": f.span[0], "end_s": f.span[1], "drop": f.drop}
                for f in flyovers
            ]

            for fly in flyovers:
                out.finding(
                    "notice",
                    f"Doppler sweep {fly.frequencies[0]:.0f}→{fly.frequencies[-1]:.0f} Hz",
                    f"A tone falls {fly.drop * 100:.0f}% in pitch between {fly.span[0]:.0f} and {fly.span[1]:.0f} s, "
                    "the signature of a source passing by (aircraft, vehicle). With an ADS-B log the "
                    "Doppler curve pins down where the recorder stood.",
                    time_s=fly.span[0],
                    time_end_s=fly.span[1],
                    tags=["doppler", "audio"],
                    data={"t": fly.times_s[::4], "f": fly.frequencies[::4]},
                )

        onset_t, onset = await asyncio.to_thread(_onset_envelope, x, rate)
        published["onsets"] = {"t": onset_t, "v": onset}
        ctx.publish(self.name, published)
        out.progress(1.0)
