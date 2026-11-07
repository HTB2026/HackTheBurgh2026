"""Synthetic evidence with known, planted anomalies (ground truth for tests/demos).

`earwitness demo --out demo/` writes a demo pack; `build_pack` documents what is
hidden in each clip so a presenter knows what the suite should find."""

import os
import subprocess
import tempfile

import numpy as np
from scipy.io import wavfile

RATE = 48_000
DTMF = {
    "1": (697, 1209),
    "9": (852, 1477),
    "8": (852, 1336),
    "4": (770, 1209),
    "#": (941, 1477),
}
PLANTED = {
    "ultrasonic_hz": 18_500,
    "dtmf": "1984#",
    "dtmf_at_s": 4.0,
    "silence_at_s": 12.0,
    "silence_s": 0.3,
    "room_jump_at_s": 20.0,
    "cut_at_s": 20.0,
    "pulse_bpm": 72.0,
}


def planted_audio(duration: float = 30.0, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(int(duration * RATE)) / RATE
    noise_level = np.where(t < PLANTED["room_jump_at_s"], 0.003, 0.03)
    x = rng.normal(0, 1, len(t)) * noise_level
    x += 0.004 * np.sin(2 * np.pi * PLANTED["ultrasonic_hz"] * t)
    x += 0.02 * np.sin(2 * np.pi * 440 * t) * (np.sin(2 * np.pi * 0.3 * t) > 0.6)

    cursor = PLANTED["dtmf_at_s"]

    for key in PLANTED["dtmf"]:
        low, high = DTMF[key]
        span = (t >= cursor) & (t < cursor + 0.15)
        x[span] += 0.2 * (
            np.sin(2 * np.pi * low * t[span]) + np.sin(2 * np.pi * high * t[span])
        )
        cursor += 0.3

    start = int(PLANTED["silence_at_s"] * RATE)
    x[start : start + int(PLANTED["silence_s"] * RATE)] = 0.0

    return x.astype(np.float32)


def planted_video(path: str, duration: float = 30.0) -> str:
    """A video with a hard cut, a pulsing 'face' patch and the planted audio."""
    fps = 30
    width, height = 320, 240
    n = int(duration * fps)
    rng = np.random.default_rng(2)
    frames = np.empty((n, height, width, 3), dtype=np.uint8)
    yy, xx = np.mgrid[0:height, 0:width]

    for i in range(n):
        t = i / fps
        scene_b = t >= PLANTED["cut_at_s"]
        base = np.zeros((height, width, 3), dtype=np.float32)
        base[..., 0] = 60 + 40 * (xx / width) + (80 if scene_b else 0)
        base[..., 1] = 80 + 50 * (yy / height)
        base[..., 2] = 120 - (60 if scene_b else 0)
        # A slowly moving block, so frames differ.
        x0 = int(40 + 100 * (0.5 + 0.5 * np.sin(t)))
        base[100:140, x0 : x0 + 40] = 200
        # Skin-coloured patch whose redness pulses at the planted heart rate.
        pulse = 1.5 * np.sin(2 * np.pi * PLANTED["pulse_bpm"] / 60 * t)
        face = (slice(20, 90), slice(200, 280))
        base[face] = [190 + pulse, 140 + pulse * 0.6, 120]
        base += rng.normal(0, 1.5, base.shape)
        frames[i] = np.clip(base, 0, 255).astype(np.uint8)

    with tempfile.TemporaryDirectory() as tmp:
        wav = os.path.join(tmp, "a.wav")
        wavfile.write(wav, RATE, planted_audio(duration))
        process = subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", f"{width}x{height}", "-r", str(fps), "-i", "-", "-i", wav,
             "-c:v", "libx264", "-preset", "veryfast", "-crf", "16", "-pix_fmt", "yuv420p",
             "-c:a", "aac", "-b:a", "256k", "-shortest", path],
            input=frames.tobytes(),
            check=False,
        )  # fmt: skip

    if process.returncode != 0:
        raise RuntimeError("ffmpeg failed to build the fixture")

    return path


def flicker_video(
    path: str,
    *,
    grid_hz: float = 50.0,
    fps: float = 30.0,
    duration: float = 12.0,
    readout_s: float = 0.02,
    depth: float = 0.04,
) -> str:
    """A static indoor scene lit by mains lamps, filmed with a rolling shutter."""
    width, height = 320, 240
    n = int(duration * fps)
    rng = np.random.default_rng(3)
    yy, xx = np.mgrid[0:height, 0:width]
    scene = 90 + 60 * np.sin(xx / 25.0) * np.cos(yy / 30.0)
    row_time = (np.arange(height) / height * readout_s)[:, None]
    frames = np.empty((n, height, width, 3), dtype=np.uint8)
    # Grid frequency wanders a little, as real grids do.
    mains = grid_hz + 0.05 * np.sin(2 * np.pi * np.arange(n) / fps / 9.0)
    phase = 0.0

    for i in range(n):
        t = i / fps
        flicker = 1 + depth * np.sin(2 * np.pi * 2 * mains[i] * (t + row_time) + phase)
        image = scene * flicker + rng.normal(0, 1.0, scene.shape)
        frames[i] = np.clip(image, 0, 255).astype(np.uint8)[..., None]

    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "14", "-pix_fmt", "yuv420p", path],
        input=frames.tobytes(), check=True,
    )  # fmt: skip

    return path


def av_events_video(
    path: str, *, audio_offset_s: float = 0.0, duration: float = 20.0
) -> str:
    """Random flashes with a click at each; audio optionally shifted."""
    fps, width, height = 30, 160, 120
    rng = np.random.default_rng(4)
    events = np.sort(rng.uniform(1, duration - 1, 18))
    n = int(duration * fps)
    frames = np.full((n, height, width, 3), 40, dtype=np.uint8)

    for event in events:
        i = int(round(event * fps))
        frames[i] = 230

    frames = np.clip(
        frames.astype(np.int16) + rng.normal(0, 2, frames.shape), 0, 255
    ).astype(np.uint8)
    t = np.arange(int(duration * RATE)) / RATE
    audio = rng.normal(0, 0.002, len(t))

    for event in events + audio_offset_s:
        k = int(event * RATE)
        burst = np.exp(-np.arange(2_000) / 300) * rng.normal(0, 0.5, 2_000)
        audio[k : k + 2_000] += burst[: len(audio[k : k + 2_000])]

    with tempfile.TemporaryDirectory() as tmp:
        wav = os.path.join(tmp, "a.wav")
        wavfile.write(wav, RATE, audio.astype(np.float32))
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", f"{width}x{height}", "-r", str(fps), "-i", "-", "-i", wav,
             "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
             "-c:a", "aac", "-shortest", path],
            input=frames.tobytes(), check=True,
        )  # fmt: skip

    return path


def grid_flicker_video(path: str, grid_values: np.ndarray, *, fps: float = 30.0) -> str:
    """Rolling-shutter flicker whose mains frequency follows a real grid record
    (one value per second), so the footage can be dated from its lighting."""
    width, height = 160, 120
    duration = len(grid_values) - 1
    n = int(duration * fps)
    rng = np.random.default_rng(5)
    yy, xx = np.mgrid[0:height, 0:width]
    scene = 100 + 50 * np.sin(xx / 13.0) * np.cos(yy / 17.0)
    row_time = (np.arange(height) / height * 0.02)[:, None]
    fine_t = np.arange(0, duration, 1e-3)
    mains = np.interp(fine_t, np.arange(len(grid_values)), grid_values)
    phase = 2 * np.pi * 2 * np.cumsum(mains) * 1e-3
    frames = np.empty((n, height, width, 3), dtype=np.uint8)

    for i in range(n):
        t = i / fps
        k = min(int(t * 1000), len(phase) - 1)
        flicker = 1 + 0.05 * np.sin(phase[k] + 2 * np.pi * 2 * mains[k] * row_time)
        image = scene * flicker + rng.normal(0, 1.0, scene.shape)
        frames[i] = np.clip(image, 0, 255).astype(np.uint8)[..., None]

    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "12", "-pix_fmt", "yuv420p", path],
        input=frames.tobytes(), check=True,
    )  # fmt: skip

    return path


def cloned_video(path: str, *, duration: float = 6.0) -> str:
    """Copy-move forgery on a richly textured, slowly panning scene."""
    from scipy import ndimage

    fps, width, height = 30, 640, 360
    rng = np.random.default_rng(6)
    world = rng.random((height + 80, width + 200, 3))
    world = np.stack(
        [ndimage.gaussian_filter(world[..., c], 1.2) for c in range(3)], axis=-1
    )
    world = (world - world.min()) / (np.ptp(world) + 1e-9) * 255
    frames = []

    for i in range(int(duration * fps)):
        x = int(i * 0.5)
        frame = world[40 : 40 + height, x : x + width].copy()
        # The forgery: a 90x70 patch stamped 300 px to the right.
        frame[200:270, 420:510] = frame[60:130, 120:210]
        frames.append(np.clip(frame + rng.normal(0, 1.5, frame.shape), 0, 255))

    data = np.stack(frames).astype(np.uint8)
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
         "-c:v", "libx264", "-crf", "14", "-pix_fmt", "yuv420p", path],
        input=data.tobytes(), check=True,
    )  # fmt: skip

    return path


async def build_pack(out_dir: str) -> list[dict[str, object]]:
    """Write every demo clip and a manifest of what each one hides."""
    import asyncio
    import json
    from datetime import datetime
    from datetime import timedelta
    from datetime import timezone

    from earwitness import grid_fetch
    from earwitness import synth

    os.makedirs(out_dir, exist_ok=True)
    pack: list[dict[str, object]] = []
    planted = planted_video(os.path.join(out_dir, "01_planted.mp4"))
    pack.append(
        {
            "file": planted,
            "hidden": [
                "18.5 kHz ultrasonic tone (inaudible)",
                f"DTMF keypad tones '{PLANTED['dtmf']}' at {PLANTED['dtmf_at_s']} s",
                f"{PLANTED['silence_s']} s of digital silence at {PLANTED['silence_at_s']} s",
                f"hard cut + room-tone jump at {PLANTED['cut_at_s']} s (corroborated edit)",
                f"{PLANTED['pulse_bpm']:.0f} bpm pulse in the skin patch",
            ],
        }
    )
    pack.append(
        {
            "file": cloned_video(os.path.join(out_dir, "02_cloned.mp4")),
            "hidden": ["copy-move forgery: a 90x70 patch stamped 300 px to the right"],
        }
    )
    pack.append(
        {
            "file": av_events_video(
                os.path.join(out_dir, "03_dubbed.mp4"), audio_offset_s=0.25
            ),
            "hidden": ["soundtrack shifted 250 ms late"],
        }
    )
    flicker50 = flicker_video(os.path.join(out_dir, "04_lamps_50hz.mp4"), grid_hz=50.0)
    flicker60 = flicker_video(
        os.path.join(out_dir, "05_lamps_60hz.mp4"), grid_hz=60.0, fps=25.0
    )
    pack.append(
        {
            "file": flicker50,
            "hidden": [
                "100 Hz lamp flicker -> 50 Hz grid (Europe/GB)",
                "20 ms sensor readout",
            ],
        }
    )
    pack.append(
        {"file": flicker60, "hidden": ["120 Hz lamp flicker -> 60 Hz grid (Americas)"]}
    )

    truth = datetime(2024, 3, 12, 23, 10, 0, tzinfo=timezone.utc)
    series, _ = await grid_fetch.fetch(
        truth - timedelta(minutes=5), truth + timedelta(minutes=6)
    )

    if series is not None:
        offset = int((truth - series.start).total_seconds())
        raw = grid_flicker_video(
            os.path.join(out_dir, "_raw.mp4"), series.values[offset : offset + 91]
        )
        dated = os.path.join(out_dir, "06_lamps_dated.mp4")
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-i", raw, "-c", "copy",
             "-metadata", "creation_time=2024-03-12T23:31:30Z", dated],
            check=True,
        )  # fmt: skip
        os.remove(raw)
        pack.append(
            {
                "file": dated,
                "hidden": [
                    "lamps follow the real GB grid of 2024-03-12 23:10:00 UTC",
                    "metadata deliberately claims 20 minutes later",
                ],
            }
        )

    synth_dir = os.path.join(out_dir, "07_hum")
    truth_hum = await synth.create_scenario(synth_dir)
    pack.append(
        {
            "file": os.path.join(synth_dir, "spliced.mp4"),
            "options": {"grid_path": os.path.join(synth_dir, "grid.csv")},
            "hidden": [
                f"mains hum dated {truth_hum['start']} against grid.csv",
                f"second half recorded 30 min later, splice at {truth_hum['splice_at_s']} s",
                "two aircraft Doppler sweeps",
            ],
        }
    )

    with open(os.path.join(out_dir, "MANIFEST.json"), "w") as handle:
        json.dump(pack, handle, indent=2)

    await asyncio.sleep(0)

    return pack
