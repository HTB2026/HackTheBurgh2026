"""Pixel-level tampering cues on selected frames.

- Error level analysis: re-compress the frame and see which regions change.
  Areas pasted in from elsewhere often recompress differently.
- Noise residual: every camera leaves a uniform grain; a region with
  noticeably different grain was processed separately (blurred, pasted,
  generated). Video codecs reshape grain too, so these maps are for an
  examiner's eye and are not turned into findings.
- Copy-move: identical keypoint patterns at two places in one frame (cloned
  objects or patched-over areas).
"""

import asyncio
from typing import Any

import cv2
import numpy as np

from earwitness.modules.base import CaseContext
from earwitness.modules.base import Reporter
from earwitness.modules.base import Skip

_WIDTH = 960
_BLOCK = 24
_SAMPLES = 6


def _heat(base: np.ndarray, heat: np.ndarray) -> np.ndarray:
    heat = np.clip(heat, 0, 1)
    colour = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    colour = cv2.resize(
        colour, (base.shape[1], base.shape[0]), interpolation=cv2.INTER_NEAREST
    )
    bgr = cv2.cvtColor(base, cv2.COLOR_RGB2BGR)
    alpha = cv2.resize(heat, (base.shape[1], base.shape[0]))[..., None] * 0.75

    return (bgr * (1 - alpha) + colour * alpha).astype(np.uint8)


def _ela(frame: np.ndarray) -> np.ndarray:
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    _, encoded = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    again = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    diff = (
        np.abs(bgr.astype(np.int16) - again.astype(np.int16))
        .max(axis=2)
        .astype(np.float32)
    )

    return cv2.blur(diff, (5, 5))


def _blocks(values: np.ndarray, block: int) -> np.ndarray:
    h, w = values.shape[0] // block, values.shape[1] // block

    return (
        values[: h * block, : w * block].reshape(h, block, w, block).mean(axis=(1, 3))
    )


def _noise_map(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY).astype(np.float32)
    residual = gray - cv2.medianBlur(gray.astype(np.uint8), 3).astype(np.float32)
    energy = _blocks(np.abs(residual), _BLOCK)
    # Grain depends on brightness and texture; compare like with like.
    brightness = _blocks(gray, _BLOCK)
    texture = _blocks(
        np.abs(cv2.Laplacian(cv2.GaussianBlur(gray, (0, 0), 2), cv2.CV_32F)), _BLOCK
    )
    design = np.stack(
        [
            np.ones_like(brightness),
            brightness / 255,
            (brightness / 255) ** 2,
            np.log1p(texture),
        ],
        axis=-1,
    )
    flat_design = design.reshape(-1, design.shape[-1])
    target = np.log1p(energy.reshape(-1))
    coeffs, *_ = np.linalg.lstsq(flat_design, target, rcond=None)
    residual_map = (target - flat_design @ coeffs).reshape(energy.shape)
    mad = np.median(np.abs(residual_map - np.median(residual_map))) * 1.4826 + 1e-6
    # Ignore saturated / black blocks: no grain there anyway.
    valid = (brightness > 12) & (brightness < 243)

    return np.where(valid, (residual_map - np.median(residual_map)) / mad, 0.0)


def _copy_move(
    frame: np.ndarray,
) -> tuple[list[tuple[tuple[int, int], tuple[int, int]]], tuple[int, int] | None]:
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    orb = cv2.ORB_create(nfeatures=3000, fastThreshold=12)
    keypoints, descriptors = orb.detectAndCompute(gray, None)

    if descriptors is None or len(keypoints) < 50:
        return [], None

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = matcher.knnMatch(descriptors, descriptors, k=3)
    pairs = []

    for group in matches:
        for match in group[1:]:
            if match.distance > 32:
                continue

            a = np.array(keypoints[match.queryIdx].pt)
            b = np.array(keypoints[match.trainIdx].pt)

            if np.linalg.norm(a - b) > 40:
                pairs.append((a, b))

    if len(pairs) < 10:
        return [], None

    offsets = np.array([np.round((b - a) / 6) for a, b in pairs]).astype(int)
    keys, counts = np.unique(offsets, axis=0, return_counts=True)
    best = int(np.argmax(counts))

    if counts[best] < 14:
        return [], None

    offset = keys[best]
    # A pasted clone puts nearly all self-matches on one offset (and its mirror).
    # Repeated real objects (lamps, windows, keys, tiles) match at many offsets.
    mirror = np.all(keys == -offset, axis=1)
    on_offset = counts[best] + (counts[mirror].sum() if mirror.any() else 0)

    if on_offset < 0.5 * len(pairs):
        return [], None
    chosen = [
        (tuple(map(int, a)), tuple(map(int, b)))
        for (a, b), o in zip(pairs, offsets)
        if np.abs(o - offset).max() <= 1
    ]
    sources = np.array([c[0] for c in chosen])
    # Repetitive texture (tiles, windows, text) repeats at many offsets; a clone
    # is one compact patch moved once.
    spread = np.ptp(sources, axis=0)

    if spread.max() > 0.5 * max(frame.shape[:2]) or len(chosen) < 14:
        return [], None

    return chosen, (int(offset[0] * 6), int(offset[1] * 6))


def _analyse(frame: np.ndarray) -> dict[str, Any]:
    ela = _ela(frame)
    ela_blocks = _blocks(ela, _BLOCK)
    ela_z = (ela_blocks - np.median(ela_blocks)) / (
        np.median(np.abs(ela_blocks - np.median(ela_blocks))) * 1.4826 + 1e-6
    )
    noise_z = _noise_map(frame)
    pairs, offset = _copy_move(frame)
    noise_hot = np.abs(noise_z) > 4.5
    # Only clustered anomalies matter; isolated blocks are texture.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        noise_hot.astype(np.uint8), connectivity=8
    )
    regions = [
        {
            "x": int(s[0]) * _BLOCK,
            "y": int(s[1]) * _BLOCK,
            "w": int(s[2]) * _BLOCK,
            "h": int(s[3]) * _BLOCK,
            "blocks": int(s[4]),
        }
        for s in stats[1:]
        if s[4] >= 6
    ]

    return {
        "ela": ela,
        "ela_z": ela_z,
        "noise_z": noise_z,
        "regions": sorted(regions, key=lambda r: -r["blocks"]),
        "noise_fraction": float(noise_hot.mean()),
        "copy_move": pairs,
        "copy_offset": offset,
    }


class Tamper:
    name = "tamper"
    title = "PIXEL FORENSICS"
    description = "Error-level analysis, noise-residual consistency and copy-move detection on key frames."
    depends = ["timeline"]

    async def run(self, ctx: CaseContext, out: Reporter) -> None:
        info = await ctx.video_info()

        if info is None:
            raise Skip("no video stream")

        timeline = await ctx.wait_for("timeline") or {}
        duration = max(info.duration_s, 0.1)
        times = list(np.linspace(0.5, max(0.5, duration - 0.5), _SAMPLES))

        for cut in (timeline.get("cuts") or [])[:4]:
            times += [max(0.0, cut - 0.2), cut + 0.2]

        times = sorted({round(t, 2) for t in times})
        width = min(_WIDTH, info.width)
        flagged = 0
        published: dict[str, Any] = {"frames": []}

        for number, time_s in enumerate(times):
            out.progress(0.05 + 0.9 * number / len(times), f"frame @ {time_s:.2f}s")
            frame = await ctx.frame_at(time_s, width=width)

            if frame is None:
                continue

            result = await asyncio.to_thread(_analyse, frame)
            noise_heat = np.clip(np.abs(result["noise_z"]) / 8, 0, 1).astype(np.float32)
            ela_heat = np.clip(
                result["ela"] / (np.percentile(result["ela"], 99.5) + 1e-6), 0, 1
            ).astype(np.float32)
            stem = f"{number:02d}_{time_s:07.2f}"
            noise_path = out.path(f"noise_{stem}.jpg")
            ela_path = out.path(f"ela_{stem}.jpg")
            cv2.imwrite(
                noise_path, _heat(frame, noise_heat), [cv2.IMWRITE_JPEG_QUALITY, 85]
            )
            cv2.imwrite(
                ela_path, _heat(frame, ela_heat), [cv2.IMWRITE_JPEG_QUALITY, 85]
            )
            out.artifact(
                f"noise_{stem}", "heatmap", f"Noise-residual map @ {time_s:.2f}s",
                file=noise_path, time_s=time_s, data={"caption": "bright = grain unlike the rest of the frame"},
            )  # fmt: skip
            out.artifact(
                f"ela_{stem}", "heatmap", f"Error-level analysis @ {time_s:.2f}s",
                file=ela_path, time_s=time_s, data={"caption": "bright = recompresses differently"},
            )  # fmt: skip
            published["frames"].append(
                {
                    "time_s": time_s,
                    "regions": result["regions"][:5],
                    "noise_fraction": result["noise_fraction"],
                }
            )

            if result["copy_move"]:
                flagged += 1
                image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR).copy()

                for a, b in result["copy_move"][:150]:
                    cv2.line(image, a, b, (80, 220, 255), 1, cv2.LINE_AA)
                    cv2.circle(image, a, 3, (60, 60, 255), -1)
                    cv2.circle(image, b, 3, (60, 255, 120), -1)

                path = out.path(f"clone_{stem}.jpg")
                cv2.imwrite(path, image)
                out.artifact(
                    f"clone_{stem}",
                    "image",
                    f"Copy-move matches @ {time_s:.2f}s",
                    file=path,
                    time_s=time_s,
                )
                out.finding(
                    "warning",
                    f"Cloned region @ {time_s:.2f}s",
                    f"{len(result['copy_move'])} identical feature points repeat at a fixed offset "
                    f"{result['copy_offset']} px within one frame. Cloning stamps one area over another "
                    "to hide or duplicate something. Repetitive real patterns can also do this.",
                    time_s=time_s,
                    tags=["tamper", "clone"],
                    data={
                        "offset": result["copy_offset"],
                        "matches": len(result["copy_move"]),
                    },
                )

        if not flagged:
            out.finding(
                "info",
                f"No cloned regions in {len(times)} sampled frames",
                "No patch of any sampled frame reappears elsewhere in the same frame. Error-level "
                "and grain heatmaps are attached for visual inspection; on phone video, compression "
                "itself shapes them, so they are not scored automatically.",
                tags=["tamper"],
            )

        ctx.publish(self.name, published)
        out.progress(1.0)
