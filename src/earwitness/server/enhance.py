"""'Enhance': multi-frame super-resolution of a region of interest.

Neighbouring frames see the same detail shifted by sub-pixel amounts and with
independent noise. Upsampling each crop, aligning them to the centre frame and
averaging recovers detail no single frame shows (plates, signs, faces at
distance), then a light unsharp mask and local contrast make it legible.
"""

import asyncio
import os
import time
from typing import Any

import cv2
import numpy as np


def _align_and_stack(
    crops: list[np.ndarray], scale: int
) -> tuple[np.ndarray, np.ndarray, int]:
    up = [
        cv2.resize(c, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC).astype(
            np.float32
        )
        for c in crops
    ]
    centre = len(up) // 2
    reference = cv2.cvtColor(up[centre], cv2.COLOR_RGB2GRAY)
    aligned = [up[centre]]
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 60, 1e-4)

    for index, image in enumerate(up):
        if index == centre:
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        warp = np.eye(2, 3, dtype=np.float32)

        try:
            score, warp = cv2.findTransformECC(
                reference, gray, warp, cv2.MOTION_EUCLIDEAN, criteria, None, 5
            )
        except cv2.error:
            continue

        if score < 0.85:
            continue

        aligned.append(
            cv2.warpAffine(
                image,
                warp,
                (image.shape[1], image.shape[0]),
                flags=cv2.INTER_CUBIC + cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_REFLECT,
            )
        )

    # Robust merge (as burst photography does): a neighbour's pixel only counts
    # where it agrees with the reference, so moving parts never ghost.
    base = up[centre]
    smooth_base = cv2.GaussianBlur(base, (0, 0), scale)
    total = base.copy()
    weight = np.ones(base.shape[:2] + (1,), dtype=np.float32)

    for image in aligned[1:]:
        difference = np.abs(cv2.GaussianBlur(image, (0, 0), scale) - smooth_base).mean(
            axis=2, keepdims=True
        )
        w = np.exp(-((difference / 10.0) ** 2))
        total += w * image
        weight += w

    merged = total / weight
    blurred = cv2.GaussianBlur(merged, (0, 0), 1.2 * scale / 2)
    sharp = cv2.addWeighted(merged, 1.8, blurred, -0.8, 0)
    sharp = np.clip(sharp, 0, 255).astype(np.uint8)
    lab = cv2.cvtColor(sharp, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    lab[..., 0] = clahe.apply(lab[..., 0])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    single = np.clip(up[centre], 0, 255).astype(np.uint8)

    return single, enhanced, len(aligned)


def _still_lenses(crop: np.ndarray, scale: int) -> dict[str, np.ndarray]:
    """Forensic views of one region of a photograph, enlarged ``scale`` times.

    There are no neighbouring frames to stack, so the gain is honest
    enlargement plus views the eye cannot form on its own: error levels,
    amplified grain, equalised tones, inverted and single-channel views, and
    the luminance gradient.
    """
    from earwitness import photo

    big = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)
    blurred = cv2.GaussianBlur(big, (0, 0), 1.0)
    sharp = np.clip(cv2.addWeighted(big, 1.6, blurred, -0.6, 0), 0, 255).astype(
        np.uint8
    )
    lab = cv2.cvtColor(sharp, cv2.COLOR_RGB2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(lab[..., 0])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    again = photo.decode_jpeg_bytes(photo.encode_jpeg_bytes(crop, 90))
    ela = (
        np.abs(
            crop.astype(np.int16)
            - (again if again is not None else crop).astype(np.int16)
        )
        .max(axis=2)
        .astype(np.float32)
    )
    ela = cv2.resize(
        np.clip(ela * (255 / (np.percentile(ela, 99.5) + 1e-6)), 0, 255).astype(
            np.uint8
        ),
        (big.shape[1], big.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY).astype(np.float32)
    grain = gray - cv2.medianBlur(gray.astype(np.uint8), 3).astype(np.float32)
    grain = cv2.resize(
        np.clip(grain * 12 + 128, 0, 255).astype(np.uint8),
        (big.shape[1], big.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    equalised = cv2.cvtColor(
        np.stack([cv2.equalizeHist(big[..., c]) for c in range(3)], axis=-1),
        cv2.COLOR_RGB2BGR,
    )
    inverted = 255 - big
    gx = cv2.Sobel(
        cv2.cvtColor(big, cv2.COLOR_RGB2GRAY).astype(np.float32), cv2.CV_32F, 1, 0
    )
    gy = cv2.Sobel(
        cv2.cvtColor(big, cv2.COLOR_RGB2GRAY).astype(np.float32), cv2.CV_32F, 0, 1
    )
    magnitude = np.hypot(gx, gy)
    edges = np.clip(
        magnitude * (255 / (np.percentile(magnitude, 99) + 1e-6)), 0, 255
    ).astype(np.uint8)
    channels = {
        name: cv2.cvtColor(big[..., index], cv2.COLOR_GRAY2BGR)
        for index, name in enumerate(("red", "green", "blue"))
    }

    return {
        "single": cv2.cvtColor(big, cv2.COLOR_RGB2BGR),
        "enhanced": cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR),
        "ela": cv2.applyColorMap(ela, cv2.COLORMAP_INFERNO),
        "grain": cv2.cvtColor(grain, cv2.COLOR_GRAY2BGR),
        "equalised": equalised,
        "inverted": cv2.cvtColor(inverted, cv2.COLOR_RGB2BGR),
        "edges": cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR),
        **channels,
    }


_LENS_CAPTIONS = {
    "ela": "Error levels: how much each pixel changes when re-saved. A pasted region glows or goes dark against its surroundings.",
    "grain": "Sensor grain amplified 12×. Real grain is even sand; smoothing, cloning and generation leave smooth or blotchy patches.",
    "equalised": "Every channel stretched to full range: faint detail in shadows and highlights becomes visible.",
    "inverted": "Negative view. Faint dark marks on light backgrounds (writing, scratches, watermarks) jump out.",
    "edges": "Edge strength. Halos and double edges around an object mean it was cut out and placed.",
    "red": "Red channel only. Skin detail, warm-light shadows and red ink separate here.",
    "green": "Green channel only: the sharpest channel on most sensors, best for reading text.",
    "blue": "Blue channel only: noisiest channel, where retouching is often most visible.",
}


async def enhance_still(
    case: Any, box: tuple[float, float, float, float]
) -> dict[str, Any] | None:
    ctx = case.ctx

    if ctx is None:
        from earwitness.modules.base import CaseContext

        ctx = CaseContext(case.id, case.evidence_path, case.workdir, case.options)
        case.ctx = ctx

    image = await asyncio.to_thread(ctx.image)

    if image is None:
        return None

    height, width = image.shape[:2]
    x, y, w, h = box
    x0, y0 = int(max(0, x) * width), int(max(0, y) * height)
    x1, y1 = int(min(1, x + w) * width), int(min(1, y + h) * height)

    if x1 - x0 < 8 or y1 - y0 < 8:
        return None

    crop = np.ascontiguousarray(image[y0:y1, x0:x1])
    scale = int(np.clip(640 / max(x1 - x0, y1 - y0), 1, 8))
    lenses = await asyncio.to_thread(_still_lenses, crop, scale)
    directory = os.path.join(case.workdir, "enhance")
    os.makedirs(directory, exist_ok=True)
    stamp = f"{int(time.time() * 1000)}"
    urls: dict[str, str] = {}

    for name, bgr in lenses.items():
        path = os.path.join(directory, f"{stamp}-{name}.png")
        cv2.imwrite(path, bgr)
        urls[name] = f"/api/cases/{case.id}/files/enhance/{os.path.basename(path)}"

    return {
        "single_url": urls["single"],
        "enhanced_url": urls["enhanced"],
        "frames_used": 1,
        "scale": scale,
        "source_px": [x1 - x0, y1 - y0],
        "time_s": 0.0,
        "kind": "image",
        "lenses": [
            {"name": name, "url": urls[name], "caption": _LENS_CAPTIONS[name]}
            for name in (
                "ela",
                "grain",
                "equalised",
                "inverted",
                "edges",
                "red",
                "green",
                "blue",
            )
        ],
    }


async def enhance(
    case: Any, time_s: float, box: tuple[float, float, float, float], frames: int
) -> dict[str, Any] | None:
    ctx = case.ctx

    if ctx is None:
        from earwitness.modules.base import CaseContext

        ctx = CaseContext(case.id, case.evidence_path, case.workdir, case.options)
        case.ctx = ctx

    if ctx.kind == "image":
        return await enhance_still(case, box)

    info = await ctx.video_info()

    if info is None:
        return None

    frames = int(min(max(frames, 3), 25))
    start = max(0.0, time_s - (frames // 2) / info.fps)
    clip = await ctx.clip_frames(start, frames)

    if clip is None or len(clip) == 0:
        return None

    height, width = clip.shape[1:3]
    x, y, w, h = box
    x0, y0 = int(max(0, x) * width), int(max(0, y) * height)
    x1, y1 = int(min(1, x + w) * width), int(min(1, y + h) * height)

    if x1 - x0 < 8 or y1 - y0 < 8:
        return None

    crops = [frame[y0:y1, x0:x1] for frame in clip]
    scale = int(np.clip(480 / max(x1 - x0, y1 - y0), 2, 6))
    single, enhanced, used = await asyncio.to_thread(_align_and_stack, crops, scale)
    directory = os.path.join(case.workdir, "enhance")
    os.makedirs(directory, exist_ok=True)
    stamp = f"{int(time.time() * 1000)}"
    paths = {}

    for name, image in (("single", single), ("enhanced", enhanced)):
        path = os.path.join(directory, f"{stamp}-{name}.png")
        cv2.imwrite(path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        paths[name] = f"/api/cases/{case.id}/files/enhance/{os.path.basename(path)}"

    return {
        "single_url": paths["single"],
        "enhanced_url": paths["enhanced"],
        "frames_used": used,
        "scale": scale,
        "source_px": [x1 - x0, y1 - y0],
        "time_s": time_s,
    }
