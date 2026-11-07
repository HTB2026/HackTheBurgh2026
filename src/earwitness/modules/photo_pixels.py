"""Pixel forensics on a still photograph, at full resolution.

Video codecs reshape grain and recompress every frame, so on video these maps
are only for the eye. A photograph is one exposure saved once: here the same
analyses can be scored.

- Error level analysis at the file's own quality: a region pasted in from
  another JPEG recompresses differently from its surroundings.
- Noise residual consistency: one sensor leaves one grain; a patch with
  different grain was processed separately.
- Copy-move: SIFT self-matching (rotation and scale tolerant) finds a patch
  stamped elsewhere in the same picture.
- Resampling traces: resizing or rotating leaves periodic correlations between
  neighbouring pixels, visible as peaks in the spectrum of the prediction error.
- Histogram combing: contrast, levels and brightness adjustments leave gaps and
  spikes in the tone histogram.
- Luminance gradient, principal-component and amplified-noise views for a human
  examiner.
"""

import asyncio
from typing import Any

import cv2
import numpy as np
from scipy.ndimage import median_filter

from earwitness import photo
from earwitness.modules.base import CaseContext
from earwitness.modules.base import Reporter
from earwitness.modules.base import Skip
from earwitness.modules.tamper import _blocks
from earwitness.modules.tamper import _noise_map

_ANALYSIS_SIDE = 1600
_CLONE_SIDE = 2000
_BLOCK = 16


def _robust_z(values: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    sample = values[valid] if valid is not None and valid.any() else values
    median = np.median(sample)
    mad = np.median(np.abs(sample - median)) * 1.4826 + 1e-6

    return (values - median) / mad


def _ela(rgb: np.ndarray, quality: int) -> tuple[np.ndarray, np.ndarray]:
    """(ELA map at pixel resolution, texture-normalised block z-scores)."""
    again = photo.decode_jpeg_bytes(photo.encode_jpeg_bytes(rgb, quality))

    if again is None or again.shape != rgb.shape:
        again = rgb

    diff = (
        np.abs(rgb.astype(np.int16) - again.astype(np.int16))
        .max(axis=2)
        .astype(np.float32)
    )
    diff = cv2.blur(diff, (5, 5))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    energy = _blocks(diff, _BLOCK)
    brightness = _blocks(gray, _BLOCK)
    texture = _blocks(
        np.abs(cv2.Laplacian(cv2.GaussianBlur(gray, (0, 0), 1.5), cv2.CV_32F)), _BLOCK
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
    flat = design.reshape(-1, design.shape[-1])
    target = np.log1p(energy.reshape(-1))
    coefficients, *_ = np.linalg.lstsq(flat, target, rcond=None)
    residual = (target - flat @ coefficients).reshape(energy.shape)
    valid = (brightness > 10) & (brightness < 245)
    z = np.where(valid, _robust_z(residual, valid), 0.0)

    return diff, z


def _overlap(a: dict[str, float], b: dict[str, float]) -> float:
    """Intersection over the smaller box, for two fractional boxes."""
    x0, y0 = max(a["x"], b["x"]), max(a["y"], b["y"])
    x1, y1 = (
        min(a["x"] + a["w"], b["x"] + b["w"]),
        min(a["y"] + a["h"], b["y"] + b["h"]),
    )
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    smaller = min(a["w"] * a["h"], b["w"] * b["h"]) + 1e-9

    return inter / smaller


def _regions(hot: np.ndarray, minimum_blocks: int, block: int) -> list[dict[str, Any]]:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        hot.astype(np.uint8), connectivity=8
    )
    regions = []

    for s in stats[1:]:
        if s[4] >= minimum_blocks:
            regions.append(
                {
                    "x": int(s[0]) * block,
                    "y": int(s[1]) * block,
                    "w": int(s[2]) * block,
                    "h": int(s[3]) * block,
                    "blocks": int(s[4]),
                }
            )

    return sorted(regions, key=lambda r: -r["blocks"])


def _copy_move_sift(gray: np.ndarray) -> dict[str, Any] | None:
    """Self-matching SIFT keypoints that agree on one rigid motion."""
    sift = cv2.SIFT_create(nfeatures=12000, contrastThreshold=0.015, edgeThreshold=12)
    keypoints, descriptors = sift.detectAndCompute(gray, None)

    if descriptors is None or len(keypoints) < 60:
        return None

    matcher = cv2.BFMatcher(cv2.NORM_L2)
    knn = matcher.knnMatch(descriptors, descriptors, k=3)
    src, dst = [], []
    minimum_apart = 0.03 * max(gray.shape)

    for group in knn:
        if len(group) < 3:
            continue

        # group[0] is the point itself; the real nearest neighbour must beat the next one clearly.
        a, b = group[1], group[2]

        if a.distance > 0.6 * b.distance or a.distance > 220:
            continue

        p = np.array(keypoints[a.queryIdx].pt)
        q = np.array(keypoints[a.trainIdx].pt)

        if np.linalg.norm(p - q) >= minimum_apart:
            src.append(p)
            dst.append(q)

    if len(src) < 12:
        return None

    src_a, dst_a = np.float32(src), np.float32(dst)
    best: dict[str, Any] | None = None

    for _ in range(
        3
    ):  # the largest cluster, then the next ones (repeated real objects come first)
        if len(src_a) < 12:
            break

        transform, inliers = cv2.estimateAffinePartial2D(
            src_a,
            dst_a,
            method=cv2.RANSAC,
            ransacReprojThreshold=3.0,
            maxIters=4000,
            confidence=0.995,
        )

        if transform is None or inliers is None:
            break

        mask = inliers.ravel().astype(bool)
        count = int(mask.sum())

        if count < 14:
            break

        a, b = transform[0, 0], transform[0, 1]
        scale = float(np.hypot(a, b))
        angle = float(np.degrees(np.arctan2(b, a)))
        sources_all, targets_all = src_a[mask], dst_a[mask]
        shift = (targets_all - sources_all).mean(axis=0)
        sources = src_a[mask]
        targets = dst_a[mask]
        spread = np.ptp(sources, axis=0)
        # Repeated real texture (windows, tiles, text) repeats everywhere; a clone is one compact patch.
        compact = spread.max() <= 0.5 * max(gray.shape)
        moved = (
            np.linalg.norm(shift) > minimum_apart
            or abs(angle) > 3
            or abs(scale - 1) > 0.05
        )

        if compact and moved and (count >= 40 or count >= 0.2 * len(src_a)):
            candidate = {
                "count": count,
                "total_pairs": int(len(src_a)),
                "scale": scale,
                "angle_deg": angle,
                "shift": [float(shift[0]), float(shift[1])],
                "source_box": [
                    float(v)
                    for v in (
                        *sources.min(axis=0),
                        *(sources.max(axis=0) - sources.min(axis=0)),
                    )
                ],
                "target_box": [
                    float(v)
                    for v in (
                        *targets.min(axis=0),
                        *(targets.max(axis=0) - targets.min(axis=0)),
                    )
                ],
                "pairs": [
                    (tuple(map(int, s)), tuple(map(int, t)))
                    for s, t in zip(sources, targets)
                ],
            }

            if best is None or candidate["count"] > best["count"]:
                best = candidate

        src_a, dst_a = src_a[~mask], dst_a[~mask]

    return best


def _resampling(gray: np.ndarray) -> dict[str, Any]:
    """Popescu & Farid: periodic correlations from interpolation show as spectral peaks."""
    h, w = gray.shape
    side = min(1024, h, w)
    y0, x0 = (h - side) // 2, (w - side) // 2
    patch = gray[y0 : y0 + side, x0 : x0 + side].astype(np.float64)
    # 3x3 linear predictor fitted on the patch itself.
    pad = np.pad(patch, 1, mode="reflect")
    neighbours = [
        pad[1 + dy : 1 + dy + side, 1 + dx : 1 + dx + side].ravel()
        for dy in (-1, 0, 1)
        for dx in (-1, 0, 1)
        if (dy, dx) != (0, 0)
    ]
    design = np.stack(neighbours, axis=1)
    target = patch.ravel()
    pick = np.random.default_rng(0).choice(
        len(target), size=min(len(target), 200_000), replace=False
    )
    coefficients, *_ = np.linalg.lstsq(design[pick], target[pick], rcond=None)
    residual = (target - design @ coefficients).reshape(side, side)
    window = np.hanning(side)
    spectrum = np.abs(
        np.fft.fftshift(np.fft.fft2(residual * window[:, None] * window[None, :]))
    )
    log = np.log1p(spectrum)
    background = median_filter(log, size=21, mode="reflect")
    peaks = log - background
    centre = side // 2
    yy, xx = np.mgrid[:side, :side]
    fy, fx = (yy - centre) / side, (xx - centre) / side
    mask = np.ones_like(peaks, dtype=bool)
    mask &= np.hypot(fx, fy) > 0.03  # DC and very low frequencies
    # JPEG's 8x8 grid puts peaks at multiples of 1/8; ignore those (and their immediate surroundings).
    near_eighth = (np.abs((fx * 8) - np.round(fx * 8)) < 0.04) & (
        np.abs((fy * 8) - np.round(fy * 8)) < 0.04
    )
    mask &= ~near_eighth
    # Peaks on the axes or symmetric pairs are what interpolation produces.
    sigma = np.std(peaks[mask]) + 1e-9
    z = peaks / sigma
    z_masked = np.where(mask, z, 0)
    index = np.unravel_index(np.argmax(z_masked), z.shape)
    peak_z = float(z_masked[index])
    peak_f = (float(fx[index]), float(fy[index]))
    strong = int((z_masked > 6).sum())
    vis = np.clip((log - log.min()) / (np.ptp(log) + 1e-9) * 255, 0, 255).astype(
        np.uint8
    )

    return {
        "peak_z": peak_z,
        "peak_frequency": peak_f,
        "strong_peaks": strong,
        "spectrum": vis,
        "coefficients": coefficients.tolist(),
    }


def _histogram(gray: np.ndarray) -> dict[str, Any]:
    hist = np.bincount(gray.ravel(), minlength=256).astype(float)
    lo, hi = np.percentile(gray, 0.5), np.percentile(gray, 99.5)
    inner = np.arange(256)
    inside = (inner > lo + 2) & (inner < hi - 2)
    gaps = int(
        ((hist == 0) & inside & (np.roll(hist, 1) > 0) & (np.roll(hist, -1) > 0)).sum()
    )
    neighbours = (np.roll(hist, 1) + np.roll(hist, -1)) / 2 + 1
    spikes = int(
        ((hist > 2.5 * neighbours) & inside & (hist > hist[inside].mean() * 0.2)).sum()
    )
    used = int(((hist > 0) & inside).sum())
    total_inside = int(inside.sum())

    return {
        "hist": hist,
        "gaps": gaps,
        "spikes": spikes,
        "levels_used": used,
        "levels_inside": total_inside,
        "clipped_black": float(hist[:2].sum() / hist.sum()),
        "clipped_white": float(hist[254:].sum() / hist.sum()),
    }


def _gradient_view(rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.hypot(gx, gy)
    angle = (np.degrees(np.arctan2(gy, gx)) + 180) / 2  # 0..180 for OpenCV hue
    value = np.clip(magnitude / (np.percentile(magnitude, 99) + 1e-6) * 255, 0, 255)
    hsv = np.stack(
        [
            angle.astype(np.uint8),
            np.full_like(angle, 255, dtype=np.uint8),
            value.astype(np.uint8),
        ],
        axis=-1,
    )

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def _pca_views(rgb: np.ndarray) -> list[np.ndarray]:
    flat = rgb.reshape(-1, 3).astype(np.float32)
    mean = flat.mean(axis=0)
    centred = flat - mean
    sample = centred[
        np.random.default_rng(0).choice(
            len(centred), size=min(len(centred), 200_000), replace=False
        )
    ]
    _, _, vt = np.linalg.svd(sample, full_matrices=False)
    views = []

    for component in vt:
        projected = centred @ component
        lo, hi = np.percentile(projected, 0.5), np.percentile(projected, 99.5)
        view = (
            np.clip((projected - lo) / (hi - lo + 1e-6) * 255, 0, 255)
            .astype(np.uint8)
            .reshape(rgb.shape[:2])
        )
        views.append(view)

    return views


def _noise_view(rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    residual = gray - cv2.medianBlur(gray.astype(np.uint8), 3).astype(np.float32)

    return np.clip(residual * 10 + 128, 0, 255).astype(np.uint8)


class PhotoPixels:
    name = "tamper"
    title = "PIXEL FORENSICS"
    description = "Error-level analysis at the file's own quality, grain consistency, SIFT copy-move, resampling traces, histogram combing, gradient/PCA/noise views."
    depends = ["jpeg", "provenance"]

    async def run(self, ctx: CaseContext, out: Reporter) -> None:
        image = ctx.image()
        info = ctx.image_info()

        if image is None or info is None:
            raise Skip("not a decodable image")

        compression = await ctx.wait_for("jpeg") or {}
        provenance = await ctx.wait_for("provenance") or {}
        screenshot = bool(provenance.get("screenshot"))
        # Regions other checks already flagged: a second, independent trace in
        # the same place is what turns "worth a look" into "pasted".
        prior_boxes = [
            b
            for b in (compression.get("grid_region"), compression.get("ghost_region"))
            if b
        ]
        is_jpeg = bool(compression.get("is_jpeg"))
        quality = int(compression.get("quality") or 90)
        full_w, full_h = image.shape[1], image.shape[0]
        analysis = photo.fit(image, _ANALYSIS_SIDE)
        scale = analysis.shape[1] / full_w
        published: dict[str, Any] = {"regions": [], "boxes": []}
        flagged = 0

        def frac(region: dict[str, Any]) -> dict[str, float]:
            return photo.box_fraction(
                region["x"] / scale,
                region["y"] / scale,
                region["w"] / scale,
                region["h"] / scale,
                full_w,
                full_h,
            )

        # ---- ELA -------------------------------------------------------------------------------
        out.progress(0.05, "error level analysis")
        ela_quality = min(max(quality, 50), 95) if is_jpeg else 90
        ela, ela_z = await asyncio.to_thread(_ela, analysis, ela_quality)
        ela_heat = np.clip(ela / (np.percentile(ela, 99.5) + 1e-6), 0, 1).astype(
            np.float32
        )
        ela_path = out.path("ela.jpg")
        photo.save_bgr(ela_path, photo.heat_overlay(analysis, ela_heat))
        out.artifact("ela", "heatmap", f"Error-level analysis (re-saved at quality {ela_quality})", file=ela_path,
                     data={"caption": "bright = recompresses differently from its surroundings", "overlay": True})  # fmt: skip
        raw_ela_path = out.path("ela_raw.png")
        cv2.imwrite(
            raw_ela_path,
            np.clip(ela * (255 / (np.percentile(ela, 99.5) + 1e-6)), 0, 255).astype(
                np.uint8
            ),
        )
        out.artifact("ela_raw", "image", "Error levels alone (no photo underneath)", file=raw_ela_path,
                     data={"caption": "Equal grey everywhere means one JPEG generation; a region that glows or goes dark was saved separately"})  # fmt: skip
        ela_regions = _regions(ela_z > 4.0, 10, _BLOCK)
        texture_ok = [
            r
            for r in ela_regions
            if r["w"] * r["h"] < 0.35 * analysis.shape[0] * analysis.shape[1]
        ]

        ela_boxes = [frac(r) for r in texture_ok[:5]]

        if texture_ok and is_jpeg and not screenshot:
            region = texture_ok[0]
            box = frac(region)
            agree = [b for b in prior_boxes if _overlap(box, b) > 0.2]
            published["boxes"].append(box)

            if agree:
                flagged += 1

            out.finding(
                "warning" if agree else "notice",
                "Region recompresses differently from the rest"
                + (" (confirmed by the block grid)" if agree else ""),
                f"After allowing for brightness and texture, a {region['w'] / scale:.0f}×{region['h'] / scale:.0f} px area ({region['blocks']} blocks) has error levels far above the picture's norm. "
                "Everything saved in one JPEG generation ages the same way; a patch that behaves differently may have been pasted in or edited and saved again. "
                + (
                    "The compression check found a shifted JPEG grid or an earlier-quality ghost in the same place: two independent traces of a paste."
                    if agree
                    else "On its own this is a lead, not proof: strong texture, text and saturated colours also raise error levels."
                )
                + (
                    f" {len(texture_ok) - 1} smaller such area{'s' if len(texture_ok) > 2 else ''} too."
                    if len(texture_ok) > 1
                    else ""
                ),
                tags=["ela"] + (["tamper", "splice"] if agree else []),
                data={
                    "box": box,
                    "blocks": region["blocks"],
                    "regions": ela_boxes,
                    "quality": ela_quality,
                    "corroborated_by": agree,
                },
            )

        # ---- grain consistency -----------------------------------------------------------------
        out.progress(0.25, "grain consistency")
        noise_z = await asyncio.to_thread(_noise_map, analysis)
        noise_heat = np.clip(np.abs(noise_z) / 8, 0, 1).astype(np.float32)
        noise_path = out.path("noise.jpg")
        photo.save_bgr(noise_path, photo.heat_overlay(analysis, noise_heat))
        out.artifact("noise", "heatmap", "Noise-residual map", file=noise_path,
                     data={"caption": "bright = grain unlike the rest of the picture (after allowing for brightness and texture)", "overlay": True})  # fmt: skip
        hot = np.abs(noise_z) > 6.0
        noise_regions = [
            r
            for r in _regions(hot, 20, 24)
            if r["w"] * r["h"] < 0.25 * analysis.shape[0] * analysis.shape[1]
        ]

        if noise_regions and not screenshot:
            region = noise_regions[0]
            direction = (
                "less"
                if np.median(
                    noise_z[
                        region["y"] // 24 : (region["y"] + region["h"]) // 24,
                        region["x"] // 24 : (region["x"] + region["w"]) // 24,
                    ]
                )
                < 0
                else "more"
            )
            box = frac(region)
            agree = [b for b in prior_boxes + ela_boxes[:1] if _overlap(box, b) > 0.2]
            published["boxes"].append(box)

            if agree:
                flagged += 1

            out.finding(
                "warning" if agree else "notice",
                f"Grain differs in one region ({direction} noise than the rest)"
                + (" (confirmed by a second trace)" if agree else ""),
                f"A {region['w'] / scale:.0f}×{region['h'] / scale:.0f} px area carries {direction} sensor grain than the model of this picture predicts for its brightness and texture. "
                "One sensor, one exposure, one grain: a patch that differs may have been smoothed, pasted from another photo, or generated. "
                + (
                    "Error levels or the block grid disagree in the same place, which is how a pasted patch looks."
                    if agree
                    else "Skin smoothing, bokeh and flat skies also do this, so alone it is a lead; check the map."
                ),
                tags=["noise"] + (["tamper", "splice"] if agree else []),
                data={
                    "box": box,
                    "blocks": region["blocks"],
                    "regions": [frac(r) for r in noise_regions[:5]],
                    "corroborated_by": agree,
                },
            )

        # ---- copy-move ---------------------------------------------------------------------------------
        out.progress(0.4, "searching for cloned patches")
        clone_image = photo.fit(image, _CLONE_SIDE)
        clone_scale = clone_image.shape[1] / full_w
        clone = await asyncio.to_thread(
            _copy_move_sift, cv2.cvtColor(clone_image, cv2.COLOR_RGB2GRAY)
        )

        if clone:
            flagged += 1
            canvas = cv2.cvtColor(clone_image, cv2.COLOR_RGB2BGR).copy()

            for a, b in clone["pairs"][:200]:
                cv2.line(canvas, a, b, (80, 220, 255), 1, cv2.LINE_AA)
                cv2.circle(canvas, a, 3, (60, 60, 255), -1)
                cv2.circle(canvas, b, 3, (60, 255, 120), -1)

            for key, colour in (
                ("source_box", (60, 60, 255)),
                ("target_box", (60, 255, 120)),
            ):
                x, y, w, h = (int(v) for v in clone[key])
                cv2.rectangle(canvas, (x, y), (x + w, y + h), colour, 2)

            path = out.path("clone.jpg")
            photo.save_bgr(path, canvas)
            out.artifact(
                "clone",
                "image",
                "Copy-move matches (red: source, green: copy)",
                file=path,
            )
            sx, sy, sw, sh = clone["source_box"]
            tx, ty, tw, th = clone["target_box"]
            src_box = photo.box_fraction(
                sx / clone_scale,
                sy / clone_scale,
                sw / clone_scale,
                sh / clone_scale,
                full_w,
                full_h,
            )
            dst_box = photo.box_fraction(
                tx / clone_scale,
                ty / clone_scale,
                tw / clone_scale,
                th / clone_scale,
                full_w,
                full_h,
            )
            published["boxes"] += [src_box, dst_box]
            motion = []

            if np.hypot(*clone["shift"]) > 5:
                motion.append(
                    f"moved ({clone['shift'][0] / clone_scale:+.0f}, {clone['shift'][1] / clone_scale:+.0f}) px"
                )

            if abs(clone["angle_deg"]) > 1:
                motion.append(f"rotated {clone['angle_deg']:+.1f}°")

            if abs(clone["scale"] - 1) > 0.02:
                motion.append(f"scaled ×{clone['scale']:.2f}")

            out.finding(
                "critical" if clone["count"] >= 40 else "warning",
                "Cloned region: one patch appears twice",
                f"{clone['count']} distinctive feature points in one area reappear elsewhere in the same picture under a single rigid motion ({', '.join(motion) or 'in place'}). "
                "Cloning stamps one area over another to hide or duplicate something. Genuinely repeated objects (tiles, windows, text) can match too; check the two boxes.",
                tags=["tamper", "clone", "splice"],
                data={"box": dst_box, "source_box": src_box, "matches": clone["count"], "of_pairs": clone["total_pairs"],
                      "shift_px": [clone["shift"][0] / clone_scale, clone["shift"][1] / clone_scale], "angle_deg": clone["angle_deg"], "scale": clone["scale"]},
            )  # fmt: skip

        # ---- resampling ----------------------------------------------------------------------------------
        out.progress(0.6, "resampling traces")
        gray_full = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        resampling = await asyncio.to_thread(_resampling, gray_full)
        spectrum_path = out.path("resampling_spectrum.png")
        cv2.imwrite(
            spectrum_path,
            cv2.resize(
                resampling["spectrum"], (512, 512), interpolation=cv2.INTER_AREA
            ),
        )
        out.artifact("resampling", "image", "Spectrum of the pixel-prediction error", file=spectrum_path,
                     data={"caption": "A camera-original photo is a smooth cloud. Bright isolated dots are periodic correlations left by resizing, rotating or upscaling."})  # fmt: skip
        published["resampling_peak_z"] = resampling["peak_z"]

        if resampling["peak_z"] >= 9.0 and resampling["strong_peaks"] <= 60:
            fx, fy = resampling["peak_frequency"]
            period = 1 / max(1e-6, np.hypot(fx, fy))
            out.finding(
                "notice",
                "Resampled after capture (resized, rotated or upscaled)",
                f"The error of predicting each pixel from its neighbours repeats every {period:.1f} px (peak {resampling['peak_z']:.0f} σ above the spectral background). "
                "Interpolation leaves exactly this rhythm; camera sensors do not. Digital zoom, resizing for the web, rotation and AI upscaling all count.",
                tags=["resampled", "software"],
                data={
                    "peak_z": resampling["peak_z"],
                    "period_px": period,
                    "frequency": [fx, fy],
                    "strong_peaks": resampling["strong_peaks"],
                },
            )

        # ---- histogram -------------------------------------------------------------------------------------
        out.progress(0.75, "tone histogram")
        hist = await asyncio.to_thread(_histogram, gray_full)
        out.artifact("histogram", "series", "Brightness histogram",
                     data={"x": list(range(256)), "series": [{"name": "pixels", "y": hist["hist"].tolist()}], "x_unit": "level", "y_unit": "pixels"})  # fmt: skip
        published["histogram"] = {k: v for k, v in hist.items() if k != "hist"}

        if hist["gaps"] + hist["spikes"] >= 12:
            out.finding(
                "notice",
                f"Tone curve applied: {hist['gaps']} gaps and {hist['spikes']} spikes in the histogram",
                "Stretching contrast, lifting shadows or changing brightness in 8-bit software leaves a comb of empty and over-full levels. A camera writes a smooth histogram. "
                "This is an adjustment, not necessarily a deception.",
                tags=["software", "adjusted"],
                data={k: v for k, v in hist.items() if k != "hist"},
            )

        if hist["clipped_white"] > 0.08 or hist["clipped_black"] > 0.12:
            out.finding(
                "info",
                f"{max(hist['clipped_white'], hist['clipped_black']) * 100:.0f}% of pixels clipped to pure {'white' if hist['clipped_white'] > hist['clipped_black'] else 'black'}",
                "Large flat areas of pure white or black hold no detail at all: nothing can be recovered there by enhancement.",
                tags=["exposure"],
                data={
                    "clipped_white": hist["clipped_white"],
                    "clipped_black": hist["clipped_black"],
                },
            )

        # ---- examiner's views ---------------------------------------------------------------------------------
        out.progress(0.85, "examiner's views")
        gradient = await asyncio.to_thread(_gradient_view, analysis)
        gradient_path = out.path("gradient.jpg")
        photo.save_bgr(gradient_path, gradient)
        out.artifact("gradient", "heatmap", "Luminance gradient (colour = direction light falls)", file=gradient_path,
                     data={"caption": "Hue shows the direction of brightness change at each edge. Lit the same way, all objects share the same hue pattern; a pasted object lit differently stands out.", "overlay": True})  # fmt: skip
        pcas = await asyncio.to_thread(_pca_views, analysis)

        for index, view in enumerate(pcas[1:], start=2):
            pca_path = out.path(f"pca{index}.jpg")
            cv2.imwrite(pca_path, view, [cv2.IMWRITE_JPEG_QUALITY, 88])
            out.artifact(f"pca{index}", "heatmap", f"Principal component {index} of the colours", file=pca_path,
                         data={"caption": "Colour variation the eye hardly sees. Edits made in a different colour space, and halos around pasted objects, show up here.", "overlay": True})  # fmt: skip

        noise_view = await asyncio.to_thread(_noise_view, analysis)
        noise_view_path = out.path("noise_view.png")
        cv2.imwrite(noise_view_path, noise_view)
        out.artifact("noise_view", "heatmap", "Sensor grain, amplified 10×", file=noise_view_path,
                     data={"caption": "The grain itself. It should look like fine, even sand everywhere; a region that is smooth, blotchy or differently textured was processed separately.", "overlay": True})  # fmt: skip

        if not flagged:
            out.finding(
                "info",
                "No pasted or cloned regions found",
                "Error levels, grain and feature matching are consistent across the picture: nothing a copy-paste or clone stamp usually leaves. "
                "A skilled edit saved at high quality can pass these tests; the maps are attached for a human eye.",
                tags=["tamper"],
            )

        published["flagged"] = flagged
        ctx.publish(self.name, published)
        out.progress(1.0)
