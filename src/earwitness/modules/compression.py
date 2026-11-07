"""The anatomy of the compression: what the encoder left behind.

A JPEG's quantisation tables say how hard it was compressed and which software
did it; its segment order, Huffman tables and chroma subsampling say whether a
camera or an editor wrote it. Recompressing the picture at every quality and
watching the error ("JPEG ghosts") reveals an earlier save at a different
quality, overall or in one pasted region. The 8×8 block grid that JPEG leaves
in the pixels can be found even in a PNG, and a region whose grid is shifted
was pasted in from another JPEG.
"""

import asyncio
from typing import Any

import cv2
import numpy as np

from earwitness import photo
from earwitness.modules.base import CaseContext
from earwitness.modules.base import Reporter
from earwitness.modules.base import Skip
from earwitness.modules.base import downsample_series

_GHOST_QUALITIES = list(range(30, 101, 2))
_GHOST_CROP = 1024
_GRID_TILE = 64


def _aligned_crops(
    gray: np.ndarray, size: int, count: int = 4
) -> list[tuple[int, int]]:
    """Top-left corners (multiples of 8) of up to ``count`` crops spread over the image."""
    h, w = gray.shape[:2]
    size = min(size, h - h % 8, w - w % 8)

    if size < 64:
        return [(0, 0)]

    if h <= size and w <= size:
        return [(0, 0)]

    ys = np.linspace(0, h - size, 2).astype(int) // 8 * 8
    xs = np.linspace(0, w - size, 2).astype(int) // 8 * 8
    corners = sorted({(int(y), int(x)) for y in ys for x in xs})

    return corners[:count]


def ghost_curve(rgb: np.ndarray) -> dict[str, Any]:
    """Mean squared recompression error at each quality, averaged over aligned crops."""
    corners = _aligned_crops(rgb, _GHOST_CROP)
    size = min(
        _GHOST_CROP, rgb.shape[0] - rgb.shape[0] % 8, rgb.shape[1] - rgb.shape[1] % 8
    )
    crops = [rgb[y : y + size, x : x + size] for y, x in corners]
    curve = np.zeros(len(_GHOST_QUALITIES))
    block_maps: dict[int, np.ndarray] = {}
    block = 16

    for index, quality in enumerate(_GHOST_QUALITIES):
        errors = []
        maps = []

        for crop in crops:
            again = photo.decode_jpeg_bytes(photo.encode_jpeg_bytes(crop, quality))

            if again is None or again.shape != crop.shape:
                continue

            diff = (crop.astype(np.float32) - again.astype(np.float32)) ** 2
            diff = diff.mean(axis=2)
            errors.append(float(diff.mean()))
            hb, wb = diff.shape[0] // block, diff.shape[1] // block
            maps.append(
                diff[: hb * block, : wb * block]
                .reshape(hb, block, wb, block)
                .mean(axis=(1, 3))
            )

        curve[index] = float(np.mean(errors)) if errors else np.nan
        block_maps[quality] = maps

    return {
        "qualities": np.array(_GHOST_QUALITIES),
        "curve": curve,
        "block_maps": block_maps,
        "corners": corners,
        "size": size,
    }


def ghost_minima(qualities: np.ndarray, curve: np.ndarray) -> list[dict[str, float]]:
    """Local dips in the ghost curve, with their depth relative to the trend."""
    out = []
    valid = np.isfinite(curve)

    if valid.sum() < 5:
        return out

    log = np.log(curve + 1e-9)

    for i in range(2, len(curve) - 2):
        if not valid[i - 2 : i + 3].all():
            continue

        window = log[i - 2 : i + 3]
        neighbours = (log[i - 2] + log[i + 2]) / 2

        if log[i] < log[i - 1] and log[i] <= log[i + 1] and log[i] < neighbours:
            depth = float(
                np.exp(neighbours - log[i]) - 1
            )  # relative dip: 0.1 = 10% below the trend

            if depth >= 0.08 and window.argmin() == 2:
                out.append({"quality": float(qualities[i]), "depth": depth})

    return sorted(out, key=lambda d: -d["depth"])


def grid_profile(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Edge energy by column phase (mod 8) and by row phase.

    Each column's mean |difference| is taken first, then the MEDIAN over the
    columns sharing a phase, so one strong edge (a wall, a pole) cannot stand in
    for a grid: a grid is many boundaries, every 8 pixels.
    """
    g = gray.astype(np.float32)
    dx = np.abs(np.diff(g, axis=1)).mean(axis=0)  # per column boundary before x
    dy = np.abs(np.diff(g, axis=0)).mean(axis=1)
    cols = np.arange(1, g.shape[1]) % 8
    rows = np.arange(1, g.shape[0]) % 8
    col_profile = np.array(
        [np.median(dx[cols == k]) if (cols == k).any() else 0 for k in range(8)]
    )
    row_profile = np.array(
        [np.median(dy[rows == k]) if (rows == k).any() else 0 for k in range(8)]
    )

    return col_profile, row_profile


def grid_strength(profile: np.ndarray) -> tuple[int, float]:
    """(phase of the strongest boundary, how far it stands above the other phases, in sigmas).

    A pattern that repeats every 4 pixels (HEVC transform blocks, UI tiles) lifts
    the opposite phase nearly as much; that is not a JPEG grid and scores zero.
    """
    phase = int(np.argmax(profile))
    others = np.delete(profile, phase)
    spread = np.std(others) + 1e-6
    strength = float((profile[phase] - others.mean()) / spread)
    opposite = profile[(phase + 4) % 8]
    rest = np.delete(profile, [phase, (phase + 4) % 8])

    if opposite >= rest.mean() + 0.6 * (profile[phase] - rest.mean()):
        return phase, 0.0

    return phase, strength


def grid_map(gray: np.ndarray, tile: int = _GRID_TILE) -> dict[str, Any]:
    """Per-tile JPEG grid phase and strength."""
    h, w = gray.shape
    hb, wb = h // tile, w // tile
    phases = np.zeros((hb, wb, 2), dtype=int)
    strengths = np.zeros((hb, wb))

    for by in range(hb):
        for bx in range(wb):
            patch = gray[by * tile : (by + 1) * tile, bx * tile : (bx + 1) * tile]

            if patch.std() < 2:  # flat sky or wall: no grid to see
                continue

            cp, rp = grid_profile(patch)
            px, sx = grid_strength(cp)
            py, sy = grid_strength(rp)
            phases[by, bx] = (px, py)
            strengths[by, bx] = min(sx, sy)

    return {"phases": phases, "strengths": strengths, "tile": tile}


def local_ghosts(
    ghost: dict[str, Any], shape: tuple[int, ...], current: int | None
) -> dict[str, Any] | None:
    """Per-block ghost anomaly: how much lower a block's recompression error is,
    at some quality below the file's own, than the picture as a whole shows
    there. Blocks are 32 px, normalised by their own mean error so smooth and
    textured areas compare, spatially median-filtered so single noisy blocks
    never count, and reported as the most anomalous connected region."""
    block = 16
    qualities = list(ghost["qualities"])
    maps = ghost["block_maps"]
    hb, wb = shape[0] // block, shape[1] // block
    cube = np.full((len(qualities), hb, wb), np.nan, dtype=np.float32)

    for index, quality in enumerate(qualities):
        for (y, x), m in zip(ghost["corners"], maps.get(quality) or []):
            h_, w_ = min(m.shape[0], hb - y // block), min(m.shape[1], wb - x // block)
            cube[index, y // block : y // block + h_, x // block : x // block + w_] = m[
                :h_, :w_
            ]

    covered = ~np.isnan(cube[0])

    if covered.sum() < 64:
        return None

    # 2x2 pooling to 32 px blocks.
    hb2, wb2 = hb // 2, wb // 2

    if hb2 < 4 or wb2 < 4:
        return None

    pooled = np.nanmean(
        cube[:, : hb2 * 2, : wb2 * 2].reshape(len(qualities), hb2, 2, wb2, 2),
        axis=(2, 4),
    )
    covered2 = ~np.isnan(pooled[0])
    pooled = np.nan_to_num(pooled)
    mean_err = pooled.mean(axis=0)
    eligible_blocks = covered2 & (
        mean_err > 1.0
    )  # flat blocks have no quantisation story to tell

    if eligible_blocks.sum() < 32:
        return None

    norm = pooled / (mean_err[None] + 1e-6)
    reference = np.array(
        [np.median(norm[i][eligible_blocks]) for i in range(len(qualities))]
    )
    ceiling = (current - 8) if current else 92
    anomaly = np.zeros((hb2, wb2), dtype=np.float32)
    at_quality = np.zeros((hb2, wb2), dtype=np.int32)

    for index, quality in enumerate(qualities):
        if (
            quality > ceiling
            or quality < 36
            or index < 1
            or index >= len(qualities) - 1
        ):
            continue

        # Smoothed over neighbouring qualities so harmonic flicker cancels.
        local = (norm[index - 1] + norm[index] + norm[index + 1]) / 3
        ref = (reference[index - 1] + reference[index] + reference[index + 1]) / 3
        score = ref - local
        better = score > anomaly
        anomaly = np.where(better, score, anomaly)
        at_quality = np.where(better, quality, at_quality)

    anomaly = np.where(eligible_blocks, anomaly, 0).astype(np.float32)
    smoothed = cv2.medianBlur(anomaly, 3)
    values = smoothed[eligible_blocks]
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)) * 1.4826) + 1e-3
    z = (smoothed - median) / mad
    hot = eligible_blocks & (z > 5) & (smoothed > 0.35)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        hot.astype(np.uint8), connectivity=4
    )
    region = None

    for label_index, s in enumerate(stats[1:], start=1):
        if s[4] < 6 or s[4] > 0.3 * eligible_blocks.sum():
            continue

        mask = labels == label_index
        qualities_here = at_quality[mask]
        mode = int(np.bincount(qualities_here).argmax())

        # One paste has one history: most of its blocks must agree on the quality.
        if (np.abs(qualities_here - mode) <= 4).mean() < 0.6:
            continue

        candidate = {
            "x": int(s[0]) * 32,
            "y": int(s[1]) * 32,
            "w": int(s[2]) * 32,
            "h": int(s[3]) * 32,
            "blocks": int(s[4]),
            "depth": float(smoothed[mask].mean()),
            "quality": mode,
        }

        if region is None or candidate["blocks"] > region["blocks"]:
            region = candidate

    heat = np.clip(smoothed / (np.percentile(values, 99.5) + 1e-6), 0, 1).astype(
        np.float32
    )
    quality = (
        region["quality"]
        if region
        else int(np.bincount(at_quality[eligible_blocks]).argmax())
    )

    return {"heat": heat, "quality": quality, "region": region}


class Compression:
    name = "jpeg"
    title = "COMPRESSION ANATOMY"
    description = "Quantisation tables and quality, encoder fingerprints, JPEG ghosts (earlier saves), and the 8×8 block grid, including shifted grids in pasted regions."

    async def run(self, ctx: CaseContext, out: Reporter) -> None:
        image = ctx.image()
        info = ctx.image_info()

        if image is None or info is None:
            raise Skip("not a decodable image")

        with open(ctx.path, "rb") as handle:
            data = handle.read()

        jpeg = photo.jpeg_anatomy(data) if data[:2] == b"\xff\xd8" else None
        published: dict[str, Any] = {"is_jpeg": jpeg is not None, "quality": None}
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        megapixels = image.shape[0] * image.shape[1] / 1e6
        bits_per_pixel = len(data) * 8 / (image.shape[0] * image.shape[1])

        # ---- JPEG tables and encoder fingerprints -----------------------------------------
        if jpeg is not None:
            out.progress(0.1, "reading quantisation tables")
            estimate = photo.estimate_quality(jpeg.quant)

            for table_id, table in sorted(jpeg.quant.items()):
                out.artifact(
                    f"quant_{table_id}", "table",
                    f"Quantisation table {table_id} ({'luminance' if table_id == 0 else 'chrominance'})",
                    data={"columns": [f"c{i}" for i in range(8)], "rows": table.reshape(8, 8).astype(int).tolist()},
                )  # fmt: skip

            out.artifact(
                "segments", "table", "JPEG segments in file order",
                data={"columns": ["offset", "segment", "bytes", "note"],
                      "rows": [[s["offset"], s.get("app") or s["name"], s["length"], s.get("text", "") or (f"{s['scan_bytes']:,} bytes of entropy-coded data" if "scan_bytes" in s else "")] for s in jpeg.segments]},
            )  # fmt: skip

            if estimate:
                published["quality"] = estimate.quality
                published["standard_tables"] = estimate.standard_tables
                kind = (
                    "libjpeg-standard tables"
                    if estimate.standard_tables
                    else "custom tables"
                )
                out.finding(
                    "info",
                    f"Saved at JPEG quality ≈ {estimate.quality} ({kind})",
                    f"The quantisation tables match libjpeg's quality {estimate.quality} scaling" + (f" to within {estimate.distance:.2f} steps" if estimate.standard_tables else f" only loosely ({estimate.distance:.1f} steps off): the encoder used its own tables") +
                    f". {jpeg.sampling} chroma subsampling, {'progressive' if jpeg.progressive else 'baseline'}, {jpeg.huffman_tables} Huffman segment{'s' if jpeg.huffman_tables != 1 else ''}"
                    f"{' (optimised)' if jpeg.optimised_huffman else ' (standard)' if jpeg.optimised_huffman is False else ''}, "
                    f"{bits_per_pixel:.2f} bits per pixel over {megapixels:.1f} megapixels"
                    + (f", restart interval {jpeg.restart_interval}" if jpeg.restart_interval else "") + ".",
                    tags=["jpeg", "quality"],
                    data={"quality": estimate.quality, "table_distance": estimate.distance, "standard_tables": estimate.standard_tables,
                          "luma_mean": estimate.luma_mean, "chroma_mean": estimate.chroma_mean, "sampling": jpeg.sampling,
                          "progressive": jpeg.progressive, "optimised_huffman": jpeg.optimised_huffman, "bits_per_pixel": bits_per_pixel,
                          "restart_interval": jpeg.restart_interval, "precision_bits": jpeg.precision, "components": jpeg.components},
                )  # fmt: skip

                if not estimate.standard_tables:
                    out.finding(
                        "notice",
                        "Compression tables are not a camera's",
                        "Cameras and most libraries scale the standard tables of the JPEG specification. These tables follow a different recipe, as "
                        "Photoshop, some web exporters and messaging apps use, so this copy was written by such software.",
                        tags=["jpeg", "software"],
                        data={
                            "table_distance": estimate.distance,
                            "luma_table": jpeg.quant[0].astype(int).tolist(),
                        },
                    )

            signs = []

            if jpeg.progressive:
                signs.append("progressive scan order (cameras write baseline)")

            if jpeg.optimised_huffman:
                signs.append(
                    "optimised Huffman tables (cameras write the standard tables)"
                )

            if jpeg.sampling == "4:4:4" and (jpeg.components == 3):
                signs.append("no chroma subsampling (phones always subsample)")

            if jpeg.adobe is not None:
                signs.append("an APP14 'Adobe' marker")

            if jpeg.arithmetic:
                signs.append("arithmetic coding (very unusual)")

            if signs:
                out.finding(
                    "notice",
                    "Encoder fingerprint: re-saved by software",
                    "The way this JPEG is written is not how a camera writes one: "
                    + "; ".join(signs)
                    + ". The picture passed through an editor, "
                    "converter, website or messaging app after capture. That alone does not mean the content changed.",
                    tags=["jpeg", "software", "resaved"],
                    data={"signs": signs, "app_segments": jpeg.app_names},
                )
            elif (
                not jpeg.optimised_huffman
                and not jpeg.progressive
                and jpeg.adobe is None
            ):
                out.finding(
                    "info",
                    "Encoder fingerprint: camera-style JPEG",
                    "Baseline scan, standard Huffman tables"
                    + (f", {jpeg.sampling} subsampling" if jpeg.sampling != "?" else "")
                    + (
                        f", restart markers every {jpeg.restart_interval} blocks (a hardware encoder habit)"
                        if jpeg.restart_interval
                        else ""
                    )
                    + ": the way cameras and phones write a JPEG. Consistent with a file that has not been re-saved.",
                    tags=["jpeg", "camera"],
                    data={"app_segments": jpeg.app_names},
                )

            if jpeg.comments:
                out.finding(
                    "notice",
                    f"Comment inside the JPEG: “{jpeg.comments[0][:80]}”",
                    "A COM segment carries free text written by software or a person. Cameras never write one; libjpeg-based tools and some editors do.",
                    tags=["jpeg", "software", "text"],
                    data={"comments": jpeg.comments[:5]},
                )

        # ---- JPEG ghosts: earlier saves ---------------------------------------------------
        out.progress(0.3, "recompressing at every quality (JPEG ghosts)")
        ghost = await asyncio.to_thread(ghost_curve, image)
        curve = ghost["curve"]
        qualities = ghost["qualities"]
        current = published.get("quality")
        marks = {}

        if current:
            marks[f"this file ≈ q{current}"] = float(current)

        dips = ghost_minima(qualities, curve)
        # Dips above the file's own quality are quantiser harmonics (a step that
        # divides the current one exactly), not earlier saves: only lower ones count.
        earlier = [
            d
            for d in dips
            if (current is None and d["depth"] >= 0.15)
            or (
                current is not None
                and d["quality"] <= current - 6
                and d["depth"] >= 0.10
            )
        ]

        if earlier:
            marks[f"earlier save ≈ q{earlier[0]['quality']:.0f}"] = earlier[0][
                "quality"
            ]

        out.artifact(
            "ghost", "series", "JPEG ghost curve: recompression error at each quality",
            data=downsample_series(qualities, {"mean squared error": curve}) | {"x_unit": "quality", "y_unit": "MSE", "marks": marks},
        )  # fmt: skip
        published["ghost_dips"] = dips

        # Local ghosts: a pasted patch that was once a lower-quality JPEG dips at
        # that quality only where it sits, even when the whole-picture curve does
        # not move. Each 16-px block gets its own curve.
        local = await asyncio.to_thread(local_ghosts, ghost, image.shape, current)
        region_box = None

        if local is not None:
            heat, best_quality, region = (
                local["heat"],
                local["quality"],
                local["region"],
            )
            path = out.path("ghost_map.jpg")
            photo.save_bgr(
                path,
                photo.heat_overlay(
                    photo.fit(image, 1600), heat, cv2.COLORMAP_INFERNO, 0.7
                ),
            )
            out.artifact(
                "ghost_map", "heatmap", f"JPEG ghost map (strongest earlier quality ≈ {best_quality})",
                file=path, data={"caption": f"Bright: blocks that recompress unusually well at quality {best_quality}, i.e. were saved at that quality before the rest of the picture", "overlay": True},
            )  # fmt: skip

            # The region feeds the pixel check as a lead for corroboration; on its
            # own it is too easily imitated by texture to be reported.
            if region is not None:
                region_box = photo.box_fraction(
                    region["x"],
                    region["y"],
                    region["w"],
                    region["h"],
                    image.shape[1],
                    image.shape[0],
                )
                published["ghost_region"] = region_box
                published["ghost_region_quality"] = best_quality

        if earlier and jpeg is not None:
            best = earlier[0]
            out.finding(
                "warning" if best["depth"] >= 0.15 else "notice",
                f"Compressed twice: traces of an earlier save at quality ≈ {best['quality']:.0f}",
                f"Recompressing the picture at every quality shows a dip of {best['depth'] * 100:.0f}% at quality {best['quality']:.0f}, apart from the file's own quality"
                f"{f' ({current})' if current else ''}. A picture saved once has one dip; a second dip is the ghost of an earlier JPEG save of the whole picture.",
                tags=["jpeg", "resaved", "double-compression"],
                data={
                    "earlier_quality": best["quality"],
                    "depth": best["depth"],
                    "current_quality": current,
                    "all_dips": dips,
                },
            )
        elif earlier and jpeg is None:
            best = earlier[0]
            out.finding(
                "notice",
                f"This {info.format} was a JPEG saved at quality ≈ {best['quality']:.0f}",
                f"Recompressing the picture shows a {best['depth'] * 100:.0f}% dip at quality {best['quality']:.0f}: the ghost of a JPEG save before the lossless conversion.",
                tags=["jpeg", "resaved", "converted"],
                data={"earlier_quality": best["quality"], "depth": best["depth"]},
            )
        elif jpeg is not None:
            out.finding(
                "info",
                "No ghost of an earlier JPEG save",
                "Recompressing at every quality shows a single dip at the file's own quality, everywhere in the picture. Nothing points to a second JPEG generation"
                " (a re-save at a much higher quality than the first would hide one).",
                tags=["jpeg"],
                data={"dips": dips},
            )

        # ---- block grid -------------------------------------------------------------------------
        out.progress(0.7, "measuring the 8×8 block grid")
        col_profile, row_profile = await asyncio.to_thread(grid_profile, gray)
        phase_x, strength_x = grid_strength(col_profile)
        phase_y, strength_y = grid_strength(row_profile)
        strength = min(strength_x, strength_y)
        out.artifact(
            "grid_profile", "bars", "Edge energy by pixel position within the 8×8 grid",
            data={"labels": [f"col {k}" for k in range(8)] + [f"row {k}" for k in range(8)],
                  "values": np.round(np.concatenate([col_profile, row_profile]), 4).tolist()},
        )  # fmt: skip
        published["grid"] = {"phase": [phase_x, phase_y], "strength": strength}
        hevc = info.format in ("HEIF", "AVIF")  # HEVC/AV1 blocks are not a JPEG grid
        grid = await asyncio.to_thread(grid_map, gray)
        strengths, phases = grid["strengths"], grid["phases"]
        confident = strengths > 4.0
        agreement = 0.0
        dominant = np.array([phase_x, phase_y])

        if confident.sum() >= 8:
            vals, counts = np.unique(
                phases[confident].reshape(-1, 2), axis=0, return_counts=True
            )
            dominant = vals[np.argmax(counts)]
            agreement = float(counts.max() / confident.sum())

        published["grid"] |= {
            "tiles_confident": int(confident.sum()),
            "agreement": agreement,
            "dominant": dominant.tolist(),
        }
        # A real grid is the same grid everywhere; screenshots and tiled textures
        # show strong but inconsistent 8-pixel rhythms.
        grid_everywhere = strength > 3.5 and confident.sum() >= 10 and agreement >= 0.6

        if (
            jpeg is None
            and not hevc
            and grid_everywhere
            and dominant.tolist() == [0, 0]
        ):
            out.finding(
                "warning",
                f"This {info.format} was once a JPEG",
                f"An 8×8 block grid shows through the pixels ({strength:.1f} σ above the other positions, the same alignment in {agreement * 100:.0f}% of "
                f"{int(confident.sum())} textured tiles). {info.format} is lossless and leaves no grid; the picture was a JPEG before it was converted. "
                "Screenshots of JPEGs and re-saved downloads look like this.",
                tags=["jpeg", "resaved", "converted"],
                data={
                    "strength": strength,
                    "agreement": agreement,
                    "col_profile": col_profile.tolist(),
                    "row_profile": row_profile.tolist(),
                },
            )
        elif jpeg is None and not hevc and grid_everywhere:
            out.finding(
                "warning",
                f"JPEG grid offset by ({dominant[0]}, {dominant[1]}) pixels in this {info.format}",
                f"The picture carries an 8×8 compression grid that does not start at the corner (same alignment in {agreement * 100:.0f}% of textured tiles): "
                "it was a JPEG, then cropped by a non-multiple of 8 pixels before being saved losslessly.",
                tags=["jpeg", "resaved", "cropped"],
                data={
                    "phase": dominant.tolist(),
                    "strength": strength,
                    "agreement": agreement,
                },
            )

        # Per-tile grid phase: a pasted JPEG region keeps its own grid alignment.
        # The file's own grid is at (0, 0) by construction, so any confident
        # tile elsewhere is foreign; a real paste gives a cluster of tiles that
        # all agree on one shifted phase.
        if jpeg is not None and confident.sum() >= 4:
            shifted = confident & np.any(phases != 0, axis=2) & (strengths > 5.0)
            fraction_shifted = float(shifted.sum() / max(1, confident.sum()))
            heat = np.where(shifted, np.clip(strengths / 10, 0, 1), 0).astype(
                np.float32
            )
            path = out.path("grid_map.jpg")
            photo.save_bgr(
                path,
                photo.heat_overlay(
                    photo.fit(image, 1600), heat, cv2.COLORMAP_COOL, 0.7
                ),
            )
            out.artifact(
                "grid_map", "heatmap", "Block-grid alignment map",
                file=path, data={"caption": "Coloured: tiles whose 8×8 grid does not start at the picture's own origin; dark: aligned or no grid visible", "overlay": True},
            )  # fmt: skip
            best_cluster = None

            for phase_value in np.unique(phases[shifted].reshape(-1, 2), axis=0):
                same = shifted & np.all(phases == phase_value, axis=2)
                count, labels, stats, _ = cv2.connectedComponentsWithStats(
                    same.astype(np.uint8), connectivity=8
                )

                for s_ in stats[1:]:
                    if s_[4] >= 3 and (
                        best_cluster is None or s_[4] > best_cluster[0][4]
                    ):
                        best_cluster = (s_, phase_value)

            if best_cluster is not None and fraction_shifted < 0.4:
                s_, phase_value = best_cluster
                tile = grid["tile"]
                box = photo.box_fraction(
                    s_[0] * tile,
                    s_[1] * tile,
                    s_[2] * tile,
                    s_[3] * tile,
                    image.shape[1],
                    image.shape[0],
                )
                out.finding(
                    "warning" if s_[4] >= 8 else "notice",
                    f"Region with a shifted JPEG grid (offset {phase_value[0]}, {phase_value[1]} px)",
                    f"{int(s_[4])} neighbouring tiles carry an 8×8 compression grid starting {phase_value[0]} px across and {phase_value[1]} px down from the picture's own grid. "
                    "Every block of one JPEG shares one grid; a patch with its own alignment was pasted in from another JPEG and landed off-grid. "
                    "Fine repeating texture (fabric, brick, screens) can mimic this, so the pixel check looks for a second trace in the same place.",
                    tags=["jpeg", "grid"]
                    + (["splice", "tamper"] if s_[4] >= 8 else []),
                    data={
                        "box": box,
                        "tiles": int(s_[4]),
                        "phase": phase_value.tolist(),
                        "fraction_shifted": fraction_shifted,
                    },
                )
                published["grid_region"] = box

        ctx.publish(self.name, published)
        out.progress(1.0)
