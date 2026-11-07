"""What is in the file that is not the picture.

A picture file can carry anything after its end marker (archives, documents,
other images, the short video of a "motion photo"), text in comment segments
and unknown application blocks, a message in the least significant bits of
lossless pixels, colour under fully transparent pixels, and personal details
(names, e-mail addresses, file paths, device identifiers) in its metadata.
Every byte that is not pixels is accounted for here.
"""

import asyncio
import os
import re
from typing import Any

import cv2
import numpy as np

from earwitness import photo
from earwitness.modules.base import CaseContext
from earwitness.modules.base import Reporter
from earwitness.modules.base import Skip

_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]{2,}\b")
_URL = re.compile(r"\bhttps?://[^\s\"'<>]{6,}", re.I)
_PATH = re.compile(
    r"(?:[A-Za-z]:\\(?:Users|Documents and Settings)\\[^\\\s\"<>]+|/(?:Users|home)/[^/\s\"<>]+)"
)
_PHONE = re.compile(
    r"(?<![\d\w])(?:\+\d{1,3}[\s-]?)?(?:\(?0\d{2,4}\)?[\s-]?)\d{3,4}[\s-]?\d{3,4}(?![\d\w])"
)
_LOSSLESS = {"PNG", "BMP", "TIFF", "GIF", "WEBP"}


def _chi_square_lsb(channel: np.ndarray) -> dict[str, Any]:
    """Westfeld-Pfitzmann: LSB embedding equalises the pair (2k, 2k+1) frequencies.

    Natural photos with real sensor noise have near-random LSBs too, so the
    equalisation alone proves little. What a message leaves behind is a STEP:
    the first part of the image (in scan order) equalised, the rest not. The
    channel is cut into 24 consecutive slices; each gets its own p-value.
    """
    from scipy.stats import chi2

    flat = channel.ravel()
    slices = 24
    step = max(1, len(flat) // slices)
    p_values = []

    for index in range(slices):
        part = flat[index * step : (index + 1) * step]
        hist = np.bincount(part, minlength=256).astype(float)
        even, odd = hist[0::2], hist[1::2]
        expected = (even + odd) / 2
        keep = expected > 4

        if keep.sum() < 8:
            p_values.append(float("nan"))

            continue

        statistic = float((((even - expected) ** 2)[keep] / expected[keep]).sum())
        p_values.append(float(chi2.sf(statistic, df=int(keep.sum()) - 1)))

    p = np.array(p_values)
    valid = np.isfinite(p)
    equalised = valid & (p > 0.01)
    # The longest run of equalised slices starting at the top, followed by at
    # least two natural slices.
    run = 0

    for value in equalised:
        if value:
            run += 1
        else:
            break

    tail_natural = int(((~equalised) & valid)[run:].sum())
    step_found = 2 <= run < slices and tail_natural >= 2

    return {
        "p_values": [round(v, 4) if np.isfinite(v) else None for v in p],
        "fraction": run / slices,
        "step": step_found,
        "all_equalised": bool(equalised[valid].all()) if valid.any() else False,
    }


def _size(count: int) -> str:
    return (
        f"{count / 1e6:.1f} MB"
        if count >= 1e6
        else f"{count / 1e3:.0f} KB"
        if count >= 1e3
        else f"{count} byte"
    )


def _lsb_plane(rgb: np.ndarray) -> np.ndarray:
    return ((rgb & 1) * 255).astype(np.uint8)


def _split_jpegs(blob: bytes, limit: int = 6) -> list[bytes]:
    """Consecutive JPEG images in a trailing blob (MPF depth maps, motion-photo covers)."""
    out = []
    position = 0

    while len(out) < limit:
        start = blob.find(b"\xff\xd8\xff", position)

        if start < 0:
            break

        end = blob.find(b"\xff\xd9", start + 2)

        if end < 0:
            out.append(blob[start:])
            break

        out.append(blob[start : end + 2])
        position = end + 2

    return out


class Hidden:
    name = "hidden"
    title = "HIDDEN DATA"
    description = "Bytes after the end of the image (archives, embedded images, motion-photo video), comments, unknown blocks, LSB steganography test, content under transparency, personal details in metadata."
    depends = ["provenance"]

    async def run(self, ctx: CaseContext, out: Reporter) -> None:
        image = ctx.image()
        info = ctx.image_info()

        if image is None or info is None:
            raise Skip("not a decodable image")

        with open(ctx.path, "rb") as handle:
            data = handle.read()

        loaded = photo.load(ctx.path)
        jpeg = photo.jpeg_anatomy(data) if data[:2] == b"\xff\xd8" else None
        png = photo.png_anatomy(data) if data[:8] == b"\x89PNG\r\n\x1a\n" else None
        published: dict[str, Any] = {"trailing_bytes": 0, "payloads": [], "hidden": []}
        pixel_bytes = image.shape[0] * image.shape[1] * 3
        accounted: list[list[Any]] = [
            ["pixel data (decoded)", f"{pixel_bytes:,}", "the picture itself"]
        ]

        # ---- trailing data -------------------------------------------------------------------
        out.progress(0.1, "looking past the end marker")
        trailing = jpeg.trailing if jpeg else png.trailing if png else b""
        published["trailing_bytes"] = len(trailing)
        has_mpf = bool(jpeg and any("MPF" in a for a in jpeg.app_names))
        samsung_trailer = (
            trailing.rstrip().endswith(b"SEFT")
            or b"Image_UTC_Data" in trailing
            or b"MotionPhoto" in trailing
        )
        samsung_keys = [
            k.decode()
            for k in re.findall(
                rb"(Image_UTC_Data|MCC_Data|Camera_Scene_Info|Color_Display_P3|Camera_Capture_Mode_Info|MotionPhoto_Data|Front_Cam_Selfie_Info|Burst_Shot_Info|DualShot_|Single_Relighting|Panorama_Shot_Info|Depth_Map)",
                trailing,
            )
        ]
        samsung_keys = list(dict.fromkeys(samsung_keys))

        if (
            samsung_trailer
            and trailing.find(b"ftyp") < 0
            and not _split_jpegs(trailing)
        ):
            utc = re.search(rb"Image_UTC_Data(\d{13})", trailing)
            accounted.append(
                [
                    f"Samsung SEF trailer (offset {len(data) - len(trailing):,})",
                    f"{len(trailing):,}",
                    ", ".join(samsung_keys),
                ]
            )
            out.finding(
                "info",
                f"Samsung camera trailer: {', '.join(samsung_keys[:4]) or 'SEF block'}",
                f"{len(trailing):,} bytes after the end marker hold Samsung's own capture record ({', '.join(samsung_keys) or 'SEF'})"
                + (
                    f", including the capture moment in UTC milliseconds ({int(utc.group(1)) // 1000}) that the provenance check compares with the camera clock."
                    if utc
                    else "."
                )
                + " Editors and messaging apps drop this block, so its presence is consistent with a camera original.",
                tags=["metadata", "camera"],
                data={
                    "bytes": len(trailing),
                    "keys": samsung_keys,
                    "utc_ms": int(utc.group(1)) if utc else None,
                },
            )
            trailing = b""

        if trailing.strip(b"\x00"):
            label = photo.identify_bytes(trailing)
            accounted.append(
                [
                    f"after the end marker (offset {len(data) - len(trailing):,})",
                    f"{len(trailing):,}",
                    label,
                ]
            )
            embedded = (
                _split_jpegs(trailing)
                if trailing.lstrip(b"\x00")[:3] == b"\xff\xd8\xff" or has_mpf
                else []
            )
            shown = 0

            for index, blob in enumerate(embedded):
                picture = photo.jpeg_from_bytes(blob)

                if picture is None or picture.size == 0:
                    continue

                path = out.path(f"embedded{index}.jpg")

                with open(path, "wb") as handle:
                    handle.write(blob)

                out.artifact(
                    f"embedded{index}", "image", f"Embedded image {index + 1}: {picture.shape[1]}×{picture.shape[0]} px, {len(blob):,} bytes",
                    file=path, data={"caption": "A whole second JPEG stored after the main picture (depth map, gain map or preview). It can show what the main picture does not."},
                )  # fmt: skip
                shown += 1

            mp4_at = trailing.find(b"ftyp")

            if mp4_at >= 4:
                video = trailing[mp4_at - 4 :]
                path = out.path("motion_photo.mp4")

                with open(path, "wb") as handle:
                    handle.write(video)

                out.artifact("motion_photo", "video", f"Video hidden inside the photo ({_size(len(video))})", file=path,
                             data={"caption": "A 'motion photo' / Live Photo clip stored after the still image. It shows the seconds around the shot, with sound if recorded."})  # fmt: skip
                out.finding(
                    "critical",
                    f"A {_size(len(video))} video is hidden inside this photo",
                    "After the still image ends, the file continues with an MP4 movie: the seconds before and after the shutter, as Samsung 'motion photos' and some Live Photo exports store them. "
                    "Anyone who receives the file receives that clip, including what was said and who moved out of frame.",
                    tags=["hidden", "video", "privacy"],
                    data={
                        "bytes": len(video),
                        "offset": len(data) - len(trailing) + mp4_at - 4,
                    },
                )
                published["hidden"].append("motion video")
            elif shown and (has_mpf or samsung_trailer):
                out.finding(
                    "notice",
                    f"{shown} extra image{'s' if shown != 1 else ''} stored after the picture",
                    f"{len(trailing):,} bytes follow the end of the main JPEG and hold {shown} more JPEG image{'s' if shown != 1 else ''} "
                    + (
                        "declared in a multi-picture (MPF) index: depth or gain maps that phones keep for portrait and HDR effects. "
                        if has_mpf
                        else "in a Samsung trailer. "
                    )
                    + "They are legitimate, but they are also pictures of the scene that the viewer never sees.",
                    tags=["hidden", "embedded"],
                    data={"bytes": len(trailing), "images": shown, "mpf": has_mpf},
                )
                published["hidden"].append("embedded images")
            elif shown:
                out.finding(
                    "warning",
                    f"{shown} undeclared image{'s' if shown != 1 else ''} appended to the file",
                    f"{len(trailing):,} bytes after the end marker hold {shown} JPEG image{'s' if shown != 1 else ''} that no index declares. Viewers show only the first image; the rest ride along unseen.",
                    tags=["hidden", "embedded", "steganography"],
                    data={"bytes": len(trailing), "images": shown},
                )
                published["hidden"].append("appended images")
            else:
                path = out.path("trailing.bin")

                with open(path, "wb") as handle:
                    handle.write(trailing)

                strings = photo.strings_in(trailing, 6, 40)
                out.artifact("trailing", "file", f"Appended data ({len(trailing):,} bytes, {label})", file=path,
                             data={"label": label, "bytes": len(trailing), "strings": strings[:40], "head_hex": trailing[:64].hex(" ")})  # fmt: skip
                severity = ("critical" if label not in ("zero padding", "unidentified binary data", "plain text") or len(trailing) > 65536
                            else "notice" if len(trailing) > 256 else "info")  # fmt: skip
                out.finding(
                    severity,
                    f"{len(trailing):,} bytes hidden after the end of the image: {label}",
                    "Every image viewer stops at the end marker; whatever follows is invisible but travels with the file. "
                    + (
                        "This looks like a complete file of another kind, the classic way to smuggle a document or archive inside a picture. "
                        if label
                        not in (
                            "zero padding",
                            "unidentified binary data",
                            "plain text",
                        )
                        else ""
                    )
                    + (
                        f"Readable fragments: {'; '.join(s[:40] for s in strings[:4])}."
                        if strings
                        else "No readable text inside it."
                    ),
                    tags=["hidden", "steganography", "appended"],
                    data={
                        "bytes": len(trailing),
                        "label": label,
                        "strings": strings[:20],
                    },
                )
                published["hidden"].append(label)
                published["payloads"].append({"label": label, "bytes": len(trailing)})

        # ---- comments and unknown application blocks -----------------------------------------
        out.progress(0.3, "comment and application segments")
        provenance_early = await ctx.wait_for("provenance") or {}
        provenance_make = bool(
            provenance_early.get("make") or provenance_early.get("model")
        )

        if jpeg:
            known = (
                "JFIF",
                "Exif",
                "XMP",
                "ICC_PROFILE",
                "Photoshop",
                "Adobe",
                "MPF",
                "Ducky",
                "JUMBF",
                "AROT",
                "HDR",
            )

            for segment in jpeg.segments:
                app = segment.get("app")

                if not app:
                    continue

                accounted.append(
                    [
                        f"{app} (offset {segment['offset']:,})",
                        f"{segment['length']:,}",
                        "",
                    ]
                )

                if segment["length"] > 2048 and not any(k in app for k in known):
                    blob = data[
                        segment["offset"] + 4 : segment["offset"] + segment["length"]
                    ]
                    strings = photo.strings_in(blob, 6, 30)
                    out.finding(
                        "info" if provenance_make else "notice",
                        f"Unknown application block {app}: {segment['length']:,} bytes",
                        "A JPEG may carry private blocks; this one is not any of the usual kinds (Exif, XMP, ICC, Photoshop, MPF). "
                        + (
                            "Camera makers (Samsung, Huawei) write such vendor blocks, so with a camera named it is expected. "
                            if provenance_make
                            else ""
                        )
                        + (
                            f"Readable fragments: {'; '.join(s[:40] for s in strings[:4])}."
                            if strings
                            else "Nothing readable inside."
                        ),
                        tags=["hidden", "metadata"],
                        data={
                            "segment": app,
                            "bytes": segment["length"],
                            "strings": strings[:20],
                        },
                    )

            for comment in jpeg.comments:
                if comment.strip():
                    out.finding(
                        "notice",
                        f"Comment segment: “{comment[:100]}”",
                        "Free text stored in a JPEG COM segment, invisible in every viewer.",
                        tags=["hidden", "text", "metadata"],
                        data={"comment": comment[:2000]},
                    )

        if png:
            for chunk in png.chunks:
                if chunk["type"] in (
                    "IHDR",
                    "IDAT",
                    "IEND",
                    "PLTE",
                    "tRNS",
                    "gAMA",
                    "cHRM",
                    "sRGB",
                    "iCCP",
                    "pHYs",
                    "bKGD",
                    "sBIT",
                    "hIST",
                    "tIME",
                    "eXIf",
                    "acTL",
                    "fcTL",
                    "fdAT",
                ):
                    continue

                accounted.append(
                    [
                        f"PNG chunk {chunk['type']} (offset {chunk['offset']:,})",
                        f"{chunk['length']:,}",
                        "",
                    ]
                )

                if (
                    chunk["type"]
                    not in (
                        "tEXt",
                        "zTXt",
                        "iTXt",
                        "caBX",
                        "orNT",
                        "vpAg",
                        "oFFs",
                        "sTER",
                        "npTc",
                        "cICP",
                        "mDCv",
                        "cLLi",
                    )
                    and chunk["length"] > 256
                ):
                    out.finding(
                        "warning",
                        f"Non-standard PNG chunk “{chunk['type']}”: {chunk['length']:,} bytes",
                        "PNG allows private chunks, and decoders ignore any they do not know. A large unknown chunk is a hiding place; nothing displays it.",
                        tags=["hidden", "steganography"],
                        data={"chunk": chunk["type"], "bytes": chunk["length"]},
                    )

            for key, value in png.text.items():
                accounted.append([f"PNG text “{key}”", f"{len(value):,}", value[:60]])

                if len(value) > 20_000:
                    out.finding(
                        "notice",
                        f"Very large text chunk “{key}” ({len(value):,} characters)",
                        "A text chunk big enough to hold a document. Generative-AI workflows (ComfyUI, Automatic1111) store their whole recipe this way; so do people hiding text.",
                        tags=["hidden", "text"],
                        data={
                            "key": key,
                            "characters": len(value),
                            "head": value[:400],
                        },
                    )

        # ---- transparency ------------------------------------------------------------------------
        out.progress(0.45, "looking under transparent pixels")

        if loaded is not None and info.mode in ("RGBA", "LA", "PA"):
            try:
                from PIL import Image

                with Image.open(ctx.path) as pil:
                    rgba = np.asarray(pil.convert("RGBA"))

                alpha = rgba[..., 3]
                hidden_mask = alpha == 0

                if hidden_mask.mean() > 0.01:
                    under = rgba[..., :3][hidden_mask]
                    varied = (
                        float(under.std()) > 12
                        and len(np.unique(under.reshape(-1, 3), axis=0)) > 200
                    )

                    if varied:
                        revealed = rgba[..., :3].copy()
                        path = out.path("under_alpha.png")
                        cv2.imwrite(path, cv2.cvtColor(revealed, cv2.COLOR_RGB2BGR))
                        out.artifact("under_alpha", "image", "The picture with transparency removed", file=path,
                                     data={"caption": "Everything stored under fully transparent pixels. Viewers never show it."})  # fmt: skip
                        out.finding(
                            "critical",
                            f"Hidden picture under transparent pixels ({hidden_mask.mean() * 100:.0f}% of the image)",
                            "Fully transparent pixels still store a colour. Here they store varied, image-like content rather than a flat fill: "
                            "something was hidden, or a cut-out kept its original background. Setting the transparency aside reveals it.",
                            tags=["hidden", "transparency", "steganography"],
                            data={
                                "transparent_fraction": float(hidden_mask.mean()),
                                "colour_std": float(under.std()),
                            },
                        )
                        published["hidden"].append("content under transparency")
            except OSError, ValueError:
                pass

        # ---- LSB steganography (lossless formats only) ---------------------------------------------
        out.progress(0.6, "least significant bits")

        if info.format in _LOSSLESS and info.bits == 8:
            small = photo.fit(image, 2048)
            tests = await asyncio.to_thread(
                lambda: [_chi_square_lsb(small[..., c]) for c in range(3)]
            )
            scores = [t["fraction"] if t["step"] else 0.0 for t in tests]
            plane = _lsb_plane(photo.fit(image, 1200))
            path = out.path("lsb_plane.png")
            cv2.imwrite(path, cv2.cvtColor(plane, cv2.COLOR_RGB2BGR))
            out.artifact("lsb", "image", "Least significant bit of every pixel", file=path,
                         data={"caption": "Natural photos give structured, image-like noise here; a message hidden by LSB replacement gives uniform random snow, often in a band from the top."})  # fmt: skip
            published["lsb_scores"] = scores
            worst = max(scores)
            out.artifact("lsb_test", "series", "LSB pair test: p-value per slice of the image (top to bottom)",
                         data={"x": list(range(1, 25)), "series": [{"name": name, "y": [v if v is not None else 0 for v in t["p_values"]]} for name, t in zip(("red", "green", "blue"), tests)], "x_unit": "slice", "y_unit": "p"})  # fmt: skip

            if worst > 0:
                out.finding(
                    "warning",
                    f"Least significant bits look like a hidden message (first {worst * 100:.0f}% of the image)",
                    "In a natural image the counts of neighbouring values (2k and 2k+1) differ; replacing the lowest bit with message bits makes them equal. "
                    f"The chi-square test finds that equalisation from the top of the picture down to {worst * 100:.0f}% of it, then natural statistics below: the signature of a message written sequentially into the lowest bits.",
                    tags=["hidden", "steganography", "lsb"],
                    data={"channel_scores": scores, "tests": tests},
                )
                published["hidden"].append("LSB message")
            else:
                out.finding(
                    "info",
                    "No sign of a message in the least significant bits",
                    "The pair-of-values test shows no step between an equalised region and a natural one in any channel"
                    + (
                        " (the lowest bits are noise-like everywhere, as sensor noise makes them)."
                        if any(t["all_equalised"] for t in tests)
                        else "."
                    ),
                    tags=["steganography"],
                    data={"tests": tests},
                )
        elif jpeg:
            out.log("LSB test skipped: lossy JPEG pixels carry no LSB message")

        # ---- personal details in metadata ------------------------------------------------------------
        out.progress(0.8, "personal details in metadata")
        provenance = await ctx.wait_for("provenance") or {}
        texts: list[str] = []

        if loaded is not None:
            exif = photo.parse_exif(loaded.exif)

            if exif:
                texts += [
                    str(v)
                    for ifd in exif.ifds.values()
                    for v in ifd.values()
                    if isinstance(v, str)
                ]

            texts += [loaded.xmp or ""]
            texts += list(loaded.text.values())
            texts += [loaded.iptc.decode("utf-8", "replace") if loaded.iptc else ""]

        if jpeg:
            texts += jpeg.comments

        joined = "\n".join(texts)
        details: dict[str, list[str]] = {}

        for label, pattern in (
            ("e-mail addresses", _EMAIL),
            ("web links", _URL),
            ("file paths", _PATH),
        ):
            found = sorted({m.group(0) for m in pattern.finditer(joined)})
            found = [
                f
                for f in found
                if not (
                    label == "web links"
                    and (
                        "ns.adobe.com" in f
                        or "w3.org" in f
                        or "purl.org" in f
                        or "iptc.org" in f
                        or "cipa.jp" in f
                        or "aomedia" in f
                        or "apple.com/" in f
                        and "quicktime" in f
                    )
                )
            ]

            if found:
                details[label] = found[:10]

        if details:
            summary = "; ".join(f"{k}: {', '.join(v[:3])}" for k, v in details.items())
            out.finding(
                "critical"
                if ("e-mail addresses" in details or "file paths" in details)
                else "warning",
                "Personal details in the metadata: " + ", ".join(details),
                f"The invisible labels of this file contain {summary}. A file path names a user account and a machine; an e-mail names a person. These survive every ordinary copy.",
                tags=["identity", "privacy", "metadata"],
                data=details,
            )
            published["hidden"].append("personal details")

        if provenance.get("identity"):
            published["hidden"].append("owner or serial")

        # ---- accounting -------------------------------------------------------------------------------
        accounted.append(["whole file", f"{len(data):,}", os.path.basename(ctx.path)])
        out.artifact(
            "accounting",
            "table",
            "Where the bytes are",
            data={"columns": ["part", "bytes", "note"], "rows": accounted},
        )

        if not published["hidden"]:
            out.finding(
                "info",
                "Nothing hidden in the file",
                "No data after the end marker, no unknown blocks or comments, nothing under transparency, no sign of LSB embedding, and no personal details in the labels.",
                tags=["hidden"],
            )

        ctx.publish(self.name, published)
        out.progress(1.0)
