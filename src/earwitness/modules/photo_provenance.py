"""Who took this photo, when, where, with what, and what has touched it since?

A still image carries far more metadata than a video: every EXIF tag of every
IFD (camera, lens, exposure, serial numbers, owner), the GPS fix with the
compass heading of the camera, XMP with an edit history, IPTC captions and
bylines, the ICC colour profile, PNG text chunks, and an embedded thumbnail
that editors often forget to update. Each is read, cross-checked against the
others and against the file name and the pixels, and reported in plain words.
"""

import asyncio
import os
import re
import struct
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from typing import Any

import cv2
import numpy as np

from earwitness import photo
from earwitness.modules.base import CaseContext
from earwitness.modules.base import Reporter
from earwitness.modules.base import Skip
from earwitness.modules.provenance import _boxes
from earwitness.modules.provenance import _reverse_geocode
from earwitness.modules.provenance import sun_position

_FILENAME_TIME = re.compile(
    r"(20\d{2})[-_.]?(\d{2})[-_.]?(\d{2})[-_ T.]?(\d{2})[-_.:]?(\d{2})[-_.:]?(\d{2})"
)
_SCREENSHOT_NAME = re.compile(r"screen\s?shot|screenshot|captura|bildschirmfoto", re.I)

# Software fingerprints in EXIF Software, XMP CreatorTool, history agents, PNG text.
_SIGNATURES: list[tuple[re.Pattern[str], str, str]] = [
    (re.compile(r"photoshop|adobe photoshop", re.I), "Adobe Photoshop", "editor"),
    (re.compile(r"lightroom", re.I), "Adobe Lightroom", "editor"),
    (re.compile(r"camera raw", re.I), "Adobe Camera Raw", "editor"),
    (re.compile(r"adobe firefly", re.I), "Adobe Firefly", "generator"),
    (re.compile(r"\bgimp\b", re.I), "GIMP", "editor"),
    (re.compile(r"snapseed", re.I), "Snapseed", "editor"),
    (re.compile(r"pixelmator", re.I), "Pixelmator", "editor"),
    (re.compile(r"affinity", re.I), "Affinity Photo", "editor"),
    (re.compile(r"luminar", re.I), "Luminar", "editor"),
    (re.compile(r"canva", re.I), "Canva", "editor"),
    (re.compile(r"picsart", re.I), "Picsart", "editor"),
    (re.compile(r"facetune", re.I), "Facetune", "editor"),
    (re.compile(r"\bvsco\b", re.I), "VSCO", "editor"),
    (re.compile(r"paint\.net", re.I), "Paint.NET", "editor"),
    (re.compile(r"capture one", re.I), "Capture One", "editor"),
    (re.compile(r"darktable|rawtherapee", re.I), "darktable / RawTherapee", "editor"),
    (
        re.compile(r"photos\s*\d|apple photos|photos app", re.I),
        "Apple Photos",
        "editor",
    ),
    (re.compile(r"windows photo|microsoft photos", re.I), "Microsoft Photos", "editor"),
    (re.compile(r"google photos|photos\.google", re.I), "Google Photos", "platform"),
    (re.compile(r"picasa", re.I), "Picasa", "editor"),
    (re.compile(r"imagemagick|graphicsmagick", re.I), "ImageMagick", "editor"),
    (re.compile(r"pillow|\bPIL\b", re.I), "Python Pillow", "editor"),
    (re.compile(r"libpng|pnglib", re.I), "libpng (software export)", "editor"),
    (re.compile(r"ffmpeg|lavc|libav", re.I), "FFmpeg", "editor"),
    (re.compile(r"instagram", re.I), "Instagram", "social"),
    (re.compile(r"whatsapp", re.I), "WhatsApp", "social"),
    (re.compile(r"telegram", re.I), "Telegram", "social"),
    (re.compile(r"facebook|meta ai", re.I), "Facebook / Meta", "social"),
    (re.compile(r"twitter|\bx corp", re.I), "Twitter / X", "social"),
    (re.compile(r"midjourney", re.I), "Midjourney", "generator"),
    (
        re.compile(r"dall[·\-\. ]?e|openai", re.I),
        "OpenAI (DALL·E / ChatGPT)",
        "generator",
    ),
    (
        re.compile(
            r"stable diffusion|automatic1111|comfyui|invokeai|novelai|sd\s?xl|flux\.1",
            re.I,
        ),
        "Stable Diffusion / diffusion tooling",
        "generator",
    ),
    (
        re.compile(r"\bimagen\b|gemini|synthid", re.I),
        "Google Imagen / Gemini",
        "generator",
    ),
    (
        re.compile(
            r"leonardo\.ai|ideogram|bing image creator|microsoft designer|grok", re.I
        ),
        "AI image generator",
        "generator",
    ),
    (re.compile(r"screenshot|screen capture", re.I), "Screenshot tool", "screenshot"),
    (
        re.compile(r"\bscanner|epson scan|vuescan|canoscan", re.I),
        "Scanner software",
        "scanner",
    ),
    (re.compile(r"samsung camera|samsung", re.I), "Samsung camera software", "camera"),
    (
        re.compile(r"^(\d{1,2}\.\d(\.\d)?)$"),
        "Apple iOS camera (version only)",
        "camera",
    ),
    (re.compile(r"hdr\+|google camera|gcam|pixel", re.I), "Google Camera", "camera"),
    (
        re.compile(r"oneplus|xiaomi|miui|huawei|oppo|vivo|realme|honor", re.I),
        "Android vendor camera",
        "camera",
    ),
]

_IDENTITY_TAGS = (
    "BodySerialNumber", "CameraOwnerName", "LensSerialNumber", "Artist", "Copyright", "ImageUniqueID",
    "OwnerName", "SerialNumber", "CameraSerialNumber", "InternalSerialNumber",
)  # fmt: skip
_IDENTITY_XMP = (
    "dc:creator", "dc:rights", "photoshop:Credit", "photoshop:AuthorsPosition", "aux:SerialNumber",
    "aux:LensSerialNumber", "aux:ImageNumber", "xmpRights:Owner", "Iptc4xmpCore:CreatorContactInfo",
    "Iptc4xmpCore:CiEmailWork", "Iptc4xmpCore:CiTelWork", "Iptc4xmpCore:CiAdrCity", "exif:CameraOwnerName",
    "exif:BodySerialNumber", "exifEX:BodySerialNumber", "exifEX:CameraOwnerName", "exifEX:LensSerialNumber",
    "drone-dji:SerialNumber", "MicrosoftPhoto:CameraSerialNumber", "mwg-rs:Name",
)  # fmt: skip

# Common phone screen sizes (portrait) that mark a screenshot when a PNG has no camera data.
_SCREENS = {
    (1170, 2532), (1179, 2556), (1206, 2622), (1284, 2778), (1290, 2796), (1320, 2868), (1125, 2436), (1242, 2688),
    (828, 1792), (750, 1334), (1080, 1920), (1080, 2340), (1080, 2400), (1440, 3200), (1440, 3120), (1440, 3088),
    (1080, 2316), (1080, 2408), (1080, 2412), (1080, 2436), (1344, 2992), (1080, 2424), (1216, 2688),
}  # fmt: skip


def _icc_description(icc: bytes | None) -> str | None:
    if not icc or len(icc) < 132:
        return None

    try:
        count = struct.unpack(">I", icc[128:132])[0]

        for index in range(min(count, 100)):
            base = 132 + index * 12
            sig, offset, size = struct.unpack(">4sII", icc[base : base + 12])

            if sig != b"desc" or offset + size > len(icc):
                continue

            block = icc[offset : offset + size]

            if block[:4] == b"desc":
                length = struct.unpack(">I", block[8:12])[0]

                return (
                    block[12 : 12 + length]
                    .split(b"\x00", 1)[0]
                    .decode("latin-1", "replace")
                    .strip()
                )

            if block[:4] == b"mluc":
                records = struct.unpack(">I", block[8:12])[0]

                if records:
                    length, rel = struct.unpack(">II", block[20:28])

                    return (
                        block[rel : rel + length]
                        .decode("utf-16-be", "replace")
                        .strip("\x00 ")
                    )
    except struct.error:
        return None

    return None


_IPTC_NAMES = {
    5: "title", 25: "keywords", 55: "date created", 60: "time created", 80: "by-line", 85: "by-line title",
    90: "city", 92: "sub-location", 95: "province/state", 101: "country", 105: "headline", 110: "credit",
    115: "source", 116: "copyright", 120: "caption", 122: "caption writer", 103: "job id", 40: "special instructions",
}  # fmt: skip


def _iptc(blob: bytes | None) -> dict[str, str]:
    out: dict[str, str] = {}

    if not blob:
        return out

    offset = 0

    while offset + 5 <= len(blob):
        if blob[offset] != 0x1C:
            offset += 1

            continue

        record, dataset = blob[offset + 1], blob[offset + 2]
        size = struct.unpack(">H", blob[offset + 3 : offset + 5])[0]

        if size & 0x8000:
            break

        value = blob[offset + 5 : offset + 5 + size].decode("utf-8", "replace")

        if record == 2 and value.strip():
            name = _IPTC_NAMES.get(dataset, f"2:{dataset}")
            out[name] = (out[name] + "; " + value) if name in out else value

        offset += 5 + size

    return out


def _exposure_words(exif: dict[str, Any]) -> str:
    bits = []
    exposure = exif.get("ExposureTime")

    if isinstance(exposure, (int, float)) and exposure > 0:
        bits.append(f"1/{round(1 / exposure)} s" if exposure < 1 else f"{exposure:g} s")

    fnumber = exif.get("FNumber")

    if isinstance(fnumber, (int, float)) and fnumber > 0:
        bits.append(f"f/{fnumber:.1f}")

    iso = exif.get("ISOSpeedRatings") or exif.get("PhotographicSensitivity")

    if isinstance(iso, (tuple, list)):
        iso = iso[0] if iso else None

    if isinstance(iso, (int, float)):
        bits.append(f"ISO {int(iso)}")

    focal = exif.get("FocalLength")
    equiv = exif.get("FocalLengthIn35mmFilm")

    if isinstance(focal, (int, float)) and focal > 0:
        bits.append(
            f"{focal:.1f} mm"
            + (
                f" ({int(equiv)} mm equivalent)"
                if isinstance(equiv, (int, float)) and equiv
                else ""
            )
        )

    zoom = exif.get("DigitalZoomRatio")

    if isinstance(zoom, (int, float)) and zoom > 1.05:
        bits.append(f"digital zoom {zoom:.2f}×")

    flash = exif.get("Flash")

    if isinstance(flash, int):
        bits.append("flash fired" if flash & 1 else "no flash")

    return ", ".join(bits)


def _software_hits(texts: list[str]) -> list[tuple[str, str, str]]:
    seen: list[tuple[str, str, str]] = []

    for text in texts:
        if not text:
            continue

        for pattern, label, role in _SIGNATURES:
            if pattern.search(text) and all(label != s[0] for s in seen):
                seen.append((label, role, text[:160]))

    return seen


def _thumbnail_check(thumb: np.ndarray, image: np.ndarray) -> dict[str, Any]:
    """Compare the embedded thumbnail with the picture it claims to represent.

    Global correlation catches crops and replacements; a block-wise difference
    catches a single removed or pasted object that leaves the rest intact.
    """
    th, tw = thumb.shape[:2]
    small = cv2.resize(image, (tw, th), interpolation=cv2.INTER_AREA)
    a = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY).astype(np.float32)
    b = cv2.cvtColor(thumb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    a = (a - a.mean()) / (a.std() + 1e-6)
    b = (b - b.mean()) / (b.std() + 1e-6)
    correlation = float((a * b).mean())
    blur_small = cv2.GaussianBlur(small, (0, 0), 1.2)
    blur_thumb = cv2.GaussianBlur(thumb, (0, 0), 1.2)
    # Thumbnails are often brighter/darker than the picture: match tone first.
    gain = (blur_small.mean() + 1e-6) / (blur_thumb.mean() + 1e-6)
    diff = np.abs(
        blur_small.astype(np.float32)
        - np.clip(blur_thumb.astype(np.float32) * gain, 0, 255).astype(np.float32)
    )
    diff_gray = diff.mean(axis=2)
    block = max(4, min(th, tw) // 12)
    hb, wb = th // block, tw // block
    blocks = (
        diff_gray[: hb * block, : wb * block]
        .reshape(hb, block, wb, block)
        .mean(axis=(1, 3))
    )
    median = float(np.median(blocks))
    peak = float(blocks.max())
    hot = blocks > max(18.0, 4 * median + 8)
    region = None

    if hot.sum() >= 2:
        ys, xs = np.where(hot)
        region = {
            "x": round(float(xs.min() * block / tw), 4), "y": round(float(ys.min() * block / th), 4),
            "w": round(float((xs.max() - xs.min() + 1) * block / tw), 4), "h": round(float((ys.max() - ys.min() + 1) * block / th), 4),
        }  # fmt: skip

    aspect_image = image.shape[1] / image.shape[0]
    aspect_thumb = tw / th

    return {
        "correlation": correlation,
        "mean_difference": float(diff_gray.mean()),
        "peak_block_difference": peak,
        "median_block_difference": median,
        "changed_blocks": int(hot.sum()),
        "region": region,
        "aspect_image": aspect_image,
        "aspect_thumb": aspect_thumb,
        "aspect_mismatch": abs(aspect_image - aspect_thumb) / aspect_image > 0.04,
        "diff_map": diff_gray,
    }


class PhotoProvenance:
    name = "provenance"
    title = "PHOTO PROVENANCE"
    description = "EXIF, XMP, IPTC, ICC and PNG metadata: camera, lens, serial numbers, GPS with compass heading, timestamps, edit history, embedded thumbnail."

    async def run(self, ctx: CaseContext, out: Reporter) -> None:
        image = ctx.image()
        info = ctx.image_info()

        if image is None or info is None:
            raise Skip("not a decodable image")

        from earwitness import photo as _photo

        loaded = _photo.load(ctx.path)

        if loaded is None:
            raise Skip("not a decodable image")

        with open(ctx.path, "rb") as handle:
            data = handle.read()

        out.progress(0.1, "reading metadata")
        exif = photo.parse_exif(loaded.exif)
        xmp = photo.parse_xmp(loaded.xmp)
        iptc = _iptc(loaded.iptc)
        icc_name = _icc_description(loaded.icc)
        ifd0 = exif.ifds.get("IFD0", {}) if exif else {}
        exif_ifd = exif.ifds.get("Exif", {}) if exif else {}
        gps_ifd = exif.ifds.get("GPS", {}) if exif else {}
        xmp_fields = xmp.fields if xmp else {}
        filename = ctx.options.get("original_name") or os.path.basename(ctx.path)
        published: dict[str, Any] = {"kind": "image"}

        # ---- raw tables -----------------------------------------------------------------
        rows: list[list[Any]] = []

        if exif:
            for ifd, tag, name, value in exif.raw_tags:
                rows.append([f"EXIF {ifd}", f"0x{tag:04X}", name, _show(value)])

        for key, value in xmp_fields.items():
            rows.append(["XMP", "", key, value[:300]])

        for key, value in iptc.items():
            rows.append(["IPTC", "", key, value[:300]])

        for key, value in loaded.text.items():
            rows.append([f"{info.format} text", "", key, value[:500]])

        if icc_name:
            rows.append(["ICC", "", "profile description", icc_name])

        out.artifact(
            "tags", "table", "Every label stored in the file",
            data={"columns": ["where", "tag", "name", "value"], "rows": rows[:600]},
        )  # fmt: skip
        report = {
            "format": info.format, "mode": info.mode, "bits": info.bits, "frames": info.frames,
            "stored_size": [info.stored_width, info.stored_height], "oriented_size": [info.width, info.height],
            "orientation": info.orientation, "exif": exif.ifds if exif else None,
            "exif_byte_order": exif.byte_order if exif else None, "makernote": exif.makernote if exif else None,
            "xmp": xmp_fields or None, "xmp_history": xmp.history if xmp else None,
            "xmp_derived_from": xmp.derived_from if xmp else None, "iptc": iptc or None,
            "icc_profile": icc_name, "icc_bytes": len(loaded.icc or b""), "text_chunks": loaded.text or None,
            "pillow_info": loaded.pil_info,
        }  # fmt: skip
        out.artifact("probe", "json", "Raw metadata report", data=report)

        # ---- file layout --------------------------------------------------------------------
        out.progress(0.2, "mapping the file layout")
        layout_rows: list[list[Any]] = []
        jpeg = photo.jpeg_anatomy(data) if data[:2] == b"\xff\xd8" else None
        png = photo.png_anatomy(data) if data[:8] == b"\x89PNG\r\n\x1a\n" else None

        if jpeg:
            layout_rows = [
                [
                    s["offset"],
                    s.get("app") or s["name"],
                    s["length"],
                    s.get("text", "")
                    or (
                        f"{s['scan_bytes']:,} bytes of image data"
                        if "scan_bytes" in s
                        else ""
                    ),
                ]
                for s in jpeg.segments
            ]
        elif png:
            layout_rows = [
                [c["offset"], c["type"], c["length"], ""]
                for c in png.chunks
                if c["type"] != "IDAT"
            ]
            idat = [c for c in png.chunks if c["type"] == "IDAT"]

            if idat:
                layout_rows.append(
                    [
                        idat[0]["offset"],
                        f"IDAT × {len(idat)}",
                        sum(c["length"] for c in idat),
                        "compressed pixel data",
                    ]
                )
        elif data[4:8] == b"ftyp":
            boxes = await asyncio.to_thread(_boxes, ctx.path)
            layout_rows = [
                [b["offset"], ("  " * b["depth"]) + b["type"], b["size"], ""]
                for b in boxes[:200]
            ]

        if layout_rows:
            out.artifact(
                "boxes", "table", "File layout (segments, chunks or boxes)",
                data={"columns": ["offset", "part", "bytes", "note"], "rows": layout_rows},
            )  # fmt: skip

        # ---- device, lens, exposure ----------------------------------------------------------
        make = _text(ifd0.get("Make")) or _text(xmp_fields.get("tiff:Make"))
        model = _text(ifd0.get("Model")) or _text(xmp_fields.get("tiff:Model"))
        software = (
            _text(ifd0.get("Software"))
            or _text(xmp_fields.get("xmp:CreatorTool"))
            or _text(loaded.text.get("Software"))
        )
        lens = (
            _text(exif_ifd.get("LensModel"))
            or _text(xmp_fields.get("aux:Lens"))
            or _text(xmp_fields.get("exifEX:LensModel"))
        )
        exposure = _exposure_words(exif_ifd)
        published.update(
            {"make": make, "model": model, "software": software, "lens": lens}
        )

        if make or model:
            device = " ".join(dict.fromkeys(x for x in (make, model) if x))
            out.finding(
                "info",
                f"Taken on {device}",
                f"The camera tags name the device{f', lens {lens}' if lens else ''}{f'; {exposure}' if exposure else ''}"
                f"{f'; software {software}' if software else ''}. These labels are written by the camera and can be edited.",
                tags=["device"],
                data={"make": make, "model": model, "lens": lens, "software": software, "exposure": exposure or None,
                      "makernote": exif.makernote if exif else None},
            )  # fmt: skip

        if exif and exif.makernote and (make or model):
            out.finding(
                "info",
                f"Maker notes from {exif.makernote['vendor_prefix']} intact ({exif.makernote['size_bytes']:,} bytes)",
                "The manufacturer's private block survives. Most editors and messaging apps drop or damage it, "
                "so its presence is consistent with a camera original (or a careful copy).",
                tags=["metadata", "camera"],
                data=exif.makernote,
            )

        # ---- identity: serials, owner, bylines ------------------------------------------------
        identity: dict[str, str] = {}

        for tag in _IDENTITY_TAGS:
            for ifd in (ifd0, exif_ifd):
                value = _text(ifd.get(tag))

                if value and value.lower() not in ("0", "none", "unknown"):
                    identity[tag] = value

        for key in _IDENTITY_XMP:
            value = _text(xmp_fields.get(key))

            if value:
                identity[key] = value

        for key in ("by-line", "copyright", "credit", "caption writer", "source"):
            if iptc.get(key):
                identity[f"IPTC {key}"] = iptc[key]

        for key in ("Author", "Artist", "Copyright", "Creator"):
            if loaded.text.get(key):
                identity[f"PNG {key}"] = loaded.text[key]

        if identity:
            serials = {k: v for k, v in identity.items() if "serial" in k.lower()}
            module_ids = {k: v for k, v in identity.items() if k == "ImageUniqueID"}
            names = {
                k: v
                for k, v in identity.items()
                if k not in serials and k not in module_ids
            }

            if names or serials:
                out.finding(
                    "critical" if names else "warning",
                    ("Name in the file: " + "; ".join(list(names.values())[:2]))
                    if names
                    else "Camera serial number stored in the file",
                    "The metadata carries "
                    + ", ".join(f"{k} = “{v}”" for k, v in list(identity.items())[:6])
                    + ". "
                    + (
                        "A name or copyright line identifies the photographer directly. "
                        if names
                        else ""
                    )
                    + (
                        "A serial number ties every photo from this camera together and can be matched to purchase or repair records."
                        if serials
                        else ""
                    ),
                    tags=["identity", "metadata", "privacy"],
                    data=identity,
                )
            else:
                out.finding(
                    "notice",
                    f"Camera module identifier: {module_ids['ImageUniqueID']}",
                    "ImageUniqueID names the camera module or sensor batch (Samsung and others write it). It is not unique to one phone, but it links photos from the same model and firmware.",
                    tags=["identity", "metadata"],
                    data=identity,
                )

        # ---- software --------------------------------------------------------------------------
        out.progress(0.35, "fingerprinting software")
        agents = [h.get("softwareAgent", "") for h in (xmp.history if xmp else [])]
        texts = [software or "", xmp_fields.get("xmp:CreatorTool", ""), *agents, loaded.text.get("Software", ""),
                 loaded.text.get("Comment", ""), loaded.text.get("comment", ""), " ".join(jpeg.comments) if jpeg else "",
                 xmp_fields.get("photoshop:History", "")]  # fmt: skip
        hits = _software_hits(texts)

        # Adobe's own JPEG markers are a fingerprint no camera writes.
        if (
            jpeg
            and (jpeg.adobe is not None or any("Ducky" in a for a in jpeg.app_names))
            and all(h[0] != "Adobe Photoshop" for h in hits)
        ):
            hits.append(
                (
                    "Adobe Photoshop",
                    "editor",
                    "APP14 'Adobe' marker"
                    + (
                        " + APP12 'Ducky' (Save for Web)"
                        if any("Ducky" in a for a in jpeg.app_names)
                        else ""
                    ),
                )
            )
        published["software_seen"] = sorted(label for label, _, _ in hits)
        published["generators"] = [
            label for label, role, _ in hits if role == "generator"
        ]

        for label, role, evidence in hits:
            if role == "camera":
                continue

            severity = "warning" if role in ("editor", "generator") else "notice"
            explanation = {
                "editor": "Camera originals name only the camera firmware; this file names editing software, so it was re-saved by that program (which does not by itself prove the picture was altered).",
                "social": "Messaging and social apps re-encode every picture and strip most metadata; this copy came through one.",
                "platform": "A photo platform exported or re-saved this copy.",
                "generator": "The metadata names an AI image generator. See the AI generation signals check.",
                "screenshot": "This file is a screenshot, not a photograph taken with a camera.",
                "scanner": "This image was scanned from paper or film.",
            }[role]
            out.finding(
                severity,
                f"{'Generated with' if role == 'generator' else 'Processed by'} {label}",
                f"{explanation} Evidence: “{evidence}”.",
                tags=["software", role] + (["editor"] if role == "generator" else []),
                data={"software": label, "role": role, "evidence": evidence},
            )

        if xmp and xmp.history:
            out.artifact(
                "history", "table", "Edit history recorded in XMP",
                data={
                    "columns": ["when", "action", "software", "parameters", "changed"],
                    "rows": [[h.get("when", ""), h.get("action", ""), h.get("softwareAgent", ""), h.get("parameters", ""), h.get("changed", "")] for h in xmp.history],
                },
            )  # fmt: skip
            steps = [
                f"{h.get('action', '?')} ({h.get('softwareAgent', 'unknown software')}{', ' + h['when'][:19] if h.get('when') else ''})"
                for h in xmp.history
            ]
            out.finding(
                "warning",
                f"Edit history: {len(xmp.history)} recorded step{'s' if len(xmp.history) != 1 else ''}",
                "The XMP block keeps a log written by Adobe-style software: "
                + "; ".join(steps[:6])
                + ". "
                "This is the software's own account of what was done and when; it can be stripped but rarely forged.",
                tags=["software", "editor", "history"],
                data={"history": xmp.history},
            )

        if xmp and xmp.derived_from:
            out.finding(
                "notice",
                "Derived from another file",
                "XMP says this image was derived from "
                + ", ".join(f"{k} {v}" for k, v in list(xmp.derived_from.items())[:3])
                + ".",
                tags=["software", "history"],
                data=xmp.derived_from,
            )

        # ---- screenshot ------------------------------------------------------------------------
        screenshot_words = any(
            _SCREENSHOT_NAME.search(str(v))
            for v in (
                filename,
                ifd0.get("ImageDescription"),
                exif_ifd.get("UserComment"),
                xmp_fields.get("exif:UserComment"),
                xmp_fields.get("dc:description"),
            )
        )
        screen_size = (info.width, info.height) in _SCREENS or (
            info.height,
            info.width,
        ) in _SCREENS
        is_screenshot = screenshot_words or (
            screen_size and not make and info.format == "PNG"
        )
        published["screenshot"] = is_screenshot

        if is_screenshot:
            out.finding(
                "notice",
                "This is a screenshot",
                (
                    "The file labels itself a screenshot. "
                    if screenshot_words
                    else f"A {info.width}×{info.height} PNG with no camera data matches a phone screen exactly. "
                )
                + "Nothing in it was captured through a lens, so camera and lighting checks do not apply; the content shown may itself be real or not.",
                tags=["screenshot", "metadata"],
                data={
                    "width": info.width,
                    "height": info.height,
                    "by_label": screenshot_words,
                    "by_size": screen_size,
                },
            )

        # ---- dimensions and orientation -------------------------------------------------------
        exif_w, exif_h = exif_ifd.get("ExifImageWidth"), exif_ifd.get("ExifImageHeight")

        if (
            isinstance(exif_w, int)
            and isinstance(exif_h, int)
            and exif_w > 0
            and exif_h > 0
        ):
            stored = {info.stored_width, info.stored_height}

            if stored != {exif_w, exif_h}:
                ratio = info.stored_width * info.stored_height / (exif_w * exif_h)
                out.finding(
                    "warning",
                    f"Resized or cropped after capture: {exif_w}×{exif_h} became {info.stored_width}×{info.stored_height}",
                    f"The camera recorded the picture as {exif_w}×{exif_h} pixels, but the file holds {info.stored_width}×{info.stored_height} "
                    f"({ratio * 100:.0f}% of the original pixel count). Something re-sampled or cropped it and kept the old labels.",
                    tags=["resized", "inconsistency", "software"],
                    data={
                        "exif_size": [exif_w, exif_h],
                        "stored_size": [info.stored_width, info.stored_height],
                        "pixel_ratio": ratio,
                    },
                )

        if info.orientation != 1:
            out.finding(
                "info",
                f"Stored rotated (orientation {info.orientation}); shown upright",
                "The camera saved the sensor image as-is and recorded how to rotate it. Every measurement here is on the upright picture.",
                tags=["metadata"],
                data={
                    "orientation": info.orientation,
                    "stored": [info.stored_width, info.stored_height],
                },
            )

        # ---- time -----------------------------------------------------------------------------
        out.progress(0.5, "reconciling clocks")
        offset_original = exif_ifd.get("OffsetTimeOriginal") or exif_ifd.get(
            "OffsetTime"
        )
        original = photo.exif_datetime(
            exif_ifd.get("DateTimeOriginal"),
            offset_original,
            exif_ifd.get("SubsecTimeOriginal"),
        )
        digitized = photo.exif_datetime(
            exif_ifd.get("DateTimeDigitized"),
            exif_ifd.get("OffsetTimeDigitized") or offset_original,
        )
        modified = photo.exif_datetime(
            ifd0.get("DateTime"), exif_ifd.get("OffsetTime") or offset_original
        )
        xmp_create = _iso(
            xmp_fields.get("xmp:CreateDate") or xmp_fields.get("photoshop:DateCreated")
        )
        xmp_modify = _iso(xmp_fields.get("xmp:ModifyDate"))
        xmp_metadata = _iso(xmp_fields.get("xmp:MetadataDate"))
        png_time = _iso(loaded.text.get("Creation Time")) or _iso(
            loaded.text.get("date:create")
        )
        fix = photo.gps_fix(gps_ifd)
        gps_time = (
            fix.time_utc if fix and fix.time_utc and fix.time_utc.year > 1900 else None
        )
        utc_clock_source = "GPS receiver"

        # Samsung phones append the capture moment in UTC milliseconds after the image.
        if (
            gps_time is None
            and jpeg is not None
            and (samsung := re.search(rb"Image_UTC_Data(\d{13})", jpeg.trailing))
        ):
            gps_time = datetime.fromtimestamp(
                int(samsung.group(1)) / 1000, tz=timezone.utc
            )
            utc_clock_source = "Samsung capture record"

        file_local = None
        found = _FILENAME_TIME.search(filename)

        if found:
            try:
                file_local = datetime(*(int(g) for g in found.groups()))
            except ValueError:
                file_local = None

        times: dict[str, Any] = {
            "date_time_original": original, "date_time_digitized": digitized, "modify_date": modified,
            "offset_time_original": offset_original, "gps_time_utc": gps_time, "xmp_create_date": xmp_create,
            "xmp_modify_date": xmp_modify, "xmp_metadata_date": xmp_metadata, "png_creation_time": png_time,
            "utc_clock_source": utc_clock_source if gps_time else None,
            "filename_local": file_local, "copy_stamp": False,
        }  # fmt: skip
        time_rows = [
            [
                k.replace("_", " "),
                v.isoformat() if isinstance(v, datetime) else (v or ""),
            ]
            for k, v in times.items()
            if v not in (None, False, "")
        ]

        if time_rows:
            out.artifact(
                "clocks",
                "table",
                "Every clock in the file",
                data={"columns": ["clock", "value"], "rows": time_rows},
            )

        taken = original or digitized or xmp_create or png_time
        offset_h: float | None = None

        if isinstance(offset_original, str) and (
            m := re.match(r"^([+-])(\d{2}):?(\d{2})$", offset_original.strip())
        ):
            offset_h = (1 if m.group(1) == "+" else -1) * (
                int(m.group(2)) + int(m.group(3)) / 60
            )

        # Local wall clock vs GPS UTC gives the time zone, as the filename did for video.
        if taken is not None and taken.tzinfo is None and gps_time is not None:
            delta_h = (taken - gps_time.replace(tzinfo=None)).total_seconds() / 3600
            rounded = round(delta_h * 4) / 4
            residual = abs(delta_h - rounded) * 3600

            if residual <= 90 and abs(rounded) <= 14:
                offset_h = rounded
                taken = taken.replace(tzinfo=timezone(timedelta(hours=rounded)))
                out.finding(
                    "notice",
                    f"Local clock was UTC{rounded:+g}",
                    f"The camera clock says {original or taken:%H:%M:%S} local and the {utc_clock_source} stamped {gps_time:%H:%M:%S} UTC within "
                    f"{residual:.0f} s, so the phone was set to UTC{rounded:+g} where it took the picture. That narrows the region on its own.",
                    tags=["time", "timezone"],
                    data={"offset_h": rounded, "residual_s": residual},
                )
            else:
                out.finding(
                    "warning",
                    "Camera clock and GPS clock disagree",
                    f"The camera says {taken} and the {utc_clock_source} {gps_time:%Y-%m-%d %H:%M:%S} UTC; no whole time-zone offset reconciles them "
                    f"(nearest leaves {residual:.0f} s). A clock was wrong or the date was edited.",
                    tags=["time", "inconsistency"],
                    data={"residual_s": residual},
                )
        elif taken is not None and taken.tzinfo is not None and gps_time is not None:
            gap = abs((taken - gps_time).total_seconds())

            if gap > 120:
                out.finding(
                    "warning",
                    f"Camera clock {gap / 60:.0f} min away from the GPS clock",
                    f"The camera stamped {taken.isoformat()} and the GPS receiver {gps_time.isoformat()}. Phones keep these within seconds; a gap this size means the date or time was edited, or the GPS fix is stale.",
                    tags=["time", "inconsistency"],
                    data={"gap_s": gap},
                )

        if taken is not None:
            when_utc = (
                taken.astimezone(timezone.utc)
                if taken.tzinfo
                else taken.replace(tzinfo=timezone.utc)
            )
            local_text = f"{taken:%A %-d %B %Y, %H:%M:%S}" + (
                f" (UTC{offset_h:+g})"
                if offset_h is not None
                else " (time zone not stated)"
            )
            out.finding(
                "info",
                f"Taken {taken:%Y-%m-%d %H:%M:%S}"
                + (f" UTC{offset_h:+g}" if offset_h is not None else " (local clock)"),
                f"DateTimeOriginal: {local_text}. "
                + (
                    "Converted to UTC it is " + f"{when_utc:%Y-%m-%d %H:%M:%S}. "
                    if offset_h is not None
                    else "Without a stated zone the hour is only as good as the phone's clock. "
                )
                + "A camera writes this at the moment of capture; editors copy it, so it is a claim, not a measurement.",
                tags=["time"],
                data={k: v for k, v in times.items() if v is not None},
            )

        if (
            original
            and digitized
            and abs(
                (
                    original.replace(tzinfo=None) - digitized.replace(tzinfo=None)
                ).total_seconds()
            )
            > 2
        ):
            out.finding(
                "notice",
                "Taken and digitised at different times",
                f"DateTimeOriginal {original} but DateTimeDigitized {digitized}. Cameras write both the same; a scan, import or re-save sets them apart.",
                tags=["time", "metadata"],
                data={"original": original, "digitized": digitized},
            )

        latest_modify = max(
            (t for t in (modified, xmp_modify, xmp_metadata) if t is not None),
            key=lambda t: t.replace(tzinfo=None),
            default=None,
        )

        if taken is not None and latest_modify is not None:
            gap_s = (
                latest_modify.replace(tzinfo=None) - taken.replace(tzinfo=None)
            ).total_seconds()

            if gap_s > 5:
                out.finding(
                    "warning",
                    f"Modified {_span(gap_s)} after it was taken",
                    f"The modification clock reads {latest_modify}, {_span(gap_s)} after the capture time {taken}. Cameras write both identical; a later value means the file was opened and saved by software.",
                    tags=["software", "modified", "time"],
                    data={"taken": taken, "modified": latest_modify, "gap_s": gap_s},
                )

        if taken is not None and file_local is not None:
            naive = taken.replace(tzinfo=None)
            delta_h = (file_local - naive).total_seconds() / 3600
            rounded = round(delta_h * 4) / 4
            residual = abs(delta_h - rounded) * 3600

            if residual > 5 or abs(rounded) > 14:
                out.finding(
                    "warning",
                    "File name and camera clock disagree",
                    f"The name {filename} encodes {file_local}, the camera clock {naive}. Phones name files from the same clock; the file was renamed or the date edited.",
                    tags=["time", "inconsistency"],
                    data={"filename_time": file_local, "camera_time": naive},
                )
            elif rounded != 0 and offset_h is None:
                out.finding(
                    "notice",
                    f"File name clock is UTC{rounded:+g} from the camera clock",
                    f"The file name encodes {file_local}, the camera clock {naive}: a {rounded:+g} h difference, as when one is UTC and the other local.",
                    tags=["time", "timezone"],
                    data={"offset_h": rounded},
                )

        if not exif or not (ifd0 or exif_ifd):
            out.finding(
                "notice",
                "No camera metadata"
                if (loaded.text or published.get("generators"))
                else "Metadata stripped",
                "No EXIF block: no camera, no date, no GPS. Social platforms, messaging apps and web downloads strip these; camera originals carry a few kilobytes of them."
                + (
                    " Only a bare JFIF header remains, the signature of a generic re-encode."
                    if jpeg and jpeg.jfif and not jpeg.exif_first
                    else ""
                ),
                tags=["metadata"],
                data={
                    "format": info.format,
                    "app_segments": jpeg.app_names if jpeg else None,
                },
            )

        if taken is None and latest_modify is not None and not (make or model):
            age_h = (
                datetime.now(timezone.utc)
                - (
                    latest_modify
                    if latest_modify.tzinfo
                    else latest_modify.replace(tzinfo=timezone.utc)
                )
            ).total_seconds() / 3600

            if -1 <= age_h <= 72:
                times["copy_stamp"] = True
                out.finding(
                    "warning",
                    "Timestamp is when this copy was made",
                    f"The only date in the file is {latest_modify}, {max(age_h, 0):.1f} h before it was examined, with no camera named. That is the moment an app saved this copy, not when it was taken.",
                    tags=["time", "metadata"],
                    data={"age_hours": age_h},
                )

        published["times"] = times
        start_guess = None

        if taken is not None and not times["copy_stamp"]:
            start_guess = (
                taken.astimezone(timezone.utc)
                if taken.tzinfo
                else taken.replace(tzinfo=timezone.utc)
            )

        published["start_guess"] = start_guess
        published["start_candidates"] = [start_guess] if start_guess else []
        published["tz_known"] = offset_h is not None
        published["utc_offset_h"] = offset_h

        # ---- place --------------------------------------------------------------------------------
        out.progress(0.7, "locating")
        published["location"] = None

        if (
            fix is None
            and xmp_fields.get("exif:GPSLatitude")
            and xmp_fields.get("exif:GPSLongitude")
        ):
            fix = _xmp_gps(xmp_fields)

        if fix:
            place = await _reverse_geocode(fix.lat, fix.lon)
            published["location"] = {
                "lat": fix.lat, "lon": fix.lon, "alt_m": fix.alt_m, "accuracy_m": fix.accuracy_m, "place": place,
                "heading_deg": fix.heading_deg, "speed_kmh": fix.speed_kmh,
            }  # fmt: skip
            detail = (
                f"{place}. " if place else ""
            ) + "The file carries the phone's GPS fix"
            detail += f" (±{fix.accuracy_m:.0f} m)" if fix.accuracy_m else ""
            detail += f", altitude {fix.alt_m:.0f} m" if fix.alt_m is not None else ""

            if fix.heading_deg is not None:
                detail += f". The camera pointed {photo.compass(fix.heading_deg)} ({fix.heading_deg:.0f}°), so the scene lies in that direction from the dot on the map"

            if fix.speed_kmh is not None and fix.speed_kmh > 2:
                detail += f". The phone was moving at {fix.speed_kmh:.0f} km/h"

            out.finding(
                "critical",
                f"GPS fix: {fix.lat:.5f}, {fix.lon:.5f}",
                detail
                + ". Anyone with this file can see exactly where the photographer stood.",
                tags=["location", "gps", "privacy"],
                data=published["location"],
            )
            out.artifact(
                "map", "map", "Where the photographer stood",
                data={"points": [{"lat": fix.lat, "lon": fix.lon, "label": "GPS (metadata)", "accuracy_m": fix.accuracy_m,
                                  "kind": "gps", "heading_deg": fix.heading_deg}]},
            )  # fmt: skip

            if start_guess is not None:
                elevation, azimuth = sun_position(start_guess, fix.lat, fix.lon)
                phase = ("daylight" if elevation > 6 else "golden hour / civil twilight" if elevation > -6
                         else "nautical twilight" if elevation > -12 else "night")  # fmt: skip
                published["sun"] = {
                    "elevation": elevation,
                    "azimuth": azimuth,
                    "phase": phase,
                }
                words = f"Computed from the claimed time and GPS fix: the sun was at {elevation:+.0f}° ({phase}), azimuth {azimuth:.0f}° ({photo.compass(azimuth)})."

                if fix.heading_deg is not None and elevation > 0:
                    relative = ((azimuth - fix.heading_deg + 180) % 360) - 180
                    side = ("in front of the camera: a backlit scene, shadows reaching toward the photographer" if abs(relative) < 45
                            else "behind the camera: shadows point away into the scene" if abs(relative) > 135
                            else f"to the camera's {'right' if relative > 0 else 'left'}: shadows should fall toward the {'left' if relative > 0 else 'right'} of the frame")  # fmt: skip
                    words += f" With the camera pointing {photo.compass(fix.heading_deg)}, the sun was {side}. Shadows that disagree mean the time, place or picture is wrong."
                    published["sun"]["relative_deg"] = relative
                elif elevation <= 0:
                    words += " It was dark there then; a bright daylight scene would contradict the stated time or place."

                out.finding(
                    "info",
                    f"Sun {elevation:+.0f}° high, {photo.compass(azimuth)} ({phase})",
                    words,
                    tags=["sun", "time", "location"],
                    data=published["sun"],
                )

        if (
            iptc.get("city")
            or iptc.get("country")
            or xmp_fields.get("photoshop:City")
            or xmp_fields.get("Iptc4xmpExt:LocationShown")
        ):
            place_words = ", ".join(
                str(v)
                for v in (
                    iptc.get("sub-location"),
                    iptc.get("city"),
                    iptc.get("province/state"),
                    iptc.get("country"),
                    xmp_fields.get("photoshop:City"),
                    xmp_fields.get("photoshop:Country"),
                )
                if v
            )
            out.finding(
                "warning",
                f"Place written in the caption data: {place_words}",
                "IPTC/XMP location fields were typed by a person or tool, not measured. They still tell a reader where the author says it was taken.",
                tags=["location", "metadata"],
                data={"place": place_words},
            )

        # ---- captions, keywords, descriptions -----------------------------------------------------
        caption = (
            iptc.get("caption")
            or xmp_fields.get("dc:description")
            or _text(ifd0.get("ImageDescription"))
            or loaded.text.get("Description")
            or loaded.text.get("Comment")
        )

        if caption and not is_screenshot and len(caption) > 3:
            out.finding(
                "notice",
                f"Caption in the file: “{caption[:90]}{'…' if len(caption) > 90 else ''}”",
                "Descriptive text stored inside the file"
                + (
                    f"; keywords: {iptc.get('keywords') or xmp_fields.get('dc:subject')}"
                    if iptc.get("keywords") or xmp_fields.get("dc:subject")
                    else ""
                )
                + ". Written by a person or an app; it travels with every copy.",
                tags=["metadata", "text"],
                data={
                    "caption": caption[:1000],
                    "keywords": iptc.get("keywords") or xmp_fields.get("dc:subject"),
                },
            )

        if icc_name:
            out.finding(
                "info",
                f"Colour profile: {icc_name}",
                "The embedded ICC profile names the colour space. Phones write Display P3 or sRGB; Adobe RGB or ProPhoto point to a camera-raw workflow or an editor."
                + (
                    " A profile this large usually comes from editing software, not a phone."
                    if loaded.icc and len(loaded.icc) > 4000
                    else ""
                ),
                tags=["metadata", "colour"],
                data={"profile": icc_name, "bytes": len(loaded.icc or b"")},
            )

        # ---- embedded thumbnail -----------------------------------------------------------------
        out.progress(0.85, "checking the embedded thumbnail")

        if exif and exif.thumbnail:
            thumb = photo.jpeg_from_bytes(exif.thumbnail)

            if thumb is not None and thumb.size:
                upright = thumb

                # The thumbnail is stored unrotated, like the main image.
                if info.orientation != 1:
                    from PIL import ImageOps

                    pil = photo.pil_from(thumb)
                    pil.getexif()[0x0112] = info.orientation
                    upright = np.asarray(ImageOps.exif_transpose(pil).convert("RGB"))

                check = await asyncio.to_thread(_thumbnail_check, upright, image)
                th, tw = upright.shape[:2]
                scale = 3
                big = cv2.resize(
                    upright, (tw * scale, th * scale), interpolation=cv2.INTER_NEAREST
                )
                now = cv2.resize(
                    image, (tw * scale, th * scale), interpolation=cv2.INTER_AREA
                )
                heat = cv2.resize(
                    np.clip(check["diff_map"] / 60.0, 0, 1).astype(np.float32),
                    (tw * scale, th * scale),
                )
                diff_vis = cv2.applyColorMap(
                    (heat * 255).astype(np.uint8), cv2.COLORMAP_INFERNO
                )
                panel = np.concatenate(
                    [
                        cv2.cvtColor(big, cv2.COLOR_RGB2BGR),
                        cv2.cvtColor(now, cv2.COLOR_RGB2BGR),
                        diff_vis,
                    ],
                    axis=1,
                )
                path = out.path("thumbnail_compare.jpg")
                photo.save_bgr(path, panel, 90)
                out.artifact(
                    "thumbnail", "image", "Embedded thumbnail (left) vs the picture now (middle) and their difference (right)",
                    file=path, data={"caption": f"Thumbnail {tw}×{th} px, written by the camera; correlation {check['correlation']:.2f}"},
                )  # fmt: skip
                mismatch = check["correlation"] < 0.75 or check["aspect_mismatch"]
                local_change = check["region"] is not None and not mismatch

                if mismatch or local_change:
                    data = {k: v for k, v in check.items() if k != "diff_map"}

                    if check["region"]:
                        data["box"] = check["region"]

                    if mismatch:
                        words = (
                            f"It matches the picture only {check['correlation'] * 100:.0f}%"
                            + (
                                f" and has a different shape ({check['aspect_thumb']:.2f} vs {check['aspect_image']:.2f})"
                                if check["aspect_mismatch"]
                                else ""
                            )
                            + ". "
                        )
                    else:
                        words = (
                            f"Overall it agrees ({check['correlation'] * 100:.0f}%), but {check['changed_blocks']} blocks differ by up to "
                            f"{check['peak_block_difference']:.0f} levels where the rest differ by {check['median_block_difference']:.1f}: "
                            "something in that area was changed after the camera wrote its preview. "
                        )

                    out.finding(
                        "critical" if check["correlation"] < 0.5 else "warning",
                        "Embedded thumbnail shows a different picture"
                        if mismatch
                        else "Embedded thumbnail differs from the picture in one region",
                        "The camera wrote a small preview into the EXIF block when it took the photo. "
                        + words
                        + "Editors that change the pixels but keep the metadata leave the old preview behind: the thumbnail is what the photo looked like before.",
                        tags=["thumbnail", "inconsistency", "edited", "splice"],
                        data=data,
                    )
                else:
                    out.finding(
                        "info",
                        f"Embedded thumbnail matches the picture ({check['correlation'] * 100:.0f}%)",
                        "The preview the camera wrote at capture agrees with the full picture, which argues against pixel edits that kept the metadata.",
                        tags=["thumbnail"],
                        data={k: v for k, v in check.items() if k != "diff_map"},
                    )

        published["exif_present"] = bool(exif and (ifd0 or exif_ifd))
        published["is_jpeg"] = jpeg is not None
        published["identity"] = identity
        ctx.publish(self.name, published)
        out.progress(1.0)


def _text(value: Any) -> str | None:
    if value is None:
        return None

    text = str(value).strip().strip("\x00")

    return text or None


def _show(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"

    if isinstance(value, (tuple, list)):
        return ", ".join(_show(v) for v in value[:16]) + (
            " …" if len(value) > 16 else ""
        )

    return str(value)[:300]


def _iso(text: str | None) -> datetime | None:
    if not text:
        return None

    cleaned = text.strip().replace("Z", "+00:00")

    try:
        parsed = datetime.fromisoformat(cleaned)
    except ValueError:
        return photo.exif_datetime(cleaned)

    return parsed


def _span(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f} s"

    if seconds < 5400:
        return f"{seconds / 60:.0f} min"

    if seconds < 172800:
        return f"{seconds / 3600:.1f} h"

    return f"{seconds / 86400:.0f} days"


def _xmp_gps(fields: dict[str, str]) -> photo.GpsFix | None:
    """XMP writes GPS as '48,51.7103N'."""

    def parse(text: str) -> float | None:
        m = re.match(r"^\s*(\d+),([\d.]+)([NSEW])\s*$", text)

        if not m:
            return None

        value = int(m.group(1)) + float(m.group(2)) / 60

        return -value if m.group(3) in "SW" else value

    lat = parse(fields.get("exif:GPSLatitude", ""))
    lon = parse(fields.get("exif:GPSLongitude", ""))

    if lat is None or lon is None:
        return None

    return photo.GpsFix(
        lat=lat,
        lon=lon,
        alt_m=None,
        time_utc=None,
        heading_deg=None,
        speed_kmh=None,
        accuracy_m=None,
    )
