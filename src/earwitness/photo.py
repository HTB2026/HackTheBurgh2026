"""Shared machinery for still photographs.

Loading (every common format, EXIF orientation applied), metadata parsing
(EXIF with every IFD including the embedded thumbnail, GPS, XMP, IPTC, ICC,
PNG text chunks) and the file anatomy of JPEG and PNG (segments and chunks,
quantisation tables, trailing bytes). Modules build on these; nothing here
emits findings.
"""

import io
import math
import os
import re
import struct
import zlib
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from typing import Any

import cv2
import numpy as np
from PIL import ExifTags
from PIL import Image
from PIL import ImageOps

from earwitness.modules.base import ImageInfo

try:
    import pillow_heif

    pillow_heif.register_heif_opener()
except ImportError:  # pragma: no cover - optional
    pass

Image.MAX_IMAGE_PIXELS = 400_000_000


@dataclass(slots=True)
class Loaded:
    pixels: np.ndarray  # RGB uint8, oriented
    info: ImageInfo
    exif: bytes | None  # TIFF blob (no "Exif\0\0" prefix)
    xmp: str | None
    icc: bytes | None
    iptc: bytes | None
    text: dict[str, str]  # PNG tEXt/iTXt, WebP/other comments
    pil_info: dict[str, Any] = field(default_factory=dict)


def load(path: str) -> Loaded | None:
    try:
        image = Image.open(path)
        image.load()
    except OSError, ValueError, Image.DecompressionBombError:
        return None

    frames = int(getattr(image, "n_frames", 1) or 1)
    stored_w, stored_h = image.size
    exif_blob = image.info.get("exif")

    if isinstance(exif_blob, bytes) and exif_blob.startswith(b"Exif\x00\x00"):
        exif_blob = exif_blob[6:]

    orientation = 1

    try:
        orientation = int(image.getexif().get(0x0112, 1) or 1)
    except ValueError, TypeError, AttributeError:
        orientation = 1

    oriented = ImageOps.exif_transpose(image) if orientation != 1 else image
    mode = image.mode
    bits = 16 if mode in ("I;16", "I;16B", "I;16L", "I", "RGB;16") else 8

    if oriented.mode == "I;16" or oriented.mode.startswith("I;16"):
        arr = np.asarray(oriented, dtype=np.uint16)
        rgb = np.repeat((arr >> 8).astype(np.uint8)[..., None], 3, axis=2)
    elif oriented.mode == "I":
        arr = np.asarray(oriented, dtype=np.int32)
        scale = max(1, int(arr.max()) // 255) if arr.size else 1
        rgb = np.repeat(
            np.clip(arr // scale, 0, 255).astype(np.uint8)[..., None], 3, axis=2
        )
    else:
        rgb = np.asarray(oriented.convert("RGB"))

    xmp = image.info.get("xmp") or image.info.get("XML:com.adobe.xmp")

    if isinstance(xmp, bytes):
        xmp = xmp.decode("utf-8", "replace")

    text: dict[str, str] = {}

    for key, value in (getattr(image, "text", None) or {}).items():
        text[str(key)] = str(value)

    for key in ("comment", "Comment"):
        if key in image.info:
            value = image.info[key]
            text[key] = (
                value.decode("utf-8", "replace")
                if isinstance(value, bytes)
                else str(value)
            )

    iptc = (
        image.info.get("photoshop", {}).get(0x0404)
        if isinstance(image.info.get("photoshop"), dict)
        else None
    )
    info = ImageInfo(
        width=rgb.shape[1],
        height=rgb.shape[0],
        format=str(image.format or os.path.splitext(path)[1].lstrip(".").upper()),
        mode=mode,
        bits=bits,
        frames=frames,
        orientation=orientation,
        stored_width=stored_w,
        stored_height=stored_h,
    )
    pil_info = {
        k: v
        for k, v in image.info.items()
        if isinstance(v, (int, float, str, bool, tuple))
    }

    return Loaded(
        pixels=np.ascontiguousarray(rgb),
        info=info,
        exif=exif_blob if isinstance(exif_blob, bytes) else None,
        xmp=xmp if isinstance(xmp, str) else None,
        icc=image.info.get("icc_profile"),
        iptc=iptc if isinstance(iptc, bytes) else None,
        text=text,
        pil_info=pil_info,
    )


# ---------------------------------------------------------------- EXIF (TIFF) walking

_TYPE_SIZE = {
    1: 1,
    2: 1,
    3: 2,
    4: 4,
    5: 8,
    6: 1,
    7: 1,
    8: 2,
    9: 4,
    10: 8,
    11: 4,
    12: 8,
    13: 4,
}
_IFD_POINTERS = {0x8769: "Exif", 0x8825: "GPS", 0xA005: "Interop"}

_GPS_TAGS = {v: k for k, v in ExifTags.GPSTAGS.items()}


@dataclass(slots=True)
class ExifData:
    ifds: dict[str, dict[str, Any]]  # "IFD0", "Exif", "GPS", "Interop", "IFD1"
    thumbnail: bytes | None
    makernote: dict[str, Any] | None
    byte_order: str
    raw_tags: list[tuple[str, int, str, Any]]  # (ifd, tag id, name, value) for tables


def _read_value(
    data: bytes, order: str, kind: int, count: int, value_field: bytes
) -> Any:
    size = _TYPE_SIZE.get(kind, 1) * count

    if size <= 4:
        raw = value_field[:size]
    else:
        offset = struct.unpack(order + "I", value_field)[0]
        raw = data[offset : offset + size]

    if len(raw) < size:
        return None

    if kind == 2:
        return raw.split(b"\x00", 1)[0].decode("utf-8", "replace")

    if kind in (1, 6, 7):
        return raw if count > 1 else raw[0]

    fmt = {3: "H", 4: "I", 8: "h", 9: "i", 11: "f", 12: "d", 13: "I"}.get(kind)

    if fmt:
        values = struct.unpack(order + fmt * count, raw)

        return values[0] if count == 1 else values

    if kind in (5, 10):
        pair = "II" if kind == 5 else "ii"
        nums = struct.unpack(order + pair * count, raw)
        values = tuple(
            (nums[i] / nums[i + 1]) if nums[i + 1] else float("nan")
            for i in range(0, len(nums), 2)
        )

        return values[0] if count == 1 else values

    return raw


def _walk_ifd(
    data: bytes,
    order: str,
    offset: int,
    name: str,
    out: dict[str, dict[int, Any]],
    depth: int = 0,
) -> int | None:
    if offset <= 0 or offset + 2 > len(data) or depth > 4 or name in out:
        return None

    count = struct.unpack(order + "H", data[offset : offset + 2])[0]

    if count > 1000:
        return None

    entries: dict[int, Any] = {}

    for index in range(count):
        base = offset + 2 + index * 12

        if base + 12 > len(data):
            break

        tag, kind, number = struct.unpack(order + "HHI", data[base : base + 8])

        if kind not in _TYPE_SIZE or number > 50_000_000:
            continue

        value = _read_value(data, order, kind, number, data[base + 8 : base + 12])
        entries[tag] = value

        if tag in _IFD_POINTERS and isinstance(value, int):
            _walk_ifd(data, order, value, _IFD_POINTERS[tag], out, depth + 1)

    out[name] = entries
    next_offset = offset + 2 + count * 12

    if next_offset + 4 <= len(data):
        return struct.unpack(order + "I", data[next_offset : next_offset + 4])[0]

    return None


def _tag_name(ifd: str, tag: int) -> str:
    if ifd == "GPS":
        return ExifTags.GPSTAGS.get(tag, f"GPS_0x{tag:04X}")

    return ExifTags.TAGS.get(tag, f"0x{tag:04X}")


def parse_exif(blob: bytes | None) -> ExifData | None:
    if not blob or len(blob) < 8:
        return None

    if blob.startswith(b"Exif\x00\x00"):
        blob = blob[6:]

    if blob[:2] == b"II":
        order = "<"
    elif blob[:2] == b"MM":
        order = ">"
    else:
        return None

    try:
        first = struct.unpack(order + "I", blob[4:8])[0]
    except struct.error:
        return None

    raw_ifds: dict[str, dict[int, Any]] = {}

    try:
        next_offset = _walk_ifd(blob, order, first, "IFD0", raw_ifds)

        if next_offset:
            _walk_ifd(blob, order, next_offset, "IFD1", raw_ifds)
    except struct.error, IndexError:
        pass

    thumbnail = None
    ifd1 = raw_ifds.get("IFD1", {})

    if 0x0201 in ifd1 and 0x0202 in ifd1:
        try:
            start, length = int(ifd1[0x0201]), int(ifd1[0x0202])
            candidate = blob[start : start + length]

            if candidate[:2] == b"\xff\xd8":
                thumbnail = candidate
        except TypeError, ValueError:
            thumbnail = None

    makernote = None
    note = raw_ifds.get("Exif", {}).get(0x927C)

    if isinstance(note, (bytes, bytearray)) and len(note) > 8:
        head = bytes(note[:12])
        vendor = (
            re.sub(rb"[^ -~]", b"", head).decode("ascii", "replace").strip()
            or "unknown"
        )
        makernote = {"vendor_prefix": vendor, "size_bytes": len(note)}

    ifds: dict[str, dict[str, Any]] = {}
    raw_tags: list[tuple[str, int, str, Any]] = []

    for ifd_name, entries in raw_ifds.items():
        named: dict[str, Any] = {}

        for tag, value in entries.items():
            name = _tag_name(ifd_name, tag)

            if tag == 0x927C:
                value = (
                    f"<MakerNote, {len(value)} bytes>"
                    if isinstance(value, (bytes, bytearray))
                    else value
                )
            elif isinstance(value, (bytes, bytearray)):
                value = _bytes_repr(bytes(value))

            named[name] = value
            raw_tags.append((ifd_name, tag, name, value))

        ifds[ifd_name] = named

    return ExifData(
        ifds=ifds,
        thumbnail=thumbnail,
        makernote=makernote,
        byte_order="little-endian" if order == "<" else "big-endian",
        raw_tags=raw_tags,
    )


def _bytes_repr(value: bytes) -> str:
    printable = value.rstrip(b"\x00")

    if printable and all(32 <= b < 127 for b in printable):
        return printable.decode("ascii")

    if len(value) <= 16:
        return value.hex(" ")

    return f"<{len(value)} bytes: {value[:12].hex(' ')}…>"


# ---------------------------------------------------------------- EXIF semantics

_EXIF_DATE = re.compile(r"^(\d{4}):(\d{2}):(\d{2})[ T](\d{2}):(\d{2}):(\d{2})")


def exif_datetime(text: Any, offset: Any = None, subsec: Any = None) -> datetime | None:
    """'2025:02:12 14:41:02' (+ '+01:00', '005') -> aware datetime (UTC if offset known, else naive)."""
    if not isinstance(text, str):
        return None

    found = _EXIF_DATE.match(text.strip())

    if not found:
        return None

    try:
        moment = datetime(*(int(g) for g in found.groups()))
    except ValueError:
        return None

    if isinstance(subsec, str) and subsec.strip().isdigit():
        digits = subsec.strip()[:6]
        moment = moment.replace(microsecond=int(digits.ljust(6, "0")))

    if isinstance(offset, str) and (
        m := re.match(r"^([+-])(\d{2}):?(\d{2})$", offset.strip())
    ):
        sign = 1 if m.group(1) == "+" else -1
        tz = timezone(sign * timedelta(hours=int(m.group(2)), minutes=int(m.group(3))))

        return moment.replace(tzinfo=tz)

    return moment


def _dms(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)

    if isinstance(value, (tuple, list)) and value:
        parts = [
            float(v)
            for v in value
            if isinstance(v, (int, float)) and math.isfinite(float(v))
        ]

        if not parts:
            return None

        return (
            parts[0]
            + (parts[1] if len(parts) > 1 else 0) / 60
            + (parts[2] if len(parts) > 2 else 0) / 3600
        )

    return None


@dataclass(slots=True)
class GpsFix:
    lat: float
    lon: float
    alt_m: float | None
    time_utc: datetime | None
    heading_deg: float | None  # which way the camera pointed (GPSImgDirection)
    speed_kmh: float | None
    accuracy_m: float | None  # GPSHPositioningError


def gps_fix(gps: dict[str, Any] | None) -> GpsFix | None:
    if not gps:
        return None

    lat = _dms(gps.get("GPSLatitude"))
    lon = _dms(gps.get("GPSLongitude"))

    if lat is None or lon is None or (lat == 0 and lon == 0):
        return None

    if str(gps.get("GPSLatitudeRef", "N")).upper().startswith("S"):
        lat = -lat

    if str(gps.get("GPSLongitudeRef", "E")).upper().startswith("W"):
        lon = -lon

    alt = gps.get("GPSAltitude")
    alt_m = float(alt) if isinstance(alt, (int, float)) and math.isfinite(alt) else None

    if alt_m is not None and str(gps.get("GPSAltitudeRef", "")) in ("01", "1", "\x01"):
        alt_m = -alt_m

    when = None
    stamp, date = gps.get("GPSTimeStamp"), gps.get("GPSDateStamp")

    if isinstance(stamp, (tuple, list)) and len(stamp) == 3 and isinstance(date, str):
        try:
            y, mo, d = (int(x) for x in re.split(r"[:\-]", date.strip())[:3])
            h, mi, s = (float(x) for x in stamp)
            when = datetime(
                y,
                mo,
                d,
                int(h),
                int(mi),
                int(s),
                int((s % 1) * 1e6),
                tzinfo=timezone.utc,
            )
        except ValueError, TypeError:
            when = None
    elif isinstance(stamp, (tuple, list)) and len(stamp) == 3:
        try:
            h, mi, s = (float(x) for x in stamp)
            when = datetime(
                1900, 1, 1, int(h), int(mi), int(s), tzinfo=timezone.utc
            )  # time of day only
        except ValueError, TypeError:
            when = None

    heading = gps.get("GPSImgDirection")
    speed = gps.get("GPSSpeed")
    ref = str(gps.get("GPSSpeedRef", "K")).upper()

    if isinstance(speed, (int, float)):
        speed = float(speed) * (
            1.852 if ref.startswith("N") else 1.609 if ref.startswith("M") else 1.0
        )

    error = gps.get("GPSHPositioningError")

    return GpsFix(
        lat=lat,
        lon=lon,
        alt_m=alt_m,
        time_utc=when,
        heading_deg=float(heading)
        if isinstance(heading, (int, float)) and math.isfinite(heading)
        else None,
        speed_kmh=speed if isinstance(speed, float) and math.isfinite(speed) else None,
        accuracy_m=float(error)
        if isinstance(error, (int, float)) and math.isfinite(error)
        else None,
    )


def compass(deg: float) -> str:
    names = [
        "N",
        "NNE",
        "NE",
        "ENE",
        "E",
        "ESE",
        "SE",
        "SSE",
        "S",
        "SSW",
        "SW",
        "WSW",
        "W",
        "WNW",
        "NW",
        "NNW",
    ]

    return names[int((deg % 360) / 22.5 + 0.5) % 16]


# ---------------------------------------------------------------- XMP

_XMP_TAG = re.compile(
    r"<\s*([\w\-]+):([\w\-]+)(?:\s[^>]*)?>([^<]{1,2000})</\s*\1:\2\s*>"
)
_XMP_ATTR = re.compile(r"\b([\w\-]+):([\w\-]+)=\"([^\"]{1,2000})\"")


@dataclass(slots=True)
class XmpData:
    fields: dict[str, str]  # "xmp:CreatorTool" -> value
    history: list[dict[str, str]]  # xmpMM:History entries
    derived_from: dict[str, str]
    raw: str


def parse_xmp(text: str | None) -> XmpData | None:
    if not text:
        return None

    fields: dict[str, str] = {}

    for namespace, name, value in _XMP_TAG.findall(text):
        if namespace in ("rdf", "x", "xmlns"):
            continue

        key = f"{namespace}:{name}"
        value = value.strip()

        if value and key not in fields:
            fields[key] = value

    for namespace, name, value in _XMP_ATTR.findall(text):
        if namespace in ("rdf", "x", "xmlns", "xml"):
            continue

        fields.setdefault(f"{namespace}:{name}", value.strip())

    # rdf:li items inside Alt/Seq/Bag for the common text properties
    for prop in (
        "dc:creator",
        "dc:description",
        "dc:title",
        "dc:subject",
        "dc:rights",
        "photoshop:Keywords",
    ):
        block = re.search(rf"<{prop}>(.*?)</{prop}>", text, re.S)

        if block:
            items = [
                re.sub(r"<[^>]+>", "", li).strip()
                for li in re.findall(
                    r"<rdf:li[^>]*>(.*?)</rdf:li>", block.group(1), re.S
                )
            ]
            items = [i for i in items if i]

            if items:
                fields[prop] = "; ".join(items)

    history: list[dict[str, str]] = []
    block = re.search(r"<xmpMM:History>(.*?)</xmpMM:History>", text, re.S)

    if block:
        for li in re.findall(
            r"<rdf:li\b([^>]*)(?:/>|>(.*?)</rdf:li>)", block.group(1), re.S
        ):
            attrs, inner = li
            entry: dict[str, str] = {}

            for _, name, value in _XMP_ATTR.findall(attrs):
                entry[name] = value

            for _, name, value in _XMP_TAG.findall(inner or ""):
                entry[name] = value.strip()

            if entry:
                history.append(entry)

    derived: dict[str, str] = {}
    block = re.search(
        r"<xmpMM:DerivedFrom\b([^>]*)(?:/>|>(.*?)</xmpMM:DerivedFrom>)", text, re.S
    )

    if block:
        for _, name, value in _XMP_ATTR.findall(block.group(1)):
            derived[name] = value

        for _, name, value in _XMP_TAG.findall(block.group(2) or ""):
            derived[name] = value.strip()

    return XmpData(fields=fields, history=history, derived_from=derived, raw=text)


# ---------------------------------------------------------------- JPEG anatomy

_MARKER_NAMES = {
    0xC0: "SOF0 baseline DCT", 0xC1: "SOF1 extended sequential", 0xC2: "SOF2 progressive DCT",
    0xC3: "SOF3 lossless", 0xC5: "SOF5", 0xC6: "SOF6", 0xC7: "SOF7", 0xC9: "SOF9 arithmetic",
    0xCA: "SOF10 progressive arithmetic", 0xCB: "SOF11", 0xCD: "SOF13", 0xCE: "SOF14", 0xCF: "SOF15",
    0xC4: "DHT Huffman tables", 0xCC: "DAC arithmetic conditioning", 0xDB: "DQT quantisation tables",
    0xDD: "DRI restart interval", 0xDA: "SOS start of scan", 0xFE: "COM comment", 0xD8: "SOI", 0xD9: "EOI",
}  # fmt: skip
_ZIGZAG = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21,
    28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61,
    54, 47, 55, 62, 63,
]  # fmt: skip
STD_LUMA = np.array(
    [16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56, 14, 17, 22, 29,
     51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92, 49, 64, 78, 87, 103, 121,
     120, 101, 72, 92, 95, 98, 112, 100, 103, 99],
    dtype=float,
)  # fmt: skip
STD_CHROMA = np.array(
    [17, 18, 24, 47, 99, 99, 99, 99, 18, 21, 26, 66, 99, 99, 99, 99, 24, 26, 56, 99, 99, 99, 99, 99, 47, 66, 99, 99,
     99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
     99, 99, 99, 99, 99, 99, 99, 99],
    dtype=float,
)  # fmt: skip


@dataclass(slots=True)
class JpegAnatomy:
    segments: list[dict[str, Any]]
    quant: dict[int, np.ndarray]  # table id -> 64 values in natural (row-major) order
    progressive: bool
    arithmetic: bool
    precision: int
    components: int
    sampling: str  # "4:2:0", "4:2:2", "4:4:4", ...
    huffman_tables: int
    optimised_huffman: bool | None
    restart_interval: int | None
    comments: list[str]
    app_names: list[str]
    eoi_offset: int | None
    trailing: bytes
    adobe: dict[str, Any] | None
    jfif: bool
    exif_first: bool | None  # True: APP1 Exif comes before APP0 JFIF (camera style)


def _std_huffman(data: bytes) -> bool:
    # The DC luminance table of Annex K starts with bits counts 0,1,5,1,1,1,1,1,1,0,...
    return data[1:17] == bytes([0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0])


def jpeg_anatomy(data: bytes) -> JpegAnatomy | None:
    if data[:2] != b"\xff\xd8":
        return None

    segments: list[dict[str, Any]] = [
        {"marker": "SOI", "name": "SOI start of image", "offset": 0, "length": 2}
    ]
    quant: dict[int, np.ndarray] = {}
    progressive = arithmetic = False
    precision, components = 8, 0
    sampling = "?"
    huffman = 0
    std_huffman: bool | None = None
    restart = None
    comments: list[str] = []
    app_names: list[str] = []
    adobe = None
    jfif = False
    exif_first = None
    eoi = None
    offset = 2

    while offset + 4 <= len(data):
        if data[offset] != 0xFF:
            offset += 1

            continue

        marker = data[offset + 1]

        if marker in (0xFF, 0x00) or 0xD0 <= marker <= 0xD7:
            offset += 1 if marker == 0xFF else 2

            continue

        if marker == 0xD9:
            segments.append(
                {
                    "marker": "EOI",
                    "name": "EOI end of image",
                    "offset": offset,
                    "length": 2,
                }
            )
            eoi = offset
            break

        if marker == 0xD8:
            offset += 2

            continue

        length = struct.unpack(">H", data[offset + 2 : offset + 4])[0]
        payload = data[offset + 4 : offset + 2 + length]
        name = _MARKER_NAMES.get(
            marker,
            f"APP{marker - 0xE0}" if 0xE0 <= marker <= 0xEF else f"0x{marker:02X}",
        )
        entry: dict[str, Any] = {
            "marker": f"0xFF{marker:02X}",
            "name": name,
            "offset": offset,
            "length": length + 2,
        }

        if 0xE0 <= marker <= 0xEF:
            ident = payload.split(b"\x00", 1)[0][:32]
            label = (
                re.sub(r"[^ -~]", "", ident.decode("latin-1")).strip()
                or f"APP{marker - 0xE0}"
            )

            if payload[:4] == b"JFIF":
                jfif = True

                if exif_first is None:
                    exif_first = False
            elif payload[:4] == b"Exif":
                label = "Exif"

                if exif_first is None:
                    exif_first = True
            elif payload.startswith(b"http://ns.adobe.com/xap/1.0/"):
                label = "XMP"
            elif payload.startswith(b"http://ns.adobe.com/xmp/extension/"):
                label = "XMP (extended)"
            elif payload[:11] == b"ICC_PROFILE":
                label = "ICC_PROFILE"
            elif payload[:13] == b"Photoshop 3.0":
                label = "Photoshop 3.0 (IPTC/IRB)"
            elif payload[:5] == b"Adobe":
                label = "Adobe"
                adobe = {"version": struct.unpack(">H", payload[5:7])[0] if len(payload) >= 7 else None,
                         "transform": payload[11] if len(payload) >= 12 else None}  # fmt: skip
            elif payload[:4] == b"MPF\x00":
                label = "MPF (multi-picture)"
            elif b"jumb" in payload[:40] or b"c2pa" in payload[:60]:
                label = "JUMBF / C2PA"
            elif payload[:5] == b"Ducky":
                label = "Ducky (Adobe Save for Web)"
            elif payload[:4] == b"HDR " or payload[:6] == b"UNICOD":
                label = "HDR / Unicode"

            entry["app"] = f"APP{marker - 0xE0} {label}"
            app_names.append(entry["app"])
        elif marker == 0xDB:
            pos = 0

            while pos + 65 <= len(payload):
                pq, tq = payload[pos] >> 4, payload[pos] & 0x0F
                size = 128 if pq else 64
                raw = payload[pos + 1 : pos + 1 + size]

                if len(raw) < size:
                    break

                values = struct.unpack(">64H" if pq else "64B", raw)
                natural = np.zeros(64)

                for index, value in enumerate(values):
                    natural[_ZIGZAG[index]] = value

                quant[tq] = natural
                pos += 1 + size

            entry["tables"] = sorted(quant)
        elif marker in (
            0xC0,
            0xC1,
            0xC2,
            0xC3,
            0xC5,
            0xC6,
            0xC7,
            0xC9,
            0xCA,
            0xCB,
            0xCD,
            0xCE,
            0xCF,
        ):
            progressive = marker in (0xC2, 0xC6, 0xCA, 0xCE)
            arithmetic = marker >= 0xC9

            if len(payload) >= 6:
                precision = payload[0]
                components = payload[5]
                factors = []

                for c in range(components):
                    base = 6 + c * 3

                    if base + 3 <= len(payload):
                        factors.append(
                            (payload[base + 1] >> 4, payload[base + 1] & 0x0F)
                        )

                entry["components"] = components
                entry["sampling_factors"] = factors
                sampling = _sampling_name(factors)
        elif marker == 0xC4:
            huffman += 1

            if std_huffman is None:
                std_huffman = _std_huffman(payload)
        elif marker == 0xDD and len(payload) >= 2:
            restart = struct.unpack(">H", payload[:2])[0]
            entry["interval"] = restart
        elif marker == 0xFE:
            comments.append(payload.decode("utf-8", "replace").strip("\x00"))
            entry["text"] = comments[-1][:200]

        segments.append(entry)

        if marker == 0xDA:
            # Entropy-coded data follows until the next non-RST marker.
            scan = offset + 2 + length

            while scan + 1 < len(data):
                if (
                    data[scan] == 0xFF
                    and data[scan + 1] not in (0x00, 0xFF)
                    and not (0xD0 <= data[scan + 1] <= 0xD7)
                ):
                    break

                scan += 1

            entry["scan_bytes"] = scan - (offset + 2 + length)
            offset = scan

            continue

        offset += 2 + length

    trailing = data[eoi + 2 :] if eoi is not None else b""

    return JpegAnatomy(
        segments=segments,
        quant=quant,
        progressive=progressive,
        arithmetic=arithmetic,
        precision=precision,
        components=components,
        sampling=sampling,
        huffman_tables=huffman,
        optimised_huffman=(not std_huffman) if std_huffman is not None else None,
        restart_interval=restart,
        comments=comments,
        app_names=app_names,
        eoi_offset=eoi,
        trailing=trailing,
        adobe=adobe,
        jfif=jfif,
        exif_first=exif_first,
    )


def _sampling_name(factors: list[tuple[int, int]]) -> str:
    if len(factors) == 1:
        return "greyscale"

    if len(factors) < 3:
        return "?"

    (h0, v0), (h1, v1) = factors[0], factors[1]

    if (h0, v0) == (h1, v1):
        return "4:4:4"

    if h0 == 2 * h1 and v0 == v1:
        return "4:2:2"

    if h0 == 2 * h1 and v0 == 2 * v1:
        return "4:2:0"

    if h0 == 4 * h1:
        return "4:1:1"

    if h0 == h1 and v0 == 2 * v1:
        return "4:4:0"

    return f"{h0}x{v0}/{h1}x{v1}"


def scaled_standard_table(base: np.ndarray, quality: int) -> np.ndarray:
    """libjpeg's quality scaling of an Annex K table."""
    quality = max(1, min(100, quality))
    scale = 5000 / quality if quality < 50 else 200 - 2 * quality
    table = np.floor((base * scale + 50) / 100)

    return np.clip(table, 1, 255)


@dataclass(slots=True)
class QualityEstimate:
    quality: int  # best-matching libjpeg quality
    distance: float  # mean |table - standard(quality)| in quantiser steps
    standard_tables: bool  # tables are libjpeg/IJG-style (cameras, most software)
    luma_mean: float
    chroma_mean: float | None
    approx_quality: (
        float  # quality-like number from the mean quantiser, works for any table
    )


def estimate_quality(quant: dict[int, np.ndarray]) -> QualityEstimate | None:
    if 0 not in quant:
        return None

    luma = quant[0]
    chroma = quant.get(1)
    best, best_distance = 0, float("inf")

    for quality in range(1, 101):
        distance = float(np.abs(luma - scaled_standard_table(STD_LUMA, quality)).mean())

        if chroma is not None:
            distance = 0.5 * (
                distance
                + float(
                    np.abs(chroma - scaled_standard_table(STD_CHROMA, quality)).mean()
                )
            )

        if distance < best_distance:
            best, best_distance = quality, distance

    # Any-table estimate: invert libjpeg's scaling on the mean quantiser, which
    # gives a comparable number even for software with its own tables.
    mean = float(luma.mean())
    base_mean = float(STD_LUMA.mean())
    approx = 100 - 50 * mean / base_mean if mean <= base_mean else 50 * base_mean / mean
    approx = float(np.clip(approx, 1, 100))

    return QualityEstimate(
        quality=best,
        distance=best_distance,
        standard_tables=best_distance < 1.5,
        luma_mean=mean,
        chroma_mean=float(chroma.mean()) if chroma is not None else None,
        approx_quality=approx,
    )


# ---------------------------------------------------------------- PNG anatomy


@dataclass(slots=True)
class PngAnatomy:
    chunks: list[dict[str, Any]]
    text: dict[str, str]
    bit_depth: int
    colour_type: int
    interlaced: bool
    iend_offset: int | None
    trailing: bytes
    has_exif: bool
    has_c2pa: bool


def png_anatomy(data: bytes) -> PngAnatomy | None:
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        return None

    chunks: list[dict[str, Any]] = []
    text: dict[str, str] = {}
    offset = 8
    bit_depth, colour_type, interlaced = 8, 2, False
    iend = None
    has_exif = has_c2pa = False

    while offset + 8 <= len(data):
        length = struct.unpack(">I", data[offset : offset + 4])[0]
        kind = data[offset + 4 : offset + 8].decode("latin-1")
        payload = data[offset + 8 : offset + 8 + length]
        chunks.append({"type": kind, "offset": offset, "length": length})

        if kind == "IHDR" and len(payload) >= 13:
            bit_depth, colour_type, interlaced = (
                payload[8],
                payload[9],
                bool(payload[12]),
            )
        elif kind == "tEXt":
            key, _, value = payload.partition(b"\x00")
            text[key.decode("latin-1")] = value.decode("latin-1", "replace")
        elif kind == "zTXt":
            key, _, rest = payload.partition(b"\x00")

            try:
                text[key.decode("latin-1")] = zlib.decompress(rest[1:]).decode(
                    "latin-1", "replace"
                )
            except zlib.error:
                pass
        elif kind == "iTXt":
            key, _, rest = payload.partition(b"\x00")

            if len(rest) >= 2:
                compressed = rest[0] == 1
                body = rest[2:]
                parts = body.split(b"\x00", 2)

                if len(parts) == 3:
                    value = parts[2]

                    if compressed:
                        try:
                            value = zlib.decompress(value)
                        except zlib.error:
                            value = b""

                    text[key.decode("latin-1")] = value.decode("utf-8", "replace")
        elif kind == "eXIf":
            has_exif = True
        elif kind == "caBX":
            has_c2pa = True
        elif kind == "IEND":
            iend = offset
            break

        offset += 12 + length

    trailing = data[iend + 12 :] if iend is not None else b""

    return PngAnatomy(chunks=chunks, text=text, bit_depth=bit_depth, colour_type=colour_type, interlaced=interlaced,
                      iend_offset=iend, trailing=trailing, has_exif=has_exif, has_c2pa=has_c2pa)  # fmt: skip


# ---------------------------------------------------------------- trailing / embedded payloads

_MAGIC = [
    (b"PK\x03\x04", "ZIP archive (or DOCX/APK/JAR)"),
    (b"Rar!\x1a\x07", "RAR archive"),
    (b"7z\xbc\xaf\x27\x1c", "7-Zip archive"),
    (b"%PDF", "PDF document"),
    (b"\xff\xd8\xff", "another JPEG image"),
    (b"\x89PNG", "a PNG image"),
    (b"GIF8", "a GIF image"),
    (b"RIFF", "RIFF container (WebP/WAV/AVI)"),
    (b"\x1f\x8b", "gzip data"),
    (b"BZh", "bzip2 data"),
    (b"MZ", "Windows executable"),
    (b"\x7fELF", "Linux executable"),
    (b"OggS", "Ogg audio"),
    (b"ID3", "MP3 audio"),
    (b"ftyp", "MP4/HEIF container"),
    (b"<?xml", "XML text"),
    (b"<x:xmpmeta", "XMP metadata"),
    (b"{", "JSON-like text"),
]


def identify_bytes(blob: bytes) -> str:
    head = blob[:16]

    for magic, label in _MAGIC:
        if head.startswith(magic) or (magic == b"ftyp" and head[4:8] == magic):
            return label

    if not blob.strip(b"\x00"):
        return "zero padding"

    printable = sum(32 <= b < 127 or b in (9, 10, 13) for b in blob[:512])

    if printable >= 0.9 * min(512, len(blob)):
        return "plain text"

    return "unidentified binary data"


def strings_in(blob: bytes, minimum: int = 8, limit: int = 200) -> list[str]:
    out: list[str] = []

    for match in re.finditer(rb"[ -~]{%d,}" % minimum, blob):
        out.append(match.group(0).decode("ascii"))

        if len(out) >= limit:
            break

    return out


# ---------------------------------------------------------------- pixels


def heat_overlay(
    base_rgb: np.ndarray,
    heat01: np.ndarray,
    colormap: int = cv2.COLORMAP_INFERNO,
    alpha: float = 0.75,
) -> np.ndarray:
    """BGR composite of a 0..1 heat map over a photo (for cv2.imwrite)."""
    heat = np.clip(heat01, 0, 1).astype(np.float32)
    colour = cv2.applyColorMap((heat * 255).astype(np.uint8), colormap)
    colour = cv2.resize(
        colour, (base_rgb.shape[1], base_rgb.shape[0]), interpolation=cv2.INTER_LINEAR
    )
    bgr = cv2.cvtColor(base_rgb, cv2.COLOR_RGB2BGR)
    weight = cv2.resize(heat, (base_rgb.shape[1], base_rgb.shape[0]))[..., None] * alpha

    return (bgr * (1 - weight) + colour * weight).astype(np.uint8)


def save_jpeg(path: str, rgb: np.ndarray, quality: int = 88) -> None:
    cv2.imwrite(
        path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality]
    )


def save_bgr(path: str, bgr: np.ndarray, quality: int = 88) -> None:
    cv2.imwrite(path, bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])


def fit(rgb: np.ndarray, max_side: int) -> np.ndarray:
    scale = max_side / max(rgb.shape[:2])

    if scale >= 1:
        return rgb

    return cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def box_fraction(
    x: float, y: float, w: float, h: float, width: int, height: int
) -> dict[str, float]:
    """A region as fractions of the picture, which is what the viewer draws."""
    return {
        "x": round(max(0.0, x / width), 4),
        "y": round(max(0.0, y / height), 4),
        "w": round(min(1.0, w / width), 4),
        "h": round(min(1.0, h / height), 4),
    }


def encode_jpeg_bytes(rgb: np.ndarray, quality: int) -> bytes:
    ok, encoded = cv2.imencode(
        ".jpg",
        cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
        [cv2.IMWRITE_JPEG_QUALITY, int(quality)],
    )

    return encoded.tobytes() if ok else b""


def decode_jpeg_bytes(data: bytes) -> np.ndarray | None:
    array = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)

    return None if array is None else cv2.cvtColor(array, cv2.COLOR_BGR2RGB)


def megapixels(width: int, height: int) -> float:
    return round(width * height / 1e6, 1)


def thumbnail_of(rgb: np.ndarray, width: int = 320) -> np.ndarray:
    height = max(1, int(round(width * rgb.shape[0] / rgb.shape[1])))

    return cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)


def pil_from(rgb: np.ndarray) -> Image.Image:
    return Image.fromarray(rgb)


def jpeg_from_bytes(data: bytes) -> np.ndarray | None:
    try:
        image = Image.open(io.BytesIO(data))
        image.load()

        return np.asarray(image.convert("RGB"))
    except OSError, ValueError:
        return None
