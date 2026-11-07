"""Synthetic photographs with planted ground truth, for the demo pack and tests.

Every picture is drawn, not photographed, so nothing here is anyone's data.
The "camera original" carries a full EXIF block (device, exposure, GPS fix with
compass heading, a GPS clock one hour behind the local clock, and the camera's
own thumbnail). The edited copy keeps that metadata while a patch is cloned, a
region pasted in from a lower-quality JPEG with a shifted grid, and a ZIP file
appended after the end marker, so the stale thumbnail, the clone, the ghost,
the grid and the trailing archive are all there to be found.
"""

import io
import struct
import zipfile
import zlib

import cv2
import numpy as np
from PIL import Image
from PIL.TiffImagePlugin import IFDRational

PLANTED = {
    "make": "Earwitness Labs",
    "model": "Demo Camera X1",
    "taken_local": "2026:03:14 15:09:26",
    "gps_utc": (14, 9, 26),  # one hour behind: the phone was on UTC+1
    "lat": (55.0, 56.0, 39.12),  # 55.9442 N
    "lon": (3.0, 11.0, 14.6),  # 3.1874 W (Edinburgh)
    "heading": 135.5,
    "clone_offset": (300, 120),
    "clone_box": (
        160,
        760,
        150,
        110,
    ),  # x, y, w, h of the source patch (textured ground)
    "paste_box": (
        640,
        800,
        320,
        256,
    ),  # grid-aligned paste from a q45 JPEG: leaves a ghost
    "paste_quality": 45,
    "paste2_box": (
        1176,
        752,
        320,
        256,
    ),  # the same, 3 px off the grid: leaves a shifted grid
    "paste2_quality": 50,
    "removed_house": 2,  # the third house is painted over with ground in the edited copy
    "secret": b"meet at the usual place, 21:00\n",
    "sd_prompt": "a lighthouse on a cliff at dusk, dramatic clouds, photorealistic, 35mm",
    "sd_seed": 3141592653,
    "lsb_message": b"EARWITNESS LSB DEMO " * 400,
}


def _scene(width: int = 1600, height: int = 1200, seed: int = 7) -> np.ndarray:
    """A photo-like scene: graded sky, textured ground, a few objects, sensor grain."""
    rng = np.random.default_rng(seed)
    y = np.linspace(0, 1, height)[:, None]
    x = np.linspace(0, 1, width)[None, :]
    sky = np.stack([120 + 60 * (1 - y), 150 + 50 * (1 - y), 210 - 40 * y], axis=-1)
    ground = np.stack([70 + 25 * x, 95 + 20 * x, 60 + 10 * x], axis=-1)
    horizon = int(height * 0.58)
    scene = np.where(np.arange(height)[:, None, None] < horizon, sky, ground).astype(
        np.float32
    )
    # Ground texture: band-limited noise so there is something for features to grip.
    texture = rng.normal(0, 1, (height // 4, width // 4)).astype(np.float32)
    texture = cv2.resize(
        cv2.GaussianBlur(texture, (0, 0), 1.2),
        (width, height),
        interpolation=cv2.INTER_CUBIC,
    )
    scene[horizon:] += 28 * texture[horizon:, :, None]
    fine = rng.normal(0, 1, (height, width)).astype(np.float32)
    scene[horizon:] += 14 * cv2.GaussianBlur(fine, (0, 0), 0.8)[horizon:, :, None]
    gravel = rng.normal(0, 1, (height // 2, width // 2)).astype(np.float32)
    scene[horizon:] += (
        18
        * cv2.resize(gravel, (width, height), interpolation=cv2.INTER_NEAREST)[
            horizon:, :, None
        ]
    )
    # Objects: houses, a tree line, a signpost, some rocks.
    canvas = np.clip(scene, 0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)

    for index in range(6):
        cx = 120 + index * 250
        cv2.rectangle(
            bgr,
            (cx, horizon - 90),
            (cx + 110, horizon + 10),
            (60 + index * 15, 70, 150 + index * 10),
            -1,
        )
        pts = np.array(
            [
                [cx - 10, horizon - 90],
                [cx + 55, horizon - 150],
                [cx + 120, horizon - 90],
            ]
        )
        cv2.fillPoly(bgr, [pts], (40, 40, 120))
        cv2.rectangle(
            bgr, (cx + 20, horizon - 70), (cx + 45, horizon - 40), (200, 230, 255), -1
        )
        cv2.rectangle(
            bgr, (cx + 65, horizon - 70), (cx + 90, horizon - 40), (200, 230, 255), -1
        )

    for index in range(40):
        rx, ry = int(rng.uniform(0, width)), int(rng.uniform(horizon + 40, height - 20))
        cv2.ellipse(
            bgr,
            (rx, ry),
            (int(rng.uniform(8, 30)), int(rng.uniform(5, 14))),
            0,
            0,
            360,
            (int(rng.uniform(40, 90)),) * 3,
            -1,
        )

    cv2.rectangle(bgr, (1180, horizon - 60), (1196, horizon + 140), (90, 90, 90), -1)
    cv2.rectangle(bgr, (1100, horizon - 120), (1290, horizon - 62), (30, 160, 230), -1)
    cv2.putText(
        bgr,
        "EH8 9DH",
        (1110, horizon - 78),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        (255, 255, 255),
        3,
        cv2.LINE_AA,
    )
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    rgb += rng.normal(0, 2.2, rgb.shape).astype(np.float32)  # sensor grain everywhere

    return np.clip(rgb, 0, 255).astype(np.uint8)


def _exif(
    thumbnail_jpeg: bytes | None,
    *,
    software: str = "X1 firmware 2.0",
    width: int = 1600,
    height: int = 1200,
) -> bytes:
    exif = Image.Exif()
    exif[0x010F] = PLANTED["make"]
    exif[0x0110] = PLANTED["model"]
    exif[0x0131] = software
    exif[0x0132] = PLANTED["taken_local"]
    exif[0x0112] = 1
    sub = exif.get_ifd(0x8769)
    sub[0x9003] = PLANTED["taken_local"]
    sub[0x9004] = PLANTED["taken_local"]
    sub[0x829A] = IFDRational(1, 250)
    sub[0x829D] = IFDRational(18, 10)
    sub[0x8827] = 100
    sub[0x920A] = IFDRational(26, 10)
    sub[0xA405] = 26
    sub[0xA002] = width
    sub[0xA003] = height
    sub[0xA433] = PLANTED["make"]
    sub[0xA434] = "X1 26mm f/1.8"
    gps = exif.get_ifd(0x8825)
    gps[1] = "N"
    gps[2] = tuple(IFDRational(int(round(v * 100)), 100) for v in PLANTED["lat"])
    gps[3] = "W"
    gps[4] = tuple(IFDRational(int(round(v * 100)), 100) for v in PLANTED["lon"])
    gps[5] = b"\x00"
    gps[6] = IFDRational(62, 1)
    gps[7] = tuple(IFDRational(int(v), 1) for v in PLANTED["gps_utc"])
    gps[16] = "T"
    gps[17] = IFDRational(int(PLANTED["heading"] * 10), 10)
    gps[29] = PLANTED["taken_local"][:10]
    blob = exif.tobytes()  # "Exif\0\0" + TIFF (little-endian, IFD0 at 8)

    if thumbnail_jpeg is None:
        return blob

    tiff = bytearray(blob[6:])
    order = "<" if tiff[:2] == b"II" else ">"
    first = struct.unpack(order + "I", tiff[4:8])[0]
    count = struct.unpack(order + "H", tiff[first : first + 2])[0]
    next_pointer = first + 2 + 12 * count
    ifd1_offset = len(tiff)
    struct.pack_into(order + "I", tiff, next_pointer, ifd1_offset)
    thumb_offset = ifd1_offset + 2 + 2 * 12 + 4
    ifd1 = struct.pack(order + "H", 2)
    ifd1 += struct.pack(order + "HHII", 0x0201, 4, 1, thumb_offset)
    ifd1 += struct.pack(order + "HHII", 0x0202, 4, 1, len(thumbnail_jpeg))
    ifd1 += struct.pack(order + "I", 0)
    tiff += ifd1 + thumbnail_jpeg

    return b"Exif\x00\x00" + bytes(tiff)


def _jpeg_bytes(
    rgb: np.ndarray, quality: int, exif: bytes | None = None, **save
) -> bytes:
    buffer = io.BytesIO()
    kwargs = {"quality": quality, "subsampling": 2}

    if exif:
        kwargs["exif"] = exif

    Image.fromarray(rgb).save(buffer, "JPEG", **kwargs, **save)

    return buffer.getvalue()


def _thumbnail(rgb: np.ndarray) -> bytes:
    small = cv2.resize(rgb, (160, 120), interpolation=cv2.INTER_AREA)

    return _jpeg_bytes(small, 80)


def clean_photo(path: str) -> str:
    """A camera-style JPEG: full EXIF, matching thumbnail, saved once."""
    scene = _scene()

    with open(path, "wb") as handle:
        handle.write(_jpeg_bytes(scene, 92, _exif(_thumbnail(scene))))

    return path


def edited_photo(path: str) -> str:
    """The same scene after a clone, a paste from a low-quality JPEG, and an appended ZIP."""
    original = _scene()
    edited = original.copy()
    # 0. Remove a house: paint ground over it (the kind of edit the thumbnail betrays).
    horizon = int(1200 * 0.58)
    hx = 120 + PLANTED["removed_house"] * 250
    edited[horizon - 160 : horizon + 12, hx - 15 : hx + 125] = _scene(seed=21)[
        horizon + 60 : horizon + 232, hx - 15 : hx + 125
    ]
    # 1. Clone: stamp a patch of ground elsewhere.
    x, y, w, h = PLANTED["clone_box"]
    dx, dy = PLANTED["clone_offset"]
    edited[y + dy : y + dy + h, x + dx : x + dx + w] = original[y : y + h, x : x + w]
    # 2. Paste: a region that was a quality-45 JPEG, placed exactly on the 8x8 grid.
    px, py, pw, ph = PLANTED["paste_box"]
    patch = _scene(seed=11)[
        120 : 120 + ph, px : px + pw
    ]  # a patch of sky from another frame
    patch = np.asarray(
        Image.open(io.BytesIO(_jpeg_bytes(patch, PLANTED["paste_quality"]))).convert(
            "RGB"
        )
    )
    edited[py : py + ph, px : px + pw] = patch
    # 2b. Another paste from a quality-50 JPEG, landing 3 px off the grid.
    qx, qy, qw, qh = PLANTED["paste2_box"]
    patch = _scene(seed=12)[140 : 140 + qh, qx : qx + qw]
    patch = np.asarray(
        Image.open(io.BytesIO(_jpeg_bytes(patch, PLANTED["paste2_quality"]))).convert(
            "RGB"
        )
    )
    edited[qy + 3 : qy + 3 + qh, qx + 3 : qx + 3 + qw] = patch
    # 3. Keep the camera's metadata, including its thumbnail of the ORIGINAL picture.
    data = _jpeg_bytes(
        edited, 92, _exif(_thumbnail(original), software="PhotoFix Pro 4.2")
    )
    # 4. Append an archive after the end marker.
    archive = io.BytesIO()

    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("secret.txt", PLANTED["secret"])

    with open(path, "wb") as handle:
        handle.write(data + archive.getvalue())

    return path


def generated_png(path: str) -> str:
    """A 1024x1024 PNG carrying a Stable Diffusion recipe in its 'parameters' chunk."""
    rng = np.random.default_rng(3)
    y = np.linspace(0, 1, 1024)[:, None]
    x = np.linspace(0, 1, 1024)[None, :]
    rgb = np.stack(
        [200 * (1 - y) * (0.6 + 0.4 * x), 120 * (1 - y) + 40 * x, 160 * y + 60 + 0 * x],
        axis=-1,
    )
    rgb = cv2.GaussianBlur(rgb.astype(np.float32), (0, 0), 3)
    bgr = cv2.cvtColor(np.clip(rgb, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.rectangle(bgr, (380, 300), (470, 760), (230, 230, 240), -1)
    cv2.circle(bgr, (425, 280), 60, (90, 200, 255), -1)
    cv2.ellipse(bgr, (512, 760), (512, 120), 0, 0, 180, (60, 70, 50), -1)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) + rng.normal(
        0, 0.6, (1024, 1024, 3)
    )
    image = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))
    from PIL.PngImagePlugin import PngInfo

    meta = PngInfo()
    meta.add_text(
        "parameters",
        f"{PLANTED['sd_prompt']}\nNegative prompt: blurry, text, watermark\n"
        f"Steps: 30, Sampler: DPM++ 2M Karras, CFG scale: 7, Seed: {PLANTED['sd_seed']}, Size: 1024x1024, Model hash: 31e35c80fc, Model: sd_xl_base_1.0",
    )
    meta.add_text("Software", "AUTOMATIC1111 stable-diffusion-webui")
    image.save(path, "PNG", pnginfo=meta)

    return path


def _poster(seed: int, width: int = 1024, height: int = 768) -> np.ndarray:
    """A graphic (gradients, shapes, a title) once saved as JPEG: the kind of
    low-noise carrier LSB steganography tools are used on."""
    rng = np.random.default_rng(seed)
    y = np.linspace(0, 1, height)[:, None]
    x = np.linspace(0, 1, width)[None, :]
    rgb = np.stack(
        [40 + 160 * x * (1 - 0.3 * y), 30 + 120 * y + 40 * x, 90 + 100 * (1 - x) * y],
        axis=-1,
    ).astype(np.float32)
    bgr = cv2.cvtColor(np.clip(rgb, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

    for _ in range(12):
        centre = (int(rng.uniform(0, width)), int(rng.uniform(0, height)))
        colour = tuple(int(v) for v in rng.uniform(40, 220, 3))
        cv2.circle(bgr, centre, int(rng.uniform(30, 120)), colour, -1, cv2.LINE_AA)

    cv2.putText(
        bgr,
        "SUMMER FESTIVAL 2026",
        (60, 400),
        cv2.FONT_HERSHEY_DUPLEX,
        2.2,
        (255, 255, 255),
        4,
        cv2.LINE_AA,
    )
    decoded = Image.open(
        io.BytesIO(_jpeg_bytes(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), 85))
    )

    return np.asarray(decoded.convert("RGB")).copy()


def stego_png(path: str) -> str:
    """A lossless PNG with a picture hidden under transparency on the right third
    and an LSB-embedded message over the first 45% of the (composed) pixels."""
    carrier = _poster(5)
    hidden = _poster(9)
    alpha = np.full(carrier.shape[:2], 255, dtype=np.uint8)
    alpha[:, 640:] = (
        0  # the right third is "transparent" but keeps the other picture's pixels
    )
    rgb = carrier.copy()
    rgb[:, 640:] = hidden[:, 640:]
    flat = rgb.reshape(-1)
    count = int(len(flat) * 0.45)
    # Steganography tools encrypt the payload first, so the embedded bits are
    # uniform: a keystream over the message text stands in for that.
    repeats = count // (8 * len(PLANTED["lsb_message"])) + 1
    message = np.frombuffer(
        (PLANTED["lsb_message"] * repeats)[: count // 8 + 1], dtype=np.uint8
    )
    keystream = np.random.default_rng(2026).integers(
        0, 256, len(message), dtype=np.uint8
    )
    bits = np.unpackbits(message ^ keystream)[:count]
    flat[:count] = (flat[:count] & 0xFE) | bits
    Image.fromarray(np.dstack([rgb, alpha]), "RGBA").save(path, "PNG")

    return path


def motion_photo(path: str, video_path: str | None = None) -> str:
    """A JPEG with an MP4 clip appended after the end marker, as motion photos store it."""
    scene = _scene(1200, 900, seed=13)
    data = _jpeg_bytes(scene, 90, _exif(None, width=1200, height=900))

    if video_path is None:
        # A tiny stand-in MP4: a valid ftyp box followed by a free box (enough to be recognised).
        payload = (
            struct.pack(">I4s4sI", 24, b"ftyp", b"isom", 512)
            + b"isomiso2mp41"
            + struct.pack(">I4s", 8 + 4000, b"free")
            + bytes(4000)
        )
    else:
        with open(video_path, "rb") as handle:
            payload = handle.read()

    with open(path, "wb") as handle:
        handle.write(data + payload)

    return path


def png_was_jpeg(path: str) -> str:
    """A PNG that was a quality-60 JPEG before conversion: the 8x8 grid survives."""
    poster = _poster(17, 1280, 960)
    decoded = np.asarray(Image.open(io.BytesIO(_jpeg_bytes(poster, 60))).convert("RGB"))
    Image.fromarray(decoded).save(path, "PNG", compress_level=6)

    return path


def crc(data: bytes) -> int:
    return zlib.crc32(data) & 0xFFFFFFFF
