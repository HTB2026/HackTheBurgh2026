"""People in the picture: how many, how recognisable, and what their eyes reflect.

A photograph holds faces at full resolution, so unlike video the question is
not "is there a pulse" but "could a stranger recognise this person": face size
in pixels and sharpness decide that. Each face's eyes are cropped and enlarged,
because corneal reflections show what was in front of the subject (windows,
screens, the photographer). Names written into the file by photo managers
(face regions with names) are reported too. Nobody is identified here.
"""

import asyncio
import re
from typing import Any

import cv2
import numpy as np

from earwitness import photo
from earwitness.modules.base import CaseContext
from earwitness.modules.base import Reporter
from earwitness.modules.base import Skip
from earwitness.modules.pulse import _model

_DETECT_SIDE = 1600
_RECOGNISABLE_PX = 80  # face width in original pixels
_SHARP = 60.0  # Laplacian variance on a 128 px face crop
_CHECK_PX = (
    120  # faces narrower than this hold too few blocks for the grain/ghost comparison
)
_SMOOTH_RATIO = (
    0.7  # face grain below 70% of its surroundings (and of other faces) is a lead
)
_GHOST_AGREE = 0.6  # under 60% of a face's blocks at the picture's quality is a lead
_GHOST_BLOCKS = 15  # textured 8x8 blocks needed before the ghost says anything

_HIGH_PASS = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], np.float32) / 8
_QUALITIES = np.array(list(range(30, 70, 5)) + list(range(70, 101)))
_TABLES = np.stack([photo.scaled_standard_table(photo.STD_LUMA.reshape(8, 8).astype(float), int(q)) for q in _QUALITIES])  # fmt: skip
_DCT = np.array(
    [[(np.sqrt(0.5) if u == 0 else 1.0) * 0.5 * np.cos((2 * x + 1) * u * np.pi / 16) for x in range(8)] for u in range(8)]
)  # fmt: skip

_REGION_NAME = re.compile(
    r"mwg-rs:Name=\"([^\"]{1,80})\"|<mwg-rs:Name>([^<]{1,80})</mwg-rs:Name>|MPReg:PersonDisplayName=\"([^\"]{1,80})\""
)


def _detect(model: str, rgb: np.ndarray) -> np.ndarray | None:
    detector = cv2.FaceDetectorYN.create(
        model, "", (rgb.shape[1], rgb.shape[0]), 0.6, 0.3, 500
    )
    _, faces = detector.detect(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    return faces


def _sharpness(crop: np.ndarray) -> float:
    if crop.size == 0:
        return 0.0

    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(
        gray, (128, int(128 * gray.shape[0] / max(1, gray.shape[1])) or 1)
    )

    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _ghost(luma: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
    """Best-fit last-save quality of each textured 8x8 block in a region (grid from 0,0).

    For each candidate quality, the block's DCT is requantised; the error, relative to
    what random coefficients would give, is near zero at the quality it was saved at.
    """
    bx0, by0, bx1, by1 = -(-x0 // 8), -(-y0 // 8), x1 // 8, y1 // 8
    found = []

    for by in range(by0, by1, 16):
        rows = min(16, by1 - by)

        if bx1 <= bx0 or rows <= 0:
            break

        block = luma[by * 8 : (by + rows) * 8, bx0 * 8 : bx1 * 8]
        block = block.reshape(rows, 8, bx1 - bx0, 8).transpose(0, 2, 1, 3) - 128.0
        coef = np.einsum("ux,abxy,vy->abuv", _DCT, block, _DCT)
        ac = (coef**2).sum((2, 3)) - coef[..., 0, 0] ** 2
        clipped = (block.min((2, 3)) <= -127.5) | (block.max((2, 3)) >= 126.5)
        c = coef[:, :, None]
        err = ((c - np.round(c / _TABLES) * _TABLES) ** 2).sum((3, 4))
        ratio = err / (np.minimum(c**2, _TABLES**2 / 12).sum((3, 4)) + 1e-3)
        q = np.where(
            (ac > 50) & ~clipped & (ratio.min(2) < 0.35), _QUALITIES[ratio.argmin(2)], 0
        )
        found.append(q[q > 0])

    return np.concatenate(found) if found else np.zeros(0, int)


def _grain(
    high: np.ndarray, x0: int, y0: int, x1: int, y1: int, keep: np.ndarray | None = None
) -> float | None:
    """Sensor grain: robust spread of the high-pass in the flattest quarter of 16 px blocks."""
    values = []

    for y in range(max(0, y0), y1 - 16, 16):
        for x in range(max(0, x0), x1 - 16, 16):
            if keep is not None and not keep[y - y0 + 8, x - x0 + 8]:
                continue

            b = high[y : y + 16, x : x + 16]
            values.append(1.4826 * float(np.median(np.abs(b - np.median(b)))))

    return float(np.percentile(values, 25)) if len(values) >= 6 else None


def _picture_quality(luma: np.ndarray) -> tuple[int | None, float]:
    """Dominant last-save quality over 60 random 128 px patches, and the share of blocks agreeing."""
    h, w = luma.shape
    rng = np.random.default_rng(0)
    found = []

    for _ in range(60 if h > 160 and w > 160 else 0):
        x = int(rng.integers(0, w - 128)) // 8 * 8
        y = int(rng.integers(0, h - 128)) // 8 * 8
        found.append(_ghost(luma, x, y, x + 128, y + 128))

    qs = np.concatenate(found) if found else np.zeros(0, int)

    if len(qs) < 100:
        return None, 0.0

    counts = np.bincount(qs)
    best = int(counts.argmax())

    return best, float(counts[max(0, best - 1) : best + 2].sum() / len(qs))


def _face_checks(image: np.ndarray, people: list[dict[str, Any]], jpeg: bool) -> None:
    """Grain and save-quality of each face against its surroundings; adds `checks`/`flags` to each person."""
    full_h, full_w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
    high = cv2.filter2D(gray, -1, _HIGH_PASS)
    luma = image[..., 0] * 0.299 + image[..., 1] * 0.587 + image[..., 2] * 0.114
    picture_q, picture_agree = _picture_quality(luma) if jpeg else (None, 0.0)

    for p in people:
        b = p["box"]
        x, y, w, h = b["x"] * full_w, b["y"] * full_h, b["w"] * full_w, b["h"] * full_h
        p["checks"] = {}
        p["flags"] = []

        if w < _CHECK_PX:
            continue

        # Inner face: cheeks, nose, chin; clear of hair and background.
        ix0, iy0, ix1, iy1 = (
            int(x + 0.2 * w),
            int(y + 0.25 * h),
            int(x + 0.8 * w),
            int(y + 0.9 * h),
        )
        rx0, ry0 = int(max(0, x - 0.8 * w)), int(max(0, y - 0.6 * h))
        rx1, ry1 = int(min(full_w, x + 1.8 * w)), int(min(full_h, y + 1.6 * h))
        ring = np.ones((ry1 - ry0, rx1 - rx0), bool)
        ring[
            max(0, int(y) - ry0) : int(y + h) - ry0,
            max(0, int(x) - rx0) : int(x + w) - rx0,
        ] = False
        face_grain = _grain(high, ix0, iy0, ix1, iy1)
        ring_grain = _grain(high, rx0, ry0, rx1, ry1, ring)
        p["checks"]["grain"] = face_grain
        p["checks"]["grain_around"] = ring_grain

        if picture_q is not None:
            qs = _ghost(luma, ix0, iy0, ix1, iy1)

            if len(qs) >= _GHOST_BLOCKS:
                agree = float(((qs >= picture_q - 1) & (qs <= picture_q + 1)).mean())
                others = qs[(qs < picture_q - 1) | (qs > picture_q + 1)]
                p["checks"]["save_quality"] = picture_q
                p["checks"]["blocks_agreeing"] = agree
                p["checks"]["blocks"] = int(len(qs))

                if agree < _GHOST_AGREE and picture_agree >= 0.8:
                    p["flags"].append("ghost")
                    p["checks"]["other_quality"] = (
                        int(np.bincount(others).argmax()) if len(others) else None
                    )

    for p in people:
        g, around = p["checks"].get("grain"), p["checks"].get("grain_around")

        if g is None or not around or around < 0.4:
            continue

        others = [
            o["checks"]["grain"]
            for o in people
            if o is not p and o["checks"].get("grain")
        ]

        if g < _SMOOTH_RATIO * around and (
            not others or g < _SMOOTH_RATIO * float(np.median(others))
        ):
            p["flags"].append("smooth")


class Faces:
    name = "faces"
    title = "PEOPLE"
    description = "Face count, size and sharpness (recognisable or not), eye-reflection crops, names tagged in the file."
    depends = ["provenance"]

    async def run(self, ctx: CaseContext, out: Reporter) -> None:
        image = ctx.image()
        info = ctx.image_info()

        if image is None or info is None:
            raise Skip("not a decodable image")

        provenance = await ctx.wait_for("provenance") or {}
        model = await _model()

        if model is None:
            raise Skip("face model unavailable (offline)")

        out.progress(0.1, "detecting faces")
        small = photo.fit(image, _DETECT_SIDE)
        scale = image.shape[1] / small.shape[1]
        faces = await asyncio.to_thread(_detect, model, small)
        faces = [] if faces is None else [f for f in faces if f[-1] >= 0.6]
        full_w, full_h = image.shape[1], image.shape[0]
        people: list[dict[str, Any]] = []
        canvas = cv2.cvtColor(small, cv2.COLOR_RGB2BGR).copy()

        for index, face in enumerate(sorted(faces, key=lambda f: -f[2] * f[3])[:12]):
            x, y, w, h = (float(v) * scale for v in face[:4])
            x0, y0 = int(max(0, x)), int(max(0, y))
            x1, y1 = int(min(full_w, x + w)), int(min(full_h, y + h))
            crop = image[y0:y1, x0:x1]
            sharp = _sharpness(crop)
            recognisable = w >= _RECOGNISABLE_PX and sharp >= _SHARP
            landmarks = [
                (float(face[4 + 2 * i]) * scale, float(face[5 + 2 * i]) * scale)
                for i in range(5)
            ]
            person = {
                "index": index + 1,
                "box": photo.box_fraction(x0, y0, x1 - x0, y1 - y0, full_w, full_h),
                "width_px": w,
                "sharpness": sharp,
                "recognisable": recognisable,
                "confidence": float(face[-1]),
                "landmarks": landmarks,
            }
            people.append(person)
            colour = (60, 60, 255) if recognisable else (80, 220, 255)
            cv2.rectangle(
                canvas,
                (int(x0 / scale), int(y0 / scale)),
                (int(x1 / scale), int(y1 / scale)),
                colour,
                2,
            )
            cv2.putText(
                canvas,
                f"{index + 1}",
                (int(x0 / scale), int(y0 / scale) - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                colour,
                2,
                cv2.LINE_AA,
            )

            if w >= 48:
                face_path = out.path(f"face{index + 1}.jpg")
                photo.save_jpeg(
                    face_path,
                    photo.fit(crop, 512)
                    if max(crop.shape[:2]) > 512
                    else cv2.resize(
                        crop,
                        None,
                        fx=max(1, 256 / max(crop.shape[:2])),
                        fy=max(1, 256 / max(crop.shape[:2])),
                        interpolation=cv2.INTER_CUBIC,
                    ),
                    92,
                )
                out.artifact(f"face{index + 1}", "image", f"Face {index + 1}: {w:.0f} px wide, {'recognisable' if recognisable else 'too small or blurred to recognise'}",
                             file=face_path, data={"caption": f"Sharpness {sharp:.0f}; detector confidence {face[-1] * 100:.0f}%", "box": person["box"]})  # fmt: skip

            # Eyes: the right eye is landmark 0, the left eye landmark 1 (image coordinates).
            if w >= 160:
                eyes = []

                for ex, ey in landmarks[:2]:
                    radius = int(w * 0.11)
                    ex0, ey0 = int(max(0, ex - radius)), int(max(0, ey - radius * 0.7))
                    ex1, ey1 = (
                        int(min(full_w, ex + radius)),
                        int(min(full_h, ey + radius * 0.7)),
                    )
                    eye = image[ey0:ey1, ex0:ex1]

                    if eye.size:
                        eyes.append(
                            cv2.resize(
                                eye,
                                (320, int(320 * eye.shape[0] / eye.shape[1]) or 1),
                                interpolation=cv2.INTER_LANCZOS4,
                            )
                        )

                if len(eyes) == 2 and eyes[0].shape[0] == eyes[1].shape[0]:
                    strip = np.concatenate(eyes, axis=1)
                    lab = cv2.cvtColor(strip, cv2.COLOR_RGB2LAB)
                    lab[..., 0] = cv2.createCLAHE(
                        clipLimit=2.0, tileGridSize=(4, 4)
                    ).apply(lab[..., 0])
                    strip = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
                    eye_path = out.path(f"eyes{index + 1}.jpg")
                    photo.save_jpeg(eye_path, strip, 94)
                    out.artifact(f"eyes{index + 1}", "image", f"Eyes of face {index + 1}, enlarged: what they reflect",
                                 file=eye_path, data={"caption": "Corneal reflections mirror what was in front of the person: windows, screens, lamps, sometimes the photographer. Local contrast boosted; no pixels invented."})  # fmt: skip
                    person["eyes_artifact"] = f"faces:eyes{index + 1}"

        if people:
            out.progress(0.5, "comparing each face with its surroundings")
            await asyncio.to_thread(_face_checks, image, people, info.format == "JPEG")
            self._report_checks(out, people)

        if people:
            path = out.path("faces.jpg")
            photo.save_bgr(path, canvas, 88)
            out.artifact("faces", "image", f"{len(people)} face{'s' if len(people) != 1 else ''} found (red: recognisable, yellow: not)", file=path,
                         data={"caption": "Boxes from the YuNet face detector. Nobody is identified; only whether a stranger could."})  # fmt: skip

        recognisable = [p for p in people if p["recognisable"]]
        out.progress(0.7, "names in metadata")
        loaded = photo.load(ctx.path)
        names: list[str] = []

        if loaded and loaded.xmp:
            for match in _REGION_NAME.finditer(loaded.xmp):
                name = next((g for g in match.groups() if g), None)

                if name and name not in names:
                    names.append(name)

        if names:
            out.finding(
                "critical",
                f"Name{'s' if len(names) != 1 else ''} tagged on face{'s' if len(names) != 1 else ''} in the file: {', '.join(names[:4])}",
                "Photo managers (Picasa, Lightroom, Windows Photo Gallery, some phones) write face regions with the person's name into the XMP block. "
                "They stay in every copy unless stripped, so the file itself says who is in it.",
                tags=["identity", "face", "privacy", "metadata"],
                data={"names": names},
            )

        if people:
            largest = people[0]
            out.finding(
                "critical" if recognisable else "notice",
                f"{len(people)} face{'s' if len(people) != 1 else ''} in the picture, {len(recognisable)} recognisable"
                if recognisable or len(people) > 1
                else "1 face, too small or blurred to recognise",
                f"The largest face is {largest['width_px']:.0f} px wide with sharpness {largest['sharpness']:.0f}. "
                + (
                    f"{len(recognisable)} face{'s are' if len(recognisable) != 1 else ' is'} large and sharp enough for a stranger (or a face-search engine) to recognise. "
                    if recognisable
                    else "No face is large and sharp enough to recognise reliably. "
                )
                + (
                    f"Eye crops are attached for {sum('eyes_artifact' in p for p in people)} face{'s' if sum('eyes_artifact' in p for p in people) != 1 else ''}; reflections there show what the person was looking at."
                    if any("eyes_artifact" in p for p in people)
                    else ""
                ),
                tags=["face", "person", "privacy"]
                + (["identifiable"] if recognisable else []),
                data={
                    "faces": [
                        {k: v for k, v in p.items() if k != "landmarks"} for p in people
                    ],
                    "box": largest["box"],
                    "recognisable": len(recognisable),
                },
            )
        else:
            subject = provenance.get("subject_location")
            out.finding(
                "info",
                "No faces found",
                "The face detector found nobody facing the camera. People seen from behind, very small in frame, or masked are not counted."
                + (
                    " The camera itself focused on a subject region, so there may be a person it saw."
                    if subject
                    else ""
                ),
                tags=["face"],
            )

        ctx.publish(
            self.name,
            {
                "faces": len(people),
                "flagged": [p["index"] for p in people if p["flags"]],
                "recognisable": len(recognisable),
                "names": names,
                "people": [
                    {k: v for k, v in p.items() if k != "landmarks"} for p in people
                ],
            },
        )
        out.progress(1.0)

    def _report_checks(self, out: Reporter, people: list[dict[str, Any]]) -> None:
        checked = [
            p
            for p in people
            if p["checks"].get("grain") is not None or "blocks" in p["checks"]
        ]

        for p in people:
            if not p["flags"]:
                continue

            c, n = p["checks"], p["index"]
            both = "smooth" in p["flags"] and "ghost" in p["flags"]
            reasons = []

            if "smooth" in p["flags"]:
                reasons.append(
                    f"its skin carries less grain ({c['grain']:.2f}) than the picture around it ({c['grain_around']:.2f}), "
                    "as if it came from a softer, smaller or more compressed photo"
                )

            if "ghost" in p["flags"]:
                other = c.get("other_quality")
                reasons.append(
                    f"only {c['blocks_agreeing'] * 100:.0f}% of its {c['blocks']} textured JPEG blocks fit the quality the rest of the picture was saved at (Q{c['save_quality']})"
                    + (f"; many fit Q{other} instead" if other else "")
                )

            out.finding(
                "warning" if both else "notice",
                f"Face {n} looks pasted in: grain and save quality both differ from the rest" if both
                else f"Face {n} has {'smoother grain' if 'smooth' in p['flags'] else 'a different JPEG save quality'} than its surroundings",
                f"Face {n} was compared with the picture around it: " + "; and ".join(reasons) + ". "
                + ("Two independent traces agree on the same face, which is what a face pasted from another photo leaves. "
                   if both else
                   "On its own this is a lead, not proof: phone beauty filters, makeup, focus and heavy skin smoothing also lower grain, "
                   "and flat or bright skin holds few usable blocks. ")
                + "Look at the face's edges (jaw, hairline) in the viewer for a seam.",
                tags=(["splice", "noise", "ghost", "face-check"] if both else
                      ["noise" if "smooth" in p["flags"] else "ghost", "face-check"]),
                data={"box": p["box"], "face": n, **{k: v for k, v in c.items() if v is not None}},
            )  # fmt: skip

        if checked and not any(p["flags"] for p in people):
            out.finding(
                "info",
                f"No pasted faces: {len(checked)} face{'s' if len(checked) != 1 else ''} match{'' if len(checked) != 1 else 'es'} the picture's grain and save quality",
                "Each face large enough to test was compared with the picture around it: the grain of its skin and the JPEG quality its blocks were last saved at. "
                "A face pasted from another photo usually differs in one or both. Faces narrower than "
                f"{_CHECK_PX} px are not tested, and a skilled swap that re-adds grain and resaves the whole picture can pass.",
                tags=["face-check"],
                data={"faces": [p["index"] for p in checked]},
            )  # fmt: skip
