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
                "recognisable": len(recognisable),
                "names": names,
                "people": [
                    {k: v for k, v in p.items() if k != "landmarks"} for p in people
                ],
            },
        )
        out.progress(1.0)
