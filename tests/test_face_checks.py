"""Per-face paste checks (grain, JPEG ghost) and video face appearances, on synthetic data."""

import cv2
import numpy as np

from earwitness import photo
from earwitness.modules.faces import _face_checks
from earwitness.modules.faces import _ghost
from earwitness.modules.pulse import _appearances


def _textured(seed: int = 0, size: int = 1024) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = cv2.GaussianBlur(
        rng.normal(128, 40, (size, size)).astype(np.float32), (0, 0), 3
    )
    grain = rng.normal(0, 4, (size, size, 3))
    return np.clip(base[..., None] + grain, 0, 255).astype(np.uint8)


def _person(x: int, y: int, w: int, size: int) -> dict:
    return {"index": 1, "box": photo.box_fraction(x, y, w, w, size, size)}


def test_ghost_reads_the_save_quality():
    img = photo.decode_jpeg_bytes(photo.encode_jpeg_bytes(_textured(), 75))
    luma = img[..., 0] * 0.299 + img[..., 1] * 0.587 + img[..., 2] * 0.114
    qs = _ghost(luma, 0, 0, 512, 512)
    assert len(qs) > 1000
    assert np.mean(np.abs(qs - 75) <= 1) > 0.8


def test_smoothed_face_is_flagged_and_untouched_face_is_not():
    img = _textured()
    clean = [_person(400, 400, 200, 1024)]
    _face_checks(img, clean, jpeg=False)
    assert clean[0]["flags"] == []

    pasted = img.copy()
    pasted[400:600, 400:600] = cv2.GaussianBlur(pasted[400:600, 400:600], (0, 0), 2)
    flagged = [_person(400, 400, 200, 1024)]
    _face_checks(pasted, flagged, jpeg=False)
    assert "smooth" in flagged[0]["flags"]


def test_small_faces_are_not_tested():
    people = [_person(10, 10, 60, 1024)]
    _face_checks(_textured(), people, jpeg=True)
    assert people[0]["checks"] == {} and people[0]["flags"] == []


def test_appearances_split_long_gaps_and_bridge_short_ones():
    fps = 30.0
    box = np.array([100.0, 100.0, 50.0, 50.0])
    # one tracker track: 0-3 s, a 1 s gap, 4-6 s, then a 5 s gap (a cut), 11-13 s
    frames = (
        list(range(0, 90, 10)) + list(range(120, 180, 10)) + list(range(330, 390, 10))
    )
    tracks = [
        {"frames": frames, "boxes": [box] * len(frames)},
        {"frames": [500, 510], "boxes": [box, box]},
    ]
    people = _appearances(tracks, fps)
    assert len(people) == 2  # the 2-detection blip is dropped
    first, second = people
    assert len(first["intervals"]) == 2 and first["first_s"] == 0
    assert second["first_s"] == 11
