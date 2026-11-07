"""The ordered list of analysis modules the case runner executes, per media kind."""

import importlib
from typing import Any

_VIDEO_MODULES = [
    ("earwitness.modules.provenance", "Provenance"),
    ("earwitness.modules.timeline", "Timeline"),
    ("earwitness.modules.spectrum", "Spectrum"),
    ("earwitness.modules.enf", "Enf"),
    ("earwitness.modules.light", "Light"),
    ("earwitness.modules.pulse", "Pulse"),
    ("earwitness.modules.tamper", "Tamper"),
    ("earwitness.modules.avsync", "AvSync"),
    ("earwitness.modules.ocr", "Ocr"),
    ("earwitness.modules.transcript", "Transcript"),
    ("earwitness.modules.sounds", "Sounds"),
    ("earwitness.modules.vision", "Vision"),
    ("earwitness.modules.geo", "Geo"),
    ("earwitness.modules.weather", "Weather"),
    ("earwitness.modules.brief", "Brief"),
]

# Still photographs: no clock, no sound, but every pixel and every byte of
# metadata at full resolution.
_IMAGE_MODULES = [
    ("earwitness.modules.photo_provenance", "PhotoProvenance"),
    ("earwitness.modules.compression", "Compression"),
    ("earwitness.modules.photo_pixels", "PhotoPixels"),
    ("earwitness.modules.hidden", "Hidden"),
    ("earwitness.modules.faces", "Faces"),
    ("earwitness.modules.synthetic", "Synthetic"),
    ("earwitness.modules.ocr", "Ocr"),
    ("earwitness.modules.vision", "Vision"),
    ("earwitness.modules.geo", "Geo"),
    ("earwitness.modules.weather", "Weather"),
    ("earwitness.modules.brief", "Brief"),
]

_BY_KIND = {"video": _VIDEO_MODULES, "image": _IMAGE_MODULES, "audio": _VIDEO_MODULES}


def all_modules(kind: str = "video") -> list[Any]:
    modules = []

    for path, cls in _BY_KIND.get(kind, _VIDEO_MODULES):
        try:
            module = importlib.import_module(path)
        except ModuleNotFoundError as error:
            if error.name == path:
                continue

            raise

        modules.append(getattr(module, cls)())

    return modules


def kinds() -> list[str]:
    return ["video", "image"]
