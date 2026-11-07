"""The ordered list of analysis modules the case runner executes."""

import importlib
from typing import Any

_MODULES = [
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


def all_modules() -> list[Any]:
    modules = []

    for path, cls in _MODULES:
        try:
            module = importlib.import_module(path)
        except ModuleNotFoundError as error:
            if error.name == path:
                continue

            raise

        modules.append(getattr(module, cls)())

    return modules
