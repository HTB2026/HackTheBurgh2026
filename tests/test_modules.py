"""End-to-end module tests on synthetic evidence with planted ground truth."""

import asyncio
import json
import os
import shutil
import subprocess

import pytest

from earwitness import demo as fixtures
from earwitness.server.jobs import CaseStore

os.environ["EARWITNESS_OFFLINE"] = "1"
os.environ.pop("OPENAI_API_KEY", None)


def run(tmp_path, video: str, modules: list[str], **options) -> list[dict]:
    async def go():
        store = CaseStore(str(tmp_path / "cases"))
        case = store.create(
            video, {"original_name": os.path.basename(video), **options}
        )
        shutil.copy(video, case.evidence_path)
        store.start(case, only=modules)
        await case.task

        return case.events

    events = asyncio.run(go())
    errors = [e for e in events if e["type"] == "module" and e["status"] == "error"]
    assert not errors, errors[0].get("trace")

    return [e["finding"] for e in events if e["type"] == "finding"]


def titles(findings: list[dict]) -> str:
    return "\n".join(f"{f['module']}: {f['title']}" for f in findings)


@pytest.fixture(scope="module")
def planted(tmp_path_factory):
    return fixtures.planted_video(str(tmp_path_factory.mktemp("v") / "planted.mp4"))


def test_planted_audio_anomalies(tmp_path, planted):
    findings = run(tmp_path, planted, ["timeline", "spectrum", "brief"])
    text = titles(findings)
    assert "Hidden tone at 18,500 Hz" in text
    assert "Keypad tones decoded: " + fixtures.PLANTED["dtmf"] in text
    silence = next(f for f in findings if "digital silence" in f["title"])
    assert abs(silence["time_s"] - fixtures.PLANTED["silence_at_s"]) < 0.05
    jump = next(f for f in findings if f["title"].startswith("Room tone jumps"))
    assert abs(jump["time_s"] - fixtures.PLANTED["room_jump_at_s"]) <= 1
    cut = next(f for f in findings if f["title"].startswith("Hard cut"))
    assert abs(cut["time_s"] - fixtures.PLANTED["cut_at_s"]) < 0.05
    assert "Edit corroborated by sound and picture" in text


def test_pulse_recovers_planted_rate(tmp_path, planted):
    findings = run(tmp_path, planted, ["pulse"])
    pulse = next(f for f in findings if f["module"] == "pulse" and "bpm" in f["title"])
    assert abs(pulse["data"]["bpm"] - fixtures.PLANTED["pulse_bpm"]) < 4


@pytest.mark.parametrize(("grid", "fps"), [(50.0, 30.0), (60.0, 25.0)])
def test_light_flicker_identifies_grid(tmp_path, grid, fps):
    video = fixtures.flicker_video(
        str(tmp_path / "flicker.mp4"), grid_hz=grid, fps=fps, readout_s=0.02
    )
    findings = run(tmp_path, video, ["light"])
    found = next(f for f in findings if f["title"].startswith("Lights flicker"))
    assert f"{int(grid)} Hz grid" in found["title"]
    readout = next(f for f in findings if f["title"].startswith("Sensor readout"))
    assert abs(readout["data"]["readout_ms"] - 20) < 2


def test_light_no_false_flicker(tmp_path, planted):
    findings = run(tmp_path, planted, ["light"])
    assert not any(f["title"].startswith("Lights flicker") for f in findings)


@pytest.mark.parametrize("offset", [0.0, 0.25])
def test_av_sync_offset(tmp_path, offset):
    video = fixtures.av_events_video(str(tmp_path / "av.mp4"), audio_offset_s=offset)
    findings = run(tmp_path, video, ["timeline", "spectrum", "avsync"])
    sync = next(f for f in findings if f["module"] == "avsync")
    assert abs(sync["data"]["lag_ms"] - offset * 1000) <= 40


def test_provenance_reads_gps_and_device(tmp_path, planted):
    tagged = str(tmp_path / "tagged.mov")
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", planted, "-c", "copy",
         "-metadata", "creation_time=2026-11-07T14:00:30Z",
         "-metadata", "com.apple.quicktime.creationdate=2026-11-07T14:00:30+0000",
         "-metadata", "com.apple.quicktime.location.ISO6709=+55.9442-003.1874+080.000/",
         "-metadata", "com.apple.quicktime.make=Apple", "-metadata", "com.apple.quicktime.model=iPhone 15",
         "-movflags", "use_metadata_tags", tagged],
        check=True,
    )  # fmt: skip
    findings = run(tmp_path, tagged, ["provenance"])
    text = titles(findings)
    assert "GPS fix: 55.94420, -3.18740" in text
    assert "iPhone 15" in text
    assert "Sun " in text


def test_enf_dates_and_detects_splice(tmp_path):
    from earwitness import synth

    truth = asyncio.run(synth.create_scenario(str(tmp_path / "syn")))
    grid = str(tmp_path / "syn" / "grid.csv")
    clean = run(
        tmp_path,
        str(tmp_path / "syn" / "clip.mp4"),
        ["provenance", "enf"],
        grid_path=grid,
    )
    dated = next(f for f in clean if f["title"].startswith("ENF dates"))
    assert truth["start"][:19].replace("T", " ") in dated["title"]
    spliced = run(
        tmp_path,
        str(tmp_path / "syn" / "spliced.mp4"),
        ["provenance", "enf"],
        grid_path=grid,
    )
    assert any("phase jumps" in f["title"] for f in spliced)
    groups = next(f for f in spliced if "different times" in f["title"])
    assert len(groups["data"]["groups"]) >= 2


def test_ocr_reads_postcode_and_plate(tmp_path):
    from PIL import Image
    from PIL import ImageDraw
    from PIL import ImageFont

    font_path = "/System/Library/Fonts/Helvetica.ttc"

    if not os.path.isfile(font_path) or shutil.which("tesseract") is None:
        pytest.skip("needs tesseract and Helvetica")

    image = Image.new("RGB", (1280, 720), (51, 68, 85))
    draw = ImageDraw.Draw(image)
    draw.text(
        (100, 120),
        "SCOTMID  12 Nicolson St  EH8 9DH",
        fill="white",
        font=ImageFont.truetype(font_path, 48),
    )
    draw.rectangle((400, 380, 820, 480), fill=(250, 210, 0))
    draw.text(
        (430, 395), "SN19 XKD", fill="black", font=ImageFont.truetype(font_path, 64)
    )
    png = str(tmp_path / "sign.png")
    image.save(png)
    video = str(tmp_path / "sign.mp4")
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-loop", "1", "-i", png, "-t", "3", "-r", "30",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", video],
        check=True,
    )  # fmt: skip
    text = titles(run(tmp_path, video, ["timeline", "ocr"]))
    assert "EH8 9DH" in text
    assert "SN19 XKD" in text


def test_events_are_json(tmp_path, planted):
    async def go():
        store = CaseStore(str(tmp_path / "cases"))
        case = store.create(planted, {})
        shutil.copy(planted, case.evidence_path)
        store.start(case, only=["provenance", "timeline", "tamper"])
        await case.task

        return case

    case = asyncio.run(go())

    with open(os.path.join(case.workdir, "events.jsonl")) as handle:
        events = [json.loads(line) for line in handle]

    assert events[-1]["type"] == "done"
    assert [e["seq"] for e in events] == list(range(len(events)))


def test_lighting_dates_silent_footage(tmp_path):
    """Lamp flicker following the real GB grid dates the footage to the second."""
    from datetime import datetime
    from datetime import timedelta
    from datetime import timezone

    from earwitness import grid_fetch

    truth = datetime(2024, 3, 12, 23, 10, 0, tzinfo=timezone.utc)
    series, _ = asyncio.run(
        grid_fetch.fetch(truth - timedelta(minutes=5), truth + timedelta(minutes=6))
    )

    if series is None:
        pytest.skip("grid record unavailable (offline)")

    offset = int((truth - series.start).total_seconds())
    raw = fixtures.grid_flicker_video(
        str(tmp_path / "raw.mp4"), series.values[offset : offset + 91]
    )
    video = str(tmp_path / "flicker.mp4")
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", raw, "-c", "copy",
         "-metadata", "creation_time=2024-03-12T23:31:30Z", video],
        check=True,
    )  # fmt: skip
    findings = run(tmp_path, video, ["provenance", "light", "enf"])
    dated = next(f for f in findings if f["title"].startswith("Lighting dates"))
    assert "2024-03-12 23:10:00" in dated["title"]
    assert any(
        f["title"] == "Metadata time contradicted by the lights" for f in findings
    )


def test_copy_move_clone_found(tmp_path):
    video = fixtures.cloned_video(str(tmp_path / "clone.mp4"))
    findings = run(tmp_path, video, ["timeline", "tamper"])
    clone = next(f for f in findings if f["title"].startswith("Cloned region"))
    dx, dy = clone["data"]["offset"]
    assert abs(abs(dx) - 300) <= 6 and abs(abs(dy) - 140) <= 6


def test_flashes_are_not_cuts(tmp_path):
    video = fixtures.av_events_video(str(tmp_path / "flash.mp4"))
    findings = run(tmp_path, video, ["timeline"])
    assert not any(f["title"].startswith("Hard cut") for f in findings)


def test_weather_reads_negations():
    from earwitness.modules.weather import _seen_conditions

    assert _seen_conditions(
        "Cloudy or hazy night; no visible rain, pavement appears dry."
    ) == (False, True)
    assert _seen_conditions("It is not raining; clear sky.") == (False, True)
    assert _seen_conditions("Heavy rain, wet pavement, umbrellas.") == (True, False)
    assert _seen_conditions("Overcast evening.") == (False, False)
