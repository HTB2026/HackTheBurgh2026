# EARWITNESS

**Discover the invisible in a video.** Drop in any clip and a forensic HUD
streams findings from the signals a viewer never notices: the hum of the power
grid, flicker of the lights, heartbeats in skin colour, tones above hearing,
the rhythm of the encoder, the physics of the camera sensor.

Built for HackTheBurgh (G-Research: *discover the invisible in data*).

```sh
uv sync                                  # Python 3.14 (uv fetches it); needs ffmpeg + tesseract
(cd web && npm install && npm run build) # the HUD
uv run earwitness serve                  # http://127.0.0.1:8765
uv run earwitness demo --out demo        # demo pack with planted ground truth
```

Optional extras, configured in `.env` (gitignored):

```sh
OPENAI_API_KEY=sk-...          # transcript, AI visual analyst, narrative brief, geolocation judge
EARWITNESS_GEOSLEUTH=~/.cache/earwitness/vendor/geo-sleuth/skills/geo-sleuth
```

For geolocation, clone https://github.com/Oldcircle/geo-sleuth (MIT) and point
`EARWITNESS_GEOSLEUTH` at its `skills/geo-sleuth` folder. Earwitness runs its
local intake (edge crops + OCR, no reverse image search), then a vision model
estimates the location **blind**, without the file's GPS. When GPS exists, the
error in km is reported. The key is read from the environment only and never
reaches the browser. Frames and audio are sent to OpenAI only when the key is
set.

The UI opens in a plain-language **Simple** view (verdict + "Has it been
edited? / When? / Where? / Who? / Anything hidden?" cards with confidence);
**Technical details** shows the full forensic HUD.

## Modules

| Module | What it finds | How |
|---|---|---|
| **Provenance** | Device, GPS (with street address), time zone, which clock stamps start vs end, editing software, rewritten container, sun position at the claimed time and place | ffprobe tags, ISO-BMFF box walk, filename/container clock reconciliation, NOAA solar model, OSM reverse geocoding |
| **Frame timeline** | Hard cuts (ignoring pans and flashes), dropped/duplicated frames, off-rhythm keyframes, bitrate | packet table, frame and histogram differencing with neighbourhood tests |
| **Spectral** | Ultrasonic/hidden tones, DTMF keypad digits, perfect digital silence, room-tone jumps, lossy-encode bandwidth cliffs, Doppler sweeps of passing aircraft | STFT, Goertzel, noise-floor step detection, Viterbi tone tracking |
| **Mains hum / ENF** | Grid (50/60 Hz), date and time **to the second**, audio splices by phase jumps and by windows that date to different times; also dates **silent** footage from lamp flicker | hum tracking vs NESO 1 s / Elexon grid records, fetched by HTTP range around the claimed time |
| **Light & flicker** | Mains flicker hidden in rolling-shutter banding: grid region, sensor readout time, a visual grid-frequency trace; lighting vs the sun | per-row temporal spectra, alias prediction, phase coherence down the frame |
| **Pulse (rPPG)** | Heart rate of each face on camera; Eulerian colour-magnification video | YuNet faces, POS method, chroma-only temporal band-pass |
| **Pixel forensics** | Copy-move (cloned) regions; error-level and noise-residual heatmaps | ORB self-matching with displacement clustering; ELA; regression-normalised residuals |
| **A/V sync** | Audio shifted or re-dubbed against the picture | onset envelope × motion energy cross-correlation, checked for consistency across the clip |
| **OCR** | Postcodes, number plates, phone numbers, URLs, dates, clock times in frame | Tesseract, two layout passes |
| **Transcript** / **AI analyst** | Speech; place, time of day, notable details and visual manipulation signs (labelled AI-inferred) | OpenAI (optional) |
| **Geo-sleuth intake** | geo-sleuth intake dossier for the sharpest frames | optional bridge; set `EARWITNESS_GEOSLEUTH` |
| **Case brief** | Verdict (EDITED / POSSIBLY EDITED / NO SIGNS OF EDITING) with WHEN / WHERE / DEVICE / PEOPLE / INTEGRITY / HIDDEN lines citing finding ids; flags edits **corroborated** by sound and picture | deterministic rules; plus an AI narrative if enabled |

The HUD has **ENHANCE**: drag a box on the paused video and multi-frame
super-resolution aligns ±N neighbouring frames (ECC) and merges them robustly,
the way burst photography does, so moving parts do not ghost.

## Verified

`uv run pytest` (14 tests, ~45 s) runs every module on synthetic clips with
planted ground truth:

| Planted | Recovered |
|---|---|
| 18.5 kHz tone, DTMF `1984#`, 0.3 s digital silence @12 s, room-tone jump + cut @20 s | all found; cut and audio break **corroborated** |
| 72 bpm skin pulse | 72 bpm |
| 100 Hz flicker @30 fps / 120 Hz @25 fps, 20 ms readout | 50 Hz / 60 Hz grid, 20.0 ms |
| Lamps following the real GB grid at 2024-03-12 23:10:00, metadata 20 min off | dated **23:10:00** from lighting alone, metadata contradiction flagged |
| Mains hum, spliced with audio from 30 min later | dated to the second; phase jump @150 s; two date clusters |
| Audio shifted +250 ms | +260 ms; aligned clip +10 ms |
| 90×70 px clone | offset (300, 140) px |
| Sign with postcode, plate, URL, hours | all read |

Checked on real iPhone/Android clips: no false cuts on fast pans, no false hum
or flicker, GPS → street address, time zone reconstructed from filename vs
container clock.

## Honest limits

- Phone microphones filter out low frequencies and iPhones lock exposure to
  the flicker period, so many phone clips carry neither hum nor visible
  flicker. The modules say so instead of guessing.
- Pulse from handheld, compressed phone video is usually low-confidence and
  is labelled as such.
- Grain/ELA heatmaps on video are shaped by the codec and are for visual
  inspection only; they never raise findings.
- Automatic grid records cover GB (NESO, Elexon). Other grids need a CSV.

## Layout

`src/earwitness/modules/`: one file per module (common `Reporter` interface in
`base.py`). `server/`: FastAPI + WebSocket event stream (`docs/PROTOCOL.md`),
enhance and HTML report. `web/`: React/Vite HUD. The original CLI (`analyse`,
`reference`, `log-flights`, `synthesise`) still works; see `uv run earwitness -h`.
