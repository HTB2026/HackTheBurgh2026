# EARWITNESS

**Discover the invisible in a video or a photograph.** Drop in any clip or
picture and a forensic HUD streams findings from the signals a viewer never
notices: the hum of the power grid, flicker of the lights, heartbeats in skin
colour, tones above hearing, the rhythm of the encoder, the physics of the
camera sensor; and, for stills, every byte of metadata, the quantisation
tables, the ghosts of earlier saves, the camera's own thumbnail, the lowest
bits of every pixel and the reflections in people's eyes.

Built for HackTheBurgh (G-Research: *discover the invisible in data*).

```sh
uv sync                                  # Python 3.14 (uv fetches it); needs ffmpeg + tesseract
cp .env.example .env                     # then paste your Tiger Data service URL
uv run earwitness db migrate             # create the schema (the server also does this)
(cd web && npm install && npm run build) # the HUD
uv run earwitness serve                  # http://127.0.0.1:8765
uv run earwitness demo --out demo        # demo pack with planted ground truth
```

For **frontend-only development**, no Python server, database, login service or
API key is needed:

```sh
cd web
npm ci
npm run dev -- --port 5174
# Open http://localhost:5174/?mock=planted&extras
```

`&extras` opens a fully populated case immediately, with up to three key findings
and evidence available inline. Technical details start closed. The preview bar
switches between video, edited photo, AI-generated photo and clean-verdict
examples; it also opens the case gallery, toggles technical details, replays
progress and resets the session. Media, maps, reports and
enhancement previews are served locally. Uploads stay in browser memory and
use synthetic findings; enhancement crops the selected region with browser
colour filters. No forensic analysis or AI requests run in this mode. Refresh
to restore deleted examples and clear uploaded files. Add `&replay` to animate
the initial load. Mock mode is only available in the Vite development server.

Optional extras, configured in `.env` (gitignored):

```sh
OPENAI_API_KEY=sk-...          # AI checks: transcript, sounds, visual analyst, brief, geolocation
GEMINI_API_KEY=...             # the same AI checks on Gemini (set one key, or both)
EARWITNESS_OLLAMA_URL=http://10.8.96.3:11434  # self-hosted Gemma 4 via Ollama
EARWITNESS_OLLAMA_MODEL=gemma4 # optional: the Ollama model tag
EARWITNESS_AI_PROVIDER=gemini  # optional: provider for cases that don't pick one (openai|gemini|ollama)
EARWITNESS_GEOSLEUTH=~/.cache/earwitness/vendor/geo-sleuth/skills/geo-sleuth
```

For geolocation, clone https://github.com/Oldcircle/geo-sleuth (MIT) and point
`EARWITNESS_GEOSLEUTH` at its `skills/geo-sleuth` folder. Earwitness runs its
local intake (edge crops + OCR, no reverse image search), then a vision model
estimates the location **blind**, without the file's GPS. When GPS exists, the
error in km is reported. The key is read from the environment only and never
reaches the browser. Frames and audio are sent to the AI provider only when its
key is set.

With two or more providers set up, each case picks its AI provider: "AI model" under **More
options** when uploading, or the picker next to **Rerun** on a case (the API
takes `ai_provider=openai|gemini|ollama` on upload and rerun). Answers are cached per
case and provider (`openai-cache/`, `gemini-cache/`, `ollama-cache/`), so switching back is free.
The Ollama provider ("Gemma" in the UI) runs every AI check on one self-hosted model,
so the transcript has a single reading instead of two cross-checked ones. Pull the
model on the Ollama host first (`ollama pull gemma4`).

## Database (Tiger Data)

Earwitness requires a [Tiger Data](https://www.tigerdata.com) (TimescaleDB)
service. Set `TIMESCALE_SERVICE_URL` in `.env`; the server refuses to start
without it. Tables live in the schema `EARWITNESS_DB_SCHEMA` (default
`earwitness`):

| Table | What | Kind |
|---|---|---|
| `grid_frequency` | Every GB grid-frequency sample fetched from NESO (1 s) / Elexon (15 s) | hypertable, segmented by source |
| `grid_coverage` | Windows already fetched; a covered window needs no HTTP | table |
| `grid_frequency_1m` | One-minute avg/min/max of the grid record | continuous aggregate |
| `cases` | Every analysed clip: hash, verdict, the time the grid dated it to | table |
| `findings` | Every finding of every case | hypertable, segmented by case |

`cases/<id>/events.jsonl` stays the record a case replays from; the database
is the shared grid record and the index across cases.

```sh
uv run earwitness db status         # hypertables, sizes, stored grid windows, counts
uv run earwitness db import-cache   # load old ~/.cache/earwitness/grid/*.npz windows
```

API: `GET /api/search?since=&until=&severity=&module=&q=` (findings across
cases), `GET /api/search/dated?start=&end=` (clips the grid dated into a
window), `GET /api/grid?start=&end=` (stored grid record, per minute).

## Login (Auth0)

Anyone can view cases; uploading, re-running and deleting need an Auth0 login.
In Auth0 create a **Single Page Application** (Allowed Callback, Logout and
Web Origins: `http://127.0.0.1:8765`, refresh-token rotation on) and an
**API** (identifier e.g. `https://earwitness.api`, RS256, Allow Offline Access
on). If the API's Application Access Policy uses "Per-app authorization" for
user-delegated access, authorize the SPA in its Application Access tab, or
login fails with "Client ... is not authorized to access resource server".
`AUTH0_AUDIENCE` must match the identifier exactly. Then set
`AUTH0_DOMAIN`, `AUTH0_CLIENT_ID` and `AUTH0_AUDIENCE` in `.env` and restart the
server; the HUD reads them from `/api/health`, so no rebuild is needed. The
server checks each access token's signature, issuer, audience and expiry, and
records the uploader's Auth0 `sub` in the case options. Without these
settings the write endpoints answer 503.

The UI opens in a plain-language **Simple** view (verdict + "Has it been
edited? / When? / Where? / Who? / Anything hidden?" cards with confidence);
**Technical details** shows the full forensic HUD.

## Photographs

A still image gets its own set of checks (`kind: image` in the case event).
They work at full resolution and ask what video cannot: has this exact file
been through an editor, is it the camera's original, what did the generator
write into it, and what does it carry besides the picture.

| Module | What it finds | How |
|---|---|---|
| **Photo provenance** | Camera, lens, exposure; serial numbers and owner names; GPS fix with the compass heading of the camera (drawn as a wedge on the map) and the sun's position relative to it; every clock in the file reconciled (camera vs GPS/Samsung UTC clock gives the time zone; modify > capture means re-saved); XMP edit history; resized/cropped vs the camera's recorded size; screenshots; **embedded EXIF thumbnail compared with the picture**, globally and block by block | EXIF (all IFDs, own TIFF walker for IFD1), XMP, IPTC IIM, ICC `desc`, PNG text chunks, Samsung SEF trailer |
| **Compression anatomy** | JPEG quality and whether the tables are libjpeg-standard; encoder habits (progressive, optimised Huffman, 4:4:4, Adobe markers) that no camera has; **JPEG ghosts** of an earlier, lower-quality save; the 8×8 block grid, including a grid in a PNG ("was once a JPEG") and tiles whose grid is shifted (pasted from another JPEG) | quantisation table matching, recompression at 36 qualities, median-robust block-boundary profiles per 64 px tile |
| **Pixel forensics** | Error-level analysis at the file's own quality, grain consistency, **SIFT copy-move** (rotation and scale tolerant), resampling spectrum, histogram combing, luminance-gradient / PCA / amplified-grain views. Region leads become findings only when two independent traces agree on the same box | texture-normalised ELA/noise z-maps, RANSAC similarity on self-matches, Popescu–Farid predictor residual |
| **Hidden data** | Bytes after the end marker (ZIP/PDF/other files, extra JPEGs from MPF depth maps, **motion-photo videos**), comments and unknown blocks, content under transparent pixels, sequential **LSB steganography** (chi-square step test), personal details (e-mails, paths, URLs) in metadata | byte accounting of the whole file |
| **People** | Face count, which faces are large and sharp enough to recognise, eye crops enlarged for corneal reflections, names tagged in XMP face regions | YuNet, Laplacian sharpness |
| **AI generation signals** | Content Credentials (C2PA/JUMBF) with generator and source type, IPTC digital source type, generator names, the **Stable Diffusion recipe** (prompt, seed, model) in PNG text, ComfyUI workflows; weak statistical signals (spectral bumps, noise floor, generator-typical sizes) reported as weak | string-level manifest parsing, radial power spectrum |
| **OCR / AI eyewitness / Geolocation / Weather** | As for video, on the whole picture plus its four quarters at full resolution; the vision model also gives an AI-generation opinion (labelled AI) | |
| **Case brief** | Verdict EDITED / POSSIBLY EDITED / NO SIGNS OF EDITING / **AI-GENERATED**, with answers: edited? made by AI? when? where (GPS, or the time zone the clock was set to)? what took it? who is in it? anything hidden? | rules; AI narrative and dossier if enabled |

The viewer replaces the video player: zoom and pan to native pixels, lay any
forensic map over the picture with opacity and blend control, see every
flagged region as a box, switch lenses (levels, negative, channels, edges,
colour ×4), a magnifier, and **Examine a detail**: a region cut from the
original at full resolution, enlarged, through eight forensic lenses (error
levels, grain ×12, equalised, negative, edges, R/G/B). Nothing invents pixels.

Verified on planted photos (`uv run pytest tests/test_photo.py`): clone offset
recovered to the pixel, the stale thumbnail, the appended ZIP, the Stable
Diffusion recipe with its seed, the LSB message, the picture under
transparency, the motion-photo video and the "PNG was a JPEG" grid. Twelve
real, unedited phone photos (iPhone HEIC/JPEG, Samsung, WhatsApp/WeChat
copies, a screenshot) raise no edit verdict.

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
| **Transcript** / **AI analyst** | Speech; place, time of day, notable details and visual manipulation signs (labelled AI-inferred) | OpenAI or Gemini (optional) |
| **Geo-sleuth intake** | geo-sleuth intake dossier for the sharpest frames | optional bridge; set `EARWITNESS_GEOSLEUTH` |
| **Case brief** | Verdict (EDITED / POSSIBLY EDITED / NO SIGNS OF EDITING) with WHEN / WHERE / DEVICE / PEOPLE / INTEGRITY / HIDDEN lines citing finding ids; flags edits **corroborated** by sound and picture | deterministic rules; plus an AI narrative if enabled |

The HUD has **ENHANCE**: drag a box on the paused video and multi-frame
super-resolution aligns ±N neighbouring frames (ECC) and merges them robustly,
the way burst photography does, so moving parts do not ghost.

## Verified

`uv run pytest` (17 tests, ~100 s; needs the database, each run uses its own
throwaway schema seeded with the stored grid record) runs every module on synthetic clips with
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
