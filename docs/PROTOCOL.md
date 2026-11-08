# EARWITNESS server ↔ UI protocol

Backend: FastAPI (`uv run earwitness serve --port 8765`). In production the
built SPA (`web/dist`) is served by the same server at `/`. In dev, Vite proxies
`/api` and `/ws` to `http://127.0.0.1:8765`.

## REST

| Method | Path | Body / query | Returns |
|---|---|---|---|
| GET | `/api/health` | | `{ok, openai: bool, modules: [{name,title,description}]}` |
| GET | `/api/cases` | | `[CaseSummary]`, newest first (includes `verdict`, `summary_plain`, and `place` / `place_confidence` (the plain "where" answer, null when unknown) once the brief exists) |
| POST | `/api/cases` | multipart `file`; optional query `approx_time` (ISO), `search_hours` | `CaseSummary` (analysis starts immediately) |
| POST | `/api/cases/from-path` | JSON `{path, options?}` (a file on the server machine) | `CaseSummary` |
| GET | `/api/cases/{id}` | | `{summary: CaseSummary, events: [Event]}` (full replay) |
| DELETE | `/api/cases/{id}` | | `{deleted}` |
| POST | `/api/cases/{id}/rerun` | | `CaseSummary` |
| GET | `/api/cases/{id}/files/{path}` | | artifact files (PNG/JPG/MP4); URLs arrive ready-made in events |
| POST | `/api/cases/{id}/enhance` | JSON `{time_s, x, y, w, h, frames?}`; x,y,w,h are 0..1 fractions of the video frame | `{single_url, enhanced_url, frames_used, scale, source_px:[w,h], time_s}` |
| GET | `/api/cases/{id}/report` | | printable HTML report (open in new tab) |

`CaseSummary = {id, filename, created (unix s), status: queued|running|done|interrupted, findings, warnings, case: CaseInfo|null}`

## WebSocket `/ws/cases/{id}`

On connect the server replays every event so far, then streams new ones live.
Every event has `seq` (0-based, increasing) and `ts` (unix seconds) plus a `type`:

### Still photographs

A case whose evidence is a picture carries `kind: "image"` (videos: `"video"`),
`image: {width, height, format, mode, bits, frames, orientation, megapixels}`,
`video: null`, `duration_s: 0`, `media_url` = an upright browser-viewable JPEG
(`proxy.jpg`) and `original_url` for the untouched file. The image module set
is separate (`/api/health` returns it as `image_modules`). Findings have no
`time_s`; a finding that points at part of the picture carries
`data.box = {x, y, w, h}` as fractions of the width and height, which the UI
draws and zooms to. Heatmap artifacts with `data.overlay: true` can be laid over
the picture. `kind: "file"` artifacts carry a downloadable `url` plus
`data {label, bytes, strings, head_hex}`. The brief adds the verdict
`AI-GENERATED` and the answers `synthetic`, `device`, `hidden`.
`POST /api/cases/{id}/enhance` on an image ignores `time_s` and returns
`lenses: [{name, url, caption}]` as well as `single_url`/`enhanced_url`.

### `case` (first event)
```json
{"type":"case","case":{"id":"…","filename":"IMG_6335.MOV","duration_s":23.1,"size_bytes":27607625,
 "has_audio":true,"video":{"width":1080,"height":1920,"fps":30.0,"codec":"hevc","frames":693}|null,
 "thumb_url":"/api/cases/…/files/thumb.jpg","media_url":null,
 "modules":[{"name":"provenance","title":"PROVENANCE","description":"…"}, …]}}
```

Since the custody update, `case` also carries `sha256`, `md5`, `ingested_at` (unix s),
`container` {format_name, format_long_name, bit_rate, size_bytes, duration_s},
`streams` [{index, type, codec, codec_long, profile, width, height, fps, pix_fmt,
color_range, color_space, bit_rate, sample_rate, channels, channel_layout, language,
handler, duration_s, frames}] and `tags` (every container tag, plus stream tags as
`stream<N>.<key>`).

### `media`
Browser-playable H.264 proxy of the evidence is ready (phones record HEVC/VP9):
`{"type":"media","media_url":"/api/cases/…/files/proxy.mp4"}` (or `null` if transcoding failed; fall back to nothing).

### `module`, status updates per analysis module
`{"type":"module","module":"timeline","status":"queued|waiting|running|done|skipped|error","progress":0..1,"message":"decoding frames","elapsed_s"?:1.2,"trace"?:"…"}`

### `finding`, the main payload
```json
{"type":"finding","finding":{
  "id":"F007","module":"spectrum","severity":"info|notice|warning|critical",
  "title":"Keypad tones decoded: 1984#","detail":"…one or two sentences…",
  "time_s":4.0|null,"time_end_s":null,
  "kind":"measured|inferred",          // inferred = AI (OpenAI) output, label it so
  "tags":["dtmf","audio"],              // e.g. splice, cut, location, gps, time, tone, ultrasonic, enf, face, pulse, text …
  "data":{…module-specific…}}}
```
Clicking a finding with `time_s` should seek the video there. Findings tagged
`location`/`gps` usually carry `data.lat/lon`.

### `artifact`, things to visualise
`{"type":"artifact","artifact":{"id":"module:name","module":"…","name":"…","kind":"…","title":"…","url":string|null,"data":any,"time_s":number|null}}`

An artifact with an id already seen replaces the old one. Kinds:

| kind | payload | render as |
|---|---|---|
| `spectrogram` | `url` PNG (low freq at bottom); `data {t0,t1,f0,f1,scale:"linear"}` | image with time/freq axes; draw a playhead synced to the video |
| `series` | `data {x:[…], series:[{name, y:[…], x?:[…]}], x_unit?, y_unit?, marks?:{label: x}}` or `{x_iso:[…], series}` (ISO time x-axis). A series with its own `x` overrides the shared x. `marks` = labelled vertical reference lines | line chart; if `x_unit=="s"` draw playhead |
| `waveform` | `data {t1, peak:[0..1 …]}` | mirrored waveform strip on the timeline |
| `filmstrip` | `url` JPG sprite: `count` tiles side-by-side; `data {tile_width,tile_height,times:[…],count}` | thumbnail track on the timeline |
| `markers` | `data {times:[…], …}` | tick marks on the timeline |
| `table` | `data {rows:[[…]], columns?:[…]}` | monospace table |
| `bars` | `data {labels:[…], values:[…]}` | bar chart |
| `map` | `data {points:[{lat,lon,label,accuracy_m?,kind}]}` | dark map (Leaflet + CARTO dark tiles), accuracy circle |
| `image` / `heatmap` | `url`, optional `data {caption, time_s, overlay?:bool}` | gallery; heatmaps can overlay the video at `time_s` |
| `video` | `url` MP4 (e.g. pulse colour magnification) | player |
| `text` | `data {text}` | prose panel |
| `json` | `data` = any JSON (e.g. `provenance:probe`, the raw ffprobe report) | collapsible tree |
| `brief` | `data {verdict: "EDITED"\|"POSSIBLY EDITED"\|"NO SIGNS OF EDITING", bullets:[str], text, corroborated:bool, headline?, narrative?, next_steps?:[str], model?}`. Bullets start with WHEN:/WHERE:/DEVICE:/PEOPLE:/INTEGRITY:/HIDDEN:/CORROBORATED: and cite finding ids like `[F012]` (make those clickable → scroll to/highlight the finding). `headline`/`narrative` exist only when OpenAI is on (AI-inferred) | the ANALYST/verdict panel (id `brief:brief`) |
| `captions` | `data {segments:[{start,end,text}]}` | subtitles over the video at currentTime + transcript list |

### `log`
`{"type":"log","module":"…","message":"…"}`, for a scrolling terminal feed.

### `done`
All modules finished.
