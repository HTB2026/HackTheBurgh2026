# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Earwitness is a video and photo forensics suite (HackTheBurgh 2026, G-Research "discover the invisible in data"): a FastAPI backend streams findings over a WebSocket to a React/Vite HUD. The demo is judged by non-experts, so every finding carries a plain-English `title`/`detail` that states its limits; nothing is hidden to simplify.

## Commands

```sh
uv sync                                   # Python 3.14; needs ffmpeg (>= 5.1) and tesseract on PATH
cp .env.example .env                      # TIMESCALE_SERVICE_URL is required; the server refuses to start without it
                                          # AUTH0_DOMAIN / AUTH0_CLIENT_ID / AUTH0_AUDIENCE enable writes (see Auth below)
uv run earwitness db migrate              # apply src/earwitness/schema.sql (idempotent; the server also does this)
uv run earwitness serve                   # http://127.0.0.1:8765, serves web/dist if built
uv run earwitness demo --out demo         # regenerate the demo pack (videos + demo/photos) and MANIFEST.json
(cd web && npm install && npm run dev)    # Vite dev server, proxies /api and /ws to :8765 (EARWITNESS_API overrides)
(cd web && npm run build)                 # tsc -b && vite build -> web/dist (committed build is what the server serves)

uv run ruff check . && uv run ruff format src tests scripts
uv run pytest -q                          # ~2 min; needs the database (each run gets a throwaway schema seeded from the main grid record)
uv run pytest tests/test_photo.py -q      # still-image suite only (~40 s)
uv run pytest tests/test_auth.py -q       # Auth0 gate, tokens signed by a local RSA key (no network)
uv run pytest tests/test_modules.py -q -k "light"          # one test by keyword
uv run python scripts/evaluate.py OUT ~/Downloads/earwitness-tests/*   # real-clip regression: target 0 false edit alarms
bash scripts/deploy.sh                    # rsync + remote_setup.sh to the server (deploy key ~/.ssh/earwitness_deploy), restarts the systemd service
```

Tests set `EARWITNESS_OFFLINE=1` and pop `OPENAI_API_KEY`, `GEMINI_API_KEY` and `GOOGLE_API_KEY`; note that `earwitness.db` calls `load_dotenv(.env)` at import, so pop the keys *after* importing `earwitness.server.jobs` in ad-hoc scripts, or AI modules will run.

AI calls go through `earwitness.ai`: modules call `ai.for_case(ctx.options)` (the case's `ai_provider`, else the server default) and use its `structured` / `transcribe` / `listen`. `ai/openai.py` and `ai/gemini.py` share that interface; modules build content in the neutral `input_text` / `input_image` (data URL) format, and the Gemini provider converts it. Never import a provider directly from a module.

## Architecture

**One case = one evidence file = one ordered run of modules.** `server/jobs.py:run_case` builds a `CaseContext` (`modules/base.py`), picks the module list from `modules/registry.py` by `ctx.kind` (`video` | `image`; audio-only uses the video set), emits a `case` event, then runs modules concurrently under a semaphore (`EARWITNESS_PARALLEL`, default 4). Every module is a class with `name`, `title`, `description`, optional `depends`, and `async run(ctx, out)`; it talks only through the `Reporter` (`out.progress/finding/artifact/log/path`). Modules must never raise past `Skip`; one failing module is recorded as `status: error` and the case continues.

**Cross-module data flows through `ctx.publish(name, value)` / `await ctx.wait_for(name)`**, keyed by module name. `run_case` publishes `None` for every module (of either kind) that is not scheduled, so `wait_for` on an absent module returns immediately instead of hanging 600 s. Dependencies listed in `depends` are awaited before the module starts; `depends = ["*"]` (brief) waits for everything.

**Events are the record.** `Case.emit` appends to `cases/<id>/events.jsonl` and fans out to WebSocket listeners; the UI (`web/src/lib/state.ts:reduce`) is a pure reducer over those events and replays them on reconnect. The DB (`db.py`, Tiger Data/TimescaleDB) is only an index (`cases`, `findings`) plus the shared GB grid-frequency record; a DB failure never sinks a case. Protocol and artifact kinds are documented in `docs/PROTOCOL.md` and mirrored in `web/src/lib/types.ts`; keep them in step when adding an event field or artifact kind.

**Decoded media is cached on the context:** `ctx.frames(width)`, `ctx.frame_at(t)`, `ctx.audio()` for video; `ctx.image()` (oriented full-res RGB), `ctx.image_scaled(w)` for stills. `frame_at` returns the still itself for images, which is how `ocr`, `vision` and `geo` serve both kinds with small `still = ctx.kind == "image"` branches. Shared photo parsing (EXIF with own TIFF walker, XMP, JPEG segments/quant tables, PNG chunks, quality estimate) lives in `photo.py`, not in modules.

**Module names double as UI keys.** The web layer maps module names to plain-language text, icons and tabs in `web/src/lib/plain.ts` (`MODULES`, `TABS`, `EXPLAIN` by artifact id prefix, `DATA_LABEL` for finding data keys). The image set reuses the names `provenance` and `tamper` (classes `PhotoProvenance`, `PhotoPixels`) so the existing UI mapping applies; new modules need entries there or they render as generic "additional check".

**Verdict and plain answers come from `modules/brief.py` alone**, by deterministic rules over finding tags/titles (`_local_brief` / `_plain_answers` for video, `_local_brief_photo` / `_photo_answers` for stills), with an optional AI (OpenAI or Gemini) narrative and "dossier" labelled AI-inferred. Verdicts: `EDITED`, `POSSIBLY EDITED`, `NO SIGNS OF EDITING`, and for stills `AI-GENERATED` (`verdictOf` in plain.ts maps them). Tags are the contract: `splice`/`cut`/`clone` findings with warning+ severity drive EDITED; `editor`/`history`/`modified`/`resized` drive POSSIBLY; `declared`+`synthetic` drive AI-GENERATED. Findings matched by title prefix (e.g. "Taken on ", "ENF dates", "Embedded thumbnail") are also part of that contract: renaming a title can silently change the verdict or an answer.

**False-alarm discipline for pixel heuristics (stills).** ELA, grain, shifted-grid and per-block ghost regions are leads (`notice`, tags `ela`/`noise`/`grid`/`ghost`) and never move the verdict on their own; `photo_pixels.py` upgrades a lead to a `splice` warning only when a second independent trace overlaps the same `data.box`. Thumbnail mismatch, SIFT copy-move, appended data and declared generators are the trusted edit signals. Before changing any image detector, rerun it over the real unedited phone photos in `~/Downloads` (iPhone HEIC/JPEG, Samsung, WhatsApp/WeChat copies, screenshots) and the planted set from `demo_photo.py`; the standard is zero false "edited" verdicts on real originals.

**Auth (Auth0).** Viewing is open; every endpoint that changes evidence (`POST /api/cases`, `/from-path`, `/rerun`, `DELETE`) depends on `server/auth.py:require_user`, which verifies an RS256 access token for `AUTH0_AUDIENCE` issued by `AUTH0_DOMAIN` (JWKS fetched and cached). Without all three env vars those endpoints answer 503, never "open". `/api/health` publishes `auth` (domain, client id, audience: public values) and the SPA wraps itself in `Auth0Provider` only when it is present (`web/src/lib/auth.tsx`); `authHeaders()` in `lib/api.ts` attaches the bearer token and starts a login redirect when logged out, so any new write call must go through it. Enhance and the WebSocket stay unauthenticated. Module tests drive `CaseStore` directly and bypass HTTP, so they need no token; HTTP-level tests patch `auth._signing_key` as `tests/test_auth.py` does. The deployed server's `.env` holds the Auth0 values; `scripts/remote_setup.sh` does not write them.

**Stills in the UI.** `CaseInfo.kind === 'image'` swaps `Player` for `PhotoStage` (zoom/pan viewer, overlay maps = heatmap artifacts with `data.overlay: true`, boxes from findings' `data.box` as 0..1 fractions, lenses, magnifier, region enhance). `lib/region.ts` is the still-image counterpart of `lib/playback.ts`: a `useSyncExternalStore` store whose snapshot must stay referentially stable, plus `MediaKindContext` so shared widgets pick "photo"/"video" wording. `/api/cases/{id}/enhance` dispatches on kind: multi-frame super-resolution for video, full-res crop through forensic lenses for images.

## Conventions

- Finding `data` must be JSON-able (the Reporter rounds floats and converts numpy); keep heavy arrays out of it and write images/CSVs via `out.path(name)` + `out.artifact(..., file=path)`.
- Regions on stills are `data.box = {x, y, w, h}` fractions of the oriented picture (`photo.box_fraction`), never pixels of an analysis-size copy.
- Blocking numpy/OpenCV work goes through `asyncio.to_thread` so the event stream stays live.
- Python style: ruff (`E,F,W,I`, isort single-line), long human-readable strings may use `# fmt: skip`. Pattern-match `try/except A, B:` (3.14 syntax) is used throughout.
- Synthetic test media come from `demo.py` (video) and `demo_photo.py` (photos, planted truth in `PLANTED`); `synth.py` builds the hum/grid scenario. Tests assert on finding titles, so title changes need test updates.

## Memory notes for this repo

Project memory lives in `~/.claude/projects/-Users-relesto-Downloads-earwitness/memory/` (hackathon context, the real clip corpus in `~/Downloads/earwitness-tests`, the photo-pipeline reliability notes). Read it before changing detectors or UI wording.
