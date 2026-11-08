# Frontend fixtures

Run Vite and open `/?mock=planted&extras`. No backend or external service is
used. The preview bar exposes four examples, gallery navigation, all detail
sections, replay and reset. The photo examples expose zoom, pan, regions,
overlays, colour lenses, magnifier and region examination; the video example
exposes playback, timed captions, filmstrip, waveform and chart seeking.

`planted.events.jsonl` supplies recorded synthetic-video measurements.
`src/lib/mockShowcase.ts` adds explicitly synthetic examples of the other
findings and artifact kinds. `src/lib/mockBackend.ts` supplies browser-only
health, case, upload, delete, rerun, crop/filter and printable-report behavior.
Uploads are not analysed and are discarded when the page is refreshed.
`&extras` loads the completed state; `&replay` animates the initial load.

The assets in `public/mock/` are small fixtures copied or derived from the
repository's `demo/` pack, plus code-generated SVG overlays and an illustrative
landmark drawing. The drawing is not a real reference photograph. Coordinate
grid map tiles are generated in the browser, so maps require no tile server.

To regenerate the video-derived fixtures from the repository root:

```sh
cp demo/01_planted.mp4 web/public/mock/planted.mp4
cp demo/photos/11_edited.jpg web/public/mock/photo.jpg
cp demo/photos/12_generated.png web/public/mock/generated.png
ffmpeg -y -ss 6 -i demo/01_planted.mp4 -frames:v 1 web/public/mock/thumb.jpg
ffmpeg -y -i demo/01_planted.mp4 -vf 'fps=1/2,scale=160:120,tile=15x1' -frames:v 1 web/public/mock/filmstrip.jpg
ffmpeg -y -i demo/01_planted.mp4 -lavfi 'showspectrumpic=s=900x300:legend=disabled' -frames:v 1 web/public/mock/spectrogram.png
```

Mock mode is gated by Vite's `import.meta.env.DEV` and the `mock` query
parameter. Normal frontend requests continue to use the real backend.
