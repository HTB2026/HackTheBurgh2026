# Test media

`clips/`: real-world phone videos used to evaluate Earwitness. **None are edited**:
any edit flag on them is a false alarm. `clips/truth.json` holds known
locations for scoring geolocation.

Score the suite on them:

```sh
uv run python scripts/evaluate.py /tmp/eval --truth media/clips/truth.json media/clips/*.mp4 media/clips/*.MOV
```

`../demo/` holds synthetic clips with planted anomalies (see `demo/MANIFEST.json`);
regenerate with `uv run earwitness demo --out demo`.

These files contain personal footage and location metadata. Keep this
repository private.
