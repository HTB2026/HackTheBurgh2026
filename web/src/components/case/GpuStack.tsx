import { useEffect, useRef, useState } from 'react'
import { drawTexture, useGpuCanvas, type Gpu } from '../../lib/gpu'
import { useCaseMedia } from '../../lib/media'
import { FrameSource, stackFrames, type Region, type StackResult } from '../../lib/stackFrames'
import { mmss } from '../../lib/format'

const COUNTS = [1, 4, 8, 16, 32]

/**
 * Live multi-frame enhancement of a video region on the GPU: choose how many frames to combine
 * and step the moment frame by frame; recomputed in about a second.
 */
export function GpuStack({ gpu, region, time }: { gpu: Gpu; region: Region; time: number }) {
  const media = useCaseMedia()
  const fps = media.fps || 30
  const [count, setCount] = useState(16)
  const [sharpen, setSharpen] = useState(0.6)
  const [t, setT] = useState(time)
  const [busy, setBusy] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [res, setRes] = useState<StackResult | null>(null)
  const src = useRef<FrameSource | null>(null)
  const single = useGpuCanvas(gpu)
  const stacked = useGpuCanvas(gpu)

  useEffect(() => {
    if (!media.mediaUrl) return
    src.current = new FrameSource(media.mediaUrl)
    return () => src.current?.dispose()
  }, [media.mediaUrl])

  useEffect(() => {
    const s = src.current
    if (!s) return
    let live = true
    setBusy(true)
    setError(null)
    stackFrames(gpu, s, region, t, fps, count, sharpen)
      .then((r) => {
        if (!live) return (r.texture.destroy(), r.single.destroy())
        setRes((old) => {
          if (old) setTimeout(() => (old.texture.destroy(), old.single.destroy()), 500)
          return r
        })
      })
      .catch((e) => live && setError(String(e?.message ?? e)))
      .finally(() => live && setBusy(false))
    return () => {
      live = false
    }
  }, [gpu, region, t, fps, count, sharpen])

  useEffect(() => {
    if (!res) return
    if (single.ctx) drawTexture(gpu, single.ctx, res.single)
    if (stacked.ctx) drawTexture(gpu, stacked.ctx, res.texture)
  }, [gpu, res, single.ctx, stacked.ctx, single.size, stacked.size])

  const aspect = `${region.w * 16} / ${region.h * 9}`
  const motion = res ? Math.max(...res.shifts.map(([x, y]) => Math.hypot(x, y))) : 0
  return (
    <div className="gpu-stack">
      <div className="gpu-stack-controls">
        <label>
          Frames{' '}
          <select className="input ctl-select" value={count} onChange={(e) => setCount(Number(e.target.value))}>
            {COUNTS.map((c) => (
              <option key={c} value={c}>
                {c}
              </option>
            ))}
          </select>
        </label>
        <button className="ctl txt" onClick={() => setT((x) => Math.max(0, x - 1 / fps))} disabled={busy}>
          ◀ frame
        </button>
        <span className="mono small">{mmss(t)}</span>
        <button className="ctl txt" onClick={() => setT((x) => x + 1 / fps)} disabled={busy}>
          frame ▶
        </button>
        <label className="lens-slider" data-tip="Sharpen the combined image">
          <span>sharpen {sharpen.toFixed(1)}</span>
          <input type="range" min={0} max={1.5} step={0.1} value={sharpen} onChange={(e) => setSharpen(Number(e.target.value))} />
        </label>
        <span className="faint small">{busy ? 'Combining frames on your GPU…' : res ? `${res.frames} frames aligned (camera moved up to ${motion.toFixed(1)} px)${res.skipped ? `; ${res.skipped} left out because the region changed (motion)` : ''}` : ''}</span>
      </div>
      {error ? (
        <p className="chart-foot">{error}</p>
      ) : (
        <div className="gpu-stack-pair">
          <figure>
            <canvas ref={single.ref} style={{ aspectRatio: aspect }} />
            <figcaption>One frame</figcaption>
          </figure>
          <figure>
            <canvas ref={stacked.ref} style={{ aspectRatio: aspect }} />
            <figcaption className="c-red-i">{res ? res.frames : count} frames combined</figcaption>
          </figure>
        </div>
      )}
      <p className="muted small">Combining real frames removes noise; it does not invent detail. Treat what you read as something to confirm.</p>
    </div>
  )
}
