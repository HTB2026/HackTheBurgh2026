import { useEffect, useMemo, useRef, useState } from 'react'
import { drawTexture, useGpu, useGpuCanvas } from '../../lib/gpu'
import { useCaseMedia } from '../../lib/media'
import { scanFrames, type ScanResult } from '../../lib/frameScan'
import { usePlayback, usePlayhead } from '../../lib/playback'
import { mmss } from '../../lib/format'
import { Icon } from '../ui/ui'

const RATES = [
  { rate: 1, label: 'Every frame (1×)' },
  { rate: 2, label: 'Faster (2×, may skip frames)' },
  { rate: 4, label: 'Quick look (4×, skips frames)' },
]

/** Inferno-like colour for 0..1 (for the flicker map). */
function heat(t: number): [number, number, number] {
  const x = Math.min(1, Math.max(0, t))
  return [Math.round(255 * Math.min(1, 1.6 * x)), Math.round(255 * Math.max(0, Math.min(1, 1.9 * x - 0.6))), Math.round(255 * Math.max(0, Math.min(1, 3 * x - 2.1)))]
}

/**
 * Reads every frame of the video on the GPU: how much each frame changed from the one before
 * (cuts, jumps, frozen or repeated frames) and where lamp flicker comes from in the picture.
 */
export function FrameScan({ thumbUrl }: { thumbUrl: string | null }) {
  const gpuState = useGpu()
  const gpu = gpuState === 'loading' ? null : gpuState
  const media = useCaseMedia()
  const pb = usePlayback()
  const playhead = usePlayhead()
  const [rate, setRate] = useState(1)
  const [progress, setProgress] = useState<number | null>(null)
  const [result, setResult] = useState<ScanResult | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [grid, setGrid] = useState<'50' | '60'>('50')
  const changesTex = useRef<GPUTexture | null>(null)
  const abort = useRef<AbortController | null>(null)
  const { ref: liveRef, ctx: liveCtx } = useGpuCanvas(gpu)
  const flickerRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => () => abort.current?.abort(), [])

  // Show the "what changed" map live while scanning.
  useEffect(() => {
    if (progress == null || !gpu || !liveCtx) return
    let raf = 0
    const tick = () => {
      if (changesTex.current) drawTexture(gpu, liveCtx, changesTex.current)
      raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf)
  }, [progress != null, gpu, liveCtx]) // eslint-disable-line react-hooks/exhaustive-deps

  const run = async () => {
    if (!gpu || !media.mediaUrl || !media.fps) return
    abort.current?.abort()
    const ac = new AbortController()
    abort.current = ac
    setError(null)
    setResult(null)
    setProgress(0)
    try {
      const r = await scanFrames(gpu, media.mediaUrl, media.originalUrl, media.fps, media.duration, {
        rate,
        signal: ac.signal,
        onProgress: setProgress,
        onChanges: (t) => (changesTex.current = t),
      })
      setResult(r)
      setGrid(r.flicker.amp50 ? '50' : '60')
    } catch (e) {
      if ((e as Error).name !== 'AbortError') setError(String((e as Error).message ?? e))
    } finally {
      setProgress(null)
      changesTex.current?.destroy()
      changesTex.current = null
    }
  }

  // Flicker map: amplitude at the chosen grid's aliased frequency, over a frame of the video.
  const amp = result ? (grid === '50' ? result.flicker.amp50 : result.flicker.amp60) : null
  const ampStats = useMemo(() => {
    if (!amp || !result) return null
    const sorted = Float32Array.from(amp).sort()
    const noise = Float32Array.from(result.flicker.control).sort()
    return { p99: sorted[Math.floor(sorted.length * 0.99)], median: sorted[Math.floor(sorted.length / 2)], noise: noise[Math.floor(noise.length / 2)] }
  }, [amp, result])
  useEffect(() => {
    const c = flickerRef.current
    if (!c || !result || !amp || !ampStats) return
    const { w, h } = result.flicker
    c.width = w
    c.height = h
    const g = c.getContext('2d')!
    const img = g.createImageData(w, h)
    // Scale to at least 10x the noise floor, so a video without flicker gives a dark map.
    const top = Math.max(ampStats.noise * 10, ampStats.p99)
    for (let i = 0; i < w * h; i++) {
      const t = amp[i] / top
      const [r, gg, b] = heat(t)
      img.data.set([r, gg, b, Math.round(255 * Math.min(0.85, t * 1.2))], i * 4)
    }
    g.putImageData(img, 0, 0)
  }, [result, amp, ampStats])

  if (!gpu || !media.fps) return null
  const dur = result ? Math.max(0.001, result.times[result.times.length - 1] ?? media.duration) : media.duration
  const W = 1000
  const H = 90
  const maxC = result ? Math.max(4, ...result.change.slice(1)) : 1
  const path = result ? result.change.map((c, k) => `${k ? 'L' : 'M'}${((result.times[k] / dur) * W).toFixed(1)},${(H - (c / maxC) * (H - 6)).toFixed(1)}`).join('') : ''
  const fa = result ? (grid === '50' ? result.flicker.fa50 : result.flicker.fa60) : 0
  // Flicker against the same measurement at a frequency where no flicker lands.
  const flickerRatio = ampStats ? ampStats.median / Math.max(0.005, ampStats.noise) : 0
  const hotRatio = ampStats ? ampStats.p99 / Math.max(0.005, ampStats.noise) : 0

  return (
    <div className="frame-scan">
      <div className="frame-scan-head">
        <b>
          <Icon name="film" /> Frame scan
        </b>
        <span className="faint small">Reads every frame on your GPU: cuts, frozen or repeated frames, and where lamp flicker comes from.</span>
        <span className="grow" />
        <select className="input ctl-select" value={rate} onChange={(e) => setRate(Number(e.target.value))} disabled={progress != null} aria-label="Scan speed">
          {RATES.map((r) => (
            <option key={r.rate} value={r.rate}>
              {r.label}
            </option>
          ))}
        </select>
        {progress == null ? (
          <button className="btn btn-blue sm" onClick={run} disabled={!media.mediaUrl}>
            <Icon name="play" /> {result ? 'Scan again' : 'Scan frames'}
          </button>
        ) : (
          <button className="btn sm" onClick={() => abort.current?.abort()}>
            Stop ({Math.round(progress * 100)}%)
          </button>
        )}
      </div>
      {error && <div className="chart-foot">{error}</div>}
      {progress != null && (
        <div className="frame-scan-live">
          <canvas ref={liveRef} className="frame-scan-changes" />
          <span className="faint small">What changed since the previous frame (bright = changed)</span>
        </div>
      )}
      {result && (
        <>
          <svg className="frame-scan-curve" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" onClick={(e) => {
            const r = e.currentTarget.getBoundingClientRect()
            pb.seek(((e.clientX - r.left) / r.width) * dur)
          }}>
            <path d={path} className="curve" vectorEffect="non-scaling-stroke" />
            {result.jumps.map((k) => (
              <line key={`j${k}`} x1={(result.times[k] / dur) * W} x2={(result.times[k] / dur) * W} y1={0} y2={H} className="jump" vectorEffect="non-scaling-stroke" />
            ))}
            {result.duplicates.map((k) => (
              <line key={`d${k}`} x1={(result.times[k] / dur) * W} x2={(result.times[k] / dur) * W} y1={H - 14} y2={H} className="dup" vectorEffect="non-scaling-stroke" />
            ))}
            <line x1={(playhead / dur) * W} x2={(playhead / dur) * W} y1={0} y2={H} className="ph" vectorEffect="non-scaling-stroke" />
          </svg>
          <p className="frame-scan-summary small">
            {result.times.length} frames read{result.expected ? ` of about ${result.expected}` : ''} from the {result.source === 'original' ? 'original file' : 'viewing copy (re-encoded: exact repeats may not survive)'}
            {rate > 1 && result.times.length < result.expected * 0.97 ? ' (some skipped at this speed: scan at 1× for every frame)' : ''}.{' '}
            {result.jumps.length ? (
              <>
                <b className="c-red-t">{result.jumps.length} abrupt change{result.jumps.length === 1 ? '' : 's'}</b> (red: cuts or sudden jumps) at{' '}
                {result.jumps.slice(0, 6).map((k, i) => (
                  <button key={k} className="linkish" onClick={() => pb.seek(result.times[k])}>
                    {i ? ', ' : ''}
                    {mmss(result.times[k])}
                  </button>
                ))}
                .{' '}
              </>
            ) : (
              'No abrupt changes. '
            )}
            {result.duplicates.length ? (
              <>
                <b className="c-orange-t">{result.duplicates.length} repeated frame{result.duplicates.length === 1 ? '' : 's'}</b> (orange: identical to the frame before; frozen video or frames padded in).
              </>
            ) : (
              'No repeated frames.'
            )}{' '}
            Click the curve to jump there.
          </p>
          <div className="frame-scan-flicker">
            <div className="frame-scan-flicker-img">
              {thumbUrl && <img src={thumbUrl} alt="" />}
              <canvas ref={flickerRef} />
            </div>
            <div className="small">
              <div className="tabs" role="radiogroup" aria-label="Grid">
                {(['50', '60'] as const).map((g) => (
                  <button key={g} className={grid === g ? 'active' : ''} onClick={() => setGrid(g)} disabled={!(g === '50' ? result.flicker.amp50 : result.flicker.amp60)}>
                    {g} Hz grid
                  </button>
                ))}
              </div>
              <p>
                <b>Lamp flicker map.</b> Lamps on a {grid} Hz grid flicker at {grid === '50' ? 100 : 120} Hz; at {media.fps?.toFixed(media.fps % 1 ? 2 : 0)} fps that shows up as a {fa.toFixed(1)} Hz pulse
                from frame to frame. Bright areas pulse at exactly that rate.
              </p>
              {!amp ? (
                <p className="faint">At this frame rate a {grid} Hz-grid flicker lands on 0 Hz and can't be told apart from steady light.</p>
              ) : (
                <p>
                  Typical pulse <b>{ampStats!.median.toFixed(2)}</b> brightness levels, strongest <b>{ampStats!.p99.toFixed(2)}</b>, against a noise floor of{' '}
                  {ampStats!.noise.toFixed(2)} (same measurement at {result.flicker.fc.toFixed(1)} Hz, where no flicker lands).{' '}
                  {flickerRatio > 5
                    ? `Clear ${grid} Hz-grid flicker across the picture (${flickerRatio.toFixed(0)}× the noise): the lights run on a ${grid} Hz grid.`
                    : hotRatio > 6
                      ? `Flicker in part of the picture (${hotRatio.toFixed(0)}× the noise at its strongest): bright areas on the map show where.`
                      : `No ${grid} Hz-grid flicker above the noise.`}
                </p>
              )}
            </div>
          </div>
        </>
      )}
    </div>
  )
}
