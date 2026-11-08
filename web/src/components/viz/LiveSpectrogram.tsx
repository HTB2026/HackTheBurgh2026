import { useEffect, useMemo, useRef, useState } from 'react'
import type { Artifact } from '../../lib/types'
import { useFrameRequest, useGpuCanvas, type Gpu } from '../../lib/gpu'
import { decodeAudio, useCaseMedia, type DecodedAudio } from '../../lib/media'
import { autoWindow, gpuSpectrogram, type SpecResult } from '../../lib/gpuSpectrogram'
import { STFT_DRAW } from '../../lib/shaders/stft.wgsl'
import { usePlayback, usePlayhead } from '../../lib/playback'
import { niceTicks, num } from '../../lib/format'

const fmtHz = (f: number) => (f >= 1000 ? `${+(f / 1000).toFixed(f >= 10000 ? 0 : 1)}k` : `${+f.toFixed(f < 10 ? 2 : f < 100 ? 1 : 0)}`)

interface Band {
  key: string
  label: string
  f0: number
  f1: (nyquist: number) => number
}
const BANDS: Band[] = [
  { key: 'full', label: 'Full band', f0: 0, f1: (n) => n },
  { key: 'speech', label: 'Speech 0–4 kHz', f0: 0, f1: () => 4000 },
  { key: 'low', label: 'Low 0–1 kHz', f0: 0, f1: () => 1000 },
  { key: 'hum', label: 'Mains hum 45–65 Hz', f0: 45, f1: () => 65 },
  { key: 'hum2', label: 'Hum ×2 95–125 Hz', f0: 95, f1: () => 125 },
  { key: 'hum3', label: 'Hum ×3 145–185 Hz', f0: 145, f1: () => 185 },
  { key: 'ultra', label: 'High 15 kHz – top', f0: 15000, f1: (n) => n },
]
const WINDOWS = [0, 0.02, 0.1, 0.5, 2, 6]

interface DrawRes {
  pipeline: GPURenderPipeline
  sampler: GPUSampler
}
const drawCache = new WeakMap<GPUDevice, DrawRes>()

export interface LiveSpec {
  result: SpecResult
  f0: number
  f1: number
}

/**
 * A spectrogram computed on the GPU from the actual soundtrack: pick any band and window,
 * real decibels, and the strongest frequency traced per moment (in the hum band that trace
 * is the mains frequency over time).
 */
export function LiveSpectrogram({ gpu, a, height, onResult }: { gpu: Gpu; a: Artifact; height: number; onResult?: (r: LiveSpec | null) => void }) {
  const media = useCaseMedia()
  const pb = usePlayback()
  const playhead = usePlayhead()
  const d = a.data ?? {}
  const startBand = (d.f1 ?? 0) <= 1500 ? 'low' : 'full'
  const [band, setBand] = useState(startBand)
  const [windowS, setWindowS] = useState(0)
  const [range, setRange] = useState(80)
  const [audio, setAudio] = useState<DecodedAudio | null>(null)
  const [busy, setBusy] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [res, setRes] = useState<LiveSpec | null>(null)
  const [hover, setHover] = useState<{ x: number; y: number } | null>(null)
  const boxRef = useRef<HTMLDivElement>(null)
  const { ref, ctx, size } = useGpuCanvas(gpu)
  const resultCb = useRef(onResult)
  resultCb.current = onResult

  useEffect(() => {
    let live = true
    decodeAudio(media)
      .then((x) => live && setAudio(x))
      .catch((e) => live && (setError(`Could not decode the soundtrack: ${e?.message ?? e}`), setBusy(false)))
    return () => {
      live = false
    }
  }, [media])

  const nyquist = audio ? audio.rate / 2 : 24000
  const b = BANDS.find((x) => x.key === band) ?? BANDS[0]
  const f0 = Math.min(b.f0, nyquist - 1)
  const f1 = Math.min(b.f1(nyquist), nyquist)
  const narrow = f1 - f0 <= 300

  // Compute (the heavy part) when the audio, band, window or range changes.
  useEffect(() => {
    if (!audio) return
    let live = true
    setBusy(true)
    setError(null)
    const cols = Math.max(256, Math.min(1400, Math.round((boxRef.current?.clientWidth ?? 900) * Math.min(2, devicePixelRatio || 1))))
    gpuSpectrogram(gpu, audio, { f0, f1, rows: narrow ? 320 : 384, cols, windowS, range })
      .then((r) => {
        if (!live) return r.texture.destroy()
        setRes((old) => {
          // Keep the old texture until the 3D view has switched to the new one.
          if (old) setTimeout(() => old.result.texture.destroy(), 1000)
          const next = { result: r, f0, f1 }
          resultCb.current?.(next)
          return next
        })
        setBusy(false)
      })
      .catch((e) => live && (setError(String(e?.message ?? e)), setBusy(false)))
    return () => {
      live = false
    }
  }, [gpu, audio, f0, f1, windowS, range, narrow])

  const bind = useMemo(() => {
    if (!res) return null
    let dr = drawCache.get(gpu.device)
    if (!dr) {
      const module = gpu.device.createShaderModule({ code: STFT_DRAW, label: 'stft draw' })
      dr = {
        pipeline: gpu.device.createRenderPipeline({
          layout: 'auto',
          vertex: { module, entryPoint: 'vs' },
          fragment: { module, entryPoint: 'fs', targets: [{ format: gpu.format }] },
        }),
        sampler: gpu.device.createSampler({ magFilter: 'linear', minFilter: 'linear' }),
      }
      drawCache.set(gpu.device, dr)
    }
    return {
      dr,
      group: gpu.device.createBindGroup({
        layout: dr.pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: res.result.texture.createView() },
          { binding: 1, resource: dr.sampler },
        ],
      }),
    }
  }, [gpu, res])

  const draw = () => {
    if (!ctx || !bind || !size.width) return
    const enc = gpu.device.createCommandEncoder()
    const pass = enc.beginRenderPass({ colorAttachments: [{ view: ctx.getCurrentTexture().createView(), loadOp: 'clear', storeOp: 'store', clearValue: { r: 0, g: 0, b: 0, a: 1 } }] })
    pass.setPipeline(bind.dr.pipeline)
    pass.setBindGroup(0, bind.group)
    pass.draw(3)
    pass.end()
    gpu.device.queue.submit([enc.finish()])
  }
  const requestDraw = useFrameRequest(draw)
  useEffect(requestDraw, [ctx, bind, size]) // eslint-disable-line react-hooks/exhaustive-deps

  // Axes and overlays (SVG), in the canvas's CSS box.
  const M = { l: 48, r: 10, t: 8, b: 24 }
  const W = Math.max(0, (boxRef.current?.clientWidth ?? 0) - M.l - M.r)
  const H = height - M.t - M.b
  const t0 = res?.result.t0 ?? 0
  const t1 = res?.result.t1 ?? (media.duration || 1)
  const sx = (t: number) => M.l + ((t - t0) / Math.max(1e-6, t1 - t0)) * W
  const sy = (f: number) => M.t + H - ((f - f0) / Math.max(1e-6, f1 - f0)) * H
  const at = hover && W > 0 && { t: t0 + ((hover.x - M.l) / W) * (t1 - t0), f: f0 + (1 - (hover.y - M.t) / H) * (f1 - f0) }
  const peakPath = useMemo(() => {
    if (!res || !narrow || W <= 0) return ''
    const p = res.result.peaks
    return Array.from(p, (f, i) => `${i ? 'L' : 'M'}${(M.l + (i / (p.length - 1)) * W).toFixed(1)},${sy(f).toFixed(1)}`).join('')
  }, [res, narrow, W, f0, f1]) // eslint-disable-line react-hooks/exhaustive-deps
  const peakAt = hover && res && narrow ? res.result.peaks[Math.max(0, Math.min(res.result.peaks.length - 1, Math.round(((hover.x - M.l) / Math.max(1, W)) * (res.result.peaks.length - 1))))] : null

  return (
    <div className="live-spec">
      <div className="live-spec-controls">
        <select className="input ctl-select" value={band} onChange={(e) => setBand(e.target.value)} aria-label="Frequency band">
          {BANDS.filter((x) => x.f0 < nyquist).map((x) => (
            <option key={x.key} value={x.key}>
              {x.label}
            </option>
          ))}
        </select>
        <select className="input ctl-select" value={windowS} onChange={(e) => setWindowS(Number(e.target.value))} aria-label="Window" data-tip="Longer windows separate close frequencies (hum); shorter ones show quick events">
          {WINDOWS.map((w) => (
            <option key={w} value={w}>
              {w === 0 ? `Window: auto (${autoWindow(f0, f1, narrow ? 320 : 384).toFixed(2)} s)` : `Window: ${w} s`}
            </option>
          ))}
        </select>
        <label className="lens-slider" data-tip="How many dB below the loudest sound are shown">
          <span>{range} dB</span>
          <input type="range" min={30} max={120} step={5} value={range} onChange={(e) => setRange(Number(e.target.value))} aria-label="Dynamic range" />
        </label>
        <span className="faint small">
          {busy ? 'Computing on your GPU…' : res && audio ? `${audio.source === 'original' ? 'original file' : 'viewing copy'} · ${Math.round(audio.rate / 100) / 10} kHz · window ${res.result.windowS.toFixed(2)} s · loudest ${Math.round(res.result.topDb)} dBFS` : ''}
        </span>
      </div>
      {error ? (
        <div className="chart-foot">{error}</div>
      ) : (
        <div className="live-spec-plot" ref={boxRef} style={{ height }}>
          <canvas ref={ref} className="live-spec-canvas" style={{ left: M.l, top: M.t, width: `calc(100% - ${M.l + M.r}px)`, height: H }} />
          <svg width="100%" height={height}>
            <rect x={M.l} y={M.t} width={Math.max(0, W)} height={Math.max(0, H)} fill="none" stroke="#3a3a3a" />
            <g className="axis">
              {niceTicks(f0, f1, 4).map((f) => (
                <text key={f} x={M.l - 8} y={sy(f) + 4} textAnchor="end">
                  {fmtHz(f)}
                </text>
              ))}
              {niceTicks(t0, t1, Math.max(2, Math.floor(W / 80))).map((t) => (
                <text key={t} x={sx(t)} y={height - 6} textAnchor="middle">
                  {+t.toFixed(2)}s
                </text>
              ))}
              <text x={4} y={M.t + 2} className="unit">
                Hz
              </text>
            </g>
            {peakPath && <path d={peakPath} className="peak-trace" />}
            {playhead >= t0 && playhead <= t1 && <line className="ph" x1={sx(playhead)} x2={sx(playhead)} y1={M.t} y2={M.t + H} />}
            {hover && (
              <>
                <line className="xhair" x1={hover.x} x2={hover.x} y1={M.t} y2={M.t + H} />
                <line className="xhair" x1={M.l} x2={M.l + W} y1={hover.y} y2={hover.y} />
              </>
            )}
            <rect
              className="seekable"
              x={M.l}
              y={M.t}
              width={Math.max(0, W)}
              height={Math.max(0, H)}
              fill="transparent"
              onPointerMove={(e) => {
                const r = (e.currentTarget.ownerSVGElement as SVGSVGElement).getBoundingClientRect()
                setHover({ x: e.clientX - r.left, y: e.clientY - r.top })
              }}
              onPointerLeave={() => setHover(null)}
              onClick={() => at && pb.seek(at.t)}
            />
          </svg>
          {hover && at && (
            <div className="tooltip" style={{ left: Math.min(hover.x + 12, (boxRef.current?.clientWidth ?? 400) - 190), top: hover.y }}>
              <div>
                <b>{at.t.toFixed(2)} s</b> into the recording
              </div>
              <div>
                pitch <b>{num(at.f)} Hz</b>
              </div>
              {peakAt != null && (
                <div>
                  strongest here <b>{peakAt.toFixed(3)} Hz</b>
                </div>
              )}
              <div className="faint">Click to jump the video here</div>
            </div>
          )}
          {narrow && res && (
            <div className="live-spec-note faint">
              Cyan line: the strongest frequency at each moment. Mains hum is often strongest at ×2 (100/120 Hz) rather than 50/60 Hz.
            </div>
          )}
        </div>
      )}
    </div>
  )
}
