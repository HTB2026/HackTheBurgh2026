import { useState } from 'react'
import type { Artifact } from '../../lib/types'
import { useElementSize } from '../../lib/hooks'
import { usePlayback, usePlayhead } from '../../lib/playback'
import { niceTicks, num } from '../../lib/format'
import { useGpu } from '../../lib/gpu'
import { Waterfall } from './Waterfall'

const fmtHz = (f: number) => (f >= 1000 ? `${+(f / 1000).toFixed(1)}k` : `${Math.round(f)}`)

export function Spectrogram({ a, height = 240 }: { a: Artifact; height?: number }) {
  const gpuState = useGpu()
  const gpu = gpuState === 'loading' ? null : gpuState
  const [mode, setMode] = useState<'2d' | '3d'>('2d')
  const switcher = gpu && a.url && (
    <div className="tabs spec-mode" role="radiogroup" aria-label="Spectrogram view">
      {(['2d', '3d'] as const).map((m) => (
        <button key={m} type="button" role="radio" aria-checked={mode === m} className={mode === m ? 'active' : ''} onClick={() => setMode(m)} data-tip={m === '3d' ? 'Loudness as height: drag to turn, scroll to zoom (WebGPU)' : 'Flat view'}>
          {m.toUpperCase()}
        </button>
      ))}
    </div>
  )
  if (gpu && mode === '3d') {
    return (
      <div className="chart spec-3d">
        {switcher}
        <Waterfall gpu={gpu} a={a} height={Math.max(320, height + 120)} />
      </div>
    )
  }
  return <Spectrogram2D a={a} height={height} switcher={switcher} />
}

function Spectrogram2D({ a, height, switcher }: { a: Artifact; height: number; switcher: React.ReactNode }) {
  const [ref, size] = useElementSize<HTMLDivElement>()
  const [hover, setHover] = useState<{ x: number; y: number } | null>(null)
  const pb = usePlayback()
  const d = a.data ?? {}
  const t0: number = d.t0 ?? 0
  const t1: number = d.t1 ?? (pb.duration || 1)
  const f0: number = d.f0 ?? 0
  const f1: number = d.f1 ?? 1
  const M = { l: 48, r: 10, t: 8, b: 24 }
  const W = Math.max(0, size.width - M.l - M.r)
  const H = height - M.t - M.b
  const sx = (t: number) => M.l + ((t - t0) / (t1 - t0)) * W
  const sy = (f: number) => M.t + H - ((f - f0) / (f1 - f0)) * H
  const tt = niceTicks(t0, t1, Math.max(2, Math.floor(W / 80)))
  const ft = niceTicks(f0, f1, 4)
  const at = hover && { t: t0 + ((hover.x - M.l) / W) * (t1 - t0), f: f0 + (1 - (hover.y - M.t) / H) * (f1 - f0) }
  return (
    <div className="chart" ref={ref}>
      {switcher}
      {size.width > 0 && a.url && (
        <svg width={size.width} height={height} role="img" aria-label={a.title}>
          <image href={a.url} x={M.l} y={M.t} width={W} height={H} preserveAspectRatio="none" />
          <rect x={M.l} y={M.t} width={W} height={H} fill="none" stroke="#3a3a3a" />
          <g className="axis">
            {ft.map((f) => (
              <text key={f} x={M.l - 8} y={sy(f) + 4} textAnchor="end">
                {fmtHz(f)}
              </text>
            ))}
            {tt.map((t) => (
              <text key={t} x={sx(t)} y={height - 6} textAnchor="middle">
                {+t.toFixed(2)}s
              </text>
            ))}
            <text x={4} y={M.t + 2} className="unit">
              Hz
            </text>
          </g>
          <SpecPlayhead sx={sx} t0={t0} t1={t1} top={M.t} bottom={M.t + H} />
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
      )}
      {hover && at && (
        <div className="tooltip" style={{ left: Math.min(hover.x + 12, size.width - 170), top: hover.y }}>
          <div>
            <b>{at.t.toFixed(2)} s</b> into the video
          </div>
          <div>
            pitch <b>{num(at.f)} Hz</b>
          </div>
          <div className="faint">Click to jump the video here</div>
        </div>
      )}
      {Array.isArray(d.db) && (
        <div className="chart-foot">
          Brightness scale: {Math.round(d.db[0])} dB (silent, dark) to {Math.round(d.db[1])} dB (loud, bright)
        </div>
      )}
    </div>
  )
}

function SpecPlayhead({ sx, t0, t1, top, bottom }: { sx: (t: number) => number; t0: number; t1: number; top: number; bottom: number }) {
  const t = usePlayhead()
  if (t < t0 || t > t1) return null
  return <line className="ph" x1={sx(t)} x2={sx(t)} y1={top} y2={bottom} />
}
