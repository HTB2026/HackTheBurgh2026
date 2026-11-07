import { useMemo, useState } from 'react'
import type { Artifact } from '../../lib/types'
import { useElementSize } from '../../lib/hooks'
import { usePlayback, usePlayhead } from '../../lib/playback'
import { niceTicks, num, pad, tickFormatter } from '../../lib/format'

// Validated on #212121 (all pairs): blue, orange, green. A 4th+ series is drawn grey and direct-labelled.
export const SERIES = ['#3987e5', '#d95926', '#199e70']
const OVERFLOW = '#8a8a8a'

interface Line {
  name: string
  xs: number[]
  ys: (number | null)[]
  color: string
}

function nearest(xs: number[], x: number) {
  let lo = 0
  let hi = xs.length - 1
  while (lo < hi) {
    const mid = (lo + hi) >> 1
    if (xs[mid] < x) lo = mid + 1
    else hi = mid
  }
  if (lo > 0 && Math.abs(xs[lo - 1] - x) < Math.abs(xs[lo] - x)) return lo - 1
  return lo
}

function isoLabel(ms: number, span: number) {
  const d = new Date(ms)
  const hm = `${pad(d.getUTCHours())}:${pad(d.getUTCMinutes())}`
  if (span < 3 * 60_000) return `${hm}:${pad(d.getUTCSeconds())}`
  if (span < 2 * 86400_000) return hm
  return `${d.getUTCDate()}/${d.getUTCMonth() + 1} ${hm}`
}

/** Min/max per pixel column so isolated spikes (cuts!) survive downsampling. */
function pathFor(l: Line, sx: (x: number) => number, sy: (y: number) => number, W: number) {
  const n = l.xs.length
  let d = ''
  let pen = false
  if (n <= W * 2) {
    for (let i = 0; i < n; i++) {
      const y = l.ys[i]
      if (y == null || !isFinite(y)) {
        pen = false
        continue
      }
      d += `${pen ? 'L' : 'M'}${sx(l.xs[i]).toFixed(1)},${sy(y).toFixed(1)}`
      pen = true
    }
    return d
  }
  let col = -1
  let mn = 0
  let mx = 0
  let first = 0
  let last = 0
  const flush = () => {
    if (col < 0) return
    d += `${pen ? 'L' : 'M'}${col},${sy(first).toFixed(1)}L${col},${sy(mn).toFixed(1)}L${col},${sy(mx).toFixed(1)}L${col},${sy(last).toFixed(1)}`
    pen = true
  }
  for (let i = 0; i < n; i++) {
    const y = l.ys[i]
    if (y == null || !isFinite(y)) {
      flush()
      col = -1
      pen = false
      continue
    }
    const c = Math.round(sx(l.xs[i]))
    if (c !== col) {
      flush()
      col = c
      mn = mx = first = last = y
    } else {
      mn = Math.min(mn, y)
      mx = Math.max(mx, y)
      last = y
    }
  }
  flush()
  return d
}

export function SeriesChart({ a, height = 190 }: { a: Artifact; height?: number }) {
  const [ref, size] = useElementSize<HTMLDivElement>()
  const [hx, setHx] = useState<number | null>(null)
  const pb = usePlayback()
  const d = a.data ?? {}
  const iso = Array.isArray(d.x_iso)
  const timeAxis = d.x_unit === 's'

  const lines: Line[] = useMemo(() => {
    const shared: number[] = iso ? (d.x_iso as string[]).map((s) => Date.parse(s)) : (d.x ?? [])
    return ((d.series ?? []) as { name: string; y: (number | null)[]; x?: (number | string)[] }[]).map((s, i) => {
      const xs = s.x ? s.x.map((v) => (typeof v === 'string' ? Date.parse(v) : v)) : shared
      return { name: s.name, xs, ys: (s.y ?? []).slice(0, xs.length), color: SERIES[i] ?? OVERFLOW }
    })
  }, [d, iso])

  const multi = lines.length > 1
  const M = { l: 52, r: multi ? 96 : 14, t: 10, b: 24 }
  const W = Math.max(0, size.width - M.l - M.r)
  const H = height - M.t - M.b

  const ext = useMemo(() => {
    let x0 = Infinity
    let x1 = -Infinity
    let y0 = Infinity
    let y1 = -Infinity
    for (const l of lines) {
      for (const x of l.xs) {
        if (x < x0) x0 = x
        if (x > x1) x1 = x
      }
      for (const y of l.ys) {
        if (y == null || !isFinite(y)) continue
        if (y < y0) y0 = y
        if (y > y1) y1 = y
      }
    }
    if (!isFinite(x0) || !isFinite(y0)) return null
    if (x0 === x1) x1 = x0 + 1
    const padY = (y1 - y0) * 0.08 || Math.abs(y0) * 0.05 || 1
    let lo = y0 - padY
    if (y0 >= 0 && y0 < (y1 - y0) * 0.5) lo = 0
    return { x0, x1, y0: lo, y1: y1 + padY }
  }, [lines])

  if (!ext || !lines.length) return <div className="faint small">No data points were recorded.</div>

  const sx = (x: number) => M.l + ((x - ext.x0) / (ext.x1 - ext.x0)) * W
  const sy = (y: number) => M.t + H - ((y - ext.y0) / (ext.y1 - ext.y0)) * H
  const yt = niceTicks(ext.y0, ext.y1, 4)
  const fmtY = tickFormatter(yt)
  const xt = iso ? [] : niceTicks(ext.x0, ext.x1, Math.max(2, Math.floor(W / 90)))
  const isoN = Math.max(2, Math.floor(W / 120))
  const isoTicks: number[] = iso ? Array.from({ length: isoN + 1 }, (_, i) => ext.x0 + ((ext.x1 - ext.x0) * i) / isoN) : []
  const fmtX = (x: number) => (iso ? isoLabel(x, ext.x1 - ext.x0) : timeAxis ? `${+x.toFixed(2)}s` : num(x, d.x_unit))

  const hoverVal = hx != null ? ext.x0 + ((hx - M.l) / W) * (ext.x1 - ext.x0) : null
  const readout = hoverVal != null ? lines.map((l) => ({ l, i: nearest(l.xs, hoverVal) })).map(({ l, i }) => ({ l, x: l.xs[i], y: l.ys[i] })) : []

  const marks = Object.entries((d.marks ?? {}) as Record<string, number | string>)
    .map(([label, raw]) => ({ label, x: typeof raw === 'string' ? Date.parse(raw) : raw, row: 0 }))
    .filter((m) => isFinite(m.x) && m.x >= ext.x0 && m.x <= ext.x1)
    .sort((p, q) => p.x - q.x)
  for (let i = 1; i < marks.length; i++) if (sx(marks[i].x) - sx(marks[i - 1].x) < marks[i - 1].label.length * 6.5 + 8) marks[i].row = (marks[i - 1].row + 1) % 3

  const ends = multi
    ? lines
        .map((l) => {
          let i = l.ys.length - 1
          while (i > 0 && (l.ys[i] == null || !isFinite(l.ys[i] as number))) i--
          return { l, y: sy((l.ys[i] as number) ?? ext.y0) }
        })
        .sort((p, q) => p.y - q.y)
    : []
  for (let i = 1; i < ends.length; i++) if (ends[i].y - ends[i - 1].y < 12) ends[i].y = ends[i - 1].y + 12

  return (
    <div className="chart" ref={ref}>
      {multi && (
        <div className="legend">
          {lines.map((l) => (
            <span key={l.name}>
              <i style={{ background: l.color }} />
              {l.name}
            </span>
          ))}
        </div>
      )}
      {size.width > 0 && (
        <svg width={size.width} height={height} role="img" aria-label={a.title}>
          <g className="grid">
            {yt.map((y) => (
              <line key={y} x1={M.l} x2={M.l + W} y1={sy(y)} y2={sy(y)} />
            ))}
          </g>
          <g className="axis">
            {yt.map((y) => (
              <text key={y} x={M.l - 8} y={sy(y) + 4} textAnchor="end">
                {fmtY(y)}
              </text>
            ))}
            {(iso ? isoTicks : xt).map((x, i) => (
              <text key={i} x={sx(x)} y={height - 6} textAnchor={iso && i === 0 ? 'start' : iso && i === isoTicks.length - 1 ? 'end' : 'middle'}>
                {fmtX(x)}
              </text>
            ))}
            {d.y_unit && (
              <text x={4} y={M.t + 2} textAnchor="start" className="unit">
                {d.y_unit}
              </text>
            )}
          </g>
          <line className="baseline" x1={M.l} x2={M.l + W} y1={M.t + H} y2={M.t + H} />
          {marks.map((m) => (
            <g key={m.label} className="mark">
              <line x1={sx(m.x)} x2={sx(m.x)} y1={M.t} y2={M.t + H} />
              <text x={sx(m.x) + 4} y={M.t + 10 + m.row * 12}>
                {m.label}
              </text>
            </g>
          ))}
          {lines.map((l) => (
            <path key={l.name} d={pathFor(l, sx, sy, W)} fill="none" stroke={l.color} strokeWidth={2} strokeLinejoin="round" strokeLinecap="round" />
          ))}
          {ends.map((e) => (
            <text key={e.l.name} x={M.l + W + 8} y={e.y + 4} className="endlabel">
              {e.l.name.length > 14 ? e.l.name.slice(0, 13) + '…' : e.l.name}
            </text>
          ))}
          {timeAxis && <Playhead sx={sx} top={M.t} bottom={M.t + H} x0={ext.x0} x1={ext.x1} />}
          {hx != null && (
            <>
              <line className="xhair" x1={hx} x2={hx} y1={M.t} y2={M.t + H} />
              {readout.map(
                (r) =>
                  r.y != null && isFinite(r.y) && <circle key={r.l.name} cx={sx(r.x)} cy={sy(r.y)} r={4} fill={r.l.color} stroke="#212121" strokeWidth={2} />,
              )}
            </>
          )}
          <rect
            className={timeAxis ? 'seekable' : ''}
            x={M.l}
            y={M.t}
            width={Math.max(0, W)}
            height={Math.max(0, H)}
            fill="transparent"
            onPointerMove={(e) => {
              const r = (e.currentTarget.ownerSVGElement as SVGSVGElement).getBoundingClientRect()
              setHx(Math.min(M.l + W, Math.max(M.l, e.clientX - r.left)))
            }}
            onPointerLeave={() => setHx(null)}
            onClick={() => timeAxis && hoverVal != null && pb.seek(hoverVal)}
          />
        </svg>
      )}
      {hx != null && readout.length > 0 && (
        <div className="tooltip" style={{ left: Math.min(hx + 12, size.width - 190), top: (multi ? 26 : 0) + M.t + 20 }}>
          <div className="faint">{fmtX(readout[0].x)}</div>
          {readout.map((r) => (
            <div key={r.l.name}>
              {multi && <i className="sw" style={{ background: r.l.color }} />}
              {multi ? `${r.l.name}: ` : ''}
              <b>{r.y == null ? '—' : num(r.y, d.y_unit)}</b>
            </div>
          ))}
          {timeAxis && <div className="faint">Click to jump the video here</div>}
        </div>
      )}
    </div>
  )
}

function Playhead({ sx, top, bottom, x0, x1 }: { sx: (x: number) => number; top: number; bottom: number; x0: number; x1: number }) {
  const t = usePlayhead()
  if (t < x0 || t > x1) return null
  return <line className="ph" x1={sx(t)} x2={sx(t)} y1={top} y2={bottom} />
}
