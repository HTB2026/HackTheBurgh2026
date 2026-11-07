import { memo, useEffect, useRef, useState } from 'react'
import type { Artifact, Finding } from '../../lib/types'
import { usePlayback, usePlayhead } from '../../lib/playback'
import { useElementSize } from '../../lib/hooks'
import { mmss, niceTicks } from '../../lib/format'
import { capture } from '../../lib/pointer'
import { SEVERITY } from '../../lib/plain'

interface Props {
  duration: number
  filmstrips: Artifact[]
  waveforms: Artifact[]
  markers: Artifact[]
  findings: Finding[]
  hovered?: string | null
  onHover?: (id: string | null) => void
  onPick?: (f: Finding) => void
  compact?: boolean
}

/** Filmstrip + sound + keyframes + flagged moments, one shared playhead. Click or drag to seek. */
export function Timeline({ duration, filmstrips, waveforms, markers, findings, hovered, onHover, onPick, compact }: Props) {
  const pb = usePlayback()
  const [ref, size] = useElementSize<HTMLDivElement>()
  const [hoverX, setHoverX] = useState<number | null>(null)
  const dragging = useRef(false)
  const W = size.width
  const dur = duration > 0 ? duration : 1
  const timed = findings.filter((f) => f.time_s != null)
  const pct = (t: number) => `${(Math.min(dur, Math.max(0, t)) / dur) * 100}%`
  const timeAt = (clientX: number) => {
    const r = ref.current!.getBoundingClientRect()
    return Math.min(dur, Math.max(0, ((clientX - r.left) / r.width) * dur))
  }

  const rows = [
    { key: 'ruler', label: 'Time', tip: 'Minutes and seconds into the video', cls: 'tl-ruler', node: <Ruler W={W} dur={dur} /> },
    ...filmstrips.map((a) => ({ key: a.id, label: 'Picture', tip: 'Thumbnails across the video', cls: 'tl-film', node: <Filmstrip a={a} W={W} dur={dur} /> })),
    ...waveforms.map((a) => ({ key: a.id, label: 'Sound', tip: 'Loudness over time (quiet clips are scaled up)', cls: 'tl-wave', node: <Waveform a={a} W={W} dur={dur} /> })),
    ...(compact
      ? []
      : markers.map((a) => ({
          key: a.id,
          label: a.title || a.name,
          tip: 'Keyframes: full pictures stored at regular intervals. Irregular spacing hints at an edit.',
          cls: 'tl-marks',
          node: (
            <>
              {((a.data?.times ?? []) as number[]).map((t, i) => (
                <i key={i} className="mk" style={{ left: pct(t) }} />
              ))}
            </>
          ),
        }))),
    {
      key: 'flags',
      label: 'Flagged',
      tip: 'Findings tied to a moment. Colour and shape show how serious they are.',
      cls: 'tl-flags',
      node: (
        <>
          {timed.map((f) => {
            const range = f.time_end_s != null && f.time_end_s > (f.time_s as number)
            return (
              <i
                key={f.id}
                className={`fm sev-${f.severity} ${range ? 'range' : ''} ${hovered === f.id ? 'hl' : ''}`}
                style={{ left: pct(f.time_s as number), width: range ? `${(((f.time_end_s as number) - (f.time_s as number)) / dur) * 100}%` : undefined }}
                data-tip={`${SEVERITY[f.severity]?.word ?? f.severity} at ${mmss(f.time_s)}: ${f.title}`}
                onPointerEnter={() => onHover?.(f.id)}
                onPointerLeave={() => onHover?.(null)}
                onPointerDown={(e) => {
                  e.stopPropagation()
                  pb.seek(f.time_s as number)
                  onPick?.(f)
                }}
              />
            )
          })}
        </>
      ),
    },
  ]

  return (
    <div className="timeline">
      <div className="tl-labels">
        {rows.map((r) => (
          <div key={r.key} className={`tl-label ${r.cls}`} data-tip={r.tip}>
            {r.label}
          </div>
        ))}
      </div>
      <div
        className="tl-tracks"
        ref={ref}
        onPointerDown={(e) => {
          dragging.current = true
          capture(e)
          pb.seek(timeAt(e.clientX))
        }}
        onPointerMove={(e) => {
          const r = ref.current!.getBoundingClientRect()
          setHoverX(e.clientX - r.left)
          if (dragging.current) pb.seek(timeAt(e.clientX))
        }}
        onPointerUp={() => (dragging.current = false)}
        onPointerLeave={() => setHoverX(null)}
      >
        {rows.map((r) => (
          <div key={r.key} className={`tl-track ${r.cls}`}>
            {r.node}
          </div>
        ))}
        <Playhead dur={dur} />
        {hoverX != null && W > 0 && (
          <div className="tl-hover" style={{ left: hoverX }}>
            <span className="mono">{mmss((hoverX / W) * dur)}</span>
          </div>
        )}
      </div>
    </div>
  )
}

function Playhead({ dur }: { dur: number }) {
  const t = usePlayhead()
  return <div className="tl-playhead" style={{ left: `${(Math.min(dur, t) / dur) * 100}%` }} />
}

const Ruler = memo(function Ruler({ W, dur }: { W: number; dur: number }) {
  if (!W) return null
  const major = niceTicks(0, dur, Math.max(2, Math.floor(W / 90)))
  return (
    <>
      {major.map((t) => (
        <span key={t} className="tick mono" style={{ left: `${(t / dur) * 100}%` }}>
          {mmss(t)}
        </span>
      ))}
    </>
  )
})

const Filmstrip = memo(function Filmstrip({ a, W, dur }: { a: Artifact; W: number; dur: number }) {
  const d = a.data ?? {}
  const times: number[] = d.times ?? []
  const count: number = d.count ?? times.length
  const tw: number = d.tile_width ?? 160
  const th: number = d.tile_height ?? 90
  if (!a.url || !count || !W) return null
  const H = 52
  const scaledW = (tw * H) / th
  const sheetW = scaledW * count
  return (
    <>
      {Array.from({ length: count }, (_, i) => {
        const t0 = times[i] ?? (i / count) * dur
        const t1 = times[i + 1] ?? dur
        const left = (t0 / dur) * W
        const width = Math.max(1, ((t1 - t0) / dur) * W)
        return (
          <div
            key={i}
            className="tile"
            style={{
              left,
              width,
              backgroundImage: `url(${a.url})`,
              backgroundSize: `${sheetW}px ${H}px`,
              backgroundPosition: `${-i * scaledW + (width - scaledW) / 2}px 0`,
            }}
          />
        )
      })}
    </>
  )
})

function drawWave(canvas: HTMLCanvasElement | null, peaks: number[], span: number, dur: number, W: number, H: number, color: string) {
  if (!canvas || !W) return
  const dpr = window.devicePixelRatio || 1
  canvas.width = Math.round(W * dpr)
  canvas.height = Math.round(H * dpr)
  canvas.style.width = `${W}px`
  canvas.style.height = `${H}px`
  const g = canvas.getContext('2d')!
  g.scale(dpr, dpr)
  g.clearRect(0, 0, W, H)
  g.fillStyle = color
  const used = (Math.min(span, dur) / dur) * W
  const max = Math.max(1e-6, ...peaks)
  const norm = max < 0.25 ? 0.9 / max : 1
  const cols = Math.max(1, Math.floor(used / 2))
  const mid = H / 2
  for (let x = 0; x < cols; x++) {
    const i0 = Math.floor((x / cols) * peaks.length)
    const i1 = Math.max(i0 + 1, Math.floor(((x + 1) / cols) * peaks.length))
    let p = 0
    for (let i = i0; i < i1 && i < peaks.length; i++) p = Math.max(p, peaks[i])
    const h = Math.max(0.5, Math.min(1, p * norm) * (mid - 1))
    g.fillRect(x * 2, mid - h, 1.4, h * 2)
  }
}

const Waveform = memo(function Waveform({ a, W, dur }: { a: Artifact; W: number; dur: number }) {
  const base = useRef<HTMLCanvasElement>(null)
  const lit = useRef<HTMLCanvasElement>(null)
  const peaks: number[] = a.data?.peak ?? []
  const span: number = a.data?.t1 ?? dur
  useEffect(() => {
    drawWave(base.current, peaks, span, dur, W, 40, '#4a4a4c')
    drawWave(lit.current, peaks, span, dur, W, 40, '#db2828')
  }, [peaks, span, dur, W])
  return (
    <>
      <canvas ref={base} className="wave" />
      <Played dur={dur}>
        <canvas ref={lit} className="wave" />
      </Played>
    </>
  )
})

function Played({ dur, children }: { dur: number; children: React.ReactNode }) {
  const t = usePlayhead()
  return <div style={{ position: 'absolute', inset: 0, clipPath: `inset(0 ${100 - (Math.min(dur, t) / dur) * 100}% 0 0)` }}>{children}</div>
}
