import { useEffect, useRef, useState } from 'react'
import { api } from '../../lib/api'
import type { EnhanceResult } from '../../lib/types'
import { mmss } from '../../lib/format'
import { capture } from '../../lib/pointer'
import { Icon, Modal } from '../ui/ui'

export interface EnhanceBox {
  x: number
  y: number
  w: number
  h: number
  time_s: number
}

const STEPS = [
  'Collecting the frames around this moment',
  'Enlarging each crop',
  'Lining them up to a fraction of a pixel',
  'Averaging away the noise',
  'Sharpening edges',
  'Boosting local contrast',
]

export function EnhanceModal({ caseId, box, onClose }: { caseId: string; box: EnhanceBox; onClose: () => void }) {
  const [result, setResult] = useState<EnhanceResult | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [stepIdx, setStepIdx] = useState(0)
  const [view, setView] = useState<'slider' | 'side'>('slider')

  useEffect(() => {
    let live = true
    api
      .enhance(caseId, { time_s: box.time_s, x: box.x, y: box.y, w: box.w, h: box.h })
      .then((r) => live && setResult(r))
      .catch((e) => live && setError((e as Error).message))
    const id = setInterval(() => setStepIdx((i) => Math.min(STEPS.length - 1, i + 1)), 450)
    return () => {
      live = false
      clearInterval(id)
    }
  }, [caseId, box])

  return (
    <Modal onClose={onClose} wide>
      <div className="enh">
        <header className="enh-head">
          <h3>
            <Icon name="magnifying-glass-plus" className="c-red-i" /> Zoom in on a detail
          </h3>
          <span className="muted">
            Region {(box.w * 100).toFixed(0)}% × {(box.h * 100).toFixed(0)}% of the frame at {mmss(box.time_s)}
          </span>
          {result && (
            <div className="tabs mini">
              <button className={view === 'slider' ? 'active' : ''} onClick={() => setView('slider')}>
                Before / after slider
              </button>
              <button className={view === 'side' ? 'active' : ''} onClick={() => setView('side')}>
                Side by side
              </button>
            </div>
          )}
        </header>
        {error ? (
          <div className="enh-wait">
            <Icon name="triangle-exclamation" className="c-red-i big" />
            <b>Could not sharpen this region</b>
            <span className="muted">{error}. Try a larger box, or a moment where the camera is steadier.</span>
          </div>
        ) : !result ? (
          <div className="enh-wait">
            <div className="enh-scan" />
            <b className="enh-title">Enhancing…</b>
            <div className="progress" style={{ width: 360 }}>
              <span style={{ width: `${((stepIdx + 1) / STEPS.length) * 100}%` }} />
            </div>
            <span className="muted">{STEPS[stepIdx]}</span>
          </div>
        ) : (
          <>
            {view === 'slider' ? <Compare a={result.single_url} b={result.enhanced_url} /> : <SideBySide r={result} />}
            <dl className="enh-stats">
              <div data-tip="Neighbouring frames that were aligned and averaged. More frames = more real detail recovered.">
                <dt>Frames combined</dt>
                <dd>{result.frames_used}</dd>
              </div>
              <div>
                <dt>Enlarged</dt>
                <dd>{result.scale}×</dd>
              </div>
              <div>
                <dt>Original crop</dt>
                <dd>
                  {result.source_px?.[0]} × {result.source_px?.[1]} px
                </dd>
              </div>
              <div>
                <dt>Moment</dt>
                <dd>{mmss(result.time_s)}</dd>
              </div>
              <a className="btn sm" href={result.enhanced_url} target="_blank" rel="noreferrer">
                <Icon name="download" /> Open enhanced image
              </a>
            </dl>
            <p className="muted small">
              Enhancing combines real detail from several frames; it does not invent pixels. Still, treat what you read here as something to
              confirm, not proof.
            </p>
          </>
        )}
      </div>
    </Modal>
  )
}

function SideBySide({ r }: { r: EnhanceResult }) {
  return (
    <div className="side-by-side">
      <figure>
        <img src={r.single_url} alt="Single frame, enlarged" />
        <figcaption>One frame, enlarged</figcaption>
      </figure>
      <figure>
        <img src={r.enhanced_url} alt="Enhanced" />
        <figcaption className="c-red-i">Enhanced from {r.frames_used} frames</figcaption>
      </figure>
    </div>
  )
}

function Compare({ a, b }: { a: string; b: string }) {
  const [pos, setPos] = useState(0.5)
  const ref = useRef<HTMLDivElement>(null)
  const dragging = useRef(false)
  const move = (clientX: number) => {
    const r = ref.current?.getBoundingClientRect()
    if (r) setPos(Math.min(1, Math.max(0, (clientX - r.left) / r.width)))
  }
  return (
    <div
      className="compare"
      ref={ref}
      onPointerDown={(e) => {
        dragging.current = true
        capture(e)
        move(e.clientX)
      }}
      onPointerMove={(e) => dragging.current && move(e.clientX)}
      onPointerUp={() => (dragging.current = false)}
    >
      <img src={a} alt="One frame" />
      <img src={b} alt="Enhanced" style={{ clipPath: `inset(0 0 0 ${pos * 100}%)` }} />
      <div className="handle" style={{ left: `${pos * 100}%` }}>
        <span>
          <Icon name="arrows-left-right" />
        </span>
      </div>
      <span className="tag l">One frame</span>
      <span className="tag r">Enhanced</span>
    </div>
  )
}
