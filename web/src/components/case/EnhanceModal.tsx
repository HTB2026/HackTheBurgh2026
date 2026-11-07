import { useEffect, useRef, useState } from 'react'
import { api } from '../../lib/api'
import type { EnhanceResult, Lens } from '../../lib/types'
import { mmss } from '../../lib/format'
import { capture } from '../../lib/pointer'
import { useMediaKind } from '../../lib/region'
import { Icon, Modal } from '../ui/ui'

export interface EnhanceBox {
  x: number
  y: number
  w: number
  h: number
  time_s: number
}

const STEPS_VIDEO = [
  'Collecting the frames around this moment',
  'Enlarging each crop',
  'Lining them up to a fraction of a pixel',
  'Averaging away the noise',
  'Sharpening edges',
  'Boosting local contrast',
]

const STEPS_PHOTO = [
  'Cutting the region from the full-resolution original',
  'Enlarging it',
  'Sharpening edges and boosting local contrast',
  'Computing error levels',
  'Amplifying the sensor grain',
  'Separating the colour channels',
]

export function EnhanceModal({ caseId, box, onClose }: { caseId: string; box: EnhanceBox; onClose: () => void }) {
  const kind = useMediaKind()
  const still = kind === 'image'
  const [result, setResult] = useState<EnhanceResult | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [stepIdx, setStepIdx] = useState(0)
  const [view, setView] = useState<'slider' | 'side' | 'lenses'>('slider')
  const [lens, setLens] = useState<Lens | null>(null)
  const steps = still ? STEPS_PHOTO : STEPS_VIDEO

  useEffect(() => {
    let live = true
    api
      .enhance(caseId, { time_s: box.time_s, x: box.x, y: box.y, w: box.w, h: box.h })
      .then((r) => {
        if (!live) return
        setResult(r)
        if (r.lenses?.length) setLens(r.lenses[0])
      })
      .catch((e) => live && setError((e as Error).message))
    const id = setInterval(() => setStepIdx((i) => Math.min(steps.length - 1, i + 1)), 450)
    return () => {
      live = false
      clearInterval(id)
    }
  }, [caseId, box, steps.length])

  const lenses = result?.lenses ?? []

  return (
    <Modal onClose={onClose} wide>
      <div className="enh">
        <header className="enh-head">
          <h3>
            <Icon name="magnifying-glass-plus" className="c-red-i" /> {still ? 'Examine a detail' : 'Zoom in on a detail'}
          </h3>
          <span className="muted">
            Region {(box.w * 100).toFixed(0)}% × {(box.h * 100).toFixed(0)}% of the {still ? 'picture' : `frame at ${mmss(box.time_s)}`}
          </span>
          {result && (
            <div className="tabs mini">
              <button className={view === 'slider' ? 'active' : ''} onClick={() => setView('slider')}>
                Before / after slider
              </button>
              <button className={view === 'side' ? 'active' : ''} onClick={() => setView('side')}>
                Side by side
              </button>
              {lenses.length > 0 && (
                <button className={view === 'lenses' ? 'active' : ''} onClick={() => setView('lenses')}>
                  Forensic lenses <span className="n">{lenses.length}</span>
                </button>
              )}
            </div>
          )}
        </header>
        {error ? (
          <div className="enh-wait">
            <Icon name="triangle-exclamation" className="c-red-i big" />
            <b>Could not {still ? 'examine' : 'sharpen'} this region</b>
            <span className="muted">{error}. Try a larger box{still ? '' : ', or a moment where the camera is steadier'}.</span>
          </div>
        ) : !result ? (
          <div className="enh-wait">
            <div className="enh-scan" />
            <b className="enh-title">{still ? 'Examining…' : 'Enhancing…'}</b>
            <div className="progress" style={{ width: 360 }}>
              <span style={{ width: `${((stepIdx + 1) / steps.length) * 100}%` }} />
            </div>
            <span className="muted">{steps[stepIdx]}</span>
          </div>
        ) : view === 'lenses' && lens ? (
          <div className="lenses">
            <div className="lens-pick">
              {lenses.map((l) => (
                <button key={l.name} className={`chip ${lens.name === l.name ? 'on' : ''}`} onClick={() => setLens(l)} data-tip={l.caption}>
                  {l.name}
                </button>
              ))}
            </div>
            <Compare a={result.enhanced_url} b={lens.url} leftLabel="Enlarged" rightLabel={lens.name} />
            <p className="muted small lens-caption">{lens.caption}</p>
          </div>
        ) : view === 'slider' ? (
          <Compare a={result.single_url} b={result.enhanced_url} leftLabel={still ? 'Plain enlargement' : 'One frame'} rightLabel="Enhanced" />
        ) : (
          <SideBySide r={result} still={still} />
        )}
        {result && !error && (
          <>
            <dl className="enh-stats">
              {!still && (
                <div data-tip="Neighbouring frames that were aligned and averaged. More frames = more real detail recovered.">
                  <dt>Frames combined</dt>
                  <dd>{result.frames_used}</dd>
                </div>
              )}
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
              {!still && (
                <div>
                  <dt>Moment</dt>
                  <dd>{mmss(result.time_s)}</dd>
                </div>
              )}
              <a className="btn sm" href={result.enhanced_url} target="_blank" rel="noreferrer">
                <Icon name="download" /> Open enhanced image
              </a>
            </dl>
            <p className="muted small">
              {still
                ? 'Nothing here invents pixels: the crop is taken from the original file at full resolution, enlarged, and shown through lenses (error levels, grain, channels, edges) that make existing detail visible. Treat what you read as something to confirm, not proof.'
                : 'Enhancing combines real detail from several frames; it does not invent pixels. Still, treat what you read here as something to confirm, not proof.'}
            </p>
          </>
        )}
      </div>
    </Modal>
  )
}

function SideBySide({ r, still }: { r: EnhanceResult; still: boolean }) {
  return (
    <div className="side-by-side">
      <figure>
        <img src={r.single_url} alt="Enlarged" />
        <figcaption>{still ? 'Plain enlargement' : 'One frame, enlarged'}</figcaption>
      </figure>
      <figure>
        <img src={r.enhanced_url} alt="Enhanced" />
        <figcaption className="c-red-i">{still ? 'Sharpened, local contrast boosted' : `Enhanced from ${r.frames_used} frames`}</figcaption>
      </figure>
    </div>
  )
}

function Compare({ a, b, leftLabel, rightLabel }: { a: string; b: string; leftLabel: string; rightLabel: string }) {
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
      <img src={a} alt={leftLabel} />
      <img src={b} alt={rightLabel} style={{ clipPath: `inset(0 0 0 ${pos * 100}%)` }} />
      <div className="handle" style={{ left: `${pos * 100}%` }}>
        <span>
          <Icon name="arrows-left-right" />
        </span>
      </div>
      <span className="tag l">{leftLabel}</span>
      <span className="tag r">{rightLabel}</span>
    </div>
  )
}
