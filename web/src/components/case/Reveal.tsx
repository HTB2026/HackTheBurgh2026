import { hasLocation } from '../../lib/geo'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import L from 'leaflet'
import type { Artifact } from '../../lib/types'
import type { CaseState } from '../../lib/state'
import { useTypewriter } from '../../lib/hooks'
import { Icon } from '../ui/ui'
import { geoFinding, winnerIndex } from './GeoBoard'

export interface DossierLine {
  text: string
  evidence: string[]
  certainty: 'measured' | 'likely' | 'possible' | string
  category: string
}

const CATEGORY_ICON: Record<string, string> = {
  where: 'location-crosshairs',
  when: 'clock',
  device: 'mobile-screen',
  body: 'heart-pulse',
  surroundings: 'tree-city',
  integrity: 'scissors',
  hidden: 'eye-slash',
}

const AUTO_MS = 3500

// ---------------------------------------------------------------- heartbeat (Web Audio, synthesised)

class Heart {
  private ctx: AudioContext | null = null
  private master: GainNode | null = null
  private timer: number | undefined
  private next = 0

  start(bpm: number) {
    if (!this.ctx) {
      const AC = window.AudioContext ?? (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext
      this.ctx = new AC()
      this.master = this.ctx.createGain()
      this.master.gain.value = 0.55
      const lp = this.ctx.createBiquadFilter()
      lp.type = 'lowpass'
      lp.frequency.value = 160
      this.master.connect(lp).connect(this.ctx.destination)
    }
    void this.ctx.resume()
    const period = 60 / bpm
    this.next = this.ctx.currentTime + 0.1
    window.clearInterval(this.timer)
    // schedule ahead in small steps so timing stays sample-accurate
    this.timer = window.setInterval(() => {
      const ctx = this.ctx!
      while (this.next < ctx.currentTime + 0.4) {
        this.thump(this.next, 58, 1.0) // lub
        this.thump(this.next + 0.26 * Math.min(1, period), 46, 0.65) // dub
        this.next += period
      }
    }, 100)
  }

  private thump(at: number, freq: number, level: number) {
    const ctx = this.ctx!
    const o = ctx.createOscillator()
    const g = ctx.createGain()
    o.type = 'sine'
    o.frequency.setValueAtTime(freq * 1.6, at)
    o.frequency.exponentialRampToValueAtTime(freq, at + 0.06)
    g.gain.setValueAtTime(0.0001, at)
    g.gain.exponentialRampToValueAtTime(level, at + 0.015)
    g.gain.exponentialRampToValueAtTime(0.0001, at + 0.22)
    o.connect(g).connect(this.master!)
    o.start(at)
    o.stop(at + 0.25)
  }

  stop() {
    window.clearInterval(this.timer)
    this.timer = undefined
    void this.ctx?.suspend()
  }

  dispose() {
    this.stop()
    void this.ctx?.close()
    this.ctx = null
  }
}

/** A red ECG trace that beats at `bpm` along the bottom of the screen. */
function Ecg({ bpm, live }: { bpm: number; live: boolean }) {
  const period = 60 / bpm
  // one beat drawn in a 200-unit tile; the strip scrolls one tile per beat
  const beat = 'L60 50 L70 50 L76 44 L82 50 L92 50 L97 58 L103 8 L109 66 L114 50 L132 50 L144 41 L158 50 L200 50'
  const tiles = 14
  const d = Array.from({ length: tiles }, (_, i) => `M${i * 200} 50 ${beat.replace(/L(\d+) /g, (_, x) => `L${Number(x) + i * 200} `)}`).join(' ')
  return (
    <div className={`ecg ${live ? 'live' : ''}`} aria-hidden>
      <svg viewBox={`0 0 ${200 * (tiles - 2)} 70`} preserveAspectRatio="none">
        <g style={{ animationDuration: `${period}s` }} className="ecg-scroll">
          <path d={d} />
        </g>
      </svg>
      <span className="ecg-bpm mono">{live ? `${Math.round(bpm)} BPM` : 'AMBIENT'}</span>
    </div>
  )
}

// ---------------------------------------------------------------- "where" visual

function WhereVisual({ sat, lat, lon }: { sat?: Artifact; lat: number; lon: number }) {
  const coords = `${lat.toFixed(5)}°, ${lon.toFixed(5)}°`
  const typed = useTypewriter(coords, true, 18)
  const mapEl = useRef<HTMLDivElement>(null)
  useEffect(() => {
    if (sat?.url || !mapEl.current) return
    const m = L.map(mapEl.current, { zoomControl: false, attributionControl: false, dragging: false, scrollWheelZoom: false, doubleClickZoom: false, boxZoom: false, keyboard: false })
    L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', { maxZoom: 18 }).addTo(m)
    m.setView([20, 0], 1)
    const t = setTimeout(() => m.flyTo([lat, lon], 15, { duration: 5, easeLinearity: 0.15 }), 500)
    return () => {
      clearTimeout(t)
      m.remove()
    }
  }, [sat?.url, lat, lon])
  return (
    <div className="rv-where">
      <div className="rv-where-img">
        {sat?.url ? <img src={sat.url} alt="Satellite view" className="rv-zoom" /> : <div ref={mapEl} className="rv-minimap" />}
        <svg className="rv-cross" viewBox="0 0 100 100" aria-hidden>
          <circle cx="50" cy="50" r="14" />
          <path d="M50 20 V38 M50 62 V80 M20 50 H38 M62 50 H80" />
        </svg>
      </div>
      <span className="rv-coords mono">{typed}</span>
    </div>
  )
}

// ---------------------------------------------------------------- the overlay

function Line({ line, current, onJump, where }: { line: DossierLine; current: boolean; onJump: (id: string) => void; where?: React.ReactNode }) {
  return (
    <li className={`rv-line ${current ? 'current' : ''}`}>
      <div className="rv-line-meta">
        <Icon name={CATEGORY_ICON[line.category] ?? 'circle'} />
        <span className={`rv-cert ${line.certainty}`}>{String(line.certainty).toUpperCase()}</span>
      </div>
      <div className="rv-line-body">
        <p className="rv-text">
          <span className="rv-burn" aria-hidden />
          {line.text}
        </p>
        {where}
        {line.evidence?.length > 0 && (
          <div className="rv-evidence">
            {line.evidence.map((id) => (
              <button key={id} className="rv-chip mono" onClick={() => onJump(id)} data-tip="Close and show this finding">
                {id}
              </button>
            ))}
          </div>
        )}
      </div>
    </li>
  )
}

export function Reveal({ state, dossier, caseId, onClose }: { state: CaseState; dossier: Artifact; caseId: string; onClose: (jumpTo?: string) => void }) {
  const d = dossier.data ?? {}
  const lines: DossierLine[] = Array.isArray(d.lines) ? d.lines : []
  const bpmRaw: number | null = typeof d.heart_bpm === 'number' && d.heart_bpm > 0 ? d.heart_bpm : null
  const bpm = Math.min(150, Math.max(40, bpmRaw ?? 50)) // ≤ 2.5 Hz: photosensitivity-safe
  const [step, setStep] = useState(-1) // -1: headline; 0..n-1: lines; n: closing
  const [auto, setAuto] = useState(false)
  const [sound, setSound] = useState(false)
  const [run, setRun] = useState(0) // replay counter
  const heart = useMemo(() => new Heart(), [])
  const listRef = useRef<HTMLOListElement>(null)
  const headline = useTypewriter(d.headline ?? 'What this video gives away', true, 28)
  const headlineDone = headline.length >= String(d.headline ?? 'What this video gives away').length

  // background: the geo evidence frame if there is one, otherwise the thumbnail
  const bg = state.artifacts['geo:frame0']?.url ?? state.info?.thumb_url ?? null

  // geo data for "where" lines
  const geo = geoFinding(state)
  const gd = geo?.data
  const best = gd ? winnerIndex(gd) : 0
  const sat = gd ? state.artifacts[`geo:ref${best}_sat`] : undefined
  const firstWhere = lines.findIndex((l) => l.category === 'where')
  const located = gd && hasLocation(gd.latitude, gd.longitude, gd.radius_km)
  const whereLat = located ? gd.latitude : null
  const whereLon = located ? gd.longitude : null

  const next = useCallback(() => setStep((s) => Math.min(lines.length, s + 1)), [lines.length])
  const prev = useCallback(() => setStep((s) => Math.max(-1, s - 1)), [])

  useEffect(() => {
    const on = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose()
      else if (e.key === ' ' || e.key === 'ArrowRight') {
        e.preventDefault()
        next()
      } else if (e.key === 'ArrowLeft') prev()
      else if (e.key === 'a' || e.key === 'A') setAuto((a) => !a)
      else if (e.key === 's' || e.key === 'S') setSound((v) => !v)
    }
    window.addEventListener('keydown', on)
    return () => window.removeEventListener('keydown', on)
  }, [next, prev, onClose])

  useEffect(() => {
    if (!auto || step >= lines.length) return
    if (step === -1 && !headlineDone) return
    const t = setTimeout(next, step === -1 ? 1600 : AUTO_MS)
    return () => clearTimeout(t)
  }, [auto, step, headlineDone, lines.length, next])

  useEffect(() => {
    if (sound && bpmRaw) heart.start(bpm)
    else heart.stop()
  }, [sound, bpm, bpmRaw, heart])
  useEffect(() => () => heart.dispose(), [heart])

  useEffect(() => {
    document.body.style.overflow = 'hidden'
    return () => {
      document.body.style.overflow = ''
    }
  }, [])

  useEffect(() => {
    const cur = listRef.current?.querySelector('.rv-line.current, .rv-closing')
    cur?.scrollIntoView({ behavior: 'smooth', block: 'center' })
  }, [step])

  return (
    <div className="reveal" role="dialog" aria-modal aria-label="What this video gives away" key={run}>
      {bg && <div className="rv-bg" style={{ backgroundImage: `url(${bg})` }} />}
      <div className="rv-vignette" style={{ animationDuration: `${60 / bpm}s` }} />

      <header className="rv-top">
        <span className="mono">CASE {caseId} · WHAT THIS VIDEO GIVES AWAY</span>
        <span className="grow" />
        {bpmRaw ? (
          <button className={`rv-btn ${sound ? 'on' : ''}`} onClick={() => setSound((v) => !v)} data-tip="Heartbeat sound at the measured rate (S)">
            <Icon name={sound ? 'volume-high' : 'volume-xmark'} /> {sound ? 'Heartbeat on' : 'Heartbeat off'}
          </button>
        ) : null}
        <button className={`rv-btn ${auto ? 'on' : ''}`} onClick={() => setAuto((a) => !a)} data-tip="Presentation mode: one line every 3.5 s (A)">
          <Icon name={auto ? 'pause' : 'play'} /> Auto
        </button>
        <button className="rv-btn" onClick={() => onClose()} aria-label="Close (Esc)">
          <Icon name="xmark" />
        </button>
      </header>

      <div className="rv-stage">
        <h1 className={`rv-headline ${headlineDone ? '' : 'typing'}`}>{headline}</h1>
        <ol className="rv-lines" ref={listRef}>
          {lines.slice(0, Math.max(0, step + 1)).map((l, i) => (
            <Line
              key={i}
              line={l}
              current={i === step}
              onJump={(id) => onClose(id)}
              where={i === firstWhere && whereLat != null && whereLon != null ? <WhereVisual sat={sat} lat={whereLat} lon={whereLon} /> : undefined}
            />
          ))}
          {step >= lines.length && (
            <li className="rv-closing">
              {d.closing && <p className="rv-closing-text">{d.closing}</p>}
              <p className="rv-note">
                Every line above is traceable to a measured finding or a labelled AI inference. Nothing was looked up about you.
              </p>
              <div className="rv-end">
                <button className="rv-btn" onClick={() => onClose()}>
                  Close
                </button>
                <button
                  className="rv-btn"
                  onClick={() => {
                    setStep(-1)
                    setRun((r) => r + 1)
                  }}
                >
                  <Icon name="rotate-left" /> Replay
                </button>
              </div>
            </li>
          )}
        </ol>
      </div>

      <footer className="rv-foot">
        <span className="mono">
          {Math.max(0, Math.min(step + 1, lines.length))} / {lines.length}
        </span>
        <span className="muted">Space or → next · ← back · A auto · {bpmRaw ? 'S sound · ' : ''}Esc close</span>
        {step < lines.length && (
          <button className="rv-btn" onClick={next}>
            Next <Icon name="arrow-right" />
          </button>
        )}
      </footer>
      <Ecg bpm={bpm} live={!!bpmRaw} />
    </div>
  )
}
