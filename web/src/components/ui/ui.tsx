import { useEffect, useRef, useState, type CSSProperties, type ReactNode } from 'react'
import { googleMapsUrl } from '../../lib/geo'
import { CONFIDENCE, GLOSSARY, SEVERITY, type Accent } from '../../lib/plain'

export function Icon({ name, className = '', style, title }: { name: string; className?: string; style?: CSSProperties; title?: string }) {
  return <i className={`fa-solid fa-${name} ${className}`} style={style} aria-hidden={title ? undefined : true} title={title} />
}

/** A technical word with a plain-English tooltip. */
export function Term({ k, children }: { k: string; children?: ReactNode }) {
  const tip = GLOSSARY[k]
  if (!tip) return <>{children ?? k}</>
  return (
    <span className="term" data-tip={tip} tabIndex={0}>
      {children ?? k}
    </span>
  )
}

/** One floating tooltip for every [data-tip] element; never clipped by panels. */
export function TipLayer() {
  const [tip, setTip] = useState<{ text: string; x: number; y: number; below: boolean } | null>(null)
  useEffect(() => {
    const show = (el: Element | null) => {
      const t = el?.closest('[data-tip]') as HTMLElement | null
      if (!t) return setTip(null)
      const r = t.getBoundingClientRect()
      const below = r.top < 90
      setTip({ text: t.dataset.tip ?? '', x: r.left + r.width / 2, y: below ? r.bottom + 8 : r.top - 8, below })
    }
    const over = (e: Event) => show(e.target as Element)
    const hide = () => setTip(null)
    document.addEventListener('mouseover', over)
    document.addEventListener('focusin', over)
    document.addEventListener('scroll', hide, true)
    return () => {
      document.removeEventListener('mouseover', over)
      document.removeEventListener('focusin', over)
      document.removeEventListener('scroll', hide, true)
    }
  }, [])
  if (!tip?.text) return null
  const x = Math.min(window.innerWidth - 150, Math.max(150, tip.x))
  return (
    <div className={`float-tip ${tip.below ? 'below' : ''}`} style={{ left: x, top: tip.y }} role="tooltip">
      {tip.text}
    </div>
  )
}

export function SectionTitle({
  icon,
  accent = 'grey',
  children,
  small,
  right,
  id,
}: {
  icon: string
  accent?: Accent
  children: ReactNode
  small?: ReactNode
  right?: ReactNode
  id?: string
}) {
  return (
    <h2 className={`section-title c-${accent}`} id={id}>
      <Icon name={icon} />
      <span>{children}</span>
      {small && <small>{small}</small>}
      {right && <span className="st-right">{right}</span>}
    </h2>
  )
}

export function Alert({ icon = 'circle-info', title, children, tone = 'blue' }: { icon?: string; title?: ReactNode; children: ReactNode; tone?: 'blue' | 'red' | 'yellow' | 'green' }) {
  return (
    <div className={`alert alert-${tone}`} role="note">
      <Icon name={icon} className="alert-icon" />
      <div>
        {title && <b className="alert-title">{title}</b>}
        <div className="alert-body">{children}</div>
      </div>
    </div>
  )
}

export function Badge({ icon, accent = 'grey', children, tip }: { icon: string; accent?: Accent; children: ReactNode; tip?: string }) {
  return (
    <span className={`badge c-${accent}`} data-tip={tip}>
      <Icon name={icon} />
      {children}
    </span>
  )
}

export function SevChip({ sev, solid = true }: { sev: string; solid?: boolean }) {
  const s = SEVERITY[sev] ?? SEVERITY.info
  return (
    <span className={`sev sev-${sev} ${solid ? 'solid' : ''}`} data-tip={s.tip}>
      <Icon name={s.icon} />
      {s.word}
    </span>
  )
}

export function SourceChip({ kind }: { kind: string }) {
  return kind === 'inferred' ? (
    <span className="src src-ai" data-tip={GLOSSARY.AI}>
      <Icon name="robot" />
      AI
    </span>
  ) : (
    <span className="src src-measured" data-tip={GLOSSARY.measured}>
      <Icon name="ruler" />
      Measured
    </span>
  )
}

export function ConfBar({ level, compact }: { level: string; compact?: boolean }) {
  const c = CONFIDENCE[level] ?? CONFIDENCE.unknown
  return (
    <span className={`conf conf-${level} ${compact ? 'compact' : ''}`} data-tip={c.tip}>
      <span className="conf-track" aria-hidden>
        {[1, 2, 3].map((i) => (
          <i key={i} className={i <= c.level ? 'on' : ''} />
        ))}
      </span>
      <span className="conf-word">{c.level ? `${c.word} evidence` : c.word}</span>
    </span>
  )
}

/** Small icon link that opens a place in Google Maps (pin for exact points, the area for wide estimates). */
export function MapsLink({ lat, lon, radiusM, label }: { lat: number; lon: number; radiusM?: number | null; label?: string }) {
  const wide = !!radiusM && radiusM >= 2000
  const tip = label ?? (wide ? 'Open this area in Google Maps' : 'Open in Google Maps')
  return (
    <a className="copy maps-link" href={googleMapsUrl(lat, lon, radiusM)} target="_blank" rel="noreferrer" onClick={(e) => e.stopPropagation()} data-tip={tip} aria-label={tip}>
      <Icon name="map-location-dot" />
    </a>
  )
}

export function CopyButton({ text, label = 'Copy' }: { text: string; label?: string }) {
  const [done, setDone] = useState(false)
  return (
    <button
      className="copy"
      onClick={(e) => {
        e.stopPropagation()
        navigator.clipboard?.writeText(text).then(() => {
          setDone(true)
          setTimeout(() => setDone(false), 1200)
        })
      }}
      data-tip={done ? 'Copied' : `${label} to clipboard`}
      aria-label={label}
    >
      <Icon name={done ? 'check' : 'copy'} />
    </button>
  )
}

export function Modal({ children, onClose, wide }: { children: ReactNode; onClose: () => void; wide?: boolean }) {
  const ref = useRef<HTMLDivElement>(null)
  useEffect(() => {
    ref.current?.focus()
    const on = (e: KeyboardEvent) => e.key === 'Escape' && onClose()
    window.addEventListener('keydown', on)
    return () => window.removeEventListener('keydown', on)
  }, [onClose])
  return (
    <div
      className="modal-back"
      ref={ref}
      tabIndex={-1}
      onMouseDown={(e) => {
        if (e.target === e.currentTarget) onClose()
      }}
    >
      <div className={`modal ${wide ? 'wide' : ''}`} role="dialog" aria-modal>
        <button className="modal-x" onClick={onClose} aria-label="Close">
          <Icon name="xmark" />
        </button>
        {children}
      </div>
    </div>
  )
}

/** Redacted bars: the loading state of a case file. */
export function Redacted({ lines = 3 }: { lines?: number }) {
  return (
    <div className="redacted" aria-label="Loading">
      {Array.from({ length: lines }, (_, i) => (
        <span key={i} style={{ width: `${55 + ((i * 37) % 40)}%` }} />
      ))}
    </div>
  )
}

export function Empty({ icon = 'folder-open', children }: { icon?: string; children: ReactNode }) {
  return (
    <div className="empty">
      <Icon name={icon} />
      <div>{children}</div>
    </div>
  )
}
