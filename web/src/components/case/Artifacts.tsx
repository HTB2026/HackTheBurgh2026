import { Component, useState, type ReactNode } from 'react'
import type { Artifact } from '../../lib/types'
import type { ModuleState } from '../../lib/state'
import { usePlayback } from '../../lib/playback'
import { useRegionFocus } from '../../lib/region'
import { mmss } from '../../lib/format'
import { explainArtifact, modulePlain } from '../../lib/plain'
import { Icon, Modal } from '../ui/ui'
import { JsonTree } from '../ui/Data'
import { SeriesChart } from '../viz/SeriesChart'
import { Spectrogram } from '../viz/Spectrogram'
import { Bars } from '../viz/Bars'
import { TableView } from '../viz/TableView'
import { MapView } from '../viz/MapView'
import { Transcript } from '../viz/Transcript'

const KIND_ICON: Record<string, string> = {
  series: 'chart-line',
  spectrogram: 'wave-square',
  bars: 'chart-bar',
  table: 'table',
  map: 'map-location-dot',
  image: 'image',
  heatmap: 'fire',
  video: 'video',
  text: 'file-lines',
  captions: 'closed-captioning',
  json: 'code',
  markers: 'grip-lines-vertical',
  waveform: 'wave-square',
  filmstrip: 'film',
  brief: 'scale-balanced',
  file: 'file-zipper',
  dossier: 'eye',
}

const VISUAL = new Set(['image', 'heatmap', 'video'])

class Boundary extends Component<{ children: ReactNode }, { error: string | null }> {
  state = { error: null as string | null }
  static getDerivedStateFromError(e: unknown) {
    return { error: String((e as Error)?.message ?? e) }
  }
  render() {
    return this.state.error ? <div className="err">This exhibit could not be drawn: {this.state.error}</div> : this.props.children
  }
}

/** All artifacts for a tab: charts and tables as sections, images/videos as a card grid. */
export function ArtifactGroup({ items, modules }: { items: Artifact[]; modules: Record<string, ModuleState> }) {
  const visual = items.filter((a) => VISUAL.has(a.kind)).sort((a, b) => (a.data?.time_s ?? a.time_s ?? 0) - (b.data?.time_s ?? b.time_s ?? 0))
  const rest = items.filter((a) => !VISUAL.has(a.kind))
  const byModule = new Map<string, Artifact[]>()
  for (const a of visual) byModule.set(a.module, [...(byModule.get(a.module) ?? []), a])
  return (
    <>
      {rest.map((a) => (
        <ArtifactSection key={a.id} a={a} modules={modules} />
      ))}
      {[...byModule.entries()].map(([m, list]) => {
        const p = modulePlain(m, modules[m]?.title)
        return (
          <section className="exhibit" key={m}>
            <header className={`exhibit-head c-${p.accent}`}>
              <Icon name="images" />
              <h3>{p.name}: exhibits</h3>
              <small>{list.length}</small>
            </header>
            <p className="explain">{explainArtifact(list[0])}</p>
            <div className="cards">
              {list.map((a) => (
                <ExhibitCard key={a.id} a={a} modules={modules} />
              ))}
            </div>
          </section>
        )
      })}
    </>
  )
}

export function ArtifactSection({ a, modules }: { a: Artifact; modules: Record<string, ModuleState> }) {
  const pb = usePlayback()
  const p = modulePlain(a.module, modules[a.module]?.title)
  const explain = explainArtifact(a)
  return (
    <section className="exhibit">
      <header className={`exhibit-head c-${p.accent}`}>
        <Icon name={KIND_ICON[a.kind] ?? 'cube'} />
        <h3>{a.title || a.name}</h3>
        <small data-tip={p.does}>{p.name}</small>
        {a.time_s != null && (
          <button className="timebtn mono" onClick={() => pb.seek(a.time_s as number)}>
            <Icon name="play" /> {mmss(a.time_s)}
          </button>
        )}
      </header>
      {explain && <p className="explain">{explain}</p>}
      <div className="exhibit-body">
        <Boundary>
          <ArtifactBody a={a} />
        </Boundary>
      </div>
    </section>
  )
}

export function ArtifactBody({ a }: { a: Artifact }) {
  switch (a.kind) {
    case 'series':
      return <SeriesChart a={a} />
    case 'spectrogram':
      return <Spectrogram a={a} />
    case 'bars':
      return <Bars a={a} />
    case 'table':
      return <TableView a={a} />
    case 'map':
      return <MapView points={a.data?.points ?? []} />
    case 'captions':
      return <Transcript a={a} />
    case 'text':
      return <TextBlock text={a.data?.text ?? ''} />
    case 'json':
      return (
        <div className="json-box">
          <JsonTree value={a.data} name={a.name || 'data'} open />
        </div>
      )
    case 'video':
      return a.url ? <video className="inline-video" src={a.url} controls loop muted playsInline /> : null
    case 'image':
    case 'heatmap':
      return a.url ? <img className="inline-img" src={a.url} alt={a.title} /> : null
    case 'file':
      return (
        <div className="file-art">
          {a.url && (
            <a className="btn sm" href={a.url} target="_blank" rel="noreferrer" download>
              <Icon name="download" /> Download the appended data ({a.data?.bytes?.toLocaleString?.('en-GB') ?? '?'} bytes)
            </a>
          )}
          {a.data?.label && <p className="muted small">Identified as: {a.data.label}</p>}
          {a.data?.head_hex && <pre className="hex mono">{a.data.head_hex}</pre>}
          {Array.isArray(a.data?.strings) && a.data.strings.length > 0 && (
            <>
              <h4 className="sub">Readable fragments</h4>
              <ul className="strings mono">
                {a.data.strings.slice(0, 40).map((s: string, i: number) => (
                  <li key={i}>{s}</li>
                ))}
              </ul>
            </>
          )}
        </div>
      )
    default:
      return (
        <div className="json-box">
          {a.url && (
            <a className="link" href={a.url} target="_blank" rel="noreferrer">
              Open file
            </a>
          )}
          <JsonTree value={a.data} name={a.kind} open />
        </div>
      )
  }
}

/** Long text: markdown-ish headings and paragraphs, collapsed past a few lines. */
export function TextBlock({ text }: { text: string }) {
  const [more, setMore] = useState(false)
  const long = text.length > 900
  const shown = long && !more ? text.slice(0, 900) + '…' : text
  return (
    <div className="prose">
      {shown.split(/\n{2,}/).map((para, i) =>
        /^#+\s/.test(para) ? (
          <h4 key={i}>{para.replace(/^#+\s*/, '')}</h4>
        ) : (
          <p key={i}>{para}</p>
        ),
      )}
      {long && (
        <button className="link small" onClick={() => setMore((m) => !m)}>
          {more ? 'Show less' : 'Read all'}
        </button>
      )}
    </div>
  )
}

/** Beatmap-style card: image on top, status pill, round accent button overlapping the image edge. */
export function ExhibitCard({ a, modules }: { a: Artifact; modules: Record<string, ModuleState> }) {
  const pb = usePlayback()
  const rf = useRegionFocus()
  const [open, setOpen] = useState(false)
  const p = modulePlain(a.module, modules[a.module]?.title)
  const t: number | null = a.data?.time_s ?? a.time_s ?? null
  const box = a.data?.box && typeof a.data.box.x === 'number' ? a.data.box : null
  return (
    <article className={`card c-${p.accent}`}>
      <div className="card-img" onClick={() => setOpen(true)} role="button" tabIndex={0} onKeyDown={(e) => e.key === 'Enter' && setOpen(true)}>
        {a.kind === 'video' ? (
          <video src={a.url ?? undefined} muted loop playsInline preload="metadata" onMouseEnter={(e) => e.currentTarget.play().catch(() => undefined)} onMouseLeave={(e) => e.currentTarget.pause()} />
        ) : (
          a.url && <img src={a.url} alt={a.title} loading="lazy" />
        )}
        <span className="pill">
          <Icon name={p.icon} /> {a.kind === 'heatmap' ? (a.data?.overlay ? 'Overlay map' : 'Tamper map') : a.kind === 'video' ? 'Video' : p.name}
        </span>
      </div>
      {t != null && (
        <button className="card-play" onClick={() => pb.seek(t)} data-tip={`Jump the video to ${mmss(t)}`} aria-label="Jump to this moment">
          <Icon name="play" />
        </button>
      )}
      {t == null && box && (
        <button className="card-play" onClick={() => rf.focus(box, a.title)} data-tip="Show this region on the picture" aria-label="Show on the picture">
          <Icon name="vector-square" />
        </button>
      )}
      <div className="card-body">
        <h4 title={a.title}>{a.title}</h4>
        {a.data?.caption && <p className="muted small">{a.data.caption}</p>}
        <div className="card-meta">
          {t != null && (
            <span>
              <Icon name="clock" /> {mmss(t)}
            </span>
          )}
          <span>
            <Icon name={KIND_ICON[a.kind] ?? 'cube'} /> {a.kind}
          </span>
          <span className="grow" />
          <button className="btn ghost sm" onClick={() => setOpen(true)}>
            <Icon name="up-right-and-down-left-from-center" /> Enlarge
          </button>
        </div>
      </div>
      {open && (
        <Modal onClose={() => setOpen(false)} wide>
          <div className="lightbox">
            <h3>{a.title}</h3>
            <p className="muted">{explainArtifact(a)}</p>
            {a.kind === 'video' ? <video src={a.url ?? undefined} controls autoPlay loop muted playsInline /> : a.url && <img src={a.url} alt={a.title} />}
            <div className="lightbox-foot">
              {a.data?.caption && <span className="muted">{a.data.caption}</span>}
              <span className="grow" />
              {t != null && (
                <button
                  className="btn sm"
                  onClick={() => {
                    pb.seek(t)
                    setOpen(false)
                  }}
                >
                  <Icon name="play" /> Show me at {mmss(t)}
                </button>
              )}
              {a.url && (
                <a className="btn ghost sm" href={a.url} target="_blank" rel="noreferrer">
                  <Icon name="download" /> Open original
                </a>
              )}
            </div>
          </div>
        </Modal>
      )}
    </article>
  )
}
