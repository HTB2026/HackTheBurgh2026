import { hasLocation } from '../../lib/geo'
import { Fragment, useMemo, useState, type ReactNode } from 'react'
import type { Answer, Artifact, Finding } from '../../lib/types'
import type { CaseState, ModuleState } from '../../lib/state'
import type { Facts } from '../../lib/facts'
import { usePlayback, usePlayhead } from '../../lib/playback'
import { useElementSize } from '../../lib/hooks'
import { bytes, dateShort, mmss, niceDate } from '../../lib/format'
import { artifactTab, BULLET_LABEL, CONFIDENCE, GLOSSARY, modulePlain, SEVERITY, TABS, type TabKey } from '../../lib/plain'
import { CopyButton, Icon, SevChip, SourceChip, Term } from '../ui/ui'
import { DataView, JsonTree } from '../ui/Data'
import { Timeline } from '../viz/Timeline'
import { MapView, radiusText, type MapPoint } from '../viz/MapView'
import { ArtifactGroup } from './Artifacts'

const arts = (state: CaseState) => state.artifactOrder.map((id) => state.artifacts[id])

/** "[F012]" citations → buttons that open the finding in the findings table. */
export function Cited({ text, openFinding }: { text: string; openFinding: (id: string) => void }) {
  const parts = text.split(/(\[F\d+\](?:\s*\[F\d+\])*)/g)
  return (
    <>
      {parts.map((p, i) => {
        const ids = p.startsWith('[') ? p.match(/F\d+/g) : null
        if (!ids) return <Fragment key={i}>{p}</Fragment>
        return (
          <Fragment key={i}>
            {ids.map((id) => (
              <button key={id} className="cite mono" onClick={() => openFinding(id)} data-tip={`Open finding ${id}`}>
                {id}
              </button>
            ))}
          </Fragment>
        )
      })}
    </>
  )
}

// ---------------------------------------------------------------- 3. key moments

interface Cluster {
  t: number
  items: Finding[]
  sev: 'critical' | 'warning'
}

/** Only warning/critical moments, grouped when they fall within a second of each other. */
export function KeyMoments({ findings, duration }: { findings: Finding[]; duration: number }) {
  const pb = usePlayback()
  const [ref, size] = useElementSize<HTMLDivElement>()
  const dur = duration > 0 ? duration : 1
  const LABEL_W = 220
  const ROW_H = 54
  const placed = useMemo(() => {
    const sorted = findings
      .filter((f) => f.time_s != null && (f.severity === 'critical' || f.severity === 'warning'))
      .sort((a, b) => (a.time_s as number) - (b.time_s as number))
    const clusters: Cluster[] = []
    for (const f of sorted) {
      const last = clusters[clusters.length - 1]
      if (last && (f.time_s as number) - last.t <= Math.max(1, dur * 0.01)) {
        last.items.push(f)
        if (f.severity === 'critical') last.sev = 'critical'
      } else clusters.push({ t: f.time_s as number, items: [f], sev: f.severity === 'critical' ? 'critical' : 'warning' })
    }
    for (const c of clusters) c.items.sort((a) => (a.severity === 'critical' ? -1 : 1))
    const rowEnd: number[] = []
    return clusters.map((c) => {
      const x = (c.t / dur) * size.width
      const left = Math.min(Math.max(0, x - 10), Math.max(0, size.width - LABEL_W))
      let row = rowEnd.findIndex((end) => left > end + 8)
      if (row < 0) {
        row = rowEnd.length
        rowEnd.push(0)
      }
      rowEnd[row] = left + LABEL_W
      return { c, x, left, row }
    })
  }, [findings, dur, size.width])
  const rows = Math.max(0, ...placed.map((p) => p.row + 1))

  return (
    <div className="moments">
      <div className="moments-head">
        <span>Key moments</span>
        <span className="muted">{placed.length ? 'Click a moment to jump there' : 'Nothing suspicious is tied to a particular moment'}</span>
      </div>
      <div
        className="moments-track"
        ref={ref}
        onClick={(e) => {
          const r = e.currentTarget.getBoundingClientRect()
          pb.seek(((e.clientX - r.left) / r.width) * dur)
        }}
      >
        <div className="moments-line" />
        <MomentsHead dur={dur} />
        {placed.map(({ c, x }) => (
          <span key={c.t} className={`m-pin sev-${c.sev}`} style={{ left: x }} aria-hidden />
        ))}
        <span className="m-end faint">{mmss(duration)}</span>
        <span className="m-start faint">0:00</span>
      </div>
      {rows > 0 && (
        <div className="moments-labels" style={{ height: rows * ROW_H }}>
          {placed.map(({ c, left, row }) => (
            <button
              key={c.t}
              className={`moment sev-${c.sev}`}
              style={{ left, top: row * ROW_H, width: LABEL_W }}
              onClick={() => pb.seek(c.t)}
              data-tip={c.items.map((f) => f.title).join(' · ')}
            >
              <span className="m-time">
                <Icon name={SEVERITY[c.sev].icon} /> {SEVERITY[c.sev].word} · {mmss(c.t)}
              </span>
              <span className="m-title">
                {c.items[0].title}
                {c.items.length > 1 ? ` (+${c.items.length - 1})` : ''}
              </span>
            </button>
          ))}
        </div>
      )}
    </div>
  )
}

function MomentsHead({ dur }: { dur: number }) {
  const t = usePlayhead()
  return <div className="moments-ph" style={{ left: `${(Math.min(dur, t) / dur) * 100}%` }} />
}

// ---------------------------------------------------------------- 4. answers

const ANSWER_ICON: Record<string, string> = {
  edited: 'scissors',
  when: 'clock',
  where: 'location-dot',
  device: 'mobile-screen',
  people: 'user',
  hidden: 'eye-slash',
  speech: 'comment-dots',
  text: 'font',
}

function Confidence({ level }: { level: string }) {
  const c = CONFIDENCE[level] ?? CONFIDENCE.unknown
  return (
    <span className="confidence" data-tip={c.tip}>
      <span className="steps" aria-hidden>
        {[1, 2, 3].map((i) => (
          <i key={i} className={i <= c.level ? 'on' : ''} />
        ))}
      </span>
      {c.word}
    </span>
  )
}

function AnswerCard({ a, byId, openFinding }: { a: Answer; byId: Record<string, Finding>; openFinding: (id: string) => void }) {
  const pb = usePlayback()
  const [open, setOpen] = useState(false)
  const cited = (a.evidence ?? []).map((id) => byId[id]).filter(Boolean)
  return (
    <article className={`qa ${a.confidence === 'unknown' ? 'unknown' : ''}`}>
      <h3>
        <Icon name={ANSWER_ICON[a.key] ?? 'circle-question'} />
        {a.question}
      </h3>
      <p className="qa-answer">{a.answer}</p>
      <div className="qa-foot">
        <Confidence level={a.confidence} />
        {a.ai && (
          <span className="qa-ai" data-tip="Partly based on an AI model's reading of the video. Treat as a lead.">
            AI-assisted
          </span>
        )}
        {a.time_s != null && (
          <button className="link" onClick={() => pb.seek(a.time_s as number)}>
            Show me at {mmss(a.time_s)}
          </button>
        )}
        {cited.length > 0 && (
          <button className="link qa-how" onClick={() => setOpen((o) => !o)} aria-expanded={open}>
            How we know <Icon name={open ? 'chevron-up' : 'chevron-down'} />
          </button>
        )}
      </div>
      {open && (
        <ul className="qa-evidence">
          {cited.map((f) => (
            <li key={f.id}>
              <div className="qa-ev-head">
                <SevChip sev={f.severity} solid={false} />
                <b>{f.title}</b>
                <SourceChip kind={f.kind} />
              </div>
              <p className="muted">{f.detail}</p>
              <DataView data={f.data} />
              <div className="qa-ev-actions">
                {f.time_s != null && (
                  <button className="link" onClick={() => pb.seek(f.time_s as number)}>
                    Play from {mmss(f.time_s)}
                  </button>
                )}
                <button className="link" onClick={() => openFinding(f.id)}>
                  Open {f.id} in every finding
                </button>
              </div>
            </li>
          ))}
        </ul>
      )}
    </article>
  )
}

export function Answers({ state, facts, openFinding }: { state: CaseState; facts: Facts; openFinding: (id: string) => void }) {
  const byId: Record<string, Finding> = {}
  for (const f of state.findings) byId[f.id] = f
  const sorted = [...facts.answers].sort((a, b) => Number(a.confidence === 'unknown') - Number(b.confidence === 'unknown'))
  if (sorted.length)
    return (
      <div className="qa-grid">
        {sorted.map((a) => (
          <AnswerCard key={a.key} a={a} byId={byId} openFinding={openFinding} />
        ))}
      </div>
    )
  // older cases: the brief's bullets, in plain words
  const bullets: string[] = Array.isArray(facts.brief?.data?.bullets) ? facts.brief!.data.bullets : []
  if (!bullets.length) return <p className="muted">No summary was written for this case. Every finding is listed under Details below.</p>
  return (
    <div className="qa-grid">
      {bullets.map((b, i) => {
        const m = b.match(/^([A-Z][A-Z ]{1,20}):\s*(.*)$/s)
        return (
          <article className="qa" key={i}>
            <h3>{m ? (BULLET_LABEL[m[1]] ?? m[1]) : 'Note'}</h3>
            <p className="qa-answer">
              <Cited text={m ? m[2] : b} openFinding={openFinding} />
            </p>
          </article>
        )
      })}
    </div>
  )
}

// ---------------------------------------------------------------- 5. details

export function Detail({
  id,
  title,
  desc,
  count,
  open,
  onToggle,
  children,
}: {
  id: string
  title: string
  desc: string
  count?: ReactNode
  open: boolean
  onToggle: () => void
  children: ReactNode
}) {
  return (
    <section className={`detail ${open ? 'open' : ''}`} id={`d-${id}`}>
      <button className="detail-head" onClick={onToggle} aria-expanded={open}>
        <Icon name={open ? 'chevron-down' : 'chevron-right'} className="detail-chev" />
        <span className="detail-title">
          <b>{title}</b>
          <span className="muted">{desc}</span>
        </span>
        {count != null && <span className="detail-count">{count}</span>}
      </button>
      {open && <div className="detail-body">{children}</div>}
    </section>
  )
}

const TOPIC_ORDER: { key: TabKey; label: string }[] = [
  { key: 'timeline', label: 'Timing and cuts' },
  { key: 'sound', label: 'Sound' },
  { key: 'light', label: 'Light' },
  { key: 'picture', label: 'Picture' },
  { key: 'compression', label: 'Compression' },
  { key: 'hidden', label: 'Hidden data' },
  { key: 'people', label: 'People' },
  { key: 'place', label: 'Place' },
  { key: 'text', label: 'Text & speech' },
]

export function exhibitItems(state: CaseState, geoBoard = false): Artifact[] {
  return arts(state).filter(
    (a) => !['brief', 'map', 'dossier'].includes(a.kind) && artifactTab(a) !== 'file' && !(geoBoard && /^geo:(frame\d+|ref\d+_(wiki|sat)|leads|clues)$/.test(a.id)),
  )
}

export function Exhibits({ state, geoBoard = false }: { state: CaseState; geoBoard?: boolean }) {
  const all = exhibitItems(state, geoBoard)
  const ofKind = (k: string) => arts(state).filter((a) => a.kind === k)
  return (
    <>
      {TOPIC_ORDER.map(({ key, label }) => {
        const items = all.filter((a) => artifactTab(a) === key && !['filmstrip', 'waveform', 'markers'].includes(a.kind))
        const intro = TABS.find((t) => t.key === key)?.intro
        const showTimeline = key === 'timeline' && (ofKind('filmstrip').length || ofKind('waveform').length)
        if (!items.length && !showTimeline) return null
        return (
          <div key={key} className="topic" id={`topic-${key}`}>
            <h3 className="topic-title">{label}</h3>
            {intro && <p className="muted">{intro}</p>}
            {showTimeline ? (
              <div className="exhibit">
                <p className="explain">Thumbnails, sound level, keyframes and every flagged moment on one line. Click or drag to move the video.</p>
                <Timeline
                  duration={state.info?.duration_s ?? 0}
                  filmstrips={ofKind('filmstrip')}
                  waveforms={ofKind('waveform')}
                  markers={ofKind('markers')}
                  findings={state.findings}
                />
              </div>
            ) : null}
            <ArtifactGroup items={items} modules={state.modules} />
          </div>
        )
      })}
    </>
  )
}

/** Every location we know: GPS from the file, map artifacts, AI estimates. De-duplicated. */
export function mapPoints(state: CaseState, facts: Facts): MapPoint[] {
  const out: MapPoint[] = []
  const add = (p: MapPoint) => {
    const cls = (k?: string) => (k === 'ai' || k === 'inferred' ? 'ai' : k === 'ai-candidate' ? 'lead' : 'gps')
    if (!out.some((q) => Math.abs(q.lat - p.lat) < 1e-4 && Math.abs(q.lon - p.lon) < 1e-4 && cls(q.kind) === cls(p.kind))) out.push(p)
  }
  if (facts.gps) add({ lat: facts.gps.lat, lon: facts.gps.lon, accuracy_m: facts.gps.accuracy_m, label: 'GPS stored in the file', kind: 'gps', heading_deg: facts.gps.heading_deg ?? null })
  for (const a of arts(state).filter((x) => x.kind === 'map')) for (const p of a.data?.points ?? []) if (hasLocation(p?.lat, p?.lon, ['ai', 'inferred', 'ai-candidate'].includes(p?.kind) ? (p.accuracy_m == null ? undefined : p.accuracy_m / 1000) : undefined)) add(p)
  for (const f of state.findings) {
    const lat = f.data?.latitude
    const lon = f.data?.longitude
    if (hasLocation(lat, lon, f.kind === 'inferred' ? f.data?.radius_km : undefined))
      add({ lat, lon, accuracy_m: f.data?.radius_km ? f.data.radius_km * 1000 : undefined, label: f.title, kind: f.kind === 'inferred' ? 'ai' : 'measured' })
  }
  return out
}

export function MapSection({ points, gps, legendOnly }: { points: MapPoint[]; gps: Facts['gps']; legendOnly?: boolean }) {
  return (
    <>
      {gps && (
        <p className="map-note">
          <Icon name="location-crosshairs" className="c-red-i" /> This file carries its own GPS position{gps.place ? `: ${gps.place}` : ''}. Anyone with the
          file can see it.{typeof gps.heading_deg === 'number' ? ` The red wedge shows which way the camera pointed (${Math.round(gps.heading_deg)}°).` : ''}
        </p>
      )}
      <p className="muted small">
        Red: GPS from the file. Purple, dashed: the answer from the picture. Grey dots: other places that were considered. Circles show uncertainty; a
        yellow line joins GPS and answer.
      </p>
      <div className="map-box">
        {!legendOnly && <MapView points={points} height={420} />}
        <table className="dtable map-legend">
          <tbody>
            {points.map((p, i) => (
              <tr key={i}>
                <td className="place">
                  <span className={`dot-k ${p.kind === 'ai-candidate' ? 'lead' : p.kind === 'ai' || p.kind === 'inferred' ? 'ai' : ''}`} /> {p.label}
                </td>
                <td className="mono num coord">
                  {p.lat.toFixed(5)}, {p.lon.toFixed(5)} <CopyButton text={`${p.lat}, ${p.lon}`} label="Copy coordinates" />
                </td>
                <td className="mono num coord">{radiusText(p.accuracy_m)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  )
}

export function NotesSection({ facts, openFinding }: { facts: Facts; openFinding: (id: string) => void }) {
  const d = facts.brief?.data ?? {}
  return (
    <div className="notes">
      {d.model && <p className="faint small">Written with {d.model}. AI-written: verify against the cited findings.</p>}
      {d.headline && (
        <p className="headline">
          <Cited text={d.headline} openFinding={openFinding} />
        </p>
      )}
      {d.narrative && (
        <p>
          <Cited text={d.narrative} openFinding={openFinding} />
        </p>
      )}
      {d.text &&
        String(d.text)
          .split(/\n{2,}/)
          .map((p: string, i: number) => (
            <p key={i}>
              <Cited text={p} openFinding={openFinding} />
            </p>
          ))}
      {Array.isArray(d.next_steps) && d.next_steps.length > 0 && (
        <>
          <h4>Suggested next steps</h4>
          <ol>
            {d.next_steps.map((s: string, i: number) => (
              <li key={i}>
                <Cited text={s} openFinding={openFinding} />
              </li>
            ))}
          </ol>
        </>
      )}
    </div>
  )
}

export function hasNotes(facts: Facts) {
  const d = facts.brief?.data ?? {}
  return !!(d.headline || d.narrative || d.text || (Array.isArray(d.next_steps) && d.next_steps.length))
}

function TagTable({ tags }: { tags: Record<string, string | number> }) {
  const [q, setQ] = useState('')
  const entries = Object.entries(tags).filter(([k, v]) => !q || `${k} ${v}`.toLowerCase().includes(q.toLowerCase()))
  return (
    <>
      <label className="search-box">
        <Icon name="magnifying-glass" />
        <input className="input" placeholder={`Search ${Object.keys(tags).length} labels`} value={q} onChange={(e) => setQ(e.target.value)} />
      </label>
      <table className="dtable">
        <tbody>
          {entries.map(([k, v]) => (
            <tr key={k}>
              <td className="key mono">{k}</td>
              <td className="pre">{String(v)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </>
  )
}

export function EvidenceFile({ state, facts, caseId }: { state: CaseState; facts: Facts; caseId: string }) {
  const info = state.info!
  const v = info.video
  const im = info.image
  const still = facts.kind === 'image'
  const fileArts = arts(state).filter((a) => artifactTab(a) === 'file')
  const facts2: [string, ReactNode, string?][] = [
    ['Case number', <span className="mono">{caseId}</span>],
    ['Original file name', info.filename],
    ['Size', `${bytes(info.size_bytes)} (${info.size_bytes?.toLocaleString('en-GB')} bytes)`],
    ['Received', info.ingested_at ? niceDate(new Date(info.ingested_at * 1000).toISOString()) : 'Not recorded'],
    ...(still
      ? ([
          ['Picture', im ? `${im.width} × ${im.height} px (${im.megapixels} megapixels), ${im.format}, ${im.mode} ${im.bits}-bit${im.frames > 1 ? `, ${im.frames} frames` : ''}${im.orientation !== 1 ? `, stored rotated (orientation ${im.orientation})` : ''}` : 'Not decoded'],
          ['JPEG quality', facts.quality != null ? `about ${facts.quality} (libjpeg scale)` : im?.format === 'JPEG' || im?.format === 'MPO' ? 'Not estimated' : 'Not a JPEG', GLOSSARY.quantisation],
          ['Faces', facts.faces ? `${facts.faces.total} found, ${facts.faces.recognisable} recognisable` : 'None found'],
          ['Declared generator', facts.generator ?? 'None'],
        ] as [string, ReactNode, string?][])
      : ([
          ['Duration', mmss(info.duration_s)],
          ['Picture', v ? `${v.width} × ${v.height}, ${Math.round(v.fps * 100) / 100} fps, ${v.codec.toUpperCase()}, ${v.frames?.toLocaleString('en-GB')} frames` : 'None', GLOSSARY.codec],
          ['Sound', info.has_audio ? 'Yes' : 'None'],
        ] as [string, ReactNode, string?][])),
    ['Container', info.container?.format_long_name ?? info.container?.format_name ?? 'Not recorded', GLOSSARY.container],
    ['Device', facts.device ? `${facts.device}${facts.software ? ` (software ${facts.software})` : ''}` : 'Not stated in the file'],
    [still ? 'Taken (claimed)' : 'Recorded (claimed)', facts.recorded ? niceDate(facts.recorded) : 'Not stated', 'From the file’s own clock, which can be wrong or edited.'],
    ['GPS', facts.gps ? `${facts.gps.lat.toFixed(5)}, ${facts.gps.lon.toFixed(5)}${facts.gps.accuracy_m ? ` (${radiusText(facts.gps.accuracy_m)})` : ''}${typeof facts.gps.heading_deg === 'number' ? `, camera pointed ${Math.round(facts.gps.heading_deg)}°` : ''}` : 'None'],
    ['Re-saved by', facts.resaved ?? 'No sign of re-saving software'],
    ...(still
      ? ([] as [string, ReactNode, string?][])
      : ([
          ['Power grid', facts.grid ? `${facts.grid} Hz` : 'Not detected', GLOSSARY.ENF],
          ['Heart rate', facts.pulse ? `about ${facts.pulse} bpm` : 'Not measured', GLOSSARY.rPPG],
          ['Speech', facts.language ?? 'None found'],
        ] as [string, ReactNode, string?][])),
    ['AI used', facts.aiUsed ? 'Yes; AI results are labelled' : 'No'],
  ]
  return (
    <>
      <h4 className="sub">The file</h4>
      <dl className="stats">
        {facts2.map(([k, val, tip]) => (
          <div key={k}>
            <dt data-tip={tip}>{k}</dt>
            <dd>{val}</dd>
          </div>
        ))}
      </dl>
      <h4 className="sub">Fingerprints</h4>
      <p className="muted small">Use these to prove exactly which file was examined.</p>
      <dl className="stats">
        {(['sha256', 'md5'] as const).map((k) => (
          <div key={k}>
            <dt>
              <Term k={k === 'sha256' ? 'SHA-256' : 'MD5'} />
            </dt>
            <dd className="hash">
              {info[k] ? (
                <>
                  <span className="mono">{info[k]}</span> <CopyButton text={info[k]!} label={`Copy ${k}`} />
                </>
              ) : (
                <span className="faint">Not recorded for this case</span>
              )}
            </dd>
          </div>
        ))}
      </dl>
      <h4 className="sub">Who has handled it</h4>
      <p className="muted small">Every upload and re-run, by the signed-in account that asked for it. The fingerprints above show the file itself never changed.</p>
      {info.history && info.history.length > 0 ? (
        <ol className="custody">
          {info.history.map((h, i) => (
            <li key={i}>
              <span className="mono faint">{dateShort(h.at)}</span>{' '}
              <b>{h.action === 'uploaded' ? 'Uploaded' : h.action === 'reran' ? (h.modules?.length ? `Re-ran ${h.modules.join(', ')}` : 'Re-ran every check') : h.action}</b>{' '}
              by {h.by?.name ?? 'a signed-in user'}
              {h.action === 'reran' && h.ai_provider ? <span className="faint"> (AI: {h.ai_provider})</span> : null}
            </li>
          ))}
        </ol>
      ) : (
        <p className="faint small">Not recorded: this case was uploaded before logins were required.</p>
      )}
      {info.container && (
        <>
          <h4 className="sub">Container</h4>
          <DataView data={info.container as Record<string, unknown>} />
        </>
      )}
      {info.streams && info.streams.length > 0 && (
        <>
          <h4 className="sub">Tracks</h4>
          <div className="streams">
            {info.streams.map((s, i) => (
              <div className="stream" key={i}>
                <b>
                  Track {s.index ?? i}: {s.type ?? 'data'}
                </b>
                <DataView data={s as Record<string, unknown>} />
              </div>
            ))}
          </div>
        </>
      )}
      {info.tags && Object.keys(info.tags).length > 0 && (
        <>
          <h4 className="sub">Every label stored in the file</h4>
          <TagTable tags={info.tags} />
        </>
      )}
      {fileArts.length > 0 && (
        <>
          <h4 className="sub">Raw readouts</h4>
          <ArtifactGroup items={fileArts} modules={state.modules} />
        </>
      )}
    </>
  )
}

function statusText(m: ModuleState): { text: string; cls: string } {
  switch (m.status) {
    case 'done':
      return { text: 'Done', cls: 'ok' }
    case 'skipped':
      return { text: `Skipped${m.message ? `: ${m.message}` : ''}`, cls: 'skip' }
    case 'error':
      return { text: `Failed${m.message ? `: ${m.message}` : ''}`, cls: 'err' }
    case 'running':
      return { text: `Running ${Math.round(m.progress * 100)}%${m.message ? ` · ${m.message}` : ''}`, cls: 'run' }
    case 'waiting':
      return { text: m.message || 'Waiting for another check', cls: 'wait' }
    default:
      return { text: 'Queued', cls: 'wait' }
  }
}

export function MethodSection({ state, facts }: { state: CaseState; facts: Facts }) {
  const [trace, setTrace] = useState<string | null>(null)
  const all = arts(state)
  return (
    <>
      <p className="muted">
        Each check runs on its own. The verdict is only “Edited” when independent checks (for example picture <i>and</i> sound) point to the same
        moment. Labels inside the file count as weak evidence because they can be changed.
      </p>
      <table className="lb method">
        <thead>
          <tr>
            <th>Check</th>
            <th>What it does</th>
            <th>Status</th>
            <th className="num">Time</th>
            <th className="num">Findings</th>
            <th className="num">Exhibits</th>
          </tr>
        </thead>
        <tbody>
          {state.moduleOrder.map((n) => {
            const m = state.modules[n]
            if (!m) return null
            const p = modulePlain(n, m.title)
            const s = statusText(m)
            return (
              <tr key={n} className={m.trace ? 'lb-row' : ''} onClick={() => m.trace && setTrace(trace === n ? null : n)}>
                <td>
                  <b>{p.name}</b>
                  <div className="faint small mono">{m.title}</div>
                </td>
                <td>
                  {p.does}
                  {p.limits && <div className="faint small">Limits: {p.limits}</div>}
                  {trace === n && m.trace && <pre className="trace">{m.trace}</pre>}
                </td>
                <td className={`status st-${s.cls}`}>
                  {s.text}
                  {m.trace && <div className="faint small">Click for the error trace</div>}
                </td>
                <td className="num mono nowrap">{m.elapsed_s != null ? `${m.elapsed_s.toFixed(2)} s` : '—'}</td>
                <td className="num mono">{state.findings.filter((f) => f.module === n).length}</td>
                <td className="num mono">{all.filter((a) => a.module === n).length}</td>
              </tr>
            )
          })}
        </tbody>
      </table>
      <h4 className="sub">Activity log</h4>
      <ol className="log">
        {state.logs.map((l) => (
          <li key={l.seq} className={l.level}>
            <span className="mono faint">{new Date(l.ts * 1000).toLocaleTimeString('en-GB')}</span>
            <span className="mono log-mod">{l.module}</span>
            <span>{l.message}</span>
          </li>
        ))}
      </ol>
      <h4 className="sub">Raw case record</h4>
      <div className="json-box">
        <JsonTree value={state.info} name="case" />
        <JsonTree value={facts.brief?.data ?? null} name="brief" />
        <JsonTree value={state.findings} name="findings" />
        <JsonTree value={all.map((a) => ({ id: a.id, kind: a.kind, title: a.title, url: a.url, time_s: a.time_s }))} name="artifacts" />
      </div>
    </>
  )
}
