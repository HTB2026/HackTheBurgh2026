import { useCallback, useEffect, useMemo, useState } from 'react'
import { useCaseStream } from '../../lib/useCaseStream'
import { Playback, PlaybackContext, usePlayback } from '../../lib/playback'
import { MediaKindContext, RegionContext, RegionFocus, useRegionFocus } from '../../lib/region'
import { api } from '../../lib/api'
import { mmss } from '../../lib/format'
import { deriveFacts, rankFindings, type Facts } from '../../lib/facts'
import { useHealth } from '../../lib/health'
import { mediaNoun, modulePlain, VERDICT } from '../../lib/plain'
import type { CaseState } from '../../lib/state'
import { boxOf, type Finding, type Artifact } from '../../lib/types'
import { Icon, SourceChip } from '../ui/ui'
import { DataView } from '../ui/Data'
import { Player } from './Player'
import { PhotoStage } from './PhotoStage'
import { FindingsBoard } from './Findings'
import { ArtifactSection } from './Artifacts'
import { GeoBoard, geoFinding } from './GeoBoard'
import { Reveal } from './Reveal'
import { EvidenceFile, Exhibits, MapSection, mapPoints, MethodSection, NotesSection } from './Sections'

export type SectionKey = 'findings' | 'exhibits' | 'geo' | 'map' | 'notes' | 'file' | 'method'
export const SECTIONS: SectionKey[] = ['findings', 'exhibits', 'geo', 'map', 'notes', 'file', 'method']
export function sectionFor(key: string | undefined): { section: SectionKey | null; topic?: string } {
  if (!key || key === 'overview') return { section: null }
  if ((SECTIONS as string[]).includes(key)) return { section: key as SectionKey }
  if (key === 'place') return { section: 'map', topic: 'place' }
  if (['timeline', 'sound', 'light', 'picture', 'compression', 'hidden', 'people', 'text'].includes(key)) return { section: 'exhibits', topic: key }
  return { section: null }
}
type Tab = 'findings' | 'exhibits' | 'location' | 'file' | 'checks'
const tabFor = (s: SectionKey): Tab => s === 'geo' || s === 'map' ? 'location' : s === 'method' ? 'checks' : s === 'notes' ? 'findings' : s

// Group nearby cut and sound-break observations as one event, while leaving unrelated claims distinct.
function eventTopic(f: Finding): string {
  const text = `${f.title} ${f.tags?.join(' ')} ${f.module}`.toLowerCase()
  if (/cut|splice|break|discontinuit|missing frames|duplicate frames|room.tone/.test(text)) return 'edit-event'
  if (/synthetic|generator|generated|c2pa/.test(text)) return 'generator'
  if (/keypad/.test(text)) return 'keypad'
  if (/ultrasonic|high.frequency tone/.test(text)) return 'tone'
  if (/hidden|appended|payload/.test(text)) return 'hidden'
  if (/location|geolocation|\bgps\b/.test(text)) return 'location'
  if (/grid|\benf\b|clock|dated/.test(text)) return 'time'
  return f.module
}
function keyFindings(findings: Finding[]): Finding[][] {
  const eligible = rankFindings(findings).filter((f) => {
    const t = `${f.title} ${f.tags?.join(' ')}`.toLowerCase()
    if (/no (sign|signal|evidence|detect)|not detect|no .*found|check complete/.test(t)) return false
    if (f.module === 'pulse' || (f.module === 'provenance' && !/editor|history|hidden|serial|owner/.test(t))) return false
    return f.severity === 'critical' || f.severity === 'warning' || /hidden|appended|payload|synthetic|generator|text|transcript|speech|recognis|keypad|tone/.test(t)
  })
  const groups: Finding[][] = []
  for (const f of eligible) {
    const topic = eventTopic(f)
    const group = groups.find((g) => eventTopic(g[0]) === topic && (topic !== 'edit-event' || (f.time_s != null && g[0].time_s != null && Math.abs(f.time_s - g[0].time_s) <= 2)))
    if (group) group.push(f)
    else groups.push([f])
  }
  groups.sort((a, b) => {
    const score = (g: Finding[]) => (eventTopic(g[0]) === 'edit-event' && g.length > 1 ? 10 : 0) + (eventTopic(g[0]) === 'generator' ? 9 : 0) + (g[0].severity === 'critical' ? 6 : g[0].severity === 'warning' ? 4 : 1)
    return score(b) - score(a)
  })
  return groups.slice(0, 3)
}
function relatedArtifacts(group: Finding[], artifacts: Artifact[]): Artifact[] {
  const modules = [...new Set(group.map((f) => f.module))]
  const candidates = artifacts.filter((a) => modules.includes(a.module) && !['brief', 'dossier', 'markers', 'waveform', 'filmstrip'].includes(a.kind) && (a.kind !== 'json' || eventTopic(group[0]) === 'generator') && (a.time_s == null || group.some((f) => f.time_s != null && Math.abs(a.time_s! - f.time_s) <= 3)))
  const priority = (a: Artifact) => /difference|noise_floor/.test(a.name) ? 0 : 1
  return modules.flatMap((module) => candidates.filter((a) => a.module === module).sort((a, b) => priority(a) - priority(b)).slice(0, 1)).slice(0, 3)
}
function KeyFinding({ group, artifacts, state, onOpen }: { group: Finding[]; artifacts: Artifact[]; state: CaseState; onOpen: (id: string) => void }) {
  const [open, setOpen] = useState(false)
  const pb = usePlayback()
  const rf = useRegionFocus()
  const f = group[0]
  const related = relatedArtifacts(group, artifacts)
  const combined = eventTopic(f) === 'edit-event' && group.some((item) => item.module === 'timeline') && group.some((item) => item.module === 'spectrum')
  const qualifier = typeof f.data?.confidence === 'string' ? f.data.confidence : null
  const caveat = f.detail.split(/(?<=[.!?])\s/).find((sentence) => /\b(weak|uncertain|estimate|limited|may not|cannot|not proof)\b/i.test(sentence) && !/synthetic preview/i.test(sentence))
  return <article className="key-finding">
    <div className="key-finding-title"><b>{combined ? `Picture and sound both change at ${mmss(f.time_s ?? 0)}` : f.title}</b><SourceChip kind={f.kind} /></div>
    {(qualifier || caveat) && <p>{qualifier && `Confidence: ${qualifier}.`}{qualifier && caveat && ' '}{caveat}</p>}
    <button className="link small" onClick={() => setOpen(!open)} aria-expanded={open}>{open ? 'Hide evidence' : 'Show evidence'}</button>
    {open && <div className="key-evidence">
      {group.map((item) => <div key={item.id} className="key-evidence-item"><p>{item.detail}</p><div className="evidence-actions">
        {item.time_s != null && state.info?.kind !== 'image' && <button className="timebtn" onClick={() => pb.seek(item.time_s!)}><Icon name="play" /> {mmss(item.time_s)}</button>}
        {boxOf(item) && <button className="timebtn" onClick={() => rf.focus(boxOf(item)!, item.title)}><Icon name="vector-square" /> View region</button>}
        <button className="link small" onClick={() => onOpen(item.id)}>Finding {item.id}</button>
      </div><details><summary>Measurements</summary><DataView data={item.data} /></details></div>)}
      {related.map((a) => <ArtifactSection key={a.id} a={a} modules={state.modules} />)}
    </div>}
  </article>
}
export function CasePage({ caseId, section, onSection }: { caseId: string; section: string | undefined; onSection: (s: SectionKey | null) => void }) {
  const [nonce, setNonce] = useState(0)
  const { state, conn } = useCaseStream(caseId, nonce)
  const pb = useMemo(() => new Playback(), [])
  const rf = useMemo(() => new RegionFocus(), [])
  const { health } = useHealth()
  const facts = useMemo(() => deriveFacts(state, health?.openai ?? null), [state, health?.openai])
  const [technical, setTechnical] = useState(false)
  const [tab, setTab] = useState<Tab>('findings')
  const [focus, setFocus] = useState<string | null>(null)
  const [revealing, setRevealing] = useState(false)
  const info = state.info
  const still = facts.kind === 'image'
  const points = useMemo(() => mapPoints(state, facts), [state, facts])
  const geo = geoFinding(state)
  const arts = state.artifactOrder.map((id) => state.artifacts[id])
  const dossier = state.artifacts['brief:dossier'] ?? arts.find((a) => a.kind === 'dossier') ?? null
  const groups = useMemo(() => keyFindings(state.findings), [state.findings])
  useEffect(() => {
    const expand = (event: Event) => setTechnical(!!(event as CustomEvent<boolean>).detail)
    const replay = () => setNonce((n) => n + 1)
    window.addEventListener('mock-sections', expand); window.addEventListener('mock-replay', replay)
    return () => { window.removeEventListener('mock-sections', expand); window.removeEventListener('mock-replay', replay) }
  }, [])
  useEffect(() => { if (info?.duration_s) pb.set({ duration: info.duration_s }) }, [pb, info?.duration_s])
  useEffect(() => { document.title = info ? `${VERDICT[facts.verdict].short} · ${info.filename} · EARWITNESS` : 'EARWITNESS' }, [info, facts.verdict])
  useEffect(() => {
    if (!info) return
    const { section: target, topic } = sectionFor(section)
    if (!target) return
    setTechnical(true); setTab(tabFor(target))
    const t = setTimeout(() => (topic && document.getElementById(`topic-${topic}`) || document.getElementById('technical-details'))?.scrollIntoView({ behavior: 'smooth', block: 'start' }), 100)
    return () => clearTimeout(t)
  }, [section, !!info])
  const openFinding = useCallback((id: string) => { setTechnical(true); setTab('findings'); setFocus(null); requestAnimationFrame(() => setFocus(id)); onSection('findings') }, [onSection])
  const incomplete = state.moduleOrder.filter((n) => ['error', 'skipped'].includes(state.modules[n]?.status)).length
  const when = facts.answers.find((a) => a.key === 'when' && a.confidence !== 'unknown')
  const where = facts.answers.find((a) => a.key === 'where' && a.confidence !== 'unknown')
  const device = facts.answers.find((a) => a.key === 'device' && a.confidence !== 'unknown')
  return <PlaybackContext.Provider value={pb}><RegionContext.Provider value={rf}><MediaKindContext.Provider value={facts.kind}>
    <section className="case-head"><div className="wrap head-top"><div className="who-block"><a href="#/" className="case-label"><Icon name="arrow-left" /> Back to cases</a><h1 className="name" title={info?.filename}>{info?.filename ?? 'Opening…'}</h1>{info && <span className="muted small">{still ? 'Photo' : 'Video'}{!still && ` · ${mmss(info.duration_s)}`}</span>}</div><div className="head-icons">
      {dossier && <button className="icon-btn" onClick={() => setRevealing(true)} aria-label="Open presentation" data-tip="Presentation"><Icon name="eye" /></button>}
      <a className="icon-btn" href={api.reportUrl(caseId)} target="_blank" rel="noreferrer" aria-label="Report" data-tip="Printable report"><Icon name="file-lines" /></a>
      {info?.original_url && <a className="icon-btn" href={info.original_url} target="_blank" rel="noreferrer" aria-label="Original file" data-tip="Original file"><Icon name="download" /></a>}
      <button className="icon-btn" onClick={async () => { await api.rerun(caseId); setNonce((n) => n + 1) }} disabled={facts.running} aria-label="Rerun" data-tip="Run checks again"><Icon name="rotate-right" /></button>
    </div></div></section>
    <main className="wrap case-page">{state.error ? <p className="err-line">This case could not be opened: {state.error}</p> : !info ? <div className="skeleton" aria-label="Loading the case"><span /><span /><span className="tall" /></div> : <>
      <VerdictLine state={state} facts={facts} conn={conn} incomplete={incomplete} />
      <div className="investigation"><div className="video-col"><div className="video-frame">{still ? <PhotoStage info={info} mediaUrl={state.mediaUrl} running={facts.running} overlays={arts.filter((a) => a.kind === 'heatmap' && a.url && a.data?.overlay !== false)} findings={state.findings} height={600} onOpenFinding={openFinding} /> : <Player info={info} mediaUrl={state.mediaUrl} mediaReady={state.mediaReady} running={facts.running} heatmaps={arts.filter((a) => a.kind === 'heatmap')} captions={arts.filter((a) => a.kind === 'captions')} height={480} />}</div></div>
      <section className="key-findings"><h2>Key findings</h2>{groups.map((g) => <KeyFinding key={g[0].id} group={g} artifacts={arts} state={state} onOpen={openFinding} />)}{!groups.length && <p className="muted">{facts.running ? 'Findings will appear as checks finish.' : 'No key findings to show.'}</p>}</section></div>
      {(when || where || device || facts.recorded || facts.gps || facts.device) && <section className="fact-strip" aria-label="File facts">
        {(when || facts.recorded) && <div><b>When</b><span>{when ? `${when.answer}${when.ai ? ' · AI estimate' : ''}` : `${facts.recorded} · file clock`}</span></div>}
        {(where || facts.gps) && <div><b>Where</b><span>{where ? `${where.answer}${where.ai ? ' · AI estimate' : ''}` : `${facts.gps?.place ?? `${facts.gps?.lat}, ${facts.gps?.lon}`} · GPS metadata`}</span></div>}
        {(device || facts.device) && <div><b>Device</b><span>{device ? `${device.answer}${device.ai ? ' · AI estimate' : ''}` : `${facts.device} · file metadata`}</span></div>}
      </section>}
      <section className="technical" id="technical-details"><button className="technical-toggle" onClick={() => setTechnical(!technical)} aria-expanded={technical}><span>Technical details</span><Icon name={technical ? 'chevron-up' : 'chevron-down'} /></button>{technical && <><div className="tabs technical-tabs" role="tablist">{(['findings', 'exhibits', ...((geo || points.length) ? ['location'] : []), 'file', 'checks'] as Tab[]).map((t) => <button key={t} role="tab" aria-selected={tab === t} className={tab === t ? 'active' : ''} onClick={() => setTab(t)}>{t === 'checks' ? 'Checks' : t[0].toUpperCase() + t.slice(1)}</button>)}</div><div className="technical-panel" role="tabpanel">
        {tab === 'findings' && <><FindingsBoard findings={state.findings} modules={state.modules} focus={focus} />{section === 'notes' && <NotesSection facts={facts} openFinding={openFinding} />}</>}
        {tab === 'exhibits' && <Exhibits state={state} geoBoard={!!geo} />}
        {tab === 'location' && (geo ? <GeoBoard state={state} facts={facts} points={points} /> : <MapSection points={points} gps={facts.gps} />)}
        {tab === 'file' && <EvidenceFile state={state} facts={facts} caseId={caseId} />}
        {tab === 'checks' && <MethodSection state={state} facts={facts} />}
      </div></>}</section>
    </>}</main>
    {revealing && dossier && <Reveal state={state} dossier={dossier} caseId={caseId} onClose={(jump) => { setRevealing(false); if (jump) openFinding(jump) }} />}
  </MediaKindContext.Provider></RegionContext.Provider></PlaybackContext.Provider>
}
function VerdictLine({ state, facts, conn, incomplete }: { state: CaseState; facts: Facts; conn: string; incomplete: number }) {
  if (facts.running) {
    const m = facts.currentModule
    const phrase = m ? modulePlain(m, state.modules[m]?.title).does.split(/[.:]/)[0] : `Opening the ${mediaNoun(facts.kind)}`
    return <section className="verdict-block"><p className="verdict-line">Analysing · {phrase}… {facts.checksDone} of {facts.checksTotal} checks</p></section>
  }
  const unfinished = state.moduleOrder.filter((n) => ['queued', 'waiting', 'running'].includes(state.modules[n]?.status)).length
  return <section className="verdict-block"><p className="verdict-line">{!(facts.summary ?? VERDICT[facts.verdict].line).toLowerCase().startsWith(VERDICT[facts.verdict].word.toLowerCase()) && <b className={`v-${facts.verdict}`}>{VERDICT[facts.verdict].word}</b>} <span>{facts.summary ?? VERDICT[facts.verdict].line}</span></p>{(incomplete > 0 || (unfinished > 0 && conn !== 'live')) && <p className="muted small">Coverage incomplete: {incomplete > 0 && `${incomplete} failed or skipped`}{incomplete > 0 && unfinished > 0 && conn !== 'live' && ' · '}{unfinished > 0 && conn !== 'live' && `${unfinished} unfinished`}</p>}</section>
}
