import { useCallback, useEffect, useMemo, useState } from 'react'
import { useCaseStream } from '../../lib/useCaseStream'
import { Playback, PlaybackContext } from '../../lib/playback'
import type { CaseState } from '../../lib/state'
import { isTerminal } from '../../lib/state'
import { api } from '../../lib/api'
import { mmss } from '../../lib/format'
import { deriveFacts, type Facts } from '../../lib/facts'
import { useHealth } from '../../lib/health'
import { modulePlain, VERDICT } from '../../lib/plain'
import { mockName } from '../../lib/mock'
import { Badge, Icon } from '../ui/ui'
import { Player } from './Player'
import { FindingsBoard } from './Findings'
import { GeoBoard, geoFinding } from './GeoBoard'
import { Reveal } from './Reveal'
import { Answers, Detail, EvidenceFile, Exhibits, exhibitItems, hasNotes, KeyMoments, MapSection, mapPoints, MethodSection, NotesSection } from './Sections'

export type SectionKey = 'findings' | 'exhibits' | 'geo' | 'map' | 'notes' | 'file' | 'method'
export const SECTIONS: SectionKey[] = ['findings', 'exhibits', 'geo', 'map', 'notes', 'file', 'method']

/** Old tab links keep working: they open the matching section. */
export function sectionFor(key: string | undefined): { section: SectionKey | null; topic?: string } {
  if (!key || key === 'overview') return { section: null }
  if ((SECTIONS as string[]).includes(key)) return { section: key as SectionKey }
  if (key === 'place') return { section: 'map', topic: 'place' }
  if (['timeline', 'sound', 'light', 'picture', 'people', 'text'].includes(key)) return { section: 'exhibits', topic: key }
  return { section: null }
}

export function CasePage({ caseId, section, onSection }: { caseId: string; section: string | undefined; onSection: (s: SectionKey | null) => void }) {
  const [nonce, setNonce] = useState(0)
  const { state, conn } = useCaseStream(caseId, nonce)
  const pb = useMemo(() => new Playback(), [])
  const { health } = useHealth()
  const facts = useMemo(() => deriveFacts(state, health?.openai ?? null), [state, health?.openai])
  const [open, setOpen] = useState<Set<SectionKey>>(new Set())
  const [focus, setFocus] = useState<string | null>(null)
  const info = state.info
  const points = useMemo(() => mapPoints(state, facts), [state, facts])
  const locationAnswered = facts.answers.some((a) => a.key === 'where' && a.confidence !== 'unknown') || !!facts.gps
  const [mapDefaulted, setMapDefaulted] = useState(false)
  const geo = geoFinding(state)
  const dossier = state.artifacts['brief:dossier'] ?? state.artifactOrder.map((id) => state.artifacts[id]).find((a) => a.kind === 'dossier') ?? null
  const [revealing, setRevealing] = useState(false)

  useEffect(() => {
    if (info?.duration_s) pb.set({ duration: info.duration_s })
  }, [pb, info?.duration_s])
  useEffect(() => {
    document.title = info ? `${VERDICT[facts.verdict].short} · ${info.filename} · EARWITNESS` : 'EARWITNESS'
  }, [info, facts.verdict])

  // The map opens by itself only when the case has a location answer.
  // The geolocation board opens by itself whenever there is a geo finding.
  useEffect(() => {
    if (mapDefaulted || facts.running) return
    if (geo) {
      setMapDefaulted(true)
      setOpen((o) => new Set(o).add('geo'))
    } else if (points.length) {
      setMapDefaulted(true)
      if (locationAnswered) setOpen((o) => new Set(o).add('map'))
    }
  }, [points.length, locationAnswered, facts.running, mapDefaulted, !!geo])

  // Deep link: #/case/{id}/{section} opens and scrolls to that item.
  useEffect(() => {
    if (!info) return
    const { section: s, topic } = sectionFor(section)
    const target = (s === 'map' || s === 'geo') && geo ? 'geo' : s === 'map' && !points.length ? 'exhibits' : s
    if (!target) return
    setOpen((o) => new Set(o).add(target))
    const t = setTimeout(() => {
      const el = (topic && document.getElementById(`topic-${topic}`)) || document.getElementById(`d-${target}`)
      el?.scrollIntoView({ behavior: 'smooth', block: 'start' })
    }, 60)
    return () => clearTimeout(t)
  }, [section, !!info, points.length, !!geo])

  const toggle = (s: SectionKey) =>
    setOpen((o) => {
      const n = new Set(o)
      if (n.has(s)) n.delete(s)
      else n.add(s)
      return n
    })

  const openFinding = useCallback(
    (id: string) => {
      setOpen((o) => new Set(o).add('findings'))
      setFocus(null)
      requestAnimationFrame(() => setFocus(id))
      onSection('findings')
    },
    [onSection],
  )

  const arts = state.artifactOrder.map((id) => state.artifacts[id])
  const ofKind = (k: string) => arts.filter((a) => a.kind === k)
  const exhibits = exhibitItems(state, !!geo)
  const notes = hasNotes(facts)

  return (
    <PlaybackContext.Provider value={pb}>
      <CaseHead
        state={state}
        facts={facts}
        caseId={caseId}
        onRerun={async () => {
          await api.rerun(caseId)
          setNonce((n) => n + 1)
        }}
      />
      <main className="wrap case-page">
        {state.error ? (
          <p className="verdict-line err-line">This case could not be opened: {state.error}</p>
        ) : !info ? (
          <div className="skeleton" aria-label="Loading the case">
            <span />
            <span />
            <span className="tall" />
          </div>
        ) : (
          <>
            <VerdictLine state={state} facts={facts} conn={conn} />
            {dossier && (
              <button className="reveal-btn" onClick={() => setRevealing(true)}>
                <Icon name="eye" /> What this video gives away
              </button>
            )}

            <section className="block">
              <div className="video-col">
                <div className="video-frame">
                  <Player
                    info={info}
                    mediaUrl={state.mediaUrl}
                    mediaReady={state.mediaReady}
                    running={facts.running}
                    heatmaps={ofKind('heatmap')}
                    captions={ofKind('captions')}
                    height={480}
                  />
                </div>
                <KeyMoments findings={state.findings} duration={info.duration_s} />
              </div>
            </section>

            <section className="block">
              <h2 className="block-title">What we found</h2>
              {facts.running && !facts.answers.length ? (
                <p className="muted">The answers are written once every check has finished.</p>
              ) : (
                <Answers state={state} facts={facts} openFinding={openFinding} />
              )}
            </section>

            <section className="block">
              <h2 className="block-title">Details</h2>
              <div className="details">
                <Detail
                  id="findings"
                  title="Every finding"
                  desc="Everything every check reported, most serious first, with all its measurements."
                  count={state.findings.length}
                  open={open.has('findings')}
                  onToggle={() => toggle('findings')}
                >
                  <FindingsBoard findings={state.findings} modules={state.modules} focus={focus} />
                </Detail>
                <Detail
                  id="exhibits"
                  title="Charts and exhibits"
                  desc="Spectrograms, charts, tamper maps and frames, grouped by topic, each with how to read it."
                  count={exhibits.length}
                  open={open.has('exhibits')}
                  onToggle={() => toggle('exhibits')}
                >
                  <Exhibits state={state} geoBoard={!!geo} />
                </Detail>
                {geo && (
                  <Detail
                    id="geo"
                    title="Where it was filmed"
                    desc="Places the AI proposed from the pixels and sounds, each checked against real-world photos and satellite views."
                    count={geo.data?.verified ? 'verified' : 'lead'}
                    open={open.has('geo')}
                    onToggle={() => toggle('geo')}
                  >
                    <GeoBoard state={state} facts={facts} points={points} />
                  </Detail>
                )}
                {!geo && points.length > 0 && (
                  <Detail
                    id="map"
                    title="Map"
                    desc={facts.gps ? 'The GPS position stored in the file, and any estimate from the picture.' : 'Where the picture suggests it was filmed.'}
                    count={points.length}
                    open={open.has('map')}
                    onToggle={() => toggle('map')}
                  >
                    <MapSection points={points} gps={facts.gps} />
                  </Detail>
                )}
                {notes && (
                  <Detail
                    id="notes"
                    title="Analyst’s note"
                    desc="The written assessment, with links to the findings it cites."
                    open={open.has('notes')}
                    onToggle={() => toggle('notes')}
                  >
                    <NotesSection facts={facts} openFinding={openFinding} />
                  </Detail>
                )}
                <Detail
                  id="file"
                  title="Evidence file"
                  desc="Fingerprints, file facts, tracks, every stored label and the raw technical readout."
                  open={open.has('file')}
                  onToggle={() => toggle('file')}
                >
                  <EvidenceFile state={state} facts={facts} caseId={caseId} />
                </Detail>
                <Detail
                  id="method"
                  title="How each check ran"
                  desc="Every check, what it does, its limits, status and timing, plus the activity log."
                  count={`${facts.checksDone}/${facts.checksTotal}`}
                  open={open.has('method')}
                  onToggle={() => toggle('method')}
                >
                  <MethodSection state={state} facts={facts} />
                </Detail>
              </div>
            </section>
          </>
        )}
      </main>
      {revealing && dossier && (
        <Reveal
          state={state}
          dossier={dossier}
          caseId={caseId}
          onClose={(jump) => {
            setRevealing(false)
            if (jump) openFinding(jump)
          }}
        />
      )}
    </PlaybackContext.Provider>
  )
}

function Seal({ facts }: { facts: Facts }) {
  const v = VERDICT[facts.verdict]
  return (
    <div className={`seal v-${facts.verdict}`} data-tip={v.line}>
      <span className="seal-icon">{facts.verdict === 'running' ? <i className="pulse-dot big" /> : <Icon name={v.icon} />}</span>
      <span className="seal-text">
        <b>{facts.verdict === 'running' ? 'Analysing' : v.word}</b>
        {facts.verdict === 'running' && (
          <small>
            {facts.checksDone} of {facts.checksTotal} checks
          </small>
        )}
      </span>
    </div>
  )
}

function CaseHead({ state, facts, caseId, onRerun }: { state: CaseState; facts: Facts; caseId: string; onRerun: () => void }) {
  const info = state.info
  return (
    <section className={`case-head v-${facts.verdict}`}>
      {info?.thumb_url && <div className="cover" style={{ backgroundImage: `url(${info.thumb_url})` }} />}
      <div className="cover-tint" />
      <div className="wrap head-top">
        {info?.thumb_url ? <img className="avatar" src={info.thumb_url} alt="" /> : <div className="avatar" />}
        <div className="who-block">
          <a href="#/" className="case-label mono" data-tip="Back to all cases">
            <Icon name="arrow-left" /> CASE {caseId}
          </a>
          <h1 className={`name v-${facts.verdict}`} title={info?.filename}>
            {info?.filename ?? 'Opening…'}
          </h1>
          {info && (
            <div className="badges">
              <Badge icon="mobile-screen" accent={facts.device ? 'blue' : 'grey'}>
                {facts.device ?? 'Unknown device'}
              </Badge>
              <Badge icon="clock" accent="teal">
                {mmss(info.duration_s)}
              </Badge>
              <Badge icon="location-dot" accent={facts.gps ? 'red' : 'grey'} tip={facts.gps ? 'The file stores its own GPS position' : 'No GPS position in the file'}>
                {facts.gps ? 'GPS in file' : 'No GPS'}
              </Badge>
            </div>
          )}
        </div>
        <div className="head-actions">
          <Seal facts={facts} />
          <div className="head-icons">
            <a className="icon-btn" href={api.reportUrl(caseId)} target="_blank" rel="noreferrer" data-tip="Printable report" aria-label="Report">
              <Icon name="file-lines" />
            </a>
            {!mockName() && (
              <button className="icon-btn" onClick={onRerun} disabled={facts.running} data-tip="Run every check again" aria-label="Rerun">
                <Icon name="rotate-right" />
              </button>
            )}
          </div>
        </div>
      </div>
    </section>
  )
}

function VerdictLine({ state, facts, conn }: { state: CaseState; facts: Facts; conn: string }) {
  if (facts.running) {
    const m = facts.currentModule
    const phrase = m ? modulePlain(m, state.modules[m]?.title).does.split(/[.:]/)[0] : 'Opening the file'
    return (
      <section className="verdict-block">
        <p className="verdict-line">Examining this video…</p>
        <div className="run-line">
          <span className="muted">
            {phrase}… {facts.checksDone} of {facts.checksTotal} checks done
          </span>
          <div className="progress">
            <span style={{ width: `${facts.checksTotal ? (facts.checksDone / facts.checksTotal) * 100 : 0}%` }} />
          </div>
        </div>
      </section>
    )
  }
  const flagged = facts.counts.critical + facts.counts.warning
  const flaggedModules = new Set(state.findings.filter((f) => f.severity === 'critical' || f.severity === 'warning').map((f) => f.module))
  const passed = state.moduleOrder.filter((n) => state.modules[n]?.status === 'done' && !flaggedModules.has(n)).length
  const unfinished = state.moduleOrder.filter((n) => !isTerminal(state.modules[n]?.status ?? 'queued')).length
  return (
    <section className="verdict-block">
      <p className={`verdict-line v-${facts.verdict}`}>{facts.summary ?? VERDICT[facts.verdict].line}</p>
      <p className="muted">
        {flagged} {flagged === 1 ? 'thing' : 'things'} worth checking · {passed} {passed === 1 ? 'check' : 'checks'} passed
        {unfinished && conn !== 'live' ? ` · ${unfinished} unfinished` : ''}
      </p>
    </section>
  )
}
