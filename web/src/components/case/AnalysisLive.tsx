import { useEffect, useState } from 'react'
import type { CaseState, ModuleState } from '../../lib/state'
import type { Facts } from '../../lib/facts'
import { moduleDoing, modulePlain } from '../../lib/plain'
import { Icon } from '../ui/ui'

const clock = (secs: number) => `${Math.floor(secs / 60)}:${String(Math.floor(secs % 60)).padStart(2, '0')}`
const hms = (ts: number) => new Date(ts * 1000).toLocaleTimeString('en-GB', { hour12: false })

function stepState(m: ModuleState): { label: string; icon: string } {
  switch (m.status) {
    case 'done':
      return { label: m.elapsed_s != null ? `${m.elapsed_s.toFixed(1)} s` : 'done', icon: 'check' }
    case 'running':
      return { label: m.progress > 0 && m.progress < 1 ? `${Math.round(m.progress * 100)}%` : 'working', icon: 'spinner' }
    case 'waiting':
      return { label: 'waiting', icon: 'hourglass-half' }
    case 'skipped':
      return { label: 'not needed', icon: 'minus' }
    case 'error':
      return { label: 'failed', icon: 'xmark' }
    default:
      return { label: 'queued', icon: 'circle' }
  }
}

/**
 * What the server is doing right now, while a case runs: the current step in
 * plain words, every check's state, and the raw activity feed underneath.
 * Before the first event it covers the opening stage (hashing, transcoding).
 */
export function AnalysisLive({ state, facts }: { state: CaseState; facts: Facts }) {
  const still = facts.kind === 'image'
  const [mounted] = useState(() => Date.now() / 1000)
  const [now, setNow] = useState(() => Date.now() / 1000)
  useEffect(() => {
    const t = setInterval(() => setNow(Date.now() / 1000), 1000)
    return () => clearInterval(t)
  }, [])

  const mods = state.moduleOrder.map((n) => state.modules[n]).filter(Boolean)
  const running = mods.filter((m) => m.status === 'running').sort((a, b) => (b.started_ts ?? 0) - (a.started_ts ?? 0))
  const lead = running[0] ?? null
  const opening = !state.info
  const allDone = mods.length > 0 && facts.checksDone >= facts.checksTotal

  // Server clock when it is plausible (a live run), else time since this page opened.
  const starts = mods.map((m) => m.started_ts).filter((t): t is number => t != null)
  const serverStart = starts.length ? Math.min(...starts) : null
  const start = serverStart != null && now - serverStart >= 0 && now - serverStart < 3 * 3600 ? serverStart : mounted
  const elapsed = Math.max(0, now - start)

  const headline = opening
    ? 'Opening the evidence'
    : lead
      ? moduleDoing(lead.name, still, lead.title)
      : allDone
        ? 'Finishing up'
        : 'Waiting for a free analyst'
  const detail = opening
    ? `Fingerprinting the file and making a copy every browser can ${still ? 'show' : 'play'}.`
    : lead?.message || modulePlain(lead?.name ?? '', lead?.title).does
  const also = running.slice(1).map((m) => modulePlain(m.name, m.title).name)
  const feed = state.logs.slice(-5)

  return (
    <section className="live" aria-live="polite">
      <div className="live-top mono">
        <span className="live-rec">
          <i className="pulse-dot" /> Analysing
        </span>
        <span>{clock(elapsed)} elapsed</span>
        {!opening && (
          <span>
            {facts.checksDone} of {facts.checksTotal} checks
          </span>
        )}
      </div>

      <h2 className="live-head" key={headline}>
        {headline}
        <span className="live-dots" aria-hidden>
          <i />
          <i />
          <i />
        </span>
      </h2>
      <p className="live-detail">
        {lead && <Icon name={modulePlain(lead.name, lead.title).icon} />} {detail}
        {also.length > 0 && <span className="faint"> · also running: {also.join(', ')}</span>}
      </p>

      <div className="live-bar" role="progressbar" aria-valuemin={0} aria-valuemax={facts.checksTotal} aria-valuenow={facts.checksDone}>
        {opening ? (
          <span className="seg running" style={{ flex: 1 }}>
            <i style={{ width: '100%' }} />
          </span>
        ) : (
          mods.map((m) => (
            <span key={m.name} className={`seg ${m.status}`} data-tip={`${modulePlain(m.name, m.title).name}: ${stepState(m).label}`}>
              <i style={{ width: m.status === 'running' ? `${Math.max(8, m.progress * 100)}%` : undefined }} />
            </span>
          ))
        )}
      </div>

      {!opening && (
        <ol className="live-steps">
          {mods.map((m) => {
            const p = modulePlain(m.name, m.title)
            const st = stepState(m)
            return (
              <li key={m.name} className={m.status} data-tip={m.status === 'waiting' && m.message ? m.message : p.does}>
                <Icon name={p.icon} className="live-step-icon" />
                <span className="live-step-name">{p.name}</span>
                <span className="live-step-state mono">
                  <Icon name={st.icon} className={m.status === 'running' ? 'fa-spin' : ''} /> {st.label}
                </span>
              </li>
            )
          })}
        </ol>
      )}

      {feed.length > 0 && (
        <div className="live-feed mono" aria-hidden>
          {feed.map((l) => (
            <div key={l.seq} className={l.level}>
              <span className="faint">{hms(l.ts)}</span> <b>{l.module}</b> {l.message}
            </div>
          ))}
          <span className="live-cursor" />
        </div>
      )}
    </section>
  )
}
