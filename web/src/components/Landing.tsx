import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { api, uploadEvidence } from '../lib/api'
import { AI_HINT, AI_LABEL, aiChoices } from './AiPicker'
import type { AiProvider } from '../lib/types'
import type { CaseSummary } from '../lib/types'
import { ago, dateShort, mmss } from '../lib/format'
import { useHealth } from '../lib/health'
import { useCaseMeta } from '../lib/caseMeta'
import { modulePlain, VERDICT, verdictOf, type Verdict } from '../lib/plain'
import { Alert, Empty, Icon, Redacted, SectionTitle } from './ui/ui'

function toIso(local: string): string | null {
  if (!local) return null
  const d = new Date(local)
  return isNaN(d.getTime()) ? null : d.toISOString()
}

/** Waveform silhouette along the hero's bottom edge. */
function HeroWave({ peaks }: { peaks: number[] | null }) {
  const pts = useMemo(() => {
    const N = 240
    if (peaks && peaks.length > 8) {
      const max = Math.max(1e-6, ...peaks)
      return Array.from({ length: N }, (_, i) => {
        const a = Math.floor((i / N) * peaks.length)
        const b = Math.max(a + 1, Math.floor(((i + 1) / N) * peaks.length))
        let p = 0
        for (let j = a; j < b; j++) p = Math.max(p, peaks[j] ?? 0)
        return 0.12 + 0.88 * Math.sqrt(p / max)
      })
    }
    // no case yet: a synthetic heartbeat-like trace
    return Array.from({ length: N }, (_, i) => 0.18 + 0.12 * Math.sin(i / 5) + (i % 37 === 0 ? 0.6 : 0) + 0.08 * Math.sin(i * 1.7))
  }, [peaks])
  const H = 40
  const W = 1000
  const d = `M0 ${H} ` + pts.map((p, i) => `L${((i / (pts.length - 1)) * W).toFixed(1)} ${(H - p * H * 0.95).toFixed(1)}`).join(' ') + ` L${W} ${H} Z`
  return (
    <div className="hero-wave" aria-hidden>
      <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none">
        <path d={d} />
      </svg>
    </div>
  )
}

function VerdictPill({ v }: { v: Verdict | undefined }) {
  const key = v ?? 'unknown'
  const d = VERDICT[key]
  return (
    <span className={`pill v-${key}`}>
      {key === 'running' ? <i className="pulse-dot" /> : <Icon name={d.icon} />} {v ? d.short : '…'}
    </span>
  )
}

function CaseCard({ c, verdict: v, summary, onDelete }: { c: CaseSummary; verdict?: Verdict; summary: string | null; onDelete: () => void }) {
  return (
    <article className={`card case-card v-${v ?? 'unknown'}`}>
      <a className="card-img" href={`#/case/${c.id}`}>
        {c.case?.thumb_url ? <img src={c.case.thumb_url} alt="" loading="lazy" /> : <div className="noimg" />}
        <VerdictPill v={v} />
      </a>
      <a className="card-play" href={`#/case/${c.id}`} aria-label={`Open ${c.filename}`} data-tip="Open the case file">
        <Icon name="eye" />
      </a>
      <div className="card-body">
        <h4 title={c.filename}>
          <a href={`#/case/${c.id}`}>{c.filename}</a>
        </h4>
        <p className="muted small mono">{c.id}</p>
        {summary && <p className="card-summary">{summary}</p>}
        <div className="card-meta">
          {c.case?.image ? (
            <span data-tip="Picture size">
              <Icon name="image" /> {c.case.image.megapixels} MP
            </span>
          ) : c.case?.duration_s ? (
            <span data-tip="Length">
              <Icon name="clock" /> {mmss(c.case.duration_s)}
            </span>
          ) : null}
          <span data-tip={`Examined ${dateShort(c.created)}`}>
            <Icon name="calendar" /> {ago(c.created)}
          </span>
          <span data-tip="Findings">
            <Icon name="list" /> {c.findings}
          </span>
          <span className={c.warnings ? 'hot' : ''} data-tip="Flagged as warning or critical">
            <Icon name="triangle-exclamation" /> {c.warnings}
          </span>
          <span className="grow" />
          <button
            className="iconbtn"
            onClick={(e) => {
              e.preventDefault()
              onDelete()
            }}
            data-tip="Delete this case"
            aria-label="Delete case"
          >
            <Icon name="trash-can" />
          </button>
          <a className="btn ghost sm" href={api.reportUrl(c.id)} target="_blank" rel="noreferrer" data-tip="Printable report" aria-label="Report">
            <Icon name="file-lines" />
          </a>
        </div>
      </div>
    </article>
  )
}

export function Landing({ examine }: { examine?: boolean }) {
  const { health, offline } = useHealth()
  const [cases, setCases] = useState<CaseSummary[] | null>(null)
  // Newer backends put the verdict in the list; only older cases (and the hero waveform) need their events fetched.
  const needMeta = useMemo(() => (cases ? cases.filter((c, i) => i === 0 || c.verdict === undefined) : null), [cases])
  const meta = useCaseMeta(needMeta)
  const verdictFor = (c: CaseSummary): Verdict | undefined =>
    c.status !== 'done' ? 'running' : c.verdict !== undefined ? (c.verdict ? verdictOf(c.verdict) : 'unknown') : meta[c.id]?.verdict
  const [approx, setApprox] = useState('')
  const [aiProvider, setAiProvider] = useState<AiProvider | null>(() => {
    try {
      return (localStorage.getItem('earwitness.ai') as AiProvider | null) ?? null
    } catch {
      return null
    }
  })
  const choices = aiChoices(health)
  // The stored choice only counts if this server still offers it.
  const aiChosen = aiProvider && choices.includes(aiProvider) ? aiProvider : (health?.ai?.default ?? null)
  const pickAi = (p: AiProvider) => {
    setAiProvider(p)
    try {
      localStorage.setItem('earwitness.ai', p)
    } catch {
      /* private mode: the choice just isn't remembered */
    }
  }
  const [path, setPath] = useState('')
  const [over, setOver] = useState(false)
  const [upload, setUpload] = useState<{ name: string; p: number } | null>(null)
  const [err, setErr] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const fileInput = useRef<HTMLInputElement>(null)
  const dropRef = useRef<HTMLDivElement>(null)

  const refresh = useCallback(() => {
    api.cases().then(setCases).catch(() => setCases((c) => c ?? []))
  }, [])

  useEffect(() => {
    refresh()
    const id = setInterval(refresh, 5000)
    return () => clearInterval(id)
  }, [refresh])

  const send = useCallback(
    async (file: File) => {
      setErr(null)
      setUpload({ name: file.name, p: 0 })
      try {
        const s = await uploadEvidence(file, toIso(approx), aiChosen, (p) => setUpload({ name: file.name, p }))
        location.hash = `#/case/${s.id}`
      } catch (e) {
        setErr(String((e as Error).message ?? e))
        setUpload(null)
      }
    },
    [approx, aiChosen],
  )

  // the whole page accepts a dropped file
  useEffect(() => {
    const over = (e: DragEvent) => {
      e.preventDefault()
      setOver(true)
    }
    const leave = (e: DragEvent) => {
      if (!e.relatedTarget) setOver(false)
    }
    const drop = (e: DragEvent) => {
      e.preventDefault()
      setOver(false)
      const f = e.dataTransfer?.files?.[0]
      if (f) send(f)
    }
    window.addEventListener('dragover', over)
    window.addEventListener('dragleave', leave)
    window.addEventListener('drop', drop)
    return () => {
      window.removeEventListener('dragover', over)
      window.removeEventListener('dragleave', leave)
      window.removeEventListener('drop', drop)
    }
  }, [send])

  useEffect(() => {
    if (!examine) return
    dropRef.current?.scrollIntoView({ behavior: 'smooth', block: 'center' })
    dropRef.current?.classList.add('flash')
    const t = setTimeout(() => dropRef.current?.classList.remove('flash'), 1600)
    return () => clearTimeout(t)
  }, [examine])

  const openPath = async () => {
    if (!path.trim()) return
    setErr(null)
    setBusy(true)
    try {
      const iso = toIso(approx)
      const s = await api.fromPath(path.trim(), { ...(iso ? { approx_time: iso } : {}), ...(aiChosen ? { ai_provider: aiChosen } : {}) })
      location.hash = `#/case/${s.id}`
    } catch (e) {
      setErr(`Could not open ${path.trim()}: ${(e as Error).message}`)
    } finally {
      setBusy(false)
    }
  }

  const remove = async (c: CaseSummary) => {
    if (!confirm(`Delete the case for ${c.filename}? Its copy of the file and all results are removed. This cannot be undone.`)) return
    await api.remove(c.id).catch(() => null)
    refresh()
  }

  const latest = cases?.[0]
  const latestPeaks = latest ? (meta[latest.id]?.peaks ?? null) : null
  const known = cases ? cases.map(verdictFor).filter(Boolean) : []
  const edits = known.filter((v) => v === 'edited').length
  const suspect = known.filter((v) => v === 'possible').length

  return (
    <main className="wrap home-page">
      {over && (
        <div className="drop-overlay">
          <Icon name="file-video" />
          <b>Drop to examine</b>
        </div>
      )}
      <input
        ref={fileInput}
        type="file"
        accept="video/*,audio/*,image/*,.mov,.mp4,.mkv,.webm,.avi"
        hidden
        onChange={(e) => {
          const f = e.target.files?.[0]
          if (f) send(f)
          e.target.value = ''
        }}
      />

      <section className="hero" style={latest?.case?.thumb_url ? { backgroundImage: `url(${latest.case.thumb_url})` } : undefined}>
        <div className="hero-inner">
          <div className="hero-text">
            <h1>Every photo and video is a witness.</h1>
            <p>
              Drop in any video or photograph. We'll tell you where it was taken, when, on what, who is in it, and what the file isn't telling you.
            </p>
            <div className="hero-actions">
              <button className="btn btn-red" onClick={() => fileInput.current?.click()} disabled={!!upload}>
                <Icon name="magnifying-glass" /> Examine a video or photo
              </button>
              <a className="btn dark" href="#/how">
                <Icon name="circle-question" /> How it works
              </a>
            </div>
          </div>
        </div>
        {latest && (
          <span className="hero-last">
            Last examined {ago(latest.created)}: <b>{latest.filename}</b>
          </span>
        )}
        <HeroWave peaks={latestPeaks} />
      </section>
      <div className="counters">
        <div className="c-blue" data-tip="Every video examined on this machine">
          <Icon name="folder-open" />
          <b>{cases ? cases.length : '…'}</b>
          Cases examined
        </div>
        <div className="c-red" data-tip="Cases whose verdict is Edited (independent checks agree)">
          <Icon name="scissors" />
          <b>
            {cases ? edits : '…'}
            {suspect ? <small> + {suspect} suspect</small> : null}
          </b>
          Edits caught
        </div>
        <div className="c-teal" data-tip="Independent checks: one set for video, one for photographs">
          <Icon name="list-check" />
          <b>
            {health ? health.modules.length : offline ? '—' : '…'}
            {health?.image_modules ? <small> + {health.image_modules.length} for photos</small> : null}
          </b>
          Checks per video
        </div>
      </div>

      <div className="home">
        <div className="home-main">
          <SectionTitle icon="file-video" accent="red" small="Video, audio or a photograph (JPEG, HEIC, PNG, WebP, TIFF). Checks start straight away.">
            Examine a video or photo
          </SectionTitle>
          {choices.length > 1 && (
            <div className="ai-choice" role="radiogroup" aria-label="AI model">
              <span className="ai-choice-label">
                <Icon name="robot" /> AI model
              </span>
              <div className="tabs ai-tabs">
                {choices.map((p) => (
                  <button
                    key={p}
                    type="button"
                    role="radio"
                    aria-checked={aiChosen === p}
                    className={aiChosen === p ? 'active' : ''}
                    disabled={!!upload}
                    data-tip={AI_HINT[p]}
                    onClick={() => pickAi(p)}
                  >
                    {AI_LABEL[p] ?? p}
                  </button>
                ))}
              </div>
              <small className="faint">Runs speech, sounds, the visual analyst, geolocation and the written summary.</small>
            </div>
          )}
          <div
            ref={dropRef}
            className={`dropzone ${upload ? 'busy' : ''}`}
            role="button"
            tabIndex={0}
            onClick={() => !upload && fileInput.current?.click()}
            onKeyDown={(e) => (e.key === 'Enter' || e.key === ' ') && fileInput.current?.click()}
          >
            {upload ? (
              <div className="upload">
                <b>
                  Uploading {upload.name}… {(upload.p * 100).toFixed(0)}%
                </b>
                <div className="progress">
                  <span style={{ width: `${upload.p * 100}%` }} />
                </div>
              </div>
            ) : (
              <>
                <Icon name="cloud-arrow-up" className="drop-icon" />
                <div>
                  <b>Drop a file here, or click to choose one</b>
                  <span className="muted">Most clips take under a minute. You can watch the checks run.</span>
                </div>
              </>
            )}
          </div>
          <details className="more-options">
            <summary>
              <Icon name="sliders" /> More options: approximate date, open a file already on this computer
            </summary>
            <div className="options-grid">
              <label>
                <span>Roughly when was it filmed? (optional)</span>
                <input className="input" type="datetime-local" step="60" value={approx} onChange={(e) => setApprox(e.target.value)} />
                <small className="faint">Helps the power-grid check search the right days. Your local time zone.</small>
              </label>
              <label>
                <span>Path to a file on the server</span>
                <div className="row">
                  <input className="input" placeholder="/Users/…/IMG_6335.MOV" value={path} onChange={(e) => setPath(e.target.value)} onKeyDown={(e) => e.key === 'Enter' && openPath()} />
                  <button className="btn btn-blue sm" disabled={!path.trim() || busy} onClick={openPath}>
                    {busy ? 'Opening…' : 'Examine'}
                  </button>
                </div>
                <small className="faint">For large files: no upload needed.</small>
              </label>
            </div>
          </details>
          {err && (
            <Alert icon="triangle-exclamation" tone="red" title="That did not work">
              {err}
            </Alert>
          )}

          <SectionTitle icon="box-archive" accent="blue" small={cases ? `${cases.length} on file` : undefined}>
            Recent cases
          </SectionTitle>
          {cases === null ? (
            <div className="cards">
              {[0, 1, 2].map((i) => (
                <div className="card" key={i}>
                  <div className="card-img" />
                  <div className="card-body">
                    <Redacted lines={3} />
                  </div>
                </div>
              ))}
            </div>
          ) : cases.length === 0 ? (
            <Empty icon="folder-open">No cases yet. Examine a video or photo to open the first file.</Empty>
          ) : (
            <div className="cards">
              {cases.map((c) => (
                <CaseCard key={c.id} c={c} verdict={verdictFor(c)} summary={c.summary_plain ?? meta[c.id]?.summary ?? null} onDelete={() => remove(c)} />
              ))}
            </div>
          )}
        </div>

        <aside className="home-side">
          <SectionTitle icon="flag-checkered" accent="green">
            Getting started
          </SectionTitle>
          <ul className="links panel c-green">
            <li>
              <button onClick={() => fileInput.current?.click()}>
                Examine your first file <small>MP4, MOV, JPEG, HEIC, PNG…</small>
              </button>
            </li>
            <li>
              <a href="#/how">
                What each check does <small>plain English</small>
              </a>
            </li>
            <li>
              <a href="#/how#verdict">
                How to read a verdict <small>strong vs weak</small>
              </a>
            </li>
            <li>
              <a href="#/how#limits">
                What it cannot prove <small>read this</small>
              </a>
            </li>
          </ul>

          <SectionTitle icon="eye" accent="red">
            What it can detect
          </SectionTitle>
          <ul className="detect panel c-red">
            {[...(health?.modules ?? []), ...(health?.image_modules ?? []).filter((m) => !(health?.modules ?? []).some((v) => v.name === m.name && v.title === m.title))].map((m) => {
              const p = modulePlain(m.name, m.title)
              return (
                <li key={`${m.kind ?? 'video'}:${m.name}`} className={`c-${p.accent}`} data-tip={p.does}>
                  <Icon name={p.icon} />
                  <div>
                    <b>
                      {p.name}
                      {m.kind === 'image' && <small className="kind-tag">photo</small>}
                    </b>
                    <span className="muted small">{p.reveals || m.description}</span>
                  </div>
                </li>
              )
            })}
            {!health && (
              <li>
                <Redacted lines={4} />
              </li>
            )}
          </ul>
        </aside>
      </div>
    </main>
  )
}
