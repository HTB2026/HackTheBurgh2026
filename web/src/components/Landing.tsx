import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { api, uploadEvidence } from '../lib/api'
import { useHealth } from '../lib/health'
import { AiPicker, aiChoices } from './AiPicker'
import type { AiProvider } from '../lib/types'
import type { CaseSummary } from '../lib/types'
import { ago, dateShort, mmss } from '../lib/format'
import { useCaseMeta } from '../lib/caseMeta'
import { VERDICT, verdictOf, type Verdict } from '../lib/plain'
import { Alert, Empty, Icon, Redacted, SectionTitle } from './ui/ui'

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
        <p className={`card-place ${c.place ? `conf-${c.place_confidence}` : 'none'}`} data-tip={c.place ? `Where it was taken (${c.place_confidence} evidence)` : undefined}>
          <Icon name="location-dot" /> {c.place ?? (c.status === 'done' ? 'Location unknown' : 'Locating…')}
        </p>
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
        </div>
      </div>
    </article>
  )
}

export function Landing({ examine }: { examine?: boolean }) {
  const { health } = useHealth()
  const [aiProvider, setAiProvider] = useState<AiProvider | null>(() => {
    try { return (localStorage.getItem('earwitness.ai') as AiProvider | null) ?? null } catch { return null }
  })
  const choices = aiChoices(health)
  const aiChosen = aiProvider && choices.includes(aiProvider) ? aiProvider : (health?.ai?.default ?? null)
  const pickAi = (provider: AiProvider) => {
    setAiProvider(provider)
    try { localStorage.setItem('earwitness.ai', provider) } catch { /* Private mode may block storage. */ }
  }
  const [cases, setCases] = useState<CaseSummary[] | null>(null)
  // Older backends need their events fetched to determine the verdict.
  const needMeta = useMemo(() => (cases ? cases.filter((c) => c.verdict === undefined) : null), [cases])
  const meta = useCaseMeta(needMeta)
  const verdictFor = (c: CaseSummary): Verdict | undefined =>
    c.status !== 'done' ? 'running' : c.verdict !== undefined ? (c.verdict ? verdictOf(c.verdict) : 'unknown') : meta[c.id]?.verdict
  const [over, setOver] = useState(false)
  const [upload, setUpload] = useState<{ name: string; p: number } | null>(null)
  const [err, setErr] = useState<string | null>(null)
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
        const s = await uploadEvidence(file, null, aiChosen, (p) => setUpload({ name: file.name, p }))
        location.hash = `#/case/${s.id}`
      } catch (e) {
        setErr(String((e as Error).message ?? e))
        setUpload(null)
      }
    },
    [aiChosen],
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

  const remove = async (c: CaseSummary) => {
    if (!confirm(`Delete the case for ${c.filename}? Its copy of the file and all results are removed. This cannot be undone.`)) return
    await api.remove(c.id).catch(() => null)
    refresh()
  }

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

      <section className="home-intro">
        <h1>Every photo and video is a witness.</h1>
        <p>Drop in a video or photograph to examine what it can reveal.</p>
      </section>
      <div className="home">
        <div className="home-main">
          <AiPicker health={health} value={aiChosen} onChange={pickAi} />
          <div
            ref={dropRef}
            className={`dropzone ${upload ? 'busy' : ''}`}
            role="button"
            tabIndex={0}
            onClick={() => !upload && fileInput.current?.click()}
            aria-disabled={!!upload}
            onKeyDown={(e) => {
              if (!upload && (e.key === 'Enter' || e.key === ' ')) {
                e.preventDefault()
                fileInput.current?.click()
              }
            }}
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

      </div>
    </main>
  )
}
