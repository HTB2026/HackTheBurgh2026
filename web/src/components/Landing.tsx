import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { api, uploadEvidence } from '../lib/api'
import { useHealth } from '../lib/health'
import { canChange, ownerName, useAuth } from '../lib/auth'
import { FeedFilter, isFiltered, matches, useFeedFilter } from './FeedFilter'
import { AI_EXPERIMENTAL, AI_HINT, AI_LABEL, aiChoices } from './AiPicker'
import { EFFORT_HINT, EFFORT_LABEL, EffortTabs, storedEffort } from './EffortPicker'
import type { AiProvider, Effort } from '../lib/types'
import type { CaseSummary } from '../lib/types'
import { ago, dateShort, mmss } from '../lib/format'
import { useCaseMeta } from '../lib/caseMeta'
import { HeroField } from './HeroField'
import { VERDICT, verdictOf, type Verdict } from '../lib/plain'
import { Alert, Empty, Icon, Redacted, SectionTitle } from './ui/ui'

/** Clipboard images all arrive as "image.png"; give each a findable, dated name. */
function pastedName(f: File): File {
  if (f.name && f.name !== 'image.png' && f.name !== 'image.jpeg') return f
  const d = new Date()
  const p = (n: number) => String(n).padStart(2, '0')
  const stamp = `${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}-${p(d.getHours())}${p(d.getMinutes())}${p(d.getSeconds())}`
  const ext = f.type.split('/')[1]?.replace('jpeg', 'jpg') || 'png'
  return new File([f], `pasted-${stamp}.${ext}`, { type: f.type, lastModified: f.lastModified })
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
  const auth = useAuth()
  const may = canChange(auth, c.owner)
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
          {c.effort && c.effort !== 'medium' && (
            <span data-tip={`${EFFORT_LABEL[c.effort]} effort: ${EFFORT_HINT[c.effort]}`}>
              <Icon name="gauge-high" /> {EFFORT_LABEL[c.effort]}
            </span>
          )}
          {c.owner && (
            <span className="card-owner" data-tip={`Uploaded by ${ownerName(c.owner)}`} aria-label={`Uploaded by ${ownerName(c.owner)}`}>
              {c.owner.picture ? <img src={c.owner.picture} alt="" referrerPolicy="no-referrer" /> : <Icon name="circle-user" />}
            </span>
          )}
          <span className="grow" />
          <button
            className="iconbtn"
            disabled={may === false}
            onClick={(e) => {
              e.preventDefault()
              onDelete()
            }}
            data-tip={
              may === false
                ? `Only ${ownerName(c.owner)}, who uploaded it, can delete this case`
                : may === 'login'
                  ? 'Log in to delete this case'
                  : 'Delete this case'
            }
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
  const [effort, setEffort] = useState<Effort>(storedEffort)
  const pickEffort = (e: Effort) => {
    setEffort(e)
    try { localStorage.setItem('earwitness.effort', e) } catch { /* Private mode may block storage. */ }
  }
  const [cases, setCases] = useState<CaseSummary[] | null>(null)
  // Older backends need their events fetched to determine the verdict.
  const needMeta = useMemo(() => (cases ? cases.filter((c) => c.verdict === undefined) : null), [cases])
  const meta = useCaseMeta(needMeta)
  // The newest finished recording shapes the homepage particle field.
  const heroCase = useMemo(() => (cases ? cases.filter((c) => c.status === 'done' && c.kind !== 'image').slice(0, 1) : null), [cases])
  const heroMeta = useCaseMeta(heroCase, 1)
  const heroPeaks = heroCase?.[0] ? (heroMeta[heroCase[0].id]?.peaks ?? null) : null
  const verdictFor = (c: CaseSummary): Verdict | undefined =>
    c.status !== 'done' ? 'running' : c.verdict !== undefined ? (c.verdict ? verdictOf(c.verdict) : 'unknown') : meta[c.id]?.verdict
  const auth = useAuth()
  const [filter, setFilter] = useFeedFilter()
  const shown = useMemo(
    () => (cases ? cases.filter((c) => matches(c, verdictFor(c), c.summary_plain ?? meta[c.id]?.summary ?? null, filter, auth.me?.id ?? null)) : null),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [cases, meta, filter, auth.me?.id],
  )
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
        const s = await uploadEvidence(file, aiChosen, (p) => setUpload({ name: file.name, p }), effort)
        location.hash = `#/case/${s.id}`
      } catch (e) {
        setErr(String((e as Error).message ?? e))
        setUpload(null)
      }
    },
    [aiChosen, effort],
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
    // ...and a pasted image or clip (Cmd/Ctrl+V), unless the paste is going into a text field.
    const paste = (e: ClipboardEvent) => {
      const target = e.target as HTMLElement | null
      if (target?.closest('input, textarea, [contenteditable="true"]')) return
      const f = [...(e.clipboardData?.files ?? [])].find((x) => /^(image|video|audio)\//.test(x.type))
      if (!f || upload) return
      e.preventDefault()
      send(pastedName(f))
    }
    window.addEventListener('dragover', over)
    window.addEventListener('dragleave', leave)
    window.addEventListener('drop', drop)
    window.addEventListener('paste', paste)
    return () => {
      window.removeEventListener('dragover', over)
      window.removeEventListener('dragleave', leave)
      window.removeEventListener('drop', drop)
      window.removeEventListener('paste', paste)
    }
  }, [send, upload])

  useEffect(() => {
    if (!examine) return
    dropRef.current?.scrollIntoView({ behavior: 'smooth', block: 'center' })
    dropRef.current?.classList.add('flash')
    const t = setTimeout(() => dropRef.current?.classList.remove('flash'), 1600)
    return () => clearTimeout(t)
  }, [examine])

  const remove = async (c: CaseSummary) => {
    if (!confirm(`Delete the case for ${c.filename}? Its copy of the file and all results are removed. This cannot be undone.`)) return
    try {
      await api.remove(c.id)
    } catch (e) {
      setErr(String((e as Error).message ?? e))
    }
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
        <HeroField peaks={heroPeaks} />
      </section>
      <div className="home">
        <div className="home-main">
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
                    {AI_EXPERIMENTAL.has(p) && <span className="exp-tag">experimental</span>}
                  </button>
                ))}
              </div>
              <small className="faint">Runs speech, sounds, the visual analyst, geolocation and the written summary.</small>
            </div>
          )}
          <EffortTabs value={effort} onChange={pickEffort} disabled={!!upload} />
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
                  <b>Drop a file here, paste an image, or click to choose one</b>
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

          <SectionTitle
            icon="box-archive"
            accent="blue"
            small={cases ? (isFiltered(filter) && shown ? `${shown.length} of ${cases.length}` : `${cases.length} on file`) : undefined}
          >
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
            <>
              <FeedFilter filter={filter} onChange={setFilter} cases={cases} verdictOf={verdictFor} />
              {shown && shown.length === 0 ? (
                <Empty icon="magnifying-glass">
                  No case matches that.{' '}
                  <button type="button" className="link" onClick={() => setFilter({ q: '', verdict: null, kind: null, located: null, mine: false })}>
                    Show every case
                  </button>
                </Empty>
              ) : (
            <div className="cards">
              {(shown ?? cases).map((c) => (
                <CaseCard key={c.id} c={c} verdict={verdictFor(c)} summary={c.summary_plain ?? meta[c.id]?.summary ?? null} onDelete={() => remove(c)} />
              ))}
            </div>
              )}
            </>
          )}
        </div>

      </div>
    </main>
  )
}
