import { Fragment, useEffect, useMemo, useState } from 'react'
import type { Finding } from '../../lib/types'
import type { ModuleState } from '../../lib/state'
import { usePlayback } from '../../lib/playback'
import { mmss } from '../../lib/format'
import { modulePlain, SEVERITY } from '../../lib/plain'
import { rankFindings } from '../../lib/facts'
import { boxOf } from '../../lib/types'
import { useMediaKind, useRegionFocus } from '../../lib/region'
import { Empty, Icon, SevChip, SourceChip } from '../ui/ui'
import { DataView } from '../ui/Data'

const SEVS = ['critical', 'warning', 'notice', 'info']

function ModuleTag({ name, modules }: { name: string; modules: Record<string, ModuleState> }) {
  const p = modulePlain(name, modules[name]?.title)
  return (
    <span className={`modtag c-${p.accent}`} data-tip={`${p.name}: ${p.does}`}>
      <Icon name={p.icon} />
      {p.name}
    </span>
  )
}

function TimeButton({ f }: { f: Finding }) {
  const pb = usePlayback()
  const kind = useMediaKind()
  const rf = useRegionFocus()
  const box = boxOf(f)
  if (kind === 'image') {
    if (!box) return <span className="faint">whole photo</span>
    return (
      <button
        className="timebtn mono"
        onClick={(e) => {
          e.stopPropagation()
          rf.focus(box, f.title)
        }}
        data-tip="Zoom the picture to this region"
      >
        <Icon name="vector-square" /> region
      </button>
    )
  }
  if (f.time_s == null) return <span className="faint">whole video</span>
  return (
    <button
      className="timebtn mono"
      onClick={(e) => {
        e.stopPropagation()
        pb.seek(f.time_s as number)
      }}
      data-tip="Jump the video to this moment"
    >
      <Icon name="play" /> {mmss(f.time_s)}
      {f.time_end_s != null ? `–${mmss(f.time_end_s)}` : ''}
    </button>
  )
}

export function FindingDetail({ f }: { f: Finding }) {
  return (
    <div className="fdetail">
      <p>{f.detail}</p>
      <div className="fdetail-meta">
        <span className="faint mono">{f.id}</span>
        {f.tags?.length > 0 && (
          <span className="tags">
            {f.tags.map((t) => (
              <span key={t} className="tag">
                {t}
              </span>
            ))}
          </span>
        )}
      </div>
      <h4>Measurements behind this finding</h4>
      <DataView data={f.data} />
    </div>
  )
}

/** Leaderboard of every finding, most serious first. Click a row for all its measurements. */
export function FindingsBoard({
  findings,
  modules,
  focus,
}: {
  findings: Finding[]
  modules: Record<string, ModuleState>
  focus?: string | null
}) {
  const [sev, setSev] = useState<string>('all')
  const [mod, setMod] = useState<string>('')
  const [q, setQ] = useState('')
  const [open, setOpen] = useState<Set<string>>(new Set())
  const ranked = useMemo(() => rankFindings(findings), [findings])
  const mods = useMemo(() => [...new Set(findings.map((f) => f.module))], [findings])
  const counts = useMemo(() => {
    const c: Record<string, number> = {}
    for (const f of findings) c[f.severity] = (c[f.severity] ?? 0) + 1
    return c
  }, [findings])

  useEffect(() => {
    if (!focus) return
    setOpen((o) => new Set(o).add(focus))
    setSev('all')
    setMod('')
    setQ('')
    requestAnimationFrame(() => document.querySelector(`[data-row="${CSS.escape(focus)}"]`)?.scrollIntoView({ block: 'center', behavior: 'smooth' }))
  }, [focus])

  const needle = q.trim().toLowerCase()
  const rows = ranked.filter(
    (f) =>
      (sev === 'all' || f.severity === sev) &&
      (!mod || f.module === mod) &&
      (!needle || `${f.title} ${f.detail} ${f.tags?.join(' ')} ${JSON.stringify(f.data)}`.toLowerCase().includes(needle)),
  )

  const toggle = (id: string) =>
    setOpen((o) => {
      const n = new Set(o)
      if (n.has(id)) n.delete(id)
      else n.add(id)
      return n
    })

  return (
    <div className="board">
      <div className="board-filters">
        <div className="tabs">
          <button className={sev === 'all' ? 'active' : ''} onClick={() => setSev('all')}>
            All <span className="n">{findings.length}</span>
          </button>
          {SEVS.map((s) => (
            <button key={s} className={`sevtab sev-${s} ${sev === s ? 'active' : ''}`} onClick={() => setSev(s)} data-tip={SEVERITY[s].tip}>
              <Icon name={SEVERITY[s].icon} /> {SEVERITY[s].word} <span className="n">{counts[s] ?? 0}</span>
            </button>
          ))}
        </div>
        <select className="input" value={mod} onChange={(e) => setMod(e.target.value)} aria-label="Filter by check">
          <option value="">Every check</option>
          {mods.map((m) => (
            <option key={m} value={m}>
              {modulePlain(m, modules[m]?.title).name}
            </option>
          ))}
        </select>
        <label className="search-box">
          <Icon name="magnifying-glass" />
          <input className="input" placeholder="Search findings and their data" value={q} onChange={(e) => setQ(e.target.value)} />
        </label>
      </div>
      {rows.length === 0 ? (
        <Empty icon="filter">No findings match these filters.</Empty>
      ) : (
        <table className="lb">
          <thead>
            <tr>
              <th>Rank</th>
              <th>Severity</th>
              <th>Check</th>
              <th>When</th>
              <th>Finding</th>
              <th>Source</th>
              <th aria-label="Expand" />
            </tr>
          </thead>
          <tbody>
            {rows.map((f, i) => {
              const isOpen = open.has(f.id)
              return (
                <Fragment key={f.id}>
                  <tr
                    data-row={f.id}
                    className={`lb-row rank-${i + 1} ${isOpen ? 'open' : ''} ${focus === f.id ? 'focus' : ''}`}
                    onClick={() => toggle(f.id)}
                    aria-expanded={isOpen}
                  >
                    <td className="rank mono">#{i + 1}</td>
                    <td>
                      <SevChip sev={f.severity} />
                    </td>
                    <td>
                      <ModuleTag name={f.module} modules={modules} />
                    </td>
                    <td>
                      <TimeButton f={f} />
                    </td>
                    <td className="lb-title">
                      <b>{f.title}</b>
                      <span className="muted">{f.detail}</span>
                    </td>
                    <td>
                      <SourceChip kind={f.kind} />
                    </td>
                    <td className="chev">
                      <Icon name={isOpen ? 'chevron-up' : 'chevron-down'} />
                    </td>
                  </tr>
                  {isOpen && (
                    <tr className="lb-expand">
                      <td colSpan={7}>
                        <FindingDetail f={f} />
                      </td>
                    </tr>
                  )}
                </Fragment>
              )
            })}
          </tbody>
        </table>
      )}
    </div>
  )
}

/** Compact expandable list of findings for a topic tab. */
export function FindingList({ findings, modules, onOpenAll }: { findings: Finding[]; modules: Record<string, ModuleState>; onOpenAll?: (id: string) => void }) {
  const [open, setOpen] = useState<string | null>(null)
  if (!findings.length) return null
  return (
    <ul className="flist">
      {rankFindings(findings).map((f) => (
        <li key={f.id} className={`flist-item sev-b-${f.severity} ${open === f.id ? 'open' : ''}`}>
          <div
            className="flist-head"
            role="button"
            tabIndex={0}
            onClick={() => setOpen(open === f.id ? null : f.id)}
            onKeyDown={(e) => (e.key === 'Enter' || e.key === ' ') && (e.preventDefault(), setOpen(open === f.id ? null : f.id))}
            aria-expanded={open === f.id}
          >
            <SevChip sev={f.severity} />
            <span className="flist-title">
              <b>{f.title}</b>
              <span className="muted">{f.detail}</span>
            </span>
            <span className="flist-side">
              <ModuleTag name={f.module} modules={modules} />
              <SourceChip kind={f.kind} />
              <TimeButton f={f} />
              <Icon name={open === f.id ? 'chevron-up' : 'chevron-down'} className="faint" />
            </span>
          </div>
          {open === f.id && (
            <div className="flist-body">
              <FindingDetail f={f} />
              {onOpenAll && (
                <button className="link small" onClick={() => onOpenAll(f.id)}>
                  Show in all findings
                </button>
              )}
            </div>
          )}
        </li>
      ))}
    </ul>
  )
}
