import { Fragment, useEffect, useMemo, useState, type ReactNode } from 'react'
import type { CaseSummary } from '../lib/types'
import { VERDICT, type Verdict } from '../lib/plain'
import { useAuth } from '../lib/auth'
import { Icon } from './ui/ui'

/** What the feed is narrowed to. Kept in the URL (#/?q=…&v=…) so a view can be shared. */
export interface Filter {
  q: string
  verdict: Verdict | null
  kind: 'video' | 'image' | null
  located: 'yes' | 'no' | null
  mine: boolean
}

const EMPTY: Filter = { q: '', verdict: null, kind: null, located: null, mine: false }
const VERDICTS: Verdict[] = ['edited', 'possible', 'clean', 'synthetic', 'running']

function fromHash(): Filter {
  const p = new URLSearchParams(location.hash.split('?')[1] ?? '')
  const v = p.get('v') as Verdict | null
  const k = p.get('kind')
  const l = p.get('loc')
  return {
    q: p.get('q') ?? '',
    verdict: v && VERDICTS.includes(v) ? v : null,
    kind: k === 'video' || k === 'image' ? k : null,
    located: l === 'yes' || l === 'no' ? l : null,
    mine: p.get('mine') === '1',
  }
}

function toHash(f: Filter): string {
  const p = new URLSearchParams()
  if (f.q.trim()) p.set('q', f.q.trim())
  if (f.verdict) p.set('v', f.verdict)
  if (f.kind) p.set('kind', f.kind)
  if (f.located) p.set('loc', f.located)
  if (f.mine) p.set('mine', '1')
  const base = location.hash.split('?')[0] || '#/'
  return p.size ? `${base}?${p}` : base
}

export const isFiltered = (f: Filter) => !!(f.q.trim() || f.verdict || f.kind || f.located || f.mine)

/** The filter, read from and written back to the URL without adding history entries. */
export function useFeedFilter(): [Filter, (f: Filter) => void] {
  const [filter, setFilter] = useState<Filter>(fromHash)
  useEffect(() => {
    const next = toHash(filter)
    if (next !== location.hash && !(next === '#/' && !location.hash)) history.replaceState(null, '', location.pathname + location.search + next)
  }, [filter])
  return [filter, setFilter]
}

/** Every word typed must appear somewhere in the case's searchable text. */
export function matches(c: CaseSummary, verdict: Verdict | undefined, summary: string | null, f: Filter, myId: string | null): boolean {
  if (f.verdict && verdict !== f.verdict) return false
  if (f.kind && (c.kind ?? c.case?.kind ?? 'video') !== f.kind) return false
  if (f.located === 'yes' && !c.place) return false
  if (f.located === 'no' && c.place) return false
  if (f.mine && (!myId || c.owner?.id !== myId)) return false
  const words = f.q.toLowerCase().split(/\s+/).filter(Boolean)
  if (!words.length) return true
  const hay = [c.filename, c.place, summary, verdict ? VERDICT[verdict].word : '', verdict ? VERDICT[verdict].short : '', c.owner?.name, c.id]
    .filter(Boolean)
    .join(' ')
    .toLowerCase()
  return words.every((w) => hay.includes(w))
}

function Chip({ on, onClick, icon, children, n }: { on: boolean; onClick: () => void; icon?: string; children: ReactNode; n?: number }) {
  return (
    <button type="button" className={`feed-chip ${on ? 'on' : ''}`} aria-pressed={on} onClick={onClick}>
      {icon && <Icon name={icon} />}
      {children}
      {n != null && <span className="n">{n}</span>}
    </button>
  )
}

export function FeedFilter({
  filter,
  onChange,
  cases,
  verdictOf,
}: {
  filter: Filter
  onChange: (f: Filter) => void
  cases: CaseSummary[]
  verdictOf: (c: CaseSummary) => Verdict | undefined
}) {
  const auth = useAuth()
  const set = (patch: Partial<Filter>) => onChange({ ...filter, ...patch })
  // Counts per chip, over the whole feed, so empty choices can be left out.
  const counts = useMemo(() => {
    const by = (fn: (c: CaseSummary) => unknown) => (v: unknown) => cases.filter((c) => fn(c) === v).length
    return {
      verdict: by(verdictOf),
      kind: by((c) => c.kind ?? c.case?.kind ?? 'video'),
      located: cases.filter((c) => c.place).length,
      mine: auth.me ? cases.filter((c) => c.owner?.id === auth.me!.id).length : 0,
    }
  }, [cases, verdictOf, auth.me])

  // A chip with nothing behind it is noise, unless it is the one switched on.
  const chip = (key: string, on: boolean, n: number, icon: string, label: string, toggle: () => void) =>
    n > 0 || on ? (
      <Chip key={key} on={on} onClick={toggle} icon={icon} n={n}>
        {label}
      </Chip>
    ) : null
  const groups = [
    VERDICTS.map((v) => chip(v, filter.verdict === v, counts.verdict(v), VERDICT[v].icon, VERDICT[v].short, () => set({ verdict: filter.verdict === v ? null : v }))),
    [
      chip('video', filter.kind === 'video', counts.kind('video'), 'video', 'Videos', () => set({ kind: filter.kind === 'video' ? null : 'video' })),
      chip('image', filter.kind === 'image', counts.kind('image'), 'image', 'Photos', () => set({ kind: filter.kind === 'image' ? null : 'image' })),
    ],
    [
      chip('yes', filter.located === 'yes', counts.located, 'location-dot', 'Located', () => set({ located: filter.located === 'yes' ? null : 'yes' })),
      chip('no', filter.located === 'no', cases.length - counts.located, 'circle-question', 'Place unknown', () => set({ located: filter.located === 'no' ? null : 'no' })),
    ],
    auth.me ? [chip('mine', filter.mine, counts.mine, 'user', 'Mine', () => set({ mine: !filter.mine }))] : [],
  ]
    .map((g) => g.filter(Boolean))
    .filter((g) => g.length)

  return (
    <div className="feed-filter" role="search">
      <label className="feed-search">
        <Icon name="magnifying-glass" />
        <input
          type="search"
          value={filter.q}
          placeholder="Search by file name, place, summary, verdict or uploader"
          onChange={(e) => set({ q: e.target.value })}
          onKeyDown={(e) => e.key === 'Escape' && set({ q: '' })}
          aria-label="Search cases"
        />
      </label>
      <div className="feed-chips">
        {groups.map((g, i) => (
          <Fragment key={i}>
            {i > 0 && <span className="feed-sep" />}
            {g}
          </Fragment>
        ))}
        {isFiltered(filter) && (
          <button type="button" className="feed-clear" onClick={() => onChange(EMPTY)}>
            <Icon name="xmark" /> Clear
          </button>
        )}
      </div>
    </div>
  )
}
