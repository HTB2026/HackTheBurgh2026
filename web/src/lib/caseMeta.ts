// The case list has no verdict; pull it (and a waveform for the hero) from each case's event log, cached.
import { useEffect, useState } from 'react'
import type { CaseEvent, CaseSummary } from './types'
import { verdictOf, type Verdict } from './plain'

export interface CaseMeta {
  verdict: Verdict
  peaks: number[] | null
  summary: string | null
}

const cache = new Map<string, CaseMeta>()
const key = (c: CaseSummary) => `${c.id}:${c.status}:${c.findings}`

async function load(c: CaseSummary): Promise<CaseMeta> {
  const k = key(c)
  const hit = cache.get(k)
  if (hit) return hit
  try {
    const stored = sessionStorage.getItem(`ew:${k}`)
    if (stored) {
      const m = JSON.parse(stored) as CaseMeta
      cache.set(k, m)
      return m
    }
  } catch {
    /* ignore */
  }
  const res = await fetch(`/api/cases/${encodeURIComponent(c.id)}`)
  const body = (await res.json()) as { events: CaseEvent[] }
  let verdict: Verdict = c.status === 'done' ? 'unknown' : 'running'
  let peaks: number[] | null = null
  let summary: string | null = null
  for (const e of body.events ?? []) {
    if (e.type !== 'artifact') continue
    const a = e.artifact
    if (a.kind === 'brief') {
      verdict = verdictOf(a.data?.verdict)
      summary = a.data?.summary_plain ?? null
    }
    if (a.kind === 'waveform' && Array.isArray(a.data?.peak)) peaks = a.data.peak
  }
  const meta = { verdict, peaks, summary }
  if (c.status === 'done') {
    cache.set(k, meta)
    try {
      // keep storage small: a coarse waveform is enough for the hero
      const coarse = peaks ? peaks.filter((_, i) => i % Math.max(1, Math.floor(peaks!.length / 300)) === 0) : null
      sessionStorage.setItem(`ew:${k}`, JSON.stringify({ ...meta, peaks: coarse }))
    } catch {
      /* quota */
    }
  }
  return meta
}

/** Metadata for the first `limit` cases, fetched 4 at a time. */
export function useCaseMeta(cases: CaseSummary[] | null, limit = 24) {
  const [meta, setMeta] = useState<Record<string, CaseMeta>>({})
  const sig = (cases ?? []).slice(0, limit).map(key).join('|')
  useEffect(() => {
    if (!cases) return
    let live = true
    const list = cases.slice(0, limit)
    let i = 0
    const worker = async () => {
      while (live && i < list.length) {
        const c = list[i++]
        try {
          const m = await load(c)
          if (live) setMeta((prev) => ({ ...prev, [c.id]: m }))
        } catch {
          /* skip */
        }
      }
    }
    Promise.all([worker(), worker(), worker(), worker()])
    return () => {
      live = false
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sig])
  return meta
}
