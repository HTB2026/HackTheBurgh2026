// Pull the headline facts out of the event stream, defensively (fields vary by case and backend version).
import type { Answer, Artifact, Finding } from './types'
import type { CaseState } from './state'
import { isTerminal } from './state'
import { verdictOf, type Verdict } from './plain'

export interface Facts {
  verdict: Verdict
  brief: Artifact | null
  answers: Answer[]
  summary: string | null
  running: boolean
  checksDone: number
  checksTotal: number
  counts: Record<string, number>
  device: string | null
  software: string | null
  resaved: string | null
  recorded: string | null
  gps: { lat: number; lon: number; accuracy_m?: number; place?: string; alt_m?: number } | null
  grid: number | null
  pulse: number | null
  language: string | null
  aiUsed: boolean
  hasAudio: boolean
  currentModule: string | null
}

const firstData = (findings: Finding[], pred: (f: Finding) => boolean) => findings.find(pred)?.data ?? null

export function deriveFacts(state: CaseState, openai: boolean | null): Facts {
  const arts = state.artifactOrder.map((id) => state.artifacts[id])
  const brief = state.artifacts['brief:brief'] ?? arts.find((a) => a.kind === 'brief') ?? null
  const fs = state.findings
  const order = state.moduleOrder
  const done = order.filter((n) => state.modules[n] && isTerminal(state.modules[n].status)).length
  const running = !!state.info && !state.done && done < order.length
  const counts: Record<string, number> = { critical: 0, warning: 0, notice: 0, info: 0 }
  for (const f of fs) counts[f.severity] = (counts[f.severity] ?? 0) + 1

  const dev = firstData(fs, (f) => f.module === 'provenance' && !!f.data?.model)
  const device = dev ? [dev.make, dev.model].filter(Boolean).join(' ') : null
  const resavedF = fs.find((f) => f.tags?.includes('software') || f.tags?.includes('editor'))
  const resaved = resavedF ? resavedF.title.replace(/^Processed by\s*/i, '').replace(/\s*\(.*\)$/, '') : null

  const clock = firstData(fs, (f) => !!f.data?.creation_time_utc)
  const gpsD = firstData(fs, (f) => f.tags?.includes('gps') && isFinite(f.data?.lat))
  const gridD = firstData(fs, (f) => typeof f.data?.grid_hz === 'number')
  const nominal = fs.find((f) => /\b(50|60) Hz grid\b/.test(f.title))
  const pulseD = firstData(fs, (f) => f.module === 'pulse' && typeof f.data?.bpm === 'number')
  const lang = firstData(fs, (f) => typeof f.data?.language === 'string')

  const answers: Answer[] = Array.isArray(brief?.data?.answers) ? brief!.data.answers : []
  const current = order.find((n) => state.modules[n]?.status === 'running') ?? null
  const aiUsed = fs.some((f) => f.kind === 'inferred') || answers.some((a) => a.ai)

  return {
    verdict: state.error ? 'unknown' : running && !brief ? 'running' : brief ? verdictOf(brief.data?.verdict) : state.done ? 'unknown' : 'running',
    brief,
    answers,
    summary: brief?.data?.summary_plain ?? brief?.data?.headline ?? null,
    running,
    checksDone: done,
    checksTotal: order.length,
    counts,
    device,
    software: dev?.software ?? null,
    resaved,
    recorded: clock?.creation_time_utc ?? null,
    gps: gpsD ? { lat: gpsD.lat, lon: gpsD.lon, accuracy_m: gpsD.accuracy_m, place: gpsD.place, alt_m: gpsD.alt_m } : null,
    grid: gridD?.grid_hz ?? (nominal ? Number(nominal.title.match(/\b(50|60) Hz grid/)![1]) : null),
    pulse: pulseD ? Math.round(pulseD.bpm) : null,
    language: lang?.language ?? null,
    aiUsed: aiUsed || !!openai,
    hasAudio: !!state.info?.has_audio,
    currentModule: current,
  }
}

/** Severity rank then time, for the findings "leaderboard". */
export function rankFindings(fs: Finding[]): Finding[] {
  const sev: Record<string, number> = { critical: 0, warning: 1, notice: 2, info: 3 }
  return [...fs].sort(
    (a, b) => (sev[a.severity] ?? 9) - (sev[b.severity] ?? 9) || (a.time_s ?? 1e9) - (b.time_s ?? 1e9) || a.id.localeCompare(b.id),
  )
}
