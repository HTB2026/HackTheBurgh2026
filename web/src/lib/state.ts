import type { Artifact, CaseEvent, CaseInfo, Finding, ModuleStatus } from './types'

export interface ModuleState {
  name: string
  title: string
  description: string
  status: ModuleStatus
  progress: number
  message: string
  elapsed_s: number | null
  started_ts: number | null
  trace: string | null
}

export interface LogLine {
  seq: number
  ts: number
  module: string
  message: string
  level: 'log' | 'module' | 'error'
}

export interface Arrival {
  /** client clock (ms) when this item was first reduced */
  at: number
}

export interface CaseState {
  info: CaseInfo | null
  mediaUrl: string | null
  mediaReady: boolean
  modules: Record<string, ModuleState>
  moduleOrder: string[]
  findings: (Finding & Arrival)[]
  artifacts: Record<string, Artifact & Arrival>
  artifactOrder: string[]
  logs: LogLine[]
  done: boolean
  error: string | null
  seen: Set<number>
  lastTs: number
}

export const emptyState = (): CaseState => ({
  info: null,
  mediaUrl: null,
  mediaReady: false,
  modules: {},
  moduleOrder: [],
  findings: [],
  artifacts: {},
  artifactOrder: [],
  logs: [],
  done: false,
  error: null,
  seen: new Set(),
  lastTs: 0,
})

export type Action = { type: 'reset' } | { type: 'events'; events: CaseEvent[] }

const MAX_LOG = 400

function blankModule(name: string, title = name.toUpperCase(), description = ''): ModuleState {
  return { name, title, description, status: 'queued', progress: 0, message: '', elapsed_s: null, started_ts: null, trace: null }
}

/** Apply a batch of events. Mutation is confined to a fresh shallow copy. */
export function reduce(state: CaseState, action: Action): CaseState {
  if (action.type === 'reset') return emptyState()

  let s: CaseState = { ...state, seen: new Set(state.seen) }
  let modules = { ...s.modules }
  let findings = s.findings
  let artifacts = s.artifacts
  let artifactOrder = s.artifactOrder
  let logs = s.logs
  const now = Date.now()

  for (const ev of action.events) {
    // Server-side rerun: drop everything; new events follow from seq 0.
    if (ev.type === 'reset') {
      s = { ...emptyState(), seen: new Set() }
      modules = {}
      findings = []
      artifacts = {}
      artifactOrder = []
      logs = []
      continue
    }
    // A rerun restarts seq at 0 with a fresh `case` event: start over.
    if (ev.type === 'case' && ev.seq === 0 && s.seen.size > 0) {
      s = { ...emptyState(), seen: new Set() }
      modules = {}
      findings = []
      artifacts = {}
      artifactOrder = []
      logs = []
    }
    if (typeof ev.seq === 'number') {
      if (s.seen.has(ev.seq)) continue
      s.seen.add(ev.seq)
    }
    if (ev.ts) s.lastTs = Math.max(s.lastTs, ev.ts)

    switch (ev.type) {
      case 'case': {
        s.info = ev.case
        if (ev.case.media_url) {
          s.mediaUrl = ev.case.media_url
          s.mediaReady = true
        }
        const order: string[] = []
        for (const m of ev.case.modules ?? []) {
          order.push(m.name)
          modules[m.name] = { ...(modules[m.name] ?? blankModule(m.name)), title: m.title, description: m.description }
        }
        s.moduleOrder = order
        break
      }
      case 'media':
        s.mediaUrl = ev.media_url
        s.mediaReady = true
        break
      case 'module': {
        const prev = modules[ev.module] ?? blankModule(ev.module)
        if (!s.moduleOrder.includes(ev.module)) s.moduleOrder = [...s.moduleOrder, ev.module]
        const next: ModuleState = {
          ...prev,
          status: ev.status,
          progress: typeof ev.progress === 'number' ? ev.progress : prev.progress,
          message: ev.message ?? (ev.status === prev.status ? prev.message : ''),
          elapsed_s: ev.elapsed_s ?? prev.elapsed_s,
          trace: ev.trace ?? prev.trace,
          started_ts: ev.status === 'running' && prev.started_ts == null ? ev.ts : prev.started_ts,
        }
        modules[ev.module] = next
        if (ev.status !== prev.status || ev.status === 'error') {
          logs = [
            ...logs,
            {
              seq: ev.seq,
              ts: ev.ts,
              module: ev.module,
              message: `${ev.status.toUpperCase()}${ev.message ? ` ${ev.message}` : ''}${ev.elapsed_s != null ? ` (${ev.elapsed_s.toFixed(2)}s)` : ''}`,
              level: ev.status === 'error' ? 'error' : 'module',
            },
          ]
        } else if (ev.message) {
          logs = [...logs, { seq: ev.seq, ts: ev.ts, module: ev.module, message: ev.message, level: 'module' }]
        }
        break
      }
      case 'finding': {
        const f = ev.finding
        const idx = findings.findIndex((x) => x.id === f.id)
        if (idx >= 0) {
          findings = findings.slice()
          findings[idx] = { ...f, at: findings[idx].at }
        } else findings = [...findings, { ...f, at: now }]
        logs = [...logs, { seq: ev.seq, ts: ev.ts, module: f.module, message: `[${f.severity.toUpperCase()}] ${f.title}`, level: 'log' }]
        break
      }
      case 'artifact': {
        const a = ev.artifact
        const prev = artifacts[a.id]
        artifacts = { ...artifacts, [a.id]: { ...a, at: prev?.at ?? now } }
        if (!prev) artifactOrder = [...artifactOrder, a.id]
        break
      }
      case 'log':
        logs = [...logs, { seq: ev.seq, ts: ev.ts, module: ev.module ?? 'core', message: ev.message, level: 'log' }]
        break
      case 'done':
        s.done = true
        logs = [...logs, { seq: ev.seq, ts: ev.ts, module: 'core', message: 'ALL MODULES COMPLETE', level: 'module' }]
        break
      case 'error':
        s.error = ev.message
        break
    }
  }

  if (logs.length > MAX_LOG) logs = logs.slice(-MAX_LOG)
  return { ...s, modules, findings, artifacts, artifactOrder, logs }
}

const TERMINAL: ModuleStatus[] = ['done', 'skipped', 'error']
export const isTerminal = (st: ModuleStatus) => TERMINAL.includes(st)
