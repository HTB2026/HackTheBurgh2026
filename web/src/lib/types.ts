// Mirrors docs/PROTOCOL.md. Keep payloads loose (`any` in data) - modules evolve.

export type Severity = 'info' | 'notice' | 'warning' | 'critical'
export type ModuleStatus = 'queued' | 'waiting' | 'running' | 'done' | 'skipped' | 'error'

export interface ModuleInfo {
  name: string
  title: string
  description: string
}

export interface VideoInfo {
  width: number
  height: number
  fps: number
  codec: string
  frames: number
}

export interface CaseInfo {
  id: string
  filename: string
  duration_s: number
  size_bytes: number
  has_audio: boolean
  video: VideoInfo | null
  thumb_url: string | null
  media_url: string | null
  modules: ModuleInfo[]
  // newer backends; optional on older cases
  sha256?: string
  md5?: string
  ingested_at?: number
  container?: {
    format_name?: string
    format_long_name?: string
    bit_rate?: number | string
    size_bytes?: number
    duration_s?: number
  }
  streams?: StreamInfo[]
  tags?: Record<string, string | number>
}

export interface StreamInfo {
  index?: number
  type?: string
  codec?: string
  codec_long?: string
  profile?: string
  width?: number
  height?: number
  fps?: number
  pix_fmt?: string
  color_range?: string
  bit_rate?: number | string
  sample_rate?: number | string
  channels?: number
  channel_layout?: string
  language?: string
  handler?: string
  duration_s?: number
}

export interface Answer {
  key: string
  question: string
  answer: string
  confidence: 'strong' | 'moderate' | 'weak' | 'unknown' | string
  evidence: string[]
  time_s: number | null
  ai: boolean
}

export interface CaseSummary {
  id: string
  filename: string
  created: number
  status: 'queued' | 'running' | 'done' | 'interrupted' | string
  findings: number
  warnings: number
  verdict?: string | null
  summary_plain?: string | null
  case: CaseInfo | null
}

export interface Finding {
  id: string
  module: string
  severity: Severity
  title: string
  detail: string
  time_s: number | null
  time_end_s: number | null
  kind: 'measured' | 'inferred'
  tags: string[]
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  data: any
}

export interface Artifact {
  id: string
  module: string
  name: string
  kind: string
  title: string
  url: string | null
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  data: any
  time_s: number | null
}

interface Base {
  seq: number
  ts: number
}

export type CaseEvent =
  | (Base & { type: 'case'; case: CaseInfo })
  | (Base & { type: 'media'; media_url: string | null })
  | (Base & {
      type: 'module'
      module: string
      status: ModuleStatus
      progress: number
      message?: string
      elapsed_s?: number
      trace?: string
    })
  | (Base & { type: 'finding'; finding: Finding })
  | (Base & { type: 'artifact'; artifact: Artifact })
  | (Base & { type: 'log'; module?: string; message: string })
  | (Base & { type: 'done' })
  | (Base & { type: 'error'; message: string })
  | (Base & { type: 'reset' })

export interface AuthConfig {
  domain: string
  client_id: string
  audience: string
}

export interface Health {
  ok: boolean
  openai: boolean
  modules: ModuleInfo[]
  /** Auth0 login settings; null when the server has no login configured. */
  auth?: AuthConfig | null
}

export interface EnhanceResult {
  single_url: string
  enhanced_url: string
  frames_used: number
  scale: number
  source_px: [number, number]
  time_s: number
}
