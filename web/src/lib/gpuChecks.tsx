/**
 * Checks the viewer's GPU runs by itself as soon as a case opens (when WebGPU is available):
 * first answers within seconds while the server's checks take minutes. Results are shared by
 * the Quick look card, the frame scan panel, the key-moments timeline and the photo viewer.
 * The server's findings remain the official record; these are labelled "found in your browser".
 */
import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from 'react'
import type { Artifact, Finding } from './types'
import type { ModuleState } from './state'
import { useGpu, type Gpu } from './gpu'
import { decodeAudio, type CaseMedia, type DecodedAudio } from './media'
import { flickerVerdict, scanFrames, type ScanResult } from './frameScan'
import { gpuSpectrogram, type SpecResult } from './gpuSpectrogram'
import { ghostFromUrl, type GhostResult } from './ghost'

export type CheckStatus = 'waiting' | 'running' | 'done' | 'failed' | 'off'

export interface HumSummary {
  freq: number // Hz, median of the steadier trace
  steady: number // fraction of moments within 0.5 Hz of it
  grid: 50 | 60 | null
}
export interface ToneSummary {
  freq: number
  steady: number
}

export interface GpuChecks {
  enabled: boolean
  frames: { status: CheckStatus; result: ScanResult | null; error: string | null; progress: number; rate: number }
  sound: { status: CheckStatus; hum: HumSummary | null; tone: ToneSummary | null; overview: SpecResult | null; error: string | null }
  ghost: { status: CheckStatus; result: GhostResult | null; error: string | null }
  /** The live "what changed" map while a frame scan runs. */
  changes: React.MutableRefObject<GPUTexture | null>
  /** Run the frame scan (again) at a given speed. */
  scan: (rate: number) => void
  stopScan: () => void
}

const idle: GpuChecks = {
  enabled: false,
  frames: { status: 'off', result: null, error: null, progress: 0, rate: 1 },
  sound: { status: 'off', hum: null, tone: null, overview: null, error: null },
  ghost: { status: 'off', result: null, error: null },
  changes: { current: null },
  scan: () => {},
  stopScan: () => {},
}
/** Provided by the case page (which runs the checks with useGpuChecksRunner). */
export const GpuChecksContext = createContext<GpuChecks>(idle)
export const useGpuChecks = () => useContext(GpuChecksContext)

/** Automatic frame scans: every frame up to a minute, 2x up to ten minutes, else on request. */
const autoRate = (duration: number) => (duration <= 60 ? 1 : duration <= 600 ? 2 : 0)

const median = (a: Float32Array) => {
  const s = Float32Array.from(a).sort()
  return s[Math.floor(s.length / 2)]
}
/** The trace's typical frequency and how much of the time it stays within 0.5 Hz of it. */
function steadiness(peaks: Float32Array, tol = 0.5) {
  const m = median(peaks)
  let n = 0
  for (const f of peaks) if (Math.abs(f - m) <= tol) n++
  return { freq: m, steady: n / Math.max(1, peaks.length) }
}
const gridOf = (f: number): 50 | 60 | null => {
  for (const g of [50, 60] as const) for (let h = 1; h <= 3; h++) if (Math.abs(f - g * h) <= 1) return g
  return null
}

async function soundChecks(gpu: Gpu, audio: DecodedAudio) {
  const nyq = audio.rate / 2
  const run = (f0: number, f1: number, rows: number, cols: number, windowS = 0) => gpuSpectrogram(gpu, audio, { f0, f1, rows, cols, windowS, range: 80 })
  // Mains hum: the fundamental or (often stronger) the second harmonic, whichever is steadier.
  let hum: HumSummary | null = null
  for (const [f0, f1] of [[45, 65], [95, 125]]) {
    const r = await run(f0, f1, 240, 400)
    const s = steadiness(r.peaks)
    r.texture.destroy()
    const g = gridOf(s.freq)
    if (g && s.steady >= 0.5 && (!hum || s.steady > hum.steady)) hum = { freq: s.freq, steady: s.steady, grid: g }
  }
  // A steady tone high up (often inaudible): a marker, a beacon, or a planted signal.
  let tone: ToneSummary | null = null
  if (nyq > 15500) {
    const r = await run(15000, nyq, 320, 400, 0.1)
    const s = steadiness(r.peaks, 40)
    r.texture.destroy()
    if (s.steady >= 0.7) tone = s
  }
  return { overview: null, hum, tone }
}

/** Run the browser checks for a case; the case page provides the result via GpuChecksContext. */
export function useGpuChecksRunner({ media, kind, hasAudio, photoSrc }: { media: CaseMedia; kind: 'video' | 'image'; hasAudio: boolean; photoSrc: string | null }): GpuChecks {
  const gpuState = useGpu()
  const gpu = gpuState === 'loading' ? null : gpuState
  const [frames, setFrames] = useState<GpuChecks['frames']>(idle.frames)
  const [sound, setSound] = useState<GpuChecks['sound']>(idle.sound)
  const [ghost, setGhost] = useState<GpuChecks['ghost']>(idle.ghost)
  const changes = useRef<GPUTexture | null>(null)
  const abort = useRef<AbortController | null>(null)
  const started = useRef<string | null>(null)

  const scan = useCallback(
    async (rate: number) => {
      if (!gpu || !media.mediaUrl || !media.fps) return
      abort.current?.abort()
      const ac = new AbortController()
      abort.current = ac
      setFrames({ status: 'running', result: null, error: null, progress: 0, rate })
      try {
        const result = await scanFrames(gpu, media.mediaUrl, media.originalUrl, media.fps, media.duration, {
          rate,
          signal: ac.signal,
          onProgress: (progress) => setFrames((f) => ({ ...f, progress })),
          onChanges: (t) => (changes.current = t),
        })
        setFrames({ status: 'done', result, error: null, progress: 1, rate })
      } catch (e) {
        const cancelled = (e as Error).name === 'AbortError'
        setFrames((f) => ({ ...f, status: cancelled ? 'waiting' : 'failed', error: cancelled ? null : String((e as Error).message ?? e) }))
      } finally {
        changes.current?.destroy()
        changes.current = null
      }
    },
    [gpu, media.mediaUrl, media.originalUrl, media.fps, media.duration],
  )
  const stopScan = useCallback(() => abort.current?.abort(), [])
  useEffect(() => () => abort.current?.abort(), [])

  // Start automatically, once per case, one check at a time.
  useEffect(() => {
    if (!gpu) return
    const key = kind === 'image' ? photoSrc : media.mediaUrl
    if (!key || started.current === key) return
    if (kind === 'video' && !media.fps) return
    started.current = key
    ;(async () => {
      if (kind === 'image') {
        if (!photoSrc) return
        setGhost({ status: 'running', result: null, error: null })
        try {
          setGhost({ status: 'done', result: await ghostFromUrl(gpu, photoSrc), error: null })
        } catch (e) {
          setGhost({ status: 'failed', result: null, error: String((e as Error).message ?? e) })
        }
        return
      }
      const rate = autoRate(media.duration)
      setFrames((f) => ({ ...f, status: rate ? 'waiting' : 'off' }))
      if (hasAudio) {
        setSound({ ...idle.sound, status: 'running' })
        try {
          const r = await soundChecks(gpu, await decodeAudio(media))
          setSound({ status: 'done', ...r, error: null })
        } catch (e) {
          setSound({ ...idle.sound, status: 'failed', error: String((e as Error).message ?? e) })
        }
      }
      if (rate) await scan(rate)
    })()
  }, [gpu, kind, photoSrc, media, hasAudio, scan])

  const value = useMemo<GpuChecks>(
    () => ({ enabled: !!gpu, frames, sound, ghost, changes, scan: (r) => void scan(r), stopScan }),
    [gpu, frames, sound, ghost, scan, stopScan],
  )
  return value
}

const NOTE = 'Measured in your browser (WebGPU) while the server worked. Not saved with the case and not part of its verdict.'
const TITLES: Record<string, string> = {
  'gpu-frames': 'FRAME SCAN (YOUR BROWSER)',
  'gpu-light': 'LAMP FLICKER (YOUR BROWSER)',
  'gpu-sound': 'HUM AND TONES (YOUR BROWSER)',
  'gpu-ghost': 'JPEG GHOST (YOUR BROWSER)',
  'gpu-speech': 'TRANSCRIPT ON THIS DEVICE',
}

export interface BrowserLayer {
  findings: Finding[]
  artifacts: Artifact[]
  modules: Record<string, ModuleState>
  order: string[]
}

/**
 * The browser checks expressed like the server's output (findings, exhibits, check states), so
 * they appear in the case page's usual places: findings, key moments, the photo's flagged
 * regions, the exhibits by topic and the live analysis panel. Callers must keep them out of
 * the verdict: they are not part of the case record.
 */
export function browserLayer(c: GpuChecks, opts: { kind: 'video' | 'image'; hasAudio: boolean; hasVideo: boolean; serverTranscript: boolean; duration: number; thumbUrl?: string | null }): BrowserLayer {
  const out: BrowserLayer = { findings: [], artifacts: [], modules: {}, order: [] }
  if (!c.enabled) return out
  let n = 0
  const when = (t: number) => (opts.duration < 60 ? `${t.toFixed(2)} s` : `${Math.floor(t / 60)}:${String(Math.floor(t % 60)).padStart(2, '0')}`)
  const finding = (module: string, severity: Finding['severity'], title: string, detail: string, extra: Partial<Finding> = {}) =>
    out.findings.push({ id: `B${++n}`, module, severity, title, detail: `${detail} ${NOTE}`, time_s: null, time_end_s: null, kind: 'measured', tags: ['browser'], data: null, ...extra })
  const artifact = (module: string, name: string, kind: string, title: string) =>
    out.artifacts.push({ id: `${module}:${name}`, module, name, kind, title, url: null, data: { thumbUrl: opts.thumbUrl ?? null }, time_s: null })
  const module = (name: string, status: CheckStatus, progress = 0, message = '') => {
    if (status === 'off') return
    out.order.push(name)
    out.modules[name] = {
      name,
      title: TITLES[name],
      description: '',
      status: status === 'failed' ? 'error' : status === 'waiting' ? 'waiting' : status,
      progress: status === 'done' ? 1 : progress,
      message,
      elapsed_s: null,
      started_ts: null,
      trace: null,
    }
  }

  if (opts.kind === 'image') {
    module('gpu-ghost', c.ghost.status, 0, c.ghost.error ?? '')
    const g = c.ghost.result
    if (g?.dominant != null) {
      finding('gpu-ghost', 'info', `Last saved at JPEG quality ${g.dominant}`, `Most 8×8 blocks fit quality ${g.dominant} best; ${g.odd} textured block${g.odd === 1 ? '' : 's'} point to another quality.`, { tags: ['browser', 'compression'] })
      for (const r of g.regions)
        finding('gpu-ghost', 'notice', `Area saved at JPEG quality ${r.quality}`, `${r.blocks} neighbouring blocks were last saved at about Q${r.quality}, unlike the rest of the picture (Q${g.dominant}): often a pasted-in area.`, {
          tags: ['browser', 'compression', 'region'],
          data: { box: { x: r.x, y: r.y, w: r.w, h: r.h } },
        })
    }
    return out
  }

  if (opts.hasVideo) {
    module('gpu-frames', c.frames.status, c.frames.progress, c.frames.status === 'running' ? `${Math.round(c.frames.progress * 100)}% of the frames` : c.frames.error ?? '')
    module('gpu-light', c.frames.status, c.frames.progress, c.frames.status === 'running' ? 'measured during the frame scan' : '')
    if (c.frames.status !== 'off') {
      artifact('gpu-frames', 'scan', 'gpu-framescan', 'Frame-by-frame change (your browser)')
      artifact('gpu-light', 'flicker', 'gpu-flicker', 'Lamp flicker map (your browser)')
    }
    const r = c.frames.result
    if (r) {
      r.jumps.forEach((k) =>
        finding('gpu-frames', 'warning', `Abrupt change at ${when(r.times[k])}`, 'The picture changed far more from one frame to the next than around it: a cut, a jump or a join.', { time_s: r.times[k], tags: ['browser', 'cut'] }),
      )
      // One finding per stretch of repeated frames (repeats less than ~half a second apart).
      const d = r.duplicates
      for (let i = 0; i < d.length; ) {
        let j = i
        while (j + 1 < d.length && d[j + 1] - d[j] <= 15) j++
        const count = j - i + 1
        const gaps = d.slice(i + 1, j + 1).map((v, k) => v - d[i + k])
        const regular = count >= 4 && gaps.every((g) => g === gaps[0])
        finding(
          'gpu-frames',
          'warning',
          `${count} repeated frame${count === 1 ? '' : 's'}${regular ? `, every ${gaps[0]}th` : ''} from ${when(r.times[d[i]])}`,
          regular ? 'Frames repeated at a regular interval: the frame rate was converted (re-timed), so this is not the camera’s original timing.' : 'Frames identical to the one before: frozen video or frames padded in.',
          { time_s: r.times[d[i]], tags: ['browser', 'splice'] },
        )
        i = j + 1
      }
      if (!r.jumps.length && !d.length) finding('gpu-frames', 'info', `Every frame read: no cuts or repeated frames (${r.times.length} frames)`, 'Each frame changes smoothly from the one before.')
      const v = flickerVerdict(r)
      if (v.grid) finding('gpu-light', 'notice', `Lights on a ${v.grid} Hz grid (flicker ${v.ratio.toFixed(0)}× the noise)`, `Mains lamps pulse at ${v.grid * 2} Hz; that pulse shows up in this video well above the noise floor. ${v.grid === 50 ? '50 Hz grids: Europe, the UK, most of Africa, Asia and Australia.' : '60 Hz grids: North America, much of South America, Korea, Taiwan, western Japan.'}`, { tags: ['browser', 'grid'] })
      else finding('gpu-light', 'info', 'No mains-lamp flicker', 'No 50 or 60 Hz-grid lamp pulse above the noise floor (daylight, battery light, or flicker-free lamps).', { tags: ['browser'] })
    }
  }

  if (opts.hasAudio) {
    module('gpu-sound', c.sound.status, 0, c.sound.error ?? '')
    if (c.sound.status === 'done') {
      const h = c.sound.hum
      if (h) finding('gpu-sound', 'notice', `Steady mains hum at ${h.freq.toFixed(2)} Hz (${h.grid} Hz grid)`, `The strongest frequency in the hum band stays within 0.5 Hz of ${h.freq.toFixed(2)} Hz ${Math.round(h.steady * 100)}% of the time: electrical hum from a ${h.grid} Hz grid.`, { tags: ['browser', 'hum'] })
      const t = c.sound.tone
      if (t) finding('gpu-sound', 'warning', `Steady tone at ${(t.freq / 1000).toFixed(2)} kHz`, 'A steady tone high in the spectrum, often inaudible: a device’s whine, a marker, or a planted signal.', { tags: ['browser'] })
    }
    // Speech on this device: offered where the server has no transcript (e.g. no AI key).
    if (!opts.serverTranscript) {
      artifact('gpu-speech', 'transcript', 'gpu-transcript', 'Transcribe on this device (experimental)')
    }
  }
  return out
}
