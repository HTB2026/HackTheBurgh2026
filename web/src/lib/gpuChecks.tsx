/**
 * Checks the viewer's GPU runs by itself as soon as a case opens (when WebGPU is available):
 * first answers within seconds while the server's checks take minutes. Results are shared by
 * the Quick look card, the frame scan panel, the key-moments timeline and the photo viewer.
 * The server's findings remain the official record; these are labelled "found in your browser".
 */
import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import type { Finding } from './types'
import { useGpu, type Gpu } from './gpu'
import { decodeAudio, useCaseMedia, type DecodedAudio } from './media'
import { scanFrames, type ScanResult } from './frameScan'
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
const Ctx = createContext<GpuChecks>(idle)
export const useGpuChecks = () => useContext(Ctx)

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
  const overview = await run(0, nyq, 256, 900)
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
  return { overview, hum, tone }
}

export function GpuChecksProvider({ kind, hasAudio, photoSrc, children }: { kind: 'video' | 'image'; hasAudio: boolean; photoSrc: string | null; children: ReactNode }) {
  const gpuState = useGpu()
  const gpu = gpuState === 'loading' ? null : gpuState
  const media = useCaseMedia()
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
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>
}

/** Frame-scan results as finding-shaped moments for the key-moments timeline. */
export function browserMoments(r: ScanResult | null): Finding[] {
  if (!r) return []
  const base = { module: 'browser', detail: 'Found in your browser by the GPU frame scan.', time_end_s: null, kind: 'measured' as const, tags: ['browser'], data: null }
  const out: Finding[] = r.jumps.map((k, i) => ({ ...base, id: `B${i + 1}`, severity: 'warning' as const, title: 'Abrupt change (found in your browser)', time_s: r.times[k] }))
  // One marker per stretch of repeated frames (repeats less than ~half a second apart).
  let i = 0
  const d = r.duplicates
  while (i < d.length) {
    let j = i
    while (j + 1 < d.length && d[j + 1] - d[j] <= 15) j++
    const n = j - i + 1
    const gaps = d.slice(i + 1, j + 1).map((v, k) => v - d[i + k])
    const regular = n >= 4 && gaps.every((g) => g === gaps[0])
    out.push({
      ...base,
      id: `BR${i}`,
      severity: 'warning',
      title: `${n} repeated frame${n === 1 ? '' : 's'}${regular ? `, every ${gaps[0]}th (frame-rate conversion)` : ''} (found in your browser)`,
      time_s: r.times[d[i]],
    })
    i = j + 1
  }
  return out
}
