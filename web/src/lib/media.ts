/**
 * The case's media for browser-side (WebGPU) analysis: URLs from the case page, the decoded
 * soundtrack, and helpers to read video frames.
 */
import { createContext, useContext } from 'react'

export interface CaseMedia {
  /** Browser-playable copy (H.264/AAC for video, JPEG for photos). */
  mediaUrl: string | null
  /** The evidence file as uploaded (may not be decodable by the browser). */
  originalUrl: string | null
  fps: number | null
  duration: number
}

export const CaseMediaContext = createContext<CaseMedia>({ mediaUrl: null, originalUrl: null, fps: null, duration: 0 })
export const useCaseMedia = () => useContext(CaseMediaContext)

export interface DecodedAudio {
  samples: Float32Array // mono
  rate: number
  /** Which file the samples came from. */
  source: 'original' | 'copy'
}

const audioCache = new Map<string, Promise<DecodedAudio>>()

async function decodeUrl(url: string): Promise<{ samples: Float32Array; rate: number }> {
  const response = await fetch(url, { credentials: 'same-origin' })
  if (!response.ok) throw new Error(`audio ${response.status}`)
  const bytes = await response.arrayBuffer()
  // An OfflineAudioContext decodes at the file's own sample rate (an AudioContext would resample).
  const ctx = new OfflineAudioContext(1, 1, 44100)
  const buffer = await ctx.decodeAudioData(bytes)
  const n = buffer.length
  const mono = new Float32Array(n)
  for (let c = 0; c < buffer.numberOfChannels; c++) {
    const ch = buffer.getChannelData(c)
    for (let i = 0; i < n; i++) mono[i] += ch[i] / buffer.numberOfChannels
  }
  return { samples: mono, rate: buffer.sampleRate }
}

/**
 * The soundtrack as mono samples at its own rate. Tries the original file first (no lossy
 * re-encode: high tones survive), then the viewing copy.
 */
export function decodeAudio(media: CaseMedia): Promise<DecodedAudio> {
  const key = `${media.originalUrl}|${media.mediaUrl}`
  let hit = audioCache.get(key)
  if (!hit) {
    hit = (async () => {
      if (media.originalUrl) {
        try {
          return { ...(await decodeUrl(media.originalUrl)), source: 'original' as const }
        } catch {
          /* the browser can't decode the original's audio: use the copy */
        }
      }
      if (!media.mediaUrl) throw new Error('no audio to decode')
      return { ...(await decodeUrl(media.mediaUrl)), source: 'copy' as const }
    })()
    hit.catch(() => audioCache.delete(key))
    audioCache.set(key, hit)
  }
  return hit
}

/**
 * Low-pass (windowed sinc) and keep every `factor`-th sample, so a narrow low band (mains hum)
 * can be analysed with long windows cheaply.
 */
export function decimate(x: Float32Array, factor: number): Float32Array {
  if (factor <= 1) return x
  const taps = Math.min(255, factor * 16 + 1)
  const half = (taps - 1) / 2
  const fc = 0.45 / factor
  const h = new Float32Array(taps)
  let sum = 0
  for (let i = 0; i < taps; i++) {
    const t = i - half
    const sinc = t === 0 ? 2 * fc : Math.sin(2 * Math.PI * fc * t) / (Math.PI * t)
    h[i] = sinc * (0.42 - 0.5 * Math.cos((2 * Math.PI * i) / (taps - 1)) + 0.08 * Math.cos((4 * Math.PI * i) / (taps - 1)))
    sum += h[i]
  }
  for (let i = 0; i < taps; i++) h[i] /= sum
  const out = new Float32Array(Math.floor(x.length / factor))
  for (let o = 0; o < out.length; o++) {
    const c = o * factor
    let s = 0
    for (let i = 0; i < taps; i++) {
      const j = c + i - half
      if (j >= 0 && j < x.length) s += h[i] * x[j]
    }
    out[o] = s
  }
  return out
}
