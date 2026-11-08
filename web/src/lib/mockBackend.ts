/** Browser-only implementations for the opt-in Vite preview. Never sends evidence anywhere. */
import { loadMock, mockName } from './mock'
import { MOCK_CASES, mockCaseName, mockModules } from './mockShowcase'
import type { CaseEvent, CaseInfo, CaseSummary, EnhanceResult, Health } from './types'

const removed = new Set<string>()
const uploads = new Map<string, CaseEvent[]>()
export const isMock = () => import.meta.env.DEV && !!mockName()
export const mockDelay = (ms: number) => new Promise<void>((resolve) => window.setTimeout(resolve, ms))

export async function mockEvents(id: string): Promise<{ events: CaseEvent[] }> {
  const events = uploads.get(id) ?? await loadMock(id)
  return { events: structuredClone(events) }
}

function summary(events: CaseEvent[]): CaseSummary {
  const first = events.find((e) => e.type === 'case')
  if (!first || first.type !== 'case') throw new Error('Preview has no case information')
  const findings = events.flatMap((e) => e.type === 'finding' ? [e.finding] : [])
  const brief = events.findLast((e) => e.type === 'artifact' && e.artifact.kind === 'brief')
  return { id: first.case.id, filename: first.case.filename, kind: first.case.kind, created: first.ts, status: 'done', findings: findings.length, warnings: findings.filter((f) => f.severity === 'warning' || f.severity === 'critical').length, verdict: brief?.type === 'artifact' ? brief.artifact.data.verdict : null, summary_plain: brief?.type === 'artifact' ? brief.artifact.data.summary_plain : null, case: first.case }
}

export const mockHealth = async (): Promise<Health> => ({ ok: true, openai: true, auth: null, modules: mockModules('video'), image_modules: mockModules('image') })
export async function mockCases(): Promise<CaseSummary[]> {
  const examples = await Promise.all(MOCK_CASES.map(async (name) => summary((await mockEvents(`mock-${name}`)).events)))
  return [...uploads.values()].map(summary).concat(examples).filter((c) => !removed.has(c.id))
}
export async function mockRemove(id: string): Promise<{ deleted: boolean }> {
  removed.add(id)
  return { deleted: true }
}
export async function mockRerun(id: string): Promise<CaseSummary> {
  return summary((await mockEvents(id)).events)
}

export async function mockUpload(file: File, onProgress: (fraction: number) => void): Promise<CaseSummary> {
  const still = file.type.startsWith('image/')
  if (!still && !file.type.startsWith('video/')) throw new Error('Preview accepts images and videos. Use a bundled example for other formats.')
  for (let i = 1; i <= 5; i++) { await mockDelay(100); onProgress(i / 5) }
  const events = await loadMock(still ? 'photo' : 'planted')
  const first = events.find((e) => e.type === 'case')!
  if (first.type !== 'case') throw new Error('Missing preview')
  const url = URL.createObjectURL(file)
  first.case = { ...first.case, id: `mock-upload-${crypto.randomUUID().slice(0, 8)}`, filename: file.name, size_bytes: file.size, media_url: url, original_url: url, thumb_url: still ? url : '/mock/thumb.jpg', sha256: undefined, md5: undefined, tags: { comment: 'Local file, synthetic findings. No analysis has run.' } }
  if (still) {
    const image = await readImage(url)
    first.case.image = { ...first.case.image!, width: image.naturalWidth, height: image.naturalHeight, megapixels: +(image.naturalWidth * image.naturalHeight / 1e6).toFixed(2) }
  } else {
    const video = await readVideo(url, 0)
    first.case.duration_s = video.duration
    first.case.video = { ...first.case.video!, width: video.videoWidth, height: video.videoHeight }
    video.removeAttribute('src'); video.load()
  }
  uploads.set(first.case.id, events)
  return summary(events)
}

function readImage(url: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const image = new Image()
    image.onload = () => resolve(image)
    image.onerror = () => reject(new Error('Could not load preview image'))
    image.src = url
  })
}
function readVideo(url: string, time: number): Promise<HTMLVideoElement> {
  return new Promise((resolve, reject) => {
    const video = document.createElement('video')
    video.muted = true; video.preload = 'auto'
    const timer = window.setTimeout(() => { video.removeAttribute('src'); video.load(); reject(new Error('Preview video did not load')) }, 8000)
    video.onerror = () => { clearTimeout(timer); reject(new Error('Could not load preview video')) }
    video.onloadeddata = () => {
      const target = Math.max(0, Math.min(time, video.duration - 0.05))
      if (target < 0.01) { clearTimeout(timer); resolve(video) }
      else { video.onseeked = () => { clearTimeout(timer); resolve(video) }; video.currentTime = target }
    }
    video.src = url
  })
}

export async function mockEnhance(id: string, box: { time_s?: number; x: number; y: number; w: number; h: number }): Promise<EnhanceResult> {
  await mockDelay(900)
  const first = (await mockEvents(id)).events.find((e) => e.type === 'case')!
  const info: CaseInfo = (first as Extract<CaseEvent, { type: 'case' }>).case
  const still = info.kind === 'image'
  const source = still ? await readImage(info.media_url!) : await readVideo(info.media_url!, box.time_s ?? 0)
  try {
    const w = source instanceof HTMLVideoElement ? source.videoWidth : source.naturalWidth
    const h = source instanceof HTMLVideoElement ? source.videoHeight : source.naturalHeight
    const sx = Math.max(0, Math.min(0.99, box.x)) * w, sy = Math.max(0, Math.min(0.99, box.y)) * h
    const sw = Math.max(1, Math.min(w - sx, box.w * w)), sh = Math.max(1, Math.min(h - sy, box.h * h))
    const canvas = document.createElement('canvas')
    const scale = Math.min(4, 1400 / Math.max(sw, sh))
    canvas.width = Math.round(sw * scale); canvas.height = Math.round(sh * scale)
    const ctx = canvas.getContext('2d')!
    const render = (filter: string) => { ctx.filter = filter; ctx.drawImage(source, sx, sy, sw, sh, 0, 0, canvas.width, canvas.height); return canvas.toDataURL('image/png') }
    const single = render('none')
    const enhanced = render('contrast(1.2) saturate(1.1)')
    return { single_url: single, enhanced_url: enhanced, frames_used: 1, scale: +scale.toFixed(1), source_px: [Math.round(sw), Math.round(sh)], time_s: box.time_s ?? 0, kind: still ? 'image' : 'video', lenses: [
      { name: 'Grayscale', url: render('grayscale(1)'), caption: 'Local preview: luminance only.' },
      { name: 'Contrast', url: render('contrast(2)'), caption: 'Local preview: amplified contrast.' },
      { name: 'Negative', url: render('invert(1)'), caption: 'Local preview: inverted channels.' },
      { name: 'Colour ×4', url: render('saturate(4)'), caption: 'Local preview: amplified colour differences.' },
      { name: 'Dark detail', url: render('brightness(1.8) contrast(0.8)'), caption: 'Local preview: brighter shadows.' },
      { name: 'Highlights', url: render('brightness(0.6) contrast(1.5)'), caption: 'Local preview: darker highlights.' },
    ] }
  } finally { if (source instanceof HTMLVideoElement) { source.removeAttribute('src'); source.load() } }
}

export function previewCaseId() { return `mock-${mockCaseName(mockName() ?? 'planted')}` }
