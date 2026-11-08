import { MODULES } from './plain'
import type { Answer, Artifact, CaseEvent, CaseInfo, Finding, MediaKind, ModuleInfo } from './types'

export const MOCK_CASES = ['planted', 'photo', 'generated', 'clean'] as const
export type MockCase = (typeof MOCK_CASES)[number]
export const mockCaseName = (name: string): MockCase => MOCK_CASES.find((n) => name === n || name === `mock-${n}`) ?? 'planted'

const videoNames = ['provenance', 'timeline', 'spectrum', 'enf', 'light', 'pulse', 'tamper', 'avsync', 'ocr', 'transcript', 'vision', 'geo', 'sounds', 'weather', 'brief']
const photoNames = ['provenance', 'jpeg', 'tamper', 'hidden', 'faces', 'synthetic', 'ocr', 'vision', 'geo', 'weather', 'brief']
export const mockModules = (kind: MediaKind): ModuleInfo[] => (kind === 'image' ? photoNames : videoNames).map((name) => ({
  name, kind, title: MODULES[name]?.name.toUpperCase() ?? name.toUpperCase(), description: MODULES[name]?.does ?? '',
  ...(name === 'pulse' ? { min_effort: 'medium' as const } : {}),
}))

/** Recorded measurements plus explicitly synthetic UI examples. No backend URLs survive. */
export function completeShowcase(recorded: CaseEvent[], requested: string, full: boolean): CaseEvent[] {
  const name = mockCaseName(requested)
  const still = name === 'photo' || name === 'generated'
  const clean = name === 'clean'
  const kind = still ? 'image' : 'video'
  const source = recorded.find((e) => e.type === 'case')!
  if (source.type !== 'case') return []
  const photo = name === 'generated' ? '/mock/generated.png' : '/mock/photo.jpg'
  const info: CaseInfo = {
    ...source.case, id: `mock-${name}`, filename: still ? `${name === 'generated' ? 'ai_generated' : 'edited_photo'}.${name === 'generated' ? 'png' : 'jpg'}` : clean ? 'camera_original.mp4' : 'planted_evidence.mp4',
    kind, duration_s: still ? 0 : 30, has_audio: !still,
    thumb_url: still ? photo : '/mock/thumb.jpg', media_url: still ? photo : '/mock/planted.mp4', original_url: still ? photo : '/mock/planted.mp4',
    video: still ? null : source.case.video,
    image: still ? { width: name === 'generated' ? 1024 : 1600, height: name === 'generated' ? 1024 : 1200, format: name === 'generated' ? 'PNG' : 'JPEG', mode: 'RGB', bits: 8, frames: 1, orientation: 1, megapixels: name === 'generated' ? 1.05 : 1.92 } : null,
    modules: full || still || clean ? mockModules(kind) : source.case.modules,
    sha256: 'a3b8d712f9064ea2396cc657a48b19d0538e406e977c24cb8d93e10a5726f40c', md5: '745c2e8b13a6415d9f7e236b80a4cd19', ingested_at: 1794067869,
    container: { format_name: still ? 'image2' : 'mov,mp4,m4a,3gp,3g2,mj2', format_long_name: still ? 'Still image' : 'QuickTime / MOV', bit_rate: 280113, size_bytes: source.case.size_bytes, duration_s: still ? 0 : 30 },
    streams: still ? [] : [{ index: 0, type: 'video', codec: 'h264', profile: 'High', width: 320, height: 240, fps: 30, pix_fmt: 'yuv420p', bit_rate: 78113 }, { index: 1, type: 'audio', codec: 'aac', sample_rate: 48000, channels: 1, channel_layout: 'mono', language: 'eng' }],
    tags: { make: 'Apple', model: 'iPhone 15 Pro', creation_time: '2024-03-12T23:30:00Z', software: 'Preview fixture', location: '+51.5041-000.1187/', comment: 'Synthetic frontend showcase — not a forensic assessment of this media.' },
  }
  let events: CaseEvent[] = [{ ...source, case: info }]
  const emit = (e: Record<string, unknown>) => events.push({ ...e, seq: events.length, ts: source.ts + events.length * 0.1 } as CaseEvent)
  const artifact = (id: string, artifactKind: string, title: string, data: unknown, url: string | null = null, time: number | null = null) => {
    const [module, artifactName] = id.split(':')
    emit({ type: 'artifact', artifact: { id, module, name: artifactName, kind: artifactKind, title, data, url, time_s: time } })
  }
  const finding = (id: string, module: string, title: string, data: unknown, severity: Finding['severity'] = 'notice', tags: string[] = [], time: number | null = null, inferred = false) => emit({
    type: 'finding', finding: { id, module, title, detail: 'Synthetic preview finding. This example illustrates the interface; it is not an analysis of the displayed media.', data, severity, tags, time_s: time, time_end_s: null, kind: inferred ? 'inferred' : 'measured' },
  })
  // Keep the recorded video measurements and geolocation/dossier examples.
  const keep = recorded.filter((e) => e.type !== 'case' && e.type !== 'done' && e.type !== 'media' && (!still || (e.type === 'artifact' && (e.artifact.module === 'geo' || e.artifact.kind === 'dossier')) || (e.type === 'finding' && e.finding.module === 'geo')))
  events.push(...keep.filter((e) => !clean || e.type === 'artifact' || e.type === 'module' || e.type === 'log'))
  if (full || still || clean) {
    finding('F100', 'provenance', 'Taken on Apple iPhone 15 Pro', { make: 'Apple', model: 'iPhone 15 Pro', software: 'iOS 17.4' }, 'info', ['device'])
    finding('F101', 'provenance', 'GPS position: South Bank, London', { lat: 51.5041, lon: -0.1187, accuracy_m: 35, heading_deg: 295, alt_m: 12, place: 'South Bank, London, UK' }, 'notice', ['gps', 'location'])
    finding('F102', 'provenance', 'Camera clock: 12 March 2024, 23:30 UTC', { creation_time_utc: '2024-03-12T23:30:00Z', date_time_original: '2024-03-12T23:30:00Z', timezone: 'Europe/London' }, 'info', ['time'])
    finding('F103', 'ocr', 'Text in frame: SE1 7PB · SOUTH BANK', { text: 'SOUTH BANK · SE1 7PB · OPEN 09:00–23:00', confidence: 0.96 }, 'notice', ['location', 'text'], still ? null : 6.2)
    artifact('ocr:readings', 'table', 'Words, places and contact clues', { columns: ['time', 'text', 'confidence', 'type'], rows: [[6.2, 'SOUTH BANK', '96%', 'sign'], [6.2, 'SE1 7PB', '94%', 'UK postcode'], [14, 'OPEN 09:00–23:00', '88%', 'opening hours']] })
    artifact('vision:analysis', 'json', 'AI eyewitness: structured observations', { scene: 'Illustrative riverside scene after dark', weather: 'Dry, with cloud cover', observations: ['Street lighting', 'Pedestrian walkway', 'Landmark silhouette'], confidence: 'moderate', preview: true })
    artifact('weather:hourly', 'table', 'Weather record at the claimed time', { columns: ['time (UTC)', 'temperature', 'rain', 'cloud', 'wind'], rows: [['23:00', '8.2 °C', '0 mm', '62%', '12 km/h'], ['00:00', '7.8 °C', '0 mm', '68%', '10 km/h']] })
    finding('F104', 'weather', 'Weather and visible conditions agree', { temperature_c: 8.2, precipitation_mm: 0, cloud_cover: 62, source: 'Bundled synthetic weather fixture' }, 'info', ['weather'])
    artifact('tamper:ela', 'heatmap', 'Error-level analysis: inconsistent region', { overlay: true, caption: 'Illustrative map of recompression residuals', box: { x: 0.54, y: 0.23, w: 0.25, h: 0.3 } }, '/mock/heatmap.svg', still ? null : 10)
    artifact('tamper:noise', 'heatmap', 'Amplified grain', { overlay: true, caption: 'Synthetic residual view for overlay controls' }, '/mock/noise.svg', still ? null : 10)
    artifact('tamper:clone', 'image', 'Matched source and copied region', { box: { x: 0.54, y: 0.23, w: 0.25, h: 0.3 }, caption: 'Example region annotation; not a detector result.' }, still ? photo : '/mock/thumb.jpg', still ? null : 10)
    if (!still) {
      finding('F105', 'enf', '50 Hz grid: dated to 23:10:00 UTC', { grid_hz: 50, dated_start: '2024-03-12T23:10:00Z', score: 0.94 }, 'warning', ['time', 'contradiction'], 0)
      finding('F106', 'pulse', 'Skin colour suggests a pulse near 72 bpm', { bpm: 72, confidence: 'weak', snr_db: 6.3 }, 'notice', ['people'], 8)
      finding('F107', 'transcript', 'Speech detected: English', { language: 'English', confidence: 0.94 }, 'info', ['speech'], 0.5)
      finding('F108', 'avsync', 'Sound follows the picture by about 250 ms', { lag_s: 0.25, agreement: 0.89 }, 'warning', ['avsync'], 20)
      finding('F109', 'sounds', 'An AI listener hears a crowd and distant bells', { environment: 'Outdoor pedestrian crowd', location_clues: ['Distant clock bells'], language: 'English' }, 'notice', ['sound', 'location'], 3, true)
      artifact('pulse:trace', 'series', 'Skin colour pulse trace', { x: Array.from({ length: 240 }, (_, i) => i / 8), x_unit: 's', y_unit: 'relative', series: [{ name: 'skin signal', y: Array.from({ length: 240 }, (_, i) => Math.sin(i / 8 * Math.PI * 2 * 1.2) + Math.sin(i * 0.3) * 0.15) }] })
      artifact('pulse:heart_rate', 'series', 'Estimated heart rate', { x: [0, 5, 10, 15, 20, 25, 30], x_unit: 's', y_unit: 'bpm', series: [{ name: 'heart rate', y: [71, 72, 73, 72, 71, 73, 72] }] })
      artifact('pulse:magnified', 'video', 'Colour-magnification preview', { caption: 'Original bundled clip used as a UI preview; no pulse processing was run.' }, '/mock/planted.mp4')
      artifact('avsync:correlation', 'series', 'Audio and motion alignment', { x: [-0.5, -0.25, 0, 0.1, 0.25, 0.4, 0.5], x_unit: 's', y_unit: 'correlation', series: [{ name: 'alignment', y: [0.1, 0.15, 0.22, 0.5, 0.89, 0.4, 0.12] }] })
      artifact('sounds:scene', 'text', 'Soundscape and audible location clues', { text: '## Synthetic listening example\n\nAn outdoor crowd, footsteps and clock bells. Voices appear to speak British English.\n\nThese are fixture observations for developing the interface, not an assessment of the bundled clip.' })
    } else {
      finding('F110', 'jpeg', 'JPEG quality is approximately 88', { quality: 88, progressive: true, subsampling: '4:2:0' }, 'info', ['quality'])
      finding('F111', 'faces', 'Two faces; one large enough to recognise', { faces: [{ box: [0.15, 0.18, 0.19, 0.3], sharpness: 142 }, { box: [0.64, 0.22, 0.1, 0.14], sharpness: 18 }], recognisable: 1 }, 'notice', ['people'])
      finding('F112', 'tamper', 'Copied region: two independent traces agree', { box: { x: 0.54, y: 0.23, w: 0.25, h: 0.3 }, matches: 47, offset: [300, 140] }, 'critical', ['clone', 'splice'])
      finding('F113', 'provenance', 'Processed by Adobe Photoshop', { software: 'Adobe Photoshop', history: ['Camera original', 'Region copied', 'JPEG re-saved'] }, 'warning', ['editor', 'history'])
      finding('F114', 'hidden', 'Appended data: a readable note after the image', { bytes: 186, label: 'Preview text attachment', offset: 184392 }, 'warning', ['hidden'])
      artifact('hidden:attachment', 'file', 'Data carried after the image', { bytes: 186, label: 'Preview note (text/plain)', head_hex: '45 41 52 57 49 54 4e 45 53 53 20 50 52 45 56 49 45 57', strings: ['EARWITNESS PREVIEW', 'Bundled fixture; no private data'] }, '/mock/attachment.txt')
      artifact('hidden:metadata', 'json', 'Hidden metadata and byte accounting', { file_bytes: 184578, image_bytes: 184392, appended_bytes: 186, comments: ['Synthetic UI fixture'], transparency: { hidden_rgb: false } })
      artifact('jpeg:ghosts', 'series', 'JPEG ghost scan across recompression qualities', { x: [40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95], x_unit: 'quality', y_unit: 'error', series: [{ name: 'whole image', y: [22, 20, 18, 15, 13, 5, 11, 12, 9, 8, 3, 7] }, { name: 'suspect region', y: [28, 25, 21, 18, 7, 16, 17, 15, 12, 10, 8, 9] }] })
      artifact('jpeg:tables', 'table', 'Quantisation tables and encoder habits', { columns: ['property', 'value'], rows: [['quality estimate', '88'], ['progressive', 'yes'], ['subsampling', '4:2:0'], ['Huffman tables', 'optimised'], ['8×8 grid', 'shifted in marked region']] })
      artifact('faces:eyes0', 'image', 'Enlarged eye region: reflection inspection', { caption: 'Preview image for the eye-reflection exhibit.' }, photo)
      artifact('provenance:thumbnail', 'image', 'Embedded camera thumbnail', { caption: 'The stored thumbnail can be compared with the current picture.' }, photo)
      if (name === 'generated') {
        finding('F115', 'synthetic', 'Stable Diffusion recipe declares an AI generator', { generator_name: 'Stable Diffusion XL', generators: ['Stable Diffusion XL'], prompt: 'A quiet riverside city after sunset', seed: 314159, steps: 30, model: 'SDXL preview' }, 'warning', ['declared', 'synthetic'])
        artifact('synthetic:recipe', 'json', 'Generation recipe stored in the picture', { generator: 'Stable Diffusion XL', prompt: 'A quiet riverside city after sunset', negative_prompt: 'blurry', seed: 314159, sampler: 'DPM++ 2M Karras', steps: 30, cfg_scale: 7, preview: true })
      }
    }
    const answers: Answer[] = [
      { key: 'edited', question: 'Has it been edited?', answer: clean ? 'No signs of editing in this example.' : still ? 'A copied region and editing history agree.' : 'Picture and sound both break at 0:20.', confidence: 'strong', evidence: still ? ['F112', 'F113'] : clean ? ['F100'] : ['F007', 'F010'], time_s: still || clean ? null : 20, ai: false },
      { key: 'when', question: 'When was it recorded?', answer: still ? '12 March 2024, 23:30 UTC (camera clock).' : 'The grid trace points to 12 March 2024, 23:10 UTC.', confidence: still ? 'weak' : 'strong', evidence: [still ? 'F102' : 'F105'], time_s: null, ai: false },
      { key: 'where', question: 'Where was it taken?', answer: 'South Bank, London, within about 600 metres.', confidence: 'moderate', evidence: ['F101', 'X010'], time_s: still ? null : 3, ai: true },
      { key: 'device', question: 'What took it?', answer: 'The file labels name an Apple iPhone 15 Pro.', confidence: 'moderate', evidence: ['F100'], time_s: null, ai: false },
      { key: 'people', question: 'Who is in it?', answer: still ? 'Two faces; one is large enough to inspect.' : 'A weak skin-colour signal suggests about 72 bpm.', confidence: 'weak', evidence: [still ? 'F111' : 'F106'], time_s: still ? null : 8, ai: false },
      { key: 'hidden', question: 'Anything hidden?', answer: still ? 'A note is appended after the image data.' : 'A high-frequency tone and a keypad sequence, 1984#.', confidence: 'strong', evidence: still ? ['F114'] : ['F008', 'F011'], time_s: still ? null : 4, ai: false },
      { key: 'text', question: 'What can be read?', answer: 'SOUTH BANK, postcode SE1 7PB and opening hours.', confidence: 'moderate', evidence: ['F103'], time_s: still ? null : 6.2, ai: false },
      ...(still ? [] : [{ key: 'speech', question: 'What was said?', answer: 'English speech with timed captions.', confidence: 'moderate', evidence: ['F107'], time_s: 0.5, ai: true }]),
      ...(name === 'generated' ? [{ key: 'ai', question: 'Was it made by AI?', answer: 'The metadata declares Stable Diffusion XL, with a prompt and seed.', confidence: 'strong', evidence: ['F115'], time_s: null, ai: false }] : []),
    ]
    const verdict = clean ? 'NO SIGNS OF EDITING' : name === 'generated' ? 'AI-GENERATED' : 'EDITED'
    artifact('brief:brief', 'brief', 'Case brief', { verdict, corroborated: !clean, answers, summary_plain: clean ? 'No signs of editing in this preview case.' : name === 'generated' ? 'The picture declares an AI generator and carries its recipe.' : still ? 'A copied region, stale thumbnail and editor history point to an altered photograph.' : 'Two recordings joined at 0:20, with hidden tones and a clock that disagrees with the grid.', headline: 'Every signal has a story to tell', narrative: 'This is a synthetic UI showcase. The example brings together every kind of result so you can explore the interface without running analysis.', text: still ? 'A marked region has matching compression and grain traces [F112]. The file also records editing software [F113].\n\nThe location label places the camera in London [F101]; treat metadata as a lead until independently checked.' : 'The picture and room tone break together [F007] [F010]. A 50 Hz reference trace supplies an independent time estimate [F105].\n\nSpeech, visible text and the AI location estimate offer leads to verify against the original source [F103] [F107].', bullets: ['INTEGRITY: Independent signals are shown side by side', 'WHERE: GPS and blind location estimate can be compared', 'HIDDEN: Inspect exhibits and download the bundled example'], next_steps: ['Request the camera original and compare its fingerprint.', 'Inspect the marked region with forensic lenses.', 'Check the location estimate against an independent source.'], model: 'local UI fixture (no model called)' })
    for (const m of info.modules) emit({ type: 'module', module: m.name, status: 'done', progress: 1, elapsed_s: 1.2 + info.modules.indexOf(m) * 0.7, message: 'Preview complete' })
    emit({ type: 'log', module: 'core', message: 'Synthetic frontend showcase loaded locally. No analysis or external API calls were made.' })
  }
  emit({ type: 'done' })
  const localUrl = (a: Artifact): string | null => {
    if (!a.url) return null
    if (a.id.startsWith('geo:ref')) return '/mock/reference.svg'
    if (a.kind === 'spectrogram') return '/mock/spectrogram.png'
    if (a.kind === 'filmstrip') return '/mock/filmstrip.jpg'
    if (a.url.startsWith('/mock/')) return a.url
    if (a.kind === 'heatmap') return '/mock/heatmap.svg'
    if (a.kind === 'video') return '/mock/planted.mp4'
    return still ? photo : '/mock/thumb.jpg'
  }
  events = events.map((e, seq) => {
    if (e.type === 'media') return { ...e, seq, media_url: info.media_url }
    if (e.type === 'artifact') return { ...e, seq, artifact: { ...e.artifact, url: localUrl(e.artifact), ...(e.artifact.kind === 'filmstrip' ? { data: { tile_width: 160, tile_height: 120, times: Array.from({ length: 15 }, (_, i) => i * 2) } } : {}) } }
    return { ...e, seq }
  })
  // The clean example demonstrates the alternative verdict without contradictory alarms.
  if (clean) events = events.filter((e) => e.type !== 'finding' || e.finding.severity === 'info' || e.finding.id === 'F101' || e.finding.id === 'F103').map((e, seq) => ({ ...e, seq }))
  return events
}
