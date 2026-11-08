import type { CaseEvent } from './types'
import { completeShowcase } from './mockShowcase'

/** Dev-only: ?mock=planted | ?mock=iphone replays a recorded event stream. */
export function mockName(): string | null {
  if (!import.meta.env.DEV) return null
  return new URLSearchParams(location.search).get('mock')
}

export async function loadMock(name: string): Promise<CaseEvent[]> {
  if (!import.meta.env.DEV) return []
  const raw =
    name === 'iphone'
      ? (await import('../../mock/iphone.events.jsonl?raw')).default
      : (await import('../../mock/planted.events.jsonl?raw')).default
  const events = raw
    .split('\n')
    .filter((l) => l.trim())
    .map((l) => JSON.parse(l) as CaseEvent)
  const full = new URLSearchParams(location.search).has('extras')
  if (full) events.push(...extras(events))
  const showcase = completeShowcase(events, name, full)
  if (new URLSearchParams(location.search).get('geo') === 'unknown') {
    for (const event of showcase) {
      if (event.type === 'finding' && event.finding.module === 'geo') {
        event.finding.title = 'Location unknown'
        event.finding.detail = 'The AI could not narrow down the location from the available evidence.'
        event.finding.data = { ...event.finding.data, place: 'Unknown', latitude: 0, longitude: 0, radius_km: 20000, confidence: 'low', verified: false, candidates: [{ place: 'Unknown', latitude: 0, longitude: 0, radius_km: 20000 }] }
      }
      if (event.type === 'artifact' && event.artifact.id === 'brief:brief' && event.artifact.data) {
        const data = event.artifact.data
        if (Array.isArray(data.answers)) data.answers = data.answers.map((answer: { key: string }) => answer.key === 'where' ? { ...answer, answer: 'The AI could not narrow down the location. The file contains an independent GPS position.', confidence: 'unknown' } : answer)
        data.bullets = (data.bullets ?? []).map((bullet: string) => bullet.startsWith('WHERE:') ? 'WHERE: The AI location is unknown; independent GPS is available' : bullet)
      }
      if (event.type === 'artifact' && event.artifact.id === 'brief:dossier' && event.artifact.data && Array.isArray(event.artifact.data.lines)) {
        event.artifact.data.lines = event.artifact.data.lines.map((line: { category: string }) => line.category === 'where' ? { ...line, text: 'The AI could not narrow down the location.', certainty: 'unknown' } : line)
      }
      if (event.type === 'artifact' && event.artifact.id === 'geo:map') {
        event.artifact.data = { points: [{ lat: 0, lon: 0, label: 'Best guess: Unknown', accuracy_m: 20000000, kind: 'ai' }, { lat: 0, lon: 0, label: 'Lead: Unknown', accuracy_m: 20000000, kind: 'ai-candidate' }] }
      }
    }
  }
  return showcase
}

/** Synthetic events for kinds the recorded streams don't contain yet (dev aid, `&extras`). */
function extras(events: CaseEvent[]): CaseEvent[] {
  let seq = events.length
  const ts = events[events.length - 1]?.ts ?? 0
  const thumb = (events[0] as Extract<CaseEvent, { type: 'case' }>).case.thumb_url
  const mk = (e: Record<string, unknown>) => ({ seq: seq++, ts, ...e }) as CaseEvent
  return [
    mk({ type: 'module', module: 'brief', status: 'running', progress: 0.5, message: 'asking the analyst' }),
    mk({
      type: 'finding',
      finding: {
        id: 'X001', module: 'vision', severity: 'notice', kind: 'inferred', tags: ['vision'],
        title: 'Street sign reads "ST LEONARD\'S ST"', detail: 'Model-read text on a sign in the upper-left of frame. Unverified; check with ENHANCE.',
        time_s: 6.2, time_end_s: 9.0, data: {},
      },
    }),
    mk({
      type: 'artifact',
      artifact: { id: 'vision:objects', module: 'vision', name: 'objects', kind: 'bars', title: 'Objects seen (frames)', url: null, time_s: null,
        data: { labels: ['person', 'car', 'street lamp', 'sign', 'bicycle'], values: [412, 230, 180, 64, 12] } },
    }),
    mk({
      type: 'artifact',
      artifact: { id: 'vision:frames', module: 'vision', name: 'frames', kind: 'image', title: 'Key frame', url: thumb, time_s: 3.0,
        data: { caption: 'Key frame at 3.0 s', time_s: 3.0 } },
    }),
    mk({
      type: 'artifact',
      artifact: { id: 'vision:heat', module: 'vision', name: 'heat', kind: 'heatmap', title: 'Attention heatmap', url: thumb, time_s: 5.0,
        data: { caption: 'Saliency', time_s: 5.0, overlay: true } },
    }),
    mk({
      type: 'artifact',
      artifact: { id: 'vision:notes', module: 'vision', name: 'notes', kind: 'text', title: 'Scene description', url: null, time_s: null,
        data: { text: 'A narrow street at night, wet tarmac, sodium lamps. One pedestrian crosses left to right at about 7 s.\n\nNo faces are resolvable at this resolution.' } },
    }),
    mk({
      type: 'artifact',
      artifact: { id: 'light:grid', module: 'light', name: 'grid', kind: 'series', title: 'Grid frequency vs reference', url: null, time_s: null,
        data: { x_iso: Array.from({ length: 40 }, (_, i) => new Date(Date.UTC(2024, 2, 12, 23, 30, i * 15)).toISOString()),
          series: [
            { name: 'evidence', y: Array.from({ length: 40 }, (_, i) => 50 + Math.sin(i / 4) * 0.03) },
            { name: 'national grid', y: Array.from({ length: 40 }, (_, i) => 50 + Math.sin(i / 4 + 0.2) * 0.028) },
          ], y_unit: 'Hz' } },
    }),
    mk({
      type: 'artifact',
      artifact: { id: 'brief:brief', module: 'brief', name: 'brief', kind: 'brief', title: 'Analyst brief', url: null, time_s: null,
        data: {
          verdict: 'EDITED',
          corroborated: true,
          text: 'Two recordings joined at 20 s: picture and room tone break at the same instant [F007] [F010], and a stretch of exact digital silence sits at 12 s [F009].',
          bullets: [
            'INTEGRITY: Hard cut and +17 dB room-tone jump coincide at 20.0 s [F007] [F010]',
            'DEVICE: Re-encoded by FFmpeg, no camera metadata [F001] [F003]',
            'HIDDEN: Keypad tones 1984# at 4.0 s [F011]; near-ultrasonic beacon at 18.5 kHz [F008]',
          ],
          headline: 'A spliced clip carrying a dialled code and a hidden beacon',
          narrative: 'The file reads as a social-media re-upload of two separately recorded segments.',
          next_steps: ['Request the camera original', 'Check the 18.5 kHz tone against known retail beacons'],
          model: 'gpt-x',
        } },
    }),
    mk({
      type: 'artifact',
      artifact: { id: 'transcript:captions', module: 'transcript', name: 'captions', kind: 'captions', title: 'Transcript', url: null, time_s: null,
        data: { segments: [ { start: 0.5, end: 3.2, text: 'Okay, it is recording now.' }, { start: 3.6, end: 6.0, text: 'Type the code in, quick.' }, { start: 20.2, end: 24.0, text: 'Different room, different voice.' } ] } },
    }),
    mk({
      type: 'artifact',
      artifact: { id: 'light:flicker', module: 'light', name: 'flicker', kind: 'series', title: 'Lamp flicker', url: null, time_s: null,
        data: { x: Array.from({ length: 60 }, (_, i) => i / 2), x_unit: 's', series: [{ name: 'brightness', y: Array.from({ length: 60 }, (_, i) => 100 + Math.sin(i) * 3) }], marks: { cut: 20, silence: 12 } } },
    }),
    ...geoMock(mk, thumb),
    mk({ type: 'module', module: 'brief', status: 'done', progress: 1, elapsed_s: 3.1 }),
    mk({ type: 'module', module: 'light', status: 'error', progress: 0.3, message: 'flicker search failed', trace: 'Traceback (most recent call last):\n  File "light.py", line 42, in run\nValueError: not enough frames' }),
  ]
}

/** Dev-only geolocation + dossier sample (shapes copied from modules/geo.py and brief.py). */
function geoMock(mk: (e: Record<string, unknown>) => CaseEvent, thumb: string | null): CaseEvent[] {
  const cands = [
    { place: 'South Bank, London, UK', landmark: 'London Eye', latitude: 51.5033, longitude: -0.1196, radius_km: 0.6, probability: 0.55, evidence: ['Large illuminated wheel', 'Thames embankment lamps'] },
    { place: 'Jubilee Gardens, London, UK', landmark: 'Jubilee Gardens', latitude: 51.5045, longitude: -0.1168, radius_km: 0.4, probability: 0.25, evidence: ['Open lawn near the wheel'] },
    { place: 'Liverpool waterfront, UK', landmark: 'Wheel of Liverpool', latitude: 53.3972, longitude: -2.9916, radius_km: 1.5, probability: 0.1, evidence: ['Observation wheel'] },
  ]
  return [
    mk({ type: 'artifact', artifact: { id: 'geo:frame0', module: 'geo', name: 'frame0', kind: 'image', title: 'Frame analysed @ 3.0s', url: thumb, time_s: 3.0, data: null } }),
    mk({ type: 'artifact', artifact: { id: 'geo:frame1', module: 'geo', name: 'frame1', kind: 'image', title: 'Frame analysed @ 9.0s', url: thumb, time_s: 9.0, data: null } }),
    mk({ type: 'artifact', artifact: { id: 'geo:clues', module: 'geo', name: 'clues', kind: 'text', title: 'Non-visual clues given to the analyst', url: null, time_s: null, data: { text: 'Voices speaking English (British accents) are audible.\nAn AI listener heard: {"environment": "outdoor crowd, river", "location_clues": ["Big Ben chimes"]}\nThe lighting flickers at 100 Hz: the mains grid is 50 Hz.' } } }),
    mk({ type: 'artifact', artifact: { id: 'geo:ref0_wiki', module: 'geo', name: 'ref0_wiki', kind: 'image', title: 'Reference: Wikipedia photo of London Eye (candidate 1)', url: thumb, time_s: null, data: null } }),
    mk({ type: 'artifact', artifact: { id: 'geo:ref0_sat', module: 'geo', name: 'ref0_sat', kind: 'image', title: 'Satellite at candidate 1: South Bank, London, UK', url: thumb, time_s: null, data: null } }),
    mk({ type: 'artifact', artifact: { id: 'geo:ref2_wiki', module: 'geo', name: 'ref2_wiki', kind: 'image', title: 'Reference: Wikipedia photo of Wheel of Liverpool (candidate 3)', url: thumb, time_s: null, data: null } }),
    mk({ type: 'artifact', artifact: { id: 'geo:leads', module: 'geo', name: 'leads', kind: 'table', title: 'Candidate places and how they held up', url: null, time_s: null, data: {
      columns: ['#', 'place', 'landmark', 'proposed', 'match', 'matches', 'mismatches'],
      rows: [
        [1, cands[0].place, 'London Eye', '55%', '86%', 'wheel spoke pattern; capsule shape; embankment lamp posts', 'tree line slightly denser'],
        [2, cands[1].place, 'Jubilee Gardens', '25%', '61%', 'open lawn', 'wheel seen from wrong angle'],
        [3, cands[2].place, 'Wheel of Liverpool', '10%', '18%', '', 'wheel too small; no river wall; wrong skyline'],
      ] } } }),
    mk({ type: 'artifact', artifact: { id: 'geo:map', module: 'geo', name: 'map', kind: 'map', title: 'Geolocation: leads, verified answer and GPS', url: null, time_s: null, data: { points: [
      ...cands.map((c, i) => ({ lat: c.latitude, lon: c.longitude, label: `Lead ${i + 1}: ${c.place}`, accuracy_m: c.radius_km * 1000, kind: 'ai-candidate' })),
      { lat: 51.5033, lon: -0.1196, label: 'Verified: South Bank, London, UK', accuracy_m: 600, kind: 'ai' },
      { lat: 51.5041, lon: -0.1187, label: 'GPS (metadata)', accuracy_m: 35, kind: 'gps' },
    ] } } }),
    mk({ type: 'finding', finding: { id: 'X010', module: 'geo', severity: 'notice', kind: 'inferred', tags: ['location', 'osint'], time_s: 3.0, time_end_s: null,
      title: 'Located: South Bank, London, UK (±1 km, high confidence)', detail: 'Wheel geometry and embankment match the London Eye. Checked against reference imagery: 86% match.',
      data: { place: 'South Bank, London, UK', latitude: 51.5033, longitude: -0.1196, radius_km: 0.6, confidence: 'high', verified: true, match: 0.86, candidates: cands, error_km: 0.12 } } }),
    mk({ type: 'artifact', artifact: { id: 'brief:dossier', module: 'brief', name: 'dossier', kind: 'dossier', title: 'What this video gives away', url: null, time_s: null, data: {
      headline: 'You think this is just a clip. It is a confession.',
      lines: [
        { text: 'You were on the South Bank in London, facing the London Eye, within a few hundred metres.', evidence: ['X010'], certainty: 'likely', category: 'where' },
        { text: 'Someone near you typed 1-9-8-4-# on a phone keypad at 0:04. We heard every digit.', evidence: ['F011'], certainty: 'measured', category: 'hidden' },
        { text: 'A tone too high for you to hear runs under the whole recording, like a tracking beacon.', evidence: ['F008'], certainty: 'measured', category: 'surroundings' },
        { text: 'At 0:20 the picture and the sound break at the same instant. This is two recordings joined together.', evidence: ['F007', 'F010'], certainty: 'measured', category: 'integrity' },
        { text: 'The person on camera had a resting heart rate of about 72 beats per minute.', evidence: ['F007'], certainty: 'possible', category: 'body' },
      ],
      closing: 'You only meant to share a moment. You shared where you were, who was near you, and what you did to it.',
      heart_bpm: 72,
    } } }),
  ]
}
