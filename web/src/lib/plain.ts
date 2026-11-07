// Plain-language layer: how each check, chart and term is explained to a non-expert.
import type { Artifact, Finding } from './types'

export type Accent = 'blue' | 'lblue' | 'orange' | 'yellow' | 'pink' | 'purple' | 'green' | 'red' | 'teal' | 'grey'

export interface ModulePlain {
  name: string // plain name
  icon: string // Font Awesome solid icon
  accent: Accent
  does: string // one line: what it does
  reveals: string // what it can reveal
  limits: string // honest limits
}

export const MODULES: Record<string, ModulePlain> = {
  provenance: {
    name: 'File history',
    icon: 'file-shield',
    accent: 'lblue',
    does: 'Reads the labels stored inside the file: which device made it, when, where, and which apps touched it since.',
    reveals: 'Phone model, recording date, GPS position, and signs the file was re-saved by an editor or messaging app.',
    limits: 'Labels can be stripped or faked, so they are leads until something else agrees with them.',
  },
  timeline: {
    name: 'Frame timing',
    icon: 'film',
    accent: 'blue',
    does: 'Checks every frame: how evenly they are spaced, where the picture suddenly changes, and how the video was compressed.',
    reveals: 'Hard cuts, dropped or duplicated frames, and joins between two recordings.',
    limits: 'A camera that restarts or a scene with a sudden flash can look like a cut.',
  },
  spectrum: {
    name: 'Sound analysis',
    icon: 'wave-square',
    accent: 'teal',
    does: 'Turns the sound into a picture and searches it for tones, silences and sudden changes in the background noise.',
    reveals: 'Hidden high-pitched tones, keypad beeps, pasted-in silence, and joins between recordings.',
    limits: 'Heavy compression and loud music can hide quiet clues.',
  },
  enf: {
    name: 'Electrical hum',
    icon: 'bolt',
    accent: 'yellow',
    does: 'Listens for the faint hum of the power grid, which drifts in a pattern unique to each moment in time.',
    reveals: 'The region (50 Hz or 60 Hz grid) and sometimes the exact date and time of recording.',
    limits: 'Phones filter out most hum; many recordings have none to measure.',
  },
  light: {
    name: 'Light flicker',
    icon: 'lightbulb',
    accent: 'yellow',
    does: 'Finds the invisible flicker of mains-powered lamps in the picture and compares the scene lighting with the sun.',
    reveals: 'Which power grid the lights run on, the camera sensor speed, and whether day or night matches the claimed time.',
    limits: 'Needs artificial light; daylight and LED drivers without flicker give no signal.',
  },
  pulse: {
    name: 'Heartbeat',
    icon: 'heart-pulse',
    accent: 'pink',
    does: 'Reads the tiny colour changes in skin that happen with every heartbeat.',
    reveals: 'Whether a face belongs to a living person, and their approximate heart rate.',
    limits: 'Movement, make-up, compression and poor light make this a weak signal; treat it as a hint.',
  },
  tamper: {
    name: 'Pixel tampering',
    icon: 'wand-magic-sparkles',
    accent: 'purple',
    does: 'Looks for parts of a frame that were copied, pasted or saved differently from the rest.',
    reveals: 'Cloned objects, patched areas and inserted graphics.',
    limits: 'Phone compression creates its own patterns; the maps need a human eye.',
  },
  avsync: {
    name: 'Sound vs picture',
    icon: 'arrows-left-right',
    accent: 'teal',
    does: 'Lines up sharp sounds with visible events to see whether audio and video belong together.',
    reveals: 'Dubbed or replaced audio, and audio slipped out of time.',
    limits: 'Needs clear events such as claps, doors or footsteps.',
  },
  ocr: {
    name: 'Text in the picture',
    icon: 'font',
    accent: 'orange',
    does: 'Reads signs, screens, number plates and labels visible in the frames.',
    reveals: 'Postcodes, phone numbers, plates, web addresses and dates that place or date the video.',
    limits: 'Small, blurred or angled text is often missed or misread.',
  },
  transcript: {
    name: 'Speech',
    icon: 'comment-dots',
    accent: 'orange',
    does: 'Writes down what is said, with timestamps, and identifies the language.',
    reveals: 'Names, places and times mentioned out loud.',
    limits: 'Uses an online AI service; needs it switched on, and can mishear.',
  },
  vision: {
    name: 'AI eyewitness',
    icon: 'robot',
    accent: 'purple',
    does: 'An AI model looks at key frames and describes the place, time of day and notable details.',
    reveals: 'Leads on location, vehicles, uniforms, weather and signs of manipulation.',
    limits: 'AI can be confidently wrong. Everything it says is labelled as AI and needs checking.',
  },
  geo: {
    name: 'Location from pixels',
    icon: 'earth-europe',
    accent: 'green',
    does: 'Estimates where the video was filmed using only what is visible: architecture, signs, plants, road markings.',
    reveals: 'A likely country, region or place, with a radius of uncertainty.',
    limits: 'An estimate, judged blind by AI; compare it with GPS when the file has it.',
  },
  sounds: {
    name: 'Sound sleuth',
    icon: 'ear-listen',
    accent: 'teal',
    does: 'An AI listener names the background sounds, accents and languages it can hear.',
    reveals: 'Sirens, birds, trains, announcements, crowds and voices that point to a place or a time of day.',
    limits: 'AI listening can mishear; each sound is a lead to check, not proof.',
  },
  weather: {
    name: 'Weather witness',
    icon: 'cloud-sun-rain',
    accent: 'blue',
    does: 'Checks the real recorded weather for the claimed place and time against what the footage shows.',
    reveals: 'Rain, wind, cloud and temperature that confirm or contradict when and where it was filmed.',
    limits: 'Needs a place and a time to look up; local showers can differ from the nearest weather station.',
  },
  brief: {
    name: 'Case summary',
    icon: 'scale-balanced',
    accent: 'red',
    does: 'Weighs every finding and writes the verdict and plain answers.',
    reveals: 'Whether independent checks agree, which is what makes a conclusion strong.',
    limits: 'Only as good as the checks underneath it.',
  },
}

export function modulePlain(name: string, title?: string): ModulePlain {
  return (
    MODULES[name] ?? {
      name: title ? title.charAt(0) + title.slice(1).toLowerCase() : name,
      icon: 'microscope',
      accent: 'grey',
      does: 'An additional check.',
      reveals: '',
      limits: '',
    }
  )
}

// ---------------------------------------------------------------- tabs

export type TabKey =
  | 'overview'
  | 'timeline'
  | 'sound'
  | 'light'
  | 'picture'
  | 'people'
  | 'place'
  | 'text'
  | 'file'
  | 'findings'
  | 'method'

export interface TabDef {
  key: TabKey
  label: string
  icon: string
  accent: Accent
  modules: string[]
  tags: string[] // findings from other modules carrying these tags are cross-listed here
  intro: string
}

export const TABS: TabDef[] = [
  { key: 'overview', label: 'Overview', icon: 'eye', accent: 'red', modules: [], tags: [], intro: '' },
  {
    key: 'timeline',
    label: 'Timeline',
    icon: 'film',
    accent: 'blue',
    modules: ['timeline'],
    tags: ['cut', 'splice'],
    intro: 'Every flagged moment in order, and the frame-by-frame measurements that reveal cuts and joins.',
  },
  {
    key: 'sound',
    label: 'Sound',
    icon: 'wave-square',
    accent: 'teal',
    modules: ['spectrum', 'enf', 'avsync', 'sounds'],
    tags: ['audio', 'dtmf', 'tone', 'sync'],
    intro: 'What the microphone picked up, including sounds people cannot hear and the hum of the power grid.',
  },
  {
    key: 'light',
    label: 'Light',
    icon: 'lightbulb',
    accent: 'yellow',
    modules: ['light'],
    tags: ['flicker', 'sun'],
    intro: 'Mains-powered lamps flicker too fast to see. That flicker, and the light of the sun, can date and place a video.',
  },
  {
    key: 'picture',
    label: 'Picture',
    icon: 'image',
    accent: 'purple',
    modules: ['tamper', 'vision'],
    tags: ['tamper', 'clone', 'scene'],
    intro: 'Signs that parts of the picture were altered, and what an AI eyewitness saw in the frames. Zoom in on any detail from the player.',
  },
  {
    key: 'people',
    label: 'People',
    icon: 'heart-pulse',
    accent: 'pink',
    modules: ['pulse'],
    tags: ['face', 'pulse', 'person'],
    intro: 'Signs of living people on camera: faces, and heartbeats read from tiny changes in skin colour.',
  },
  {
    key: 'place',
    label: 'Place',
    icon: 'location-dot',
    accent: 'green',
    modules: ['geo', 'weather'],
    tags: ['location', 'gps', 'osint', 'weather'],
    intro: 'Where it was filmed: the GPS stored in the file, and independent clues in the picture.',
  },
  {
    key: 'text',
    label: 'Text & speech',
    icon: 'comment-dots',
    accent: 'orange',
    modules: ['ocr', 'transcript'],
    tags: ['text', 'speech', 'language', 'contact'],
    intro: 'Words that appear on screen and words that are spoken.',
  },
  {
    key: 'file',
    label: 'File',
    icon: 'file-shield',
    accent: 'lblue',
    modules: ['provenance'],
    tags: ['metadata', 'software', 'editor', 'container', 'device'],
    intro: 'The file itself: fingerprints, every stored label, and the raw technical readout.',
  },
  { key: 'findings', label: 'All findings', icon: 'list-ol', accent: 'red', modules: [], tags: [], intro: '' },
  { key: 'method', label: 'Method', icon: 'flask', accent: 'grey', modules: [], tags: [], intro: '' },
]

const MODULE_TAB: Record<string, TabKey> = {}
for (const t of TABS) for (const m of t.modules) MODULE_TAB[m] = t.key

/** The tab an artifact lives in. Maps go to Place, captions to Text. */
export function artifactTab(a: Artifact): TabKey {
  if (a.kind === 'map') return 'place'
  if (a.kind === 'captions') return 'text'
  if (a.module === 'geo') return 'place'
  if (a.id === 'provenance:probe' || a.module === 'provenance') return 'file'
  return MODULE_TAB[a.module] ?? 'picture'
}

export function findingInTab(f: Finding, tab: TabDef): boolean {
  if (tab.modules.includes(f.module)) return true
  if (!tab.modules.length) return false
  return (f.tags ?? []).some((t) => tab.tags.includes(t))
}

// ---------------------------------------------------------------- artifact explanations

const EXPLAIN: Record<string, string> = {
  'timeline:intervals':
    'Time between one frame and the next. A flat line is a steady camera clock; spikes mean frames were dropped, duplicated or joined.',
  'timeline:bitrate':
    'How much data each half-second of video uses. A sudden step can mean two recordings were joined or the file was re-encoded.',
  'timeline:difference':
    'How much the picture changes from one frame to the next. A tall, isolated spike is a hard cut: the camera could not have done that on its own.',
  'timeline:keyframes': 'Full reference frames the encoder stores at regular intervals. Irregular spacing hints at an edit.',
  'timeline:filmstrip': 'A thumbnail at regular intervals across the whole video.',
  'timeline:light_curve':
    'Average brightness of every frame. Lights switching, clouds or a cut show up as steps; a camera auto-exposing shows up as slow drift.',
  'spectrum:loudness':
    'How loud the sound is over time. Sudden drops to silence, or a jump in level with no visible cause, are worth a look.',
  'spectrum:spectrogram':
    'Sound drawn as a picture: time runs left to right, pitch from bottom (deep) to top (high), brightness is loudness. Steady horizontal lines are tones; sharp vertical breaks are cuts.',
  'spectrum:spectrogram_low':
    'The same picture zoomed into the deepest sounds (0–1,000 Hz), where electrical hum, engines and motors live.',
  'spectrum:noise_floor':
    'The quietest background level in each second. A room sounds the same throughout; a sudden step means the sound came from somewhere else.',
  'spectrum:waveform': 'Loudness over time.',
  'avsync:xcorr':
    'How well sharp sounds line up with visible changes at each delay. One clear peak near 0 ms means sound and picture belong together.',
  'light:flicker_spectrum':
    'Brightness ripples across the rows of the image. A peak at the right frequency is mains lighting, which reveals the power grid (50 Hz or 60 Hz).',
  'pulse:waveform': 'Tiny colour changes in skin over time. A regular rhythm is a heartbeat.',
  'pulse:bpm': 'Heart rate estimated across the video. A believable, fairly steady value suggests a living person.',
  'pulse:magnified': 'Left: the original. Right: skin-colour changes amplified so the pulse becomes visible.',
  'tamper:noise':
    'Sensor grain across the frame. Bright patches have grain unlike the rest of the picture, which can mean they were pasted in.',
  'tamper:ela':
    'Error-level analysis: the frame is re-saved and compared with itself. Areas edited after recording often glow because they recompress differently.',
  'provenance:tags': 'Every label stored inside the file, exactly as written by the device or app that made it.',
  'provenance:streams': 'The separate tracks inside the file (picture, sound, data) and how each is encoded.',
  'provenance:boxes':
    'The order of the building blocks ("boxes") inside the file. Cameras write them in a typical order; editors and apps rewrite it.',
  'provenance:probe': 'The complete raw technical readout of the file from ffprobe. Expand any branch to inspect it.',
  'provenance:map': 'The GPS position stored in the file. The circle is the accuracy the phone reported.',
  'ocr:text': 'Every piece of text read from the frames, when it first appeared, and in how many frames.',
  'transcript:segments': 'What was said, and when.',
  'vision:report': 'What the AI eyewitness wrote after looking at key frames. AI output: verify before relying on it.',
  'geo:map': 'Where the clues in the picture point. The circle shows how uncertain the estimate is.',
  'geo:intake': 'The working notes of the location search for one frame.',
  'geo:frame': 'A frame the location search examined.',
}

const KIND_EXPLAIN: Record<string, string> = {
  series: 'Hover the chart to read exact values.',
  spectrogram: 'Time runs left to right, pitch bottom to top, brightness is loudness.',
  table: 'Raw values as measured.',
  bars: 'Longer bar = more.',
  map: 'Drag to pan, scroll to zoom.',
  image: 'Click to enlarge.',
  heatmap: 'Bright areas are where the check reacted most. Click to enlarge.',
  video: 'Press play.',
  text: '',
  captions: 'Click a line to jump to it.',
  json: 'Expand a branch to see what is inside.',
}

export function explainArtifact(a: Artifact): string {
  if (EXPLAIN[a.id]) return EXPLAIN[a.id]
  const stem = `${a.module}:${a.name.replace(/[_\d.]+$/, '').replace(/_\d.*$/, '')}`
  if (EXPLAIN[stem]) return EXPLAIN[stem]
  const prefix = Object.keys(EXPLAIN).find((k) => a.id.startsWith(k))
  if (prefix) return EXPLAIN[prefix]
  return KIND_EXPLAIN[a.kind] ?? ''
}

// ---------------------------------------------------------------- glossary (tooltips)

export const GLOSSARY: Record<string, string> = {
  ENF: 'Electrical Network Frequency: the hum of the power grid, which drifts slightly in a way that can date a recording.',
  ELA: 'Error-level analysis: re-saving an image and comparing it with itself to find areas saved differently.',
  rPPG: 'Remote photoplethysmography: measuring a heartbeat from tiny skin-colour changes on video.',
  GOP: 'Group of pictures: the run of frames between two full keyframes.',
  keyframe: 'A complete picture the encoder stores; the frames in between only store changes.',
  codec: 'The compression method used to store the picture or sound (for example H.264, HEVC, AAC).',
  bitrate: 'How much data per second the file uses. More data usually means better quality.',
  spectrogram: 'A picture of sound: time left to right, pitch bottom to top, brightness is loudness.',
  dBFS: 'Decibels relative to the loudest sound the file can hold. 0 is the maximum; more negative is quieter.',
  dB: 'Decibels: a scale for loudness or signal strength.',
  Hz: 'Hertz: cycles per second. Low numbers are deep sounds, high numbers are high-pitched.',
  kHz: 'Kilohertz: thousands of cycles per second.',
  DTMF: 'The tones a phone keypad makes. Each key has its own pair of tones, so they can be decoded.',
  fps: 'Frames per second: how many pictures the video shows each second.',
  container: 'The file wrapper (MP4, MOV…) that holds the picture, sound and labels together.',
  'SHA-256': 'A fingerprint of the exact file. Change one byte and it changes completely; use it to prove which file was examined.',
  MD5: 'An older, shorter file fingerprint, still common in evidence logs.',
  GPS: 'A position written into the file by the phone when it recorded.',
  'grid 50 Hz': 'Europe, most of Asia, Africa and Australia run their power at 50 Hz.',
  'grid 60 Hz': 'The Americas, parts of Japan and a few other places run their power at 60 Hz.',
  measured: 'Calculated directly from the file by a fixed method.',
  AI: 'Produced by an AI model. Useful leads, but it can be wrong.',
  'rolling shutter': 'Phone sensors read the image row by row, so fast flicker shows up as stripes.',
  ffprobe: 'A standard tool that reads the technical structure of video files.',
}

/** Plain labels for finding data keys. */
export const DATA_LABEL: Record<string, string> = {
  lat: 'Latitude',
  lon: 'Longitude',
  latitude: 'Latitude',
  longitude: 'Longitude',
  alt_m: 'Altitude',
  accuracy_m: 'Accuracy',
  radius_km: 'Uncertainty radius',
  place: 'Address / place',
  bpm: 'Heart rate',
  snr_db: 'Signal strength over noise',
  box: 'Region in frame (x, y, w, h)',
  hz: 'Frequency',
  excess_db: 'Louder than surroundings by',
  presence: 'Present for',
  grid_hz: 'Power grid',
  alias_hz: 'Measured flicker alias',
  expected_alias_hz: 'Expected flicker alias',
  readout_ms: 'Sensor readout time',
  coherence: 'Consistency',
  lag_ms: 'Sound offset (positive = sound late)',
  halves_ms: 'Offset in first / second half',
  consistent: 'Consistent across the video',
  z: 'Peak strength (z-score)',
  histogram: 'Colour change between frames',
  pixel_diff: 'Pixel change between frames',
  jump_db: 'Jump in background level',
  runs: 'Silent stretches (start, length)',
  keys: 'Keys pressed',
  cutoff_hz: 'Sound cut off above',
  rate: 'Sample rate',
  creation_time_utc: 'Created (UTC)',
  apple_creationdate: 'Apple creation date',
  filename_local: 'Time in file name',
  duration_s: 'Duration',
  utc_offset_h: 'Time zone offset',
  offset_h: 'Time zone offset',
  residual_s: 'Leftover difference',
  elevation: 'Sun height above horizon',
  azimuth: 'Sun direction (compass)',
  phase: 'Daylight phase',
  make: 'Maker',
  model: 'Model',
  software: 'Software version',
  evidence: 'Evidence',
  top_level: 'Top-level boxes',
  start: 'Start',
  confident: 'Confident',
  metadata_diff_s: 'Differs from file clock by',
  luma: 'Brightness',
  red_blue: 'Warmth (red/blue ratio)',
  sun_elevation: 'Sun height',
  artificial: 'Artificial light',
  offset: 'Copied by (x, y pixels)',
  matches: 'Matching points',
  guess: 'Best guess',
  confidence: 'Confidence',
  country: 'Country',
  region: 'Region',
  language: 'Language',
  lines: 'Text lines',
  kind: 'Type',
  value: 'Value',
  label: 'Label',
  tag: 'Category',
  time_s: 'Time in video',
  reason: 'Reason',
  index: 'Index',
  fps: 'Frame rate',
}

/** Unit for a data key, from its suffix. */
export function unitFor(key: string): string {
  if (key === 'bpm') return 'bpm'
  if (/_hz$|^hz$/.test(key)) return 'Hz'
  if (/_db$/.test(key)) return 'dB'
  if (/_ms$/.test(key)) return 'ms'
  if (/_s$/.test(key)) return 's'
  if (/_km$/.test(key)) return 'km'
  if (/_m$/.test(key)) return 'm'
  if (/_h$/.test(key)) return 'h'
  if (key === 'elevation' || key === 'azimuth' || key === 'sun_elevation') return '°'
  return ''
}

export const CONFIDENCE: Record<string, { word: string; level: number; tip: string }> = {
  strong: { word: 'Strong', level: 3, tip: 'Two or more independent checks agree.' },
  moderate: { word: 'Moderate', level: 2, tip: 'One reliable check supports this.' },
  weak: { word: 'Weak', level: 1, tip: 'A hint only, or based on labels that can be faked.' },
  unknown: { word: 'Not enough evidence', level: 0, tip: 'Nothing in the file answers this.' },
}

export const SEVERITY: Record<string, { word: string; icon: string; tip: string }> = {
  critical: { word: 'Critical', icon: 'burst', tip: 'Strong sign of editing, or highly sensitive information.' },
  warning: { word: 'Warning', icon: 'triangle-exclamation', tip: 'Suspicious, or contradicts something else.' },
  notice: { word: 'Notice', icon: 'circle-exclamation', tip: 'Worth knowing; not suspicious on its own.' },
  info: { word: 'Info', icon: 'circle-info', tip: 'Background fact about the file.' },
}

export type Verdict = 'edited' | 'possible' | 'clean' | 'running' | 'unknown'

export const VERDICT: Record<Verdict, { word: string; short: string; icon: string; line: string }> = {
  edited: { word: 'Edited', short: 'Edited', icon: 'scissors', line: 'Independent checks show this video was cut or altered.' },
  possible: { word: 'Possibly edited', short: 'Suspect', icon: 'triangle-exclamation', line: 'Something does not add up, but it is not proven.' },
  clean: { word: 'No signs of editing', short: 'Clean', icon: 'circle-check', line: 'Nothing measured suggests a cut or alteration.' },
  running: { word: 'Analysing', short: 'Analysing', icon: 'circle-notch', line: 'Checks are still running.' },
  unknown: { word: 'No verdict', short: 'No verdict', icon: 'circle-question', line: 'No summary was produced for this case.' },
}

export function verdictOf(v: unknown): Verdict {
  const s = String(v ?? '').toUpperCase()
  if (s === 'EDITED') return 'edited'
  if (s === 'POSSIBLY EDITED') return 'possible'
  if (s === 'NO SIGNS OF EDITING') return 'clean'
  return 'unknown'
}

export const BULLET_LABEL: Record<string, string> = {
  WHEN: 'When',
  WHERE: 'Where',
  DEVICE: 'Device',
  PEOPLE: 'People',
  INTEGRITY: 'Editing',
  HIDDEN: 'Hidden',
  CORROBORATED: 'Cross-checked',
}
