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
  // ---- still photographs
  jpeg: {
    name: 'Compression anatomy',
    icon: 'table-cells',
    accent: 'blue',
    does: 'Reads how the picture was compressed: the quality tables, the encoder\u2019s habits, ghosts of earlier saves, and the 8\u00d78 block grid.',
    reveals: 'Which software wrote the file, whether it was saved more than once, and patches pasted in from another JPEG.',
    limits: 'A re-save at very high quality hides the earlier one; fine repeating textures can imitate a grid.',
  },
  hidden: {
    name: 'Hidden data',
    icon: 'eye-slash',
    accent: 'red',
    does: 'Accounts for every byte that is not the picture: data after the end marker, comments, private blocks, the lowest bits, and what sits under transparency.',
    reveals: 'Appended archives and videos, extra embedded images, hidden messages, and personal details in the labels.',
    limits: 'A message hidden with a modern tool in a JPEG is not detectable here; only the classic hiding places are checked.',
  },
  faces: {
    name: 'People',
    icon: 'user',
    accent: 'pink',
    does: 'Finds faces, measures whether each is large and sharp enough to recognise, and enlarges the eyes to show their reflections.',
    reveals: 'How many people are identifiable, names that photo software tagged into the file, and what the subject was looking at.',
    limits: 'Nobody is identified. Faces turned away, masked or tiny are not counted.',
  },
  synthetic: {
    name: 'AI generation signals',
    icon: 'robot',
    accent: 'purple',
    does: 'Looks for the labels generators write (Content Credentials, recipes, names) and for weak statistical traces of synthetic pixels.',
    reveals: 'Whether the file itself says it was generated, and with which prompt, seed and model.',
    limits: 'Without a label, pixel statistics only raise suspicion; they never prove generation.',
  },
  // Checks run by the viewer's own GPU (WebGPU) when a case opens; labelled as browser results.
  'gpu-frames': {
    name: 'Frame scan (your browser)',
    icon: 'film',
    accent: 'blue',
    does: 'Your GPU reads every frame and compares it with the one before.',
    reveals: 'Cuts and sudden jumps, and frames repeated exactly (frozen video, frames padded in, frame-rate conversion).',
    limits: 'Runs in this browser and is not saved with the case. Exact repeats survive only in the original file.',
  },
  'gpu-light': {
    name: 'Lamp flicker (your browser)',
    icon: 'lightbulb',
    accent: 'yellow',
    does: 'Your GPU measures, for every pixel, the pulse that mains lights leave in the video.',
    reveals: 'Whether the lights run on a 50 Hz or a 60 Hz grid, and where in the picture the flicker comes from.',
    limits: 'Runs in this browser and is not saved with the case. Daylight and some LED lights do not flicker.',
  },
  'gpu-sound': {
    name: 'Hum and tones (your browser)',
    icon: 'wave-square',
    accent: 'purple',
    does: 'Your GPU computes the spectrum of the soundtrack and follows the strongest frequency in the mains-hum and high bands.',
    reveals: 'A steady mains hum (and so the grid frequency), and steady high tones that are often inaudible.',
    limits: 'Runs in this browser and is not saved with the case.',
  },
  'gpu-ghost': {
    name: 'JPEG ghost (your browser)',
    icon: 'layer-group',
    accent: 'blue',
    does: 'Your GPU works out, for every 8×8 block, the JPEG quality it was last saved at.',
    reveals: 'The quality the picture was saved at, and areas saved at a different quality (often pasted in).',
    limits: 'Runs in this browser and is not saved with the case. Flat areas carry no evidence either way.',
  },
  'gpu-speech': {
    name: 'Transcript on this device',
    icon: 'microphone-lines',
    accent: 'grey',
    does: 'Whisper runs on your GPU in this browser; the audio never leaves your machine.',
    reveals: 'What is said, with timestamps.',
    limits: 'Experimental. Not saved with the case. The small model is rough on names.',
  },
}

/** What a check is doing right now, as the live analysis headline ("…" is added by the UI). */
const DOING: Record<string, string | ((still: boolean) => string)> = {
  provenance: 'Reading the file’s history and hidden labels',
  timeline: 'Checking the timing of every frame for cuts',
  spectrum: 'Listening across the whole sound spectrum',
  enf: 'Matching the electrical hum to the power grid',
  light: 'Measuring light flicker frame by frame',
  pulse: 'Looking for a heartbeat in skin colour',
  tamper: (still) => (still ? 'Scanning every pixel for pasted-in regions' : 'Scanning frames for tampered pixels'),
  avsync: 'Lining up the sound against the picture',
  ocr: 'Reading text in the picture',
  transcript: 'Transcribing what is said',
  vision: (still) => (still ? 'AI eyewitness is studying the photo' : 'AI eyewitness is studying key frames'),
  geo: (still) => `Working out where this was ${still ? 'taken' : 'filmed'}`,
  sounds: 'Identifying sounds in the background',
  weather: 'Checking the weather against the claimed date',
  brief: 'Writing up the case',
  jpeg: 'Dissecting the JPEG compression',
  hidden: 'Searching for data hidden inside the file',
  faces: 'Finding faces and people',
  'gpu-frames': 'Your GPU is reading every frame',
  'gpu-light': 'Your GPU is measuring lamp flicker',
  'gpu-sound': 'Your GPU is listening for hum and tones',
  'gpu-ghost': 'Your GPU is reading the JPEG history',
  synthetic: 'Looking for signs the picture was AI-generated',
}

export function moduleDoing(name: string, still: boolean, title?: string): string {
  const d = DOING[name]
  if (typeof d === 'function') return d(still)
  return d ?? `Running ${modulePlain(name, title).name.toLowerCase()}`
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
  | 'compression'
  | 'hidden'
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
    modules: ['timeline', 'gpu-frames'],
    tags: ['cut', 'splice'],
    intro: 'Every flagged moment in order, and the frame-by-frame measurements that reveal cuts and joins.',
  },
  {
    key: 'sound',
    label: 'Sound',
    icon: 'wave-square',
    accent: 'teal',
    modules: ['spectrum', 'enf', 'avsync', 'sounds', 'gpu-sound'],
    tags: ['audio', 'dtmf', 'tone', 'sync'],
    intro: 'What the microphone picked up, including sounds people cannot hear and the hum of the power grid.',
  },
  {
    key: 'light',
    label: 'Light',
    icon: 'lightbulb',
    accent: 'yellow',
    modules: ['light', 'gpu-light'],
    tags: ['flicker', 'sun'],
    intro: 'Mains-powered lamps flicker too fast to see. That flicker, and the light of the sun, can date and place a video.',
  },
  {
    key: 'picture',
    label: 'Picture',
    icon: 'image',
    accent: 'purple',
    modules: ['tamper', 'vision', 'synthetic'],
    tags: ['tamper', 'clone', 'scene', 'synthetic'],
    intro: 'Signs that parts of the picture were altered or generated, and what an AI eyewitness saw. Zoom in on any detail from the viewer.',
  },
  {
    key: 'compression',
    label: 'Compression',
    icon: 'table-cells',
    accent: 'blue',
    modules: ['jpeg', 'gpu-ghost'],
    tags: ['jpeg', 'ghost', 'grid', 'resaved'],
    intro: 'How the file was compressed: the tables and habits of the encoder, ghosts of earlier saves, and the 8\u00d78 grid that every JPEG leaves behind.',
  },
  {
    key: 'hidden',
    label: 'Hidden data',
    icon: 'eye-slash',
    accent: 'red',
    modules: ['hidden'],
    tags: ['hidden', 'steganography', 'appended', 'lsb'],
    intro: 'Everything in the file that is not the picture: appended files, extra images, comments, the lowest bits, and what sits under transparency.',
  },
  {
    key: 'people',
    label: 'People',
    icon: 'heart-pulse',
    accent: 'pink',
    modules: ['pulse', 'faces'],
    tags: ['face', 'pulse', 'person', 'identifiable'],
    intro: 'Signs of people: faces and whether they are recognisable, eye reflections, and (on video) heartbeats read from tiny changes in skin colour.',
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
    modules: ['ocr', 'transcript', 'gpu-speech'],
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
  'provenance:streams': 'The separate tracks inside the file (picture, sound, data) and how each is encoded.',
  'provenance:probe': 'The complete raw technical readout of the file from ffprobe. Expand any branch to inspect it.',
  'provenance:map': 'The GPS position stored in the file. The circle is the accuracy the phone reported.',
  'ocr:text': 'Every piece of text read from the frames, when it first appeared, and in how many frames.',
  'transcript:segments': 'What was said, and when.',
  'vision:report': 'What the AI eyewitness wrote after looking at key frames. AI output: verify before relying on it.',
  'geo:map': 'Where the clues in the picture point. The circle shows how uncertain the estimate is.',
  'geo:intake': 'The working notes of the location search for one frame.',
  'geo:frame': 'A frame the location search examined.',
  // ---- still photographs
  'provenance:tags': 'Every label stored inside the file, exactly as the camera or app wrote it: EXIF, XMP, IPTC, ICC and text chunks.',
  'provenance:clocks': 'Every clock the file carries. A camera writes them all within a second of each other; editors leave later ones.',
  'provenance:history': 'The edit log Adobe-style software keeps inside the file: what was done, when, and by which program.',
  'provenance:thumbnail': 'Left: the small preview the camera wrote at capture. Middle: the picture now, shrunk to match. Right: where they differ (bright = changed).',
  'provenance:boxes': 'The order of the blocks (segments, chunks or boxes) inside the file. Cameras write a typical order; editors and apps rewrite it.',
  'jpeg:quant': 'The 8\u00d78 quantisation table: how coarsely each frequency was stored. Small numbers = high quality. Its exact values fingerprint the encoder.',
  'jpeg:segments': 'The JPEG segments in file order, with their sizes and what each holds.',
  'jpeg:ghost': 'The picture re-saved at every quality and compared with itself. One dip at the file\u2019s own quality is normal; a second dip at a lower quality is the ghost of an earlier save.',
  'jpeg:ghost_map': 'Where the ghost lives: blocks that recompress unusually well at the earlier quality were saved at that quality before. Toggle it as an overlay on the picture.',
  'jpeg:grid_profile': 'Edge energy at each of the 8 positions within a block, across and down. A JPEG grid makes position 0 stand out; a lossless file has a flat profile.',
  'jpeg:grid_map': 'Tiles whose 8\u00d78 grid does not line up with the file\u2019s own grid. A pasted JPEG patch keeps its own alignment.',
  'tamper:ela': 'Error-level analysis at the file\u2019s own quality: the picture is re-saved and compared with itself. Areas saved separately glow or go dark.',
  'tamper:ela_raw': 'The error levels alone. Even grey everywhere means one JPEG generation.',
  'tamper:noise': 'Sensor grain after allowing for brightness and texture. Bright patches have grain unlike the rest of the picture.',
  'tamper:noise_view': 'The grain itself, amplified. It should look like even sand everywhere.',
  'tamper:clone': 'Feature points (red) that reappear elsewhere (green) under one rigid motion, joined by lines, with both regions boxed.',
  'tamper:resampling': 'The spectrum of the pixel-prediction error. A smooth cloud is a camera original; bright isolated dots are the rhythm of resizing or rotation.',
  'tamper:histogram': 'How many pixels sit at each brightness level. A camera writes a smooth curve; contrast and level adjustments leave combs of gaps and spikes.',
  'tamper:gradient': 'The direction light changes at every edge, as colour. Objects lit the same way share the same pattern.',
  'tamper:pca': 'Colour variation the eye hardly sees. Edits made in another colour space, and halos around pasted objects, show here.',
  'hidden:trailing': 'The bytes that follow the end of the image. Download them to inspect.',
  'hidden:embedded': 'A whole second image stored after the main one (depth map, gain map, preview).',
  'hidden:motion_photo': 'The movie clip stored inside the photo file.',
  'hidden:lsb': 'Only the lowest bit of every pixel. Natural photos give structured noise; a hidden message gives uniform snow.',
  'hidden:lsb_test': 'The chi-square pair test slice by slice from the top of the picture. A step from high to near zero is a sequentially written message.',
  'hidden:under_alpha': 'The picture with transparency removed: whatever was stored under fully transparent pixels.',
  'hidden:accounting': 'Where every byte of the file is: picture, labels, private blocks, trailing data.',
  'faces:faces': 'Faces found by the detector. Red boxes are large and sharp enough to recognise; yellow are not.',
  'faces:face': 'One face, cropped from the full-resolution picture.',
  'faces:eyes': 'Both eyes enlarged with local contrast boosted. Reflections on the cornea show what was in front of the person.',
  'synthetic:spectrum': 'Power at each spatial frequency, averaged around the centre. Camera optics give a smooth fall-off; upsampling networks leave bumps near the top.',
  'synthetic:c2pa': 'The readable parts of the Content Credentials manifest signed into the file.',
  'synthetic:recipe': 'The generation recipe stored in the file: prompt, seed, sampler, model.',
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
  'gpu-framescan': 'Measured in your browser. Red lines: abrupt changes; orange: frames identical to the one before. Click to jump there.',
  'gpu-flicker': 'Measured in your browser. Bright areas pulse at the rate mains lights leave in video; compare them with the lamps in the picture.',
  'gpu-transcript': 'Runs in your browser only when you ask: the audio never leaves this machine.',
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
  EXIF: 'The block of labels a camera writes into a photo: device, lens, exposure, date, GPS and more.',
  XMP: 'Adobe\u2019s metadata format, where editors keep their edit history and creators their names.',
  C2PA: 'Content Credentials: a signed record of how a file was made, written by generators and editors that support it.',
  'JPEG ghost': 'Re-saving a picture at every quality and watching the error. An earlier save shows up as a dip at its own quality.',
  quantisation: 'The table that decides how coarsely each frequency is stored in a JPEG. It fingerprints the encoder and its quality setting.',
  LSB: 'Least significant bit: the lowest bit of each pixel value, where simple steganography tools hide messages.',
  'copy-move': 'Cloning one part of a picture onto another, to hide or duplicate something.',
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
  // ---- still photographs
  box: 'Region of the picture',
  source_box: 'Copied from',
  shift_px: 'Moved by (x, y pixels)',
  angle_deg: 'Rotated by',
  scale: 'Scaled by',
  of_pairs: 'Candidate matches considered',
  quality: 'JPEG quality',
  earlier_quality: 'Earlier save quality',
  current_quality: 'This file\u2019s quality',
  table_distance: 'Distance from standard tables',
  standard_tables: 'Standard (libjpeg) tables',
  sampling: 'Chroma subsampling',
  progressive: 'Progressive',
  optimised_huffman: 'Optimised Huffman tables',
  bits_per_pixel: 'Bits per pixel',
  restart_interval: 'Restart interval',
  heading_deg: 'Camera pointed (compass)',
  speed_kmh: 'Speed',
  relative_deg: 'Sun relative to camera direction',
  date_time_original: 'Taken (camera clock)',
  date_time_digitized: 'Digitised',
  modify_date: 'Modified',
  offset_time_original: 'Time zone written by camera',
  gps_time_utc: 'GPS clock (UTC)',
  xmp_create_date: 'XMP create date',
  xmp_modify_date: 'XMP modify date',
  filename_local: 'Time in file name',
  copy_stamp: 'Timestamp is a copy time',
  correlation: 'Thumbnail similarity',
  peak_block_difference: 'Largest local difference',
  changed_blocks: 'Changed blocks',
  exif_size: 'Size the camera recorded',
  stored_size: 'Size stored now',
  pixel_ratio: 'Pixels kept',
  claim_generator: 'Signed by',
  generator_name: 'Generator',
  digital_source_type: 'Declared source type',
  prompt: 'Prompt',
  negative_prompt: 'Negative prompt',
  seed: 'Seed',
  steps: 'Steps',
  sampler: 'Sampler',
  cfg_scale: 'CFG scale',
  noise_floor: 'Grain level',
  peak_z: 'Peak strength (sigma)',
  period_px: 'Repeats every',
  gaps: 'Empty levels',
  spikes: 'Over-full levels',
  clipped_white: 'Pure white pixels',
  clipped_black: 'Pure black pixels',
  recognisable: 'Recognisable faces',
  width_px: 'Face width',
  sharpness: 'Sharpness',
  faces: 'Faces',
  names: 'Names tagged',
  bytes: 'Size',
  strings: 'Readable fragments',
  channel_scores: 'Fraction equalised per channel',
  transparent_fraction: 'Transparent area',
  makernote: 'Maker notes',
  lens: 'Lens',
  exposure: 'Exposure',
  signs: 'Encoder habits',
  app_segments: 'Application segments',
  history: 'Edit history',
  identity: 'Identity fields',
  corroborated_by: 'Agrees with',
  regions: 'All regions',
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

export type Verdict = 'edited' | 'possible' | 'clean' | 'synthetic' | 'running' | 'unknown'

export const VERDICT: Record<Verdict, { word: string; short: string; icon: string; line: string }> = {
  edited: { word: 'Edited', short: 'Edited', icon: 'scissors', line: 'Independent checks show this file was cut or altered.' },
  possible: { word: 'Possibly edited', short: 'Suspect', icon: 'triangle-exclamation', line: 'Something does not add up, but it is not proven.' },
  clean: { word: 'No signs of editing', short: 'Clean', icon: 'circle-check', line: 'Nothing measured suggests a cut or alteration.' },
  synthetic: { word: 'AI-generated', short: 'AI-made', icon: 'robot', line: 'The file itself declares it was made by an AI model.' },
  running: { word: 'Analysing', short: 'Analysing', icon: 'circle-notch', line: 'Checks are still running.' },
  unknown: { word: 'No verdict', short: 'No verdict', icon: 'circle-question', line: 'No summary was produced for this case.' },
}

export function verdictOf(v: unknown): Verdict {
  const s = String(v ?? '').toUpperCase()
  if (s === 'EDITED') return 'edited'
  if (s === 'POSSIBLY EDITED') return 'possible'
  if (s === 'NO SIGNS OF EDITING') return 'clean'
  if (s === 'AI-GENERATED') return 'synthetic'
  return 'unknown'
}

/** The word for the evidence: "video", "photo" or "recording". */
export function mediaNoun(kind: string | null | undefined): 'video' | 'photo' | 'recording' {
  if (kind === 'image') return 'photo'
  if (kind === 'audio') return 'recording'
  return 'video'
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
