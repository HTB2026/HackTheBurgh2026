import { Fragment, useEffect, type ReactNode } from 'react'
import { useHealth } from '../lib/health'
import { modulePlain, VERDICT } from '../lib/plain'
import type { Effort, ModuleInfo } from '../lib/types'
import { EFFORT_LABEL, EFFORT_TIME, EFFORTS } from './EffortPicker'
import { Alert, Icon, Redacted, SectionTitle } from './ui/ui'

/** A cell of the comparison: runs (true), does not (false), or how it runs. */
type Cell = boolean | string
interface Row {
  what: string
  detail?: string
  cells: [Cell, Cell, Cell]
}

const ALL: [Cell, Cell, Cell] = [true, true, true]
const DEEP: [Cell, Cell, Cell] = [false, false, true]

const COMPARE: { group: string; icon: string; rows: Row[] }[] = [
  {
    group: 'Every case',
    icon: 'list-check',
    rows: [
      { what: 'Typical time for a short clip', cells: [EFFORT_TIME.low, EFFORT_TIME.medium, EFFORT_TIME.high] },
      { what: 'Edits, cuts, metadata, compression, hidden data, signs of AI generation', detail: 'The forensic checks behind the verdict.', cells: ALL },
      { what: 'Power-grid hum dating, light flicker, sound and speech', cells: ALL },
      { what: 'Heart rate of people on camera', detail: 'Slow and rarely decisive, so Quick leaves it out.', cells: [false, true, true] },
      { what: 'AI listener', cells: [true, true, 'also names birds, insects and farm animals'] },
      { what: 'AI reasoning for the visual analyst and the summary', cells: ['short', 'standard', 'standard'] },
    ],
  },
  {
    group: 'Where it was taken',
    icon: 'location-dot',
    rows: [
      { what: 'Frames studied (video)', cells: ['3', '6', '10 + enlarged quarters'] },
      { what: 'Zoomed edge and corner crops', detail: 'geo-sleuth: small signs and plates made readable.', cells: [false, '2 frames', '4 frames'] },
      { what: 'Clue inventory', detail: 'Poles, road lines, walls, crops, soil, terrain, sun and stars, listed before any guess.', cells: DEEP },
      { what: 'Independent analysts proposing places', detail: 'Places several analysts agree on rank higher.', cells: ['1', '1', '3'] },
      { what: 'Wikipedia photo of a named landmark', cells: ALL },
      { what: 'Satellite view of each place', cells: ['1 scale', '1 scale', '2 scales'] },
      { what: 'Photos taken near each place', detail: 'Wikimedia Commons and Mapillary street level: works where there is no landmark.', cells: DEEP },
      { what: 'Features on the map', detail: 'OpenStreetMap: are the wind turbines, pylons, railway or coast seen in the frames really there?', cells: DEEP },
      { what: 'Skyline against the terrain', detail: 'The horizon each place would show, rendered from an elevation model.', cells: DEEP },
      { what: 'Sun position', detail: 'Needs a trustworthy capture time: its height draws a ring the camera stood on.', cells: DEEP },
      { what: 'Stars', detail: 'The star field is identified offline; with the horizon, the height of the celestial pole is the latitude. No date needed.', cells: DEEP },
      { what: 'Weather at each place', detail: 'Needs a capture time: rain in the frames but a dry record there counts against it.', cells: DEEP },
      { what: 'Zoom-in round', detail: 'When the answer is only a region, a second look for the exact spot.', cells: DEEP },
      { what: 'AI reasoning for location', cells: ['short', 'long', 'long'] },
    ],
  },
]

const STEPS: { icon: string; title: string; text: ReactNode; levels: Effort[] }[] = [
  {
    icon: 'eye',
    title: 'Look',
    levels: ['low', 'medium', 'high'],
    text: 'The sharpest frames from across the clip (or the photo and its four quarters) are chosen. The file’s GPS and other metadata are kept back: the AI never sees them.',
  },
  {
    icon: 'ear-listen',
    title: 'Listen and read',
    levels: ['low', 'medium', 'high'],
    text: 'Text in the picture, the spoken language, background sounds and the mains frequency of the lights become clues.',
  },
  {
    icon: 'clipboard-list',
    title: 'Take an inventory',
    levels: ['high'],
    text: 'Before guessing, the analyst lists what a geolocation expert reads in a scene with no text: utility poles, road lines, walls and hedges, crops, soil, terrain, the skyline, the sun and the stars.',
  },
  {
    icon: 'star',
    title: 'Measure the sky',
    levels: ['high'],
    text: (
      <>
        Stars are matched against a catalogue of 42,000 (offline, in seconds). The height of the celestial pole above the horizon <i>is</i> the latitude. With a
        trustworthy capture time, the stars give the longitude too, and the sun’s height draws a ring the camera stood on.
      </>
    ),
  },
  {
    icon: 'lightbulb',
    title: 'Propose places',
    levels: ['low', 'medium', 'high'],
    text: 'A reasoning model names candidate places in different cities, each with coordinates and an honest radius. Deep asks three analysts, each shown different frames, and clusters what they say.',
  },
  {
    icon: 'magnifying-glass-location',
    title: 'Check against the real world',
    levels: ['low', 'medium', 'high'],
    text: 'Each place is compared with a Wikipedia photo of its landmark and a satellite view, by a second, sceptical model. Deep adds photos taken nearby, OpenStreetMap, the terrain skyline, the sun, the stars and the weather, and reports each as passed or failed.',
  },
  {
    icon: 'crosshairs',
    title: 'Zoom in',
    levels: ['high'],
    text: 'If only a region fits, satellite views of it are searched for the exact spot, which then goes through the same checks.',
  },
  {
    icon: 'circle-check',
    title: 'Decide',
    levels: ['low', 'medium', 'high'],
    text: 'A place is verified only with a clear match (60% or more, and 20 points ahead of any place elsewhere). Two failed automatic checks rule a place out. If the file has GPS, the distance to it is shown afterwards, as a score.',
  },
]

const SOURCES: { name: string; use: string; levels: Effort[]; sends: string }[] = [
  { name: 'OpenAI, Gemini or Gemma', use: 'Speech, sounds, the visual analyst, location and the summary', levels: ['low', 'medium', 'high'], sends: 'Frames and audio of the evidence (Gemma runs on our own server)' },
  { name: 'Wikipedia', use: 'Reference photos of named landmarks', levels: ['low', 'medium', 'high'], sends: 'Landmark names' },
  { name: 'Esri World Imagery', use: 'Satellite views', levels: ['low', 'medium', 'high'], sends: 'Coordinates of candidate places' },
  { name: 'GB grid frequency record (NESO)', use: 'Dating recordings by their mains hum', levels: ['low', 'medium', 'high'], sends: 'Nothing (stored on our database)' },
  { name: 'Open-Meteo', use: 'Recorded weather: at the answer, and at Deep at every candidate', levels: ['low', 'medium', 'high'], sends: 'Coordinates and an hour' },
  { name: 'Wikimedia Commons', use: 'Photos taken near each candidate', levels: ['high'], sends: 'Coordinates of candidate places' },
  { name: 'Mapillary', use: 'Street-level photos near each candidate', levels: ['high'], sends: 'Coordinates of candidate places' },
  { name: 'OpenStreetMap (Overpass)', use: 'Which features exist near each candidate', levels: ['high'], sends: 'Coordinates and feature types' },
  { name: 'Terrain tiles (AWS open data)', use: 'Elevation, to render the skyline', levels: ['high'], sends: 'Map tile numbers' },
  { name: 'Hipparcos star catalogue (tetra3)', use: 'Identifying star fields', levels: ['high'], sends: 'Nothing: runs on our server' },
]

/** Which levels run something, as small coloured chips. */
function Levels({ levels }: { levels: Effort[] }) {
  return (
    <span className="mode-chips">
      {levels.length === 3 ? (
        <span className="mode-chip all">All levels</span>
      ) : (
        levels.map((l) => (
          <span key={l} className={`mode-chip lvl-${l}`}>
            {EFFORT_LABEL[l]}
          </span>
        ))
      )}
    </span>
  )
}

function CellMark({ cell, level }: { cell: Cell; level: Effort }) {
  if (cell === true) return <Icon name="check" className={`mark-yes lvl-${level}`} />
  if (cell === false) return <span className="mark-no">–</span>
  return <span className="mark-text">{cell}</span>
}

const runsAt = (m: ModuleInfo): Effort[] => EFFORTS.slice(EFFORTS.indexOf(m.min_effort ?? 'low'))

/** Extra work a check does at a given level, beyond what its card says. */
const DEEPER: Record<string, string> = {
  geo: 'Quick looks at fewer frames and reasons briefly. Deep adds the clue inventory, three analysts, photos taken nearby, OpenStreetMap, the terrain skyline, the sun, the stars, the weather and a zoom-in round.',
  sounds: 'Deep also names birds, insects and farm animals: species have ranges.',
  vision: 'Quick reasons briefly.',
  brief: 'Quick reasons briefly.',
}

function ModuleCards({ modules, prefix }: { modules: ModuleInfo[]; prefix: string }) {
  return (
    <>
      {modules.map((m) => {
        const p = modulePlain(m.name, m.title)
        return (
          <article key={`${prefix}:${m.name}`} className={`panel how-card c-${p.accent}`}>
            <header>
              <Icon name={p.icon} />
              <h3>{p.name}</h3>
              <Levels levels={runsAt(m)} />
            </header>
            <p>{p.does}</p>
            {p.reveals && (
              <p>
                <b>Can reveal:</b> <span className="muted">{p.reveals}</span>
              </p>
            )}
            {p.limits && (
              <p>
                <b>Limits:</b> <span className="muted">{p.limits}</span>
              </p>
            )}
            {DEEPER[m.name] && (
              <p className="how-deeper">
                <Icon name="gauge-high" /> {DEEPER[m.name]}
              </p>
            )}
          </article>
        )
      })}
    </>
  )
}

export function HowItWorks({ section }: { section?: string }) {
  const { health } = useHealth()
  useEffect(() => {
    if (!section) return
    const frame = requestAnimationFrame(() => document.getElementById(section)?.scrollIntoView({ block: 'start' }))
    return () => cancelAnimationFrame(frame)
  }, [section, health])
  const browser = ['gpu-frames', 'gpu-light', 'gpu-sound', 'gpu-ghost', 'gpu-speech'].map((k) => [k, modulePlain(k)] as const)
  return (
    <>
      <section className="banner">
        <div className="wrap">
          <div>
            <h1>How it works</h1>
            <p>
              {health ? `${health.modules.length} independent witnesses for video, ${health.image_modules?.length ?? 0} for photographs` : 'Independent witnesses'},
              questioned one by one, at the depth you choose.
            </p>
          </div>
        </div>
      </section>
      <main className="wrap how">
        <Alert title="Leads, not proof">
          EARWITNESS measures things a person cannot see or hear. Each finding says how it was measured and how sure it is. Use it to decide where to
          look, then confirm with the original source.
        </Alert>

        <SectionTitle icon="gauge-high" accent="red" id="effort" small="Chosen with the slider before uploading, and again on any re-run">
          Three levels of effort
        </SectionTitle>
        <div className="effort-cards">
          {EFFORTS.map((l) => (
            <div key={l} className={`panel effort-card lvl-${l}`}>
              <b>{EFFORT_LABEL[l]}</b>
              <span className="muted">{EFFORT_TIME[l]}</span>
              <p>
                {l === 'low'
                  ? 'A fast first look: fewer frames, short AI reasoning, no pulse check.'
                  : l === 'medium'
                    ? 'Every check, with locations verified against landmark photos and satellite imagery. The default.'
                    : 'For hard cases, such as countryside with no text or landmarks, and night skies: every extra location method we have.'}
              </p>
            </div>
          ))}
        </div>
        <div className="panel effort-table-wrap">
          <table className="effort-table">
            <thead>
              <tr>
                <th />
                {EFFORTS.map((l) => (
                  <th key={l} className={`lvl-${l}`}>
                    {EFFORT_LABEL[l]}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {COMPARE.map((g) => (
                <Fragment key={g.group}>
                  <tr className="group">
                    <th colSpan={4}>
                      <Icon name={g.icon} /> {g.group}
                    </th>
                  </tr>
                  {g.rows.map((r) => (
                    <tr key={r.what}>
                      <th scope="row">
                        {r.what}
                        {r.detail && <small>{r.detail}</small>}
                      </th>
                      {r.cells.map((c, i) => (
                        <td key={i}>
                          <CellMark cell={c} level={EFFORTS[i]} />
                        </td>
                      ))}
                    </tr>
                  ))}
                </Fragment>
              ))}
            </tbody>
          </table>
        </div>

        <SectionTitle icon="location-dot" accent="red" id="where" small="The headline answer, and how it earns the word verified">
          How it finds where
        </SectionTitle>
        <ol className="how-steps">
          {STEPS.map((s, i) => (
            <li key={s.title} className={`panel ${s.levels.length === 1 ? 'deep-only' : ''}`}>
              <span className="how-step-n">{i + 1}</span>
              <div>
                <header>
                  <Icon name={s.icon} />
                  <h3>{s.title}</h3>
                  <Levels levels={s.levels} />
                </header>
                <p>{s.text}</p>
              </div>
            </li>
          ))}
        </ol>

        <SectionTitle icon="list-check" accent="teal" small="Every check runs on every video, except where the level says otherwise">
          The checks for video
        </SectionTitle>
        <div className="how-grid">
          <ModuleCards modules={health?.modules ?? []} prefix="video" />
          {!health && <Redacted lines={6} />}
        </div>

        <SectionTitle icon="image" accent="purple" small="A photograph is one exposure at full resolution: different questions, different checks">
          The checks for photographs
        </SectionTitle>
        <div className="how-grid">
          <ModuleCards modules={health?.image_modules ?? []} prefix="image" />
          {!health && <Redacted lines={6} />}
        </div>

        <SectionTitle icon="microchip" accent="blue" id="browser" small="Same at every level: they depend on your device, not on the setting">
          Checks in your own browser
        </SectionTitle>
        <p className="muted how-lead">
          When you open a case on a device with WebGPU, your graphics chip runs its own quick checks while the server works. Their results appear among the
          findings labelled “your browser”; they are not saved with the case and never change the verdict.
        </p>
        <div className="how-grid">
          {browser.map(([k, p]) => (
            <article key={k} className={`panel how-card c-${p.accent}`}>
              <header>
                <Icon name={p.icon} />
                <h3>{p.name}</h3>
              </header>
              <p>{p.does}</p>
              {p.limits && (
                <p>
                  <b>Limits:</b> <span className="muted">{p.limits}</span>
                </p>
              )}
            </article>
          ))}
        </div>

        <SectionTitle icon="globe" accent="grey" id="sources" small="Outside services, and exactly what each one receives">
          Where outside data comes from
        </SectionTitle>
        <div className="panel effort-table-wrap">
          <table className="effort-table sources">
            <thead>
              <tr>
                <th>Source</th>
                <th>Used for</th>
                <th>Levels</th>
                <th>What it receives</th>
              </tr>
            </thead>
            <tbody>
              {SOURCES.map((s) => (
                <tr key={s.name}>
                  <th scope="row">{s.name}</th>
                  <td>{s.use}</td>
                  <td>
                    <Levels levels={s.levels} />
                  </td>
                  <td className="muted">{s.sends}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <SectionTitle icon="scale-balanced" accent="red" id="verdict" small="The verdict is the strongest claim the evidence supports">
          Reading the verdict
        </SectionTitle>
        <div className="panel pad">
          <dl className="stats">
            {(['edited', 'possible', 'clean', 'synthetic'] as const).map((v) => (
              <div key={v}>
                <dt>
                  <span className={`pill inline v-${v}`}>
                    <Icon name={VERDICT[v].icon} /> {VERDICT[v].word}
                  </span>
                </dt>
                <dd className="left">{VERDICT[v].line}</dd>
              </div>
            ))}
          </dl>
          <p className="muted how-note">
            <Icon name="gauge-high" /> The effort level never moves the verdict: it comes from the same forensic checks at every level.
          </p>
        </div>

        <SectionTitle icon="ban" accent="grey" id="limits" small="Be honest about what a file can and cannot tell you">
          What it cannot prove
        </SectionTitle>
        <div className="panel pad prose">
          <p>
            <b>No cuts found is not the same as authentic.</b> A video can be staged, or an AI-generated clip can be recorded off a screen in one take.
          </p>
          <p>
            <b>A photo with no signs of editing can still be generated.</b> Only a label the generator wrote (Content Credentials, a recipe, a name)
            proves AI origin; pixel statistics merely raise suspicion, and a generated picture re-saved through a phone loses even those.
          </p>
          <p>
            <b>Labels inside a file can be edited.</b> Dates, device names and GPS are treated as weak evidence unless another check (the power grid,
            the sun, the picture) agrees.
          </p>
          <p>
            <b>AI results are labelled AI.</b> They are useful leads, especially for location, but can be confidently wrong.
          </p>
          <p>
            <b>The sky needs a clock.</b> Without a trustworthy capture time, the sun says little and the stars give only the latitude: places on the same
            parallel, thousands of kilometres apart, look the same to them.
          </p>
          <p>
            <b>Deep is slower, not infallible.</b> Its extra checks rule places out more often than they prove one; an empty landscape can still defeat it.
          </p>
          <p>
            <b>Re-uploads destroy clues.</b> Messaging apps and social networks strip metadata and recompress picture and sound. Ask for the original
            file whenever you can.
          </p>
        </div>
      </main>
    </>
  )
}
