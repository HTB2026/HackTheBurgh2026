import { hasLocation } from '../../lib/geo'
import { useState, type ReactNode } from 'react'
import type { Artifact, Finding } from '../../lib/types'
import type { CaseState } from '../../lib/state'
import { usePlayback } from '../../lib/playback'
import { mmss } from '../../lib/format'
import { Icon, Modal } from '../ui/ui'
import { distanceText, radiusText, type MapPoint } from '../viz/MapView'
import { MapSection } from './Sections'
import { MapOrGlobe } from './MapOrGlobe'
import { isTerminal } from '../../lib/state'
import type { Facts } from '../../lib/facts'

export interface Candidate {
  place: string
  landmark?: string | null
  latitude: number
  longitude: number
  radius_km: number
  probability?: number
  evidence?: string[]
}

export function geoFinding(state: CaseState): Finding | null {
  return state.findings.find((f) => f.module === 'geo' && Array.isArray(f.data?.candidates)) ?? null
}

/** Index of the candidate the answer came from: the one nearest the answer's coordinates. */
export function winnerIndex(d: { latitude?: number; longitude?: number; candidates?: Candidate[] }): number {
  const cs = d.candidates ?? []
  if (!cs.length || typeof d.latitude !== 'number') return 0
  let best = 0
  let bestD = Infinity
  cs.forEach((c, i) => {
    const dd = (c.latitude - d.latitude!) ** 2 + (c.longitude - (d.longitude ?? 0)) ** 2
    if (dd < bestD) {
      bestD = dd
      best = i
    }
  })
  return best
}

interface Lead {
  match: number | null
  matches: string[]
  mismatches: string[]
}

function leadsFrom(a: Artifact | undefined): Record<number, Lead> {
  const out: Record<number, Lead> = {}
  const cols: string[] = a?.data?.columns ?? []
  const col = (n: string) => cols.indexOf(n)
  for (const row of (a?.data?.rows ?? []) as unknown[][]) {
    const idx = Number(row[col('#') >= 0 ? col('#') : 0]) - 1
    const m = String(row[col('match')] ?? '')
    const pct = /(\d+(\.\d+)?)%/.exec(m)
    const split = (v: unknown) =>
      String(v ?? '')
        .split(/;\s*/)
        .map((x) => x.trim())
        .filter(Boolean)
    out[idx] = { match: pct ? Number(pct[1]) / 100 : null, matches: split(row[col('matches')]), mismatches: split(row[col('mismatches')]) }
  }
  return out
}

function Thumb({ a, label, empty }: { a?: Artifact; label: string; empty: string }) {
  const [open, setOpen] = useState(false)
  if (!a?.url)
    return (
      <figure className="gb-img empty">
        <div>{empty}</div>
        <figcaption>{label}</figcaption>
      </figure>
    )
  return (
    <>
      <figure className="gb-img" onClick={() => setOpen(true)} role="button" tabIndex={0} data-tip={a.title}>
        <img src={a.url} alt={a.title} loading="lazy" />
        <figcaption>{label}</figcaption>
      </figure>
      {open && (
        <Modal onClose={() => setOpen(false)} wide>
          <div className="lightbox">
            <h3>{a.title}</h3>
            <img src={a.url} alt={a.title} />
          </div>
        </Modal>
      )}
    </>
  )
}

function MatchBar({ match, ok }: { match: number | null | undefined; ok: boolean }) {
  return (
    <div className="gb-match" data-tip="How well the real-world photo and satellite view match the evidence">
      <span>Match</span>
      <span className="gb-bar">
        <i style={{ width: `${Math.round((match ?? 0) * 100)}%` }} className={ok ? 'ok' : ''} />
      </span>
      <b className="mono">{match != null ? `${Math.round(match * 100)}%` : 'not checked'}</b>
    </div>
  )
}

function Ticks({ lead }: { lead: Lead | undefined }) {
  return (
    <>
      {(lead?.matches.length ?? 0) > 0 && (
        <ul className="gb-list ok">
          {lead!.matches.map((m, k) => (
            <li key={k}>
              <Icon name="check" /> {m}
            </li>
          ))}
        </ul>
      )}
      {(lead?.mismatches.length ?? 0) > 0 && (
        <ul className="gb-list no">
          {lead!.mismatches.map((m, k) => (
            <li key={k}>
              <Icon name="xmark" /> {m}
            </li>
          ))}
        </ul>
      )}
    </>
  )
}

/** One proposed place: the evidence beside a landmark photo and a satellite view. */
function CandidateRow({ state, c, i, rank, lead, frame, still }: { state: CaseState; c: Candidate; i: number; rank: number; lead: Lead | undefined; frame: Artifact | undefined; still: boolean }) {
  const pb = usePlayback()
  return (
    <article className="gb-row">
      <div className="gb-imgs">
        <figure className="gb-img evidence" onClick={() => frame?.time_s != null && pb.seek(frame.time_s)} role="button" tabIndex={0} data-tip={still ? 'The photo' : 'Evidence frame from the video. Click to jump there.'}>
          {frame?.url ? <img src={frame.url} alt="Evidence frame" /> : <div>No frame</div>}
          <figcaption>{still ? 'The photo' : 'The video'}</figcaption>
        </figure>
        <Icon name="arrows-left-right" className="gb-vs" />
        <Thumb a={state.artifacts[`geo:ref${i}_wiki`]} label={c.landmark ? `Wikipedia: ${c.landmark}` : 'Landmark photo'} empty="No landmark photo" />
        <Thumb a={state.artifacts[`geo:ref${i}_sat`]} label="Satellite view" empty={c.radius_km > 25 ? 'Area too large for satellite' : 'No satellite view'} />
      </div>
      <div className="gb-info">
        <div className="gb-title">
          <span className="gb-rank mono">#{rank}</span>
          <b>{c.place}</b>
        </div>
        <div className="muted small">
          {c.landmark ? `${c.landmark} · ` : ''}
          <span className="mono">
            {c.latitude.toFixed(4)}, {c.longitude.toFixed(4)}
          </span>{' '}
          · within {radiusText(c.radius_km * 1000).slice(1)}
          {typeof c.probability === 'number' ? ` · proposed at ${Math.round(c.probability * 100)}%` : ''}
        </div>
        <MatchBar match={lead?.match} ok={false} />
        <Ticks lead={lead} />
        <Why c={c} />
      </div>
    </article>
  )
}

function Why({ c }: { c: Candidate | undefined }) {
  if (!c?.evidence?.length) return null
  return (
    <details className="gb-why">
      <summary>Why the AI proposed it</summary>
      <ul>
        {c.evidence.map((e, k) => (
          <li key={k}>{e}</li>
        ))}
      </ul>
    </details>
  )
}

/**
 * The lead section of a case: where the evidence was taken. It is the
 * headline of the product, so it sits above everything else and is never
 * folded away; the other checks follow in a quieter supporting role.
 */
export function LocationHero({ state, facts, points, children }: { state: CaseState; facts: Facts; points: MapPoint[]; children?: ReactNode }) {
  const pb = usePlayback()
  const still = facts.kind === 'image'
  const filmed = still ? 'taken' : 'filmed'
  const noun = still ? 'photo' : facts.kind === 'audio' ? 'recording' : 'video'
  const f = geoFinding(state)
  const d = f?.data ?? {}
  const cands: Candidate[] = d.candidates ?? []
  const located = !!f && hasLocation(d.latitude, d.longitude, d.radius_km)
  const verified = located && !!d.verified
  const win = located ? winnerIndex(d) : -1
  const leads = leadsFrom(state.artifacts['geo:leads'])
  const frames = state.artifactOrder
    .filter((id) => /^geo:frame\d+$/.test(id))
    .map((id) => state.artifacts[id])
    .sort((a, b) => Number(a.id.replace(/\D/g, '')) - Number(b.id.replace(/\D/g, '')))
  const [frameIdx, setFrameIdx] = useState(0)
  const frame = frames[frameIdx]
  const clues: string = state.artifacts['geo:clues']?.data?.text ?? ''
  const others = cands
    .map((_, i) => i)
    .filter((i) => i !== win && hasLocation(cands[i].latitude, cands[i].longitude, cands[i].radius_km))
    .sort((a, b) => (leads[b]?.match ?? -1) - (leads[a]?.match ?? -1))
  const geoModule = state.modules.geo
  const searching = !f && !!geoModule && !isTerminal(geoModule.status)
  const gps = facts.gps
  const w = win >= 0 ? cands[win] : undefined

  // Nothing to say and nothing coming: one quiet line, the rest of the page takes over.
  if (!f && !gps && !searching && !points.length)
    return (
      <section className="loc-hero none" id="d-geo">
        <p className="loc-kicker">
          <Icon name="location-dot" /> Where it was {filmed}
        </p>
        <p className="muted">
          {geoModule ? `Nothing in this ${noun} narrowed down where it was ${filmed}.` : `The location search did not run for this ${noun}: it needs an AI model on the server.`}
        </p>
        {children}
      </section>
    )

  const tone = verified ? 'ok' : located ? 'lead' : gps ? 'gps' : searching ? 'search' : 'none'
  const stamp = verified ? 'Verified' : located ? 'Unverified lead' : gps ? 'GPS in the file' : searching ? 'Searching' : 'Location unknown'
  const title = located ? (d.place ?? f!.title) : gps ? (gps.place ?? `${gps.lat.toFixed(4)}, ${gps.lon.toFixed(4)}`) : searching ? 'Working out where this was ' + filmed + '…' : 'Nothing narrows the location down'

  return (
    <section className={`loc-hero ${tone}`} id="d-geo">
      <p className="loc-kicker">
        <Icon name="location-dot" /> Where this {noun} was {filmed}
      </p>
      <div className="loc-head">
        <div className="loc-title">
          <span className={`gb-stamp small ${verified ? 'ok' : tone === 'gps' ? 'gps' : 'lead'}`}>
            {searching ? <i className="pulse-dot" /> : <Icon name={verified ? 'circle-check' : gps && !located ? 'satellite' : 'magnifying-glass-location'} />} {stamp}
          </span>
          <h2 className="loc-place">{title}</h2>
        </div>
        <div className="loc-meta muted">
          {located && (
            <span className="mono">
              {d.latitude.toFixed(5)}, {d.longitude?.toFixed(5)}
            </span>
          )}
          {located && d.radius_km ? ` · within ${radiusText(d.radius_km * 1000).slice(1)}` : ''}
          {located && d.confidence ? ` · ${d.confidence} confidence` : ''}
          {located && typeof d.match === 'number' ? ` · ${Math.round(d.match * 100)}% match with real-world imagery` : ''}
          {searching && (geoModule?.message || `Reading the ${noun}’s pixels and sounds for clues.`)}
          {!located && !searching && gps && (
            <>
              <span className="mono">
                {gps.lat.toFixed(5)}, {gps.lon.toFixed(5)}
              </span>{' '}
              · stored in the file itself. Anyone with the file can read it.
            </>
          )}
        </div>
        {located && typeof d.error_km === 'number' && (
          <div className={`gb-gps ${d.error_km <= Math.max(1, d.radius_km ?? 0) ? 'ok' : 'off'}`}>
            <Icon name="location-crosshairs" /> The file’s own GPS, never shown to the AI, is {distanceText(d.error_km)} from this answer.
          </div>
        )}
        {located && gps && typeof d.error_km !== 'number' && (
          <div className="gb-gps off">
            <Icon name="satellite" /> The file also carries its own GPS position{gps.place ? `: ${gps.place}` : ''}.
          </div>
        )}
      </div>

      <div className={`loc-grid ${located ? '' : 'map-only'}`}>
        <div className="loc-map">
          {points.length ? <MapOrGlobe points={points} height={340} /> : <div className="loc-map-empty">{searching ? <i className="pulse-dot big" /> : <Icon name="earth-europe" />}</div>}
        </div>
        {located && (
          <div className="loc-proof">
            <div className="loc-shots">
              <figure className="gb-img evidence" onClick={() => frame?.time_s != null && pb.seek(frame.time_s)} role="button" tabIndex={0} data-tip={still ? undefined : 'Click to jump to this moment'}>
                {frame?.url ? <img src={frame.url} alt="Evidence frame" /> : <div>No frame</div>}
                <figcaption>{still ? 'The photo' : `The video${frame?.time_s != null ? ` at ${mmss(frame.time_s)}` : ''}`}</figcaption>
              </figure>
              <Thumb a={state.artifacts[`geo:ref${win}_wiki`]} label={w?.landmark ? `Wikipedia: ${w.landmark}` : 'Landmark photo'} empty="No landmark photo" />
              <Thumb a={state.artifacts[`geo:ref${win}_sat`]} label="Satellite view" empty={(w?.radius_km ?? 0) > 25 ? 'Area too large for satellite' : 'No satellite view'} />
            </div>
            {frames.length > 1 && (
              <div className="gb-frames">
                {frames.map((fr, i) => (
                  <button key={fr.id} className={`gb-frame-pick ${i === frameIdx ? 'on' : ''}`} onClick={() => setFrameIdx(i)} data-tip={fr.title}>
                    <img src={fr.url ?? undefined} alt="" />
                    {!still && <span className="mono">{mmss(fr.time_s ?? fr.data?.time_s)}</span>}
                  </button>
                ))}
              </div>
            )}
            <MatchBar match={leads[win]?.match ?? d.match} ok={verified} />
            <Ticks lead={leads[win]} />
            <Why c={w} />
          </div>
        )}
      </div>

      {children}

      <div className="loc-more">
        <details>
          <summary>
            <Icon name="circle-question" /> How this was worked out
          </summary>
          <p className="loc-how muted">
            The AI looks only at the pixels and sounds (never the file’s GPS), proposes places, then each place is checked against a Wikipedia photo of its
            landmark and a satellite view. A place is <b>verified</b> only when the real-world pictures clearly match the {noun}.
          </p>
        </details>
        {others.length > 0 && (
          <details>
            <summary>
              <Icon name="list-ol" /> {others.length} other place{others.length === 1 ? '' : 's'} the AI considered
            </summary>
            <div className="gb-rows">
              {others.map((i, rank) => (
                <CandidateRow key={i} state={state} c={cands[i]} i={i} rank={rank + 2} lead={leads[i]} frame={frame} still={still} />
              ))}
            </div>
          </details>
        )}
        {clues && (
          <details className="gb-clues">
            <summary>
              <Icon name="ear-listen" /> Non-visual clues the AI was given (sound, text, light)
            </summary>
            <pre>{clues}</pre>
          </details>
        )}
        {points.length > 0 && (
          <details>
            <summary>
              <Icon name="map-location-dot" /> Map key and exact coordinates
            </summary>
            <MapSection points={points} gps={gps} legendOnly />
          </details>
        )}
      </div>
    </section>
  )
}
