import { useState } from 'react'
import type { Artifact, Finding } from '../../lib/types'
import type { CaseState } from '../../lib/state'
import { usePlayback } from '../../lib/playback'
import { mmss } from '../../lib/format'
import { Icon, Modal } from '../ui/ui'
import { distanceText, radiusText, type MapPoint } from '../viz/MapView'
import { MapSection } from './Sections'
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

/** The geo artifacts the board shows itself (kept out of the generic exhibits list). */


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

export function GeoBoard({ state, facts, points }: { state: CaseState; facts: Facts; points: MapPoint[] }) {
  const pb = usePlayback()
  const f = geoFinding(state)!
  const d = f.data ?? {}
  const cands: Candidate[] = d.candidates ?? []
  const win = winnerIndex(d)
  const leads = leadsFrom(state.artifacts['geo:leads'])
  const frames = state.artifactOrder
    .filter((id) => /^geo:frame\d+$/.test(id))
    .map((id) => state.artifacts[id])
    .sort((a, b) => Number(a.id.replace(/\D/g, '')) - Number(b.id.replace(/\D/g, '')))
  const [frameIdx, setFrameIdx] = useState(0)
  const clues: string = state.artifacts['geo:clues']?.data?.text ?? ''
  const verified = !!d.verified
  const order = cands.map((_, i) => i).sort((a, b) => (a === win ? -1 : b === win ? 1 : (leads[b]?.match ?? -1) - (leads[a]?.match ?? -1)))
  const frame = frames[frameIdx]

  return (
    <div className="geo-board">
      <div className={`gb-answer ${verified ? 'ok' : 'lead'}`}>
        <span className={`gb-stamp ${verified ? 'ok' : 'lead'}`}>
          <Icon name={verified ? 'circle-check' : 'magnifying-glass-location'} /> {verified ? 'Verified' : 'Unverified lead'}
        </span>
        <div>
          <b className="gb-place">{d.place ?? f.title}</b>
          <div className="muted">
            {typeof d.latitude === 'number' && (
              <span className="mono">
                {d.latitude.toFixed(5)}, {d.longitude?.toFixed(5)}
              </span>
            )}
            {d.radius_km ? ` · within ${radiusText(d.radius_km * 1000).slice(1)}` : ''}
            {d.confidence ? ` · ${d.confidence} confidence` : ''}
            {typeof d.match === 'number' ? ` · ${Math.round(d.match * 100)}% match with real-world imagery` : ''}
          </div>
          {typeof d.error_km === 'number' && (
            <div className={`gb-gps ${d.error_km <= Math.max(1, d.radius_km ?? 0) ? 'ok' : 'off'}`}>
              <Icon name="location-crosshairs" /> The file’s own GPS, never shown to the AI, is {distanceText(d.error_km)} from this answer.
            </div>
          )}
        </div>
      </div>

      <p className="muted">
        How it works: the AI looks only at the pixels and sounds (never the file’s GPS), proposes places, then each place is checked against a Wikipedia
        photo of its landmark and a satellite view. A place is <b>verified</b> only when the real-world pictures clearly match the video.
      </p>

      {clues && (
        <details className="gb-clues">
          <summary>
            <Icon name="ear-listen" /> Non-visual clues the AI was given (sound, text, light)
          </summary>
          <pre>{clues}</pre>
        </details>
      )}

      {frames.length > 0 && (
        <div className="gb-frames">
          <span className="muted small">Evidence frames</span>
          {frames.map((fr, i) => (
            <button key={fr.id} className={`gb-frame-pick ${i === frameIdx ? 'on' : ''}`} onClick={() => setFrameIdx(i)} data-tip={fr.title}>
              <img src={fr.url ?? undefined} alt="" />
              <span className="mono">{mmss(fr.time_s ?? fr.data?.time_s)}</span>
            </button>
          ))}
        </div>
      )}

      <div className="gb-rows">
        {order.map((i, rank) => {
          const c = cands[i]
          const l = leads[i]
          const isWin = i === win
          return (
            <article key={i} className={`gb-row ${isWin ? (verified ? 'win ok' : 'win lead') : ''}`}>
              <div className="gb-imgs">
                <figure className="gb-img evidence" onClick={() => frame?.time_s != null && pb.seek(frame.time_s)} role="button" tabIndex={0} data-tip="Evidence frame from the video. Click to jump there.">
                  {frame?.url ? <img src={frame.url} alt="Evidence frame" /> : <div>No frame</div>}
                  <figcaption>The video</figcaption>
                </figure>
                <Icon name="arrows-left-right" className="gb-vs" />
                <Thumb a={state.artifacts[`geo:ref${i}_wiki`]} label={c.landmark ? `Wikipedia: ${c.landmark}` : 'Landmark photo'} empty="No landmark photo" />
                <Thumb a={state.artifacts[`geo:ref${i}_sat`]} label="Satellite view" empty={c.radius_km > 25 ? 'Area too large for satellite' : 'No satellite view'} />
              </div>
              <div className="gb-info">
                <div className="gb-title">
                  <span className="gb-rank mono">#{rank + 1}</span>
                  <b>{c.place}</b>
                  {isWin && (
                    <span className={`gb-stamp small ${verified ? 'ok' : 'lead'}`}>{verified ? 'Verified' : 'Best lead'}</span>
                  )}
                </div>
                <div className="muted small">
                  {c.landmark ? `${c.landmark} · ` : ''}
                  <span className="mono">
                    {c.latitude.toFixed(4)}, {c.longitude.toFixed(4)}
                  </span>{' '}
                  · within {radiusText(c.radius_km * 1000).slice(1)}
                  {typeof c.probability === 'number' ? ` · proposed at ${Math.round(c.probability * 100)}%` : ''}
                </div>
                <div className="gb-match" data-tip="How well the real-world photo and satellite view match the video">
                  <span>Match</span>
                  <span className="gb-bar">
                    <i style={{ width: `${Math.round((l?.match ?? 0) * 100)}%` }} className={isWin && verified ? 'ok' : ''} />
                  </span>
                  <b className="mono">{l?.match != null ? `${Math.round(l.match * 100)}%` : 'not checked'}</b>
                </div>
                {(l?.matches.length ?? 0) > 0 && (
                  <ul className="gb-list ok">
                    {l!.matches.map((m, k) => (
                      <li key={k}>
                        <Icon name="check" /> {m}
                      </li>
                    ))}
                  </ul>
                )}
                {(l?.mismatches.length ?? 0) > 0 && (
                  <ul className="gb-list no">
                    {l!.mismatches.map((m, k) => (
                      <li key={k}>
                        <Icon name="xmark" /> {m}
                      </li>
                    ))}
                  </ul>
                )}
                {c.evidence && c.evidence.length > 0 && (
                  <details className="gb-why">
                    <summary>Why the AI proposed it</summary>
                    <ul>
                      {c.evidence.map((e, k) => (
                        <li key={k}>{e}</li>
                      ))}
                    </ul>
                  </details>
                )}
              </div>
            </article>
          )
        })}
      </div>

      {points.length > 0 && (
        <>
          <h4 className="sub">On the map</h4>
          <MapSection points={points} gps={facts.gps} />
        </>
      )}
    </div>
  )
}
