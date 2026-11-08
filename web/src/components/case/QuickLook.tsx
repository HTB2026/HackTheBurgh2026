import { useEffect, useState } from 'react'
import { drawTexture, useGpu, useGpuCanvas } from '../../lib/gpu'
import { useGpuChecks, type CheckStatus } from '../../lib/gpuChecks'
import { flickerVerdict } from '../../lib/frameScan'
import { usePlayback } from '../../lib/playback'
import { useRegionFocus } from '../../lib/region'
import { mmss } from '../../lib/format'
import { Icon } from '../ui/ui'

function Line({ status, icon, label, children }: { status: CheckStatus; icon: string; label: string; children?: React.ReactNode }) {
  return (
    <li className={`ql-line ${status}`}>
      <Icon name={status === 'running' ? 'spinner' : status === 'done' ? icon : status === 'failed' ? 'triangle-exclamation' : 'hourglass-half'} className={status === 'running' ? 'fa-spin' : ''} />
      <b>{label}</b> <span>{status === 'running' ? 'checking…' : status === 'waiting' ? 'waiting its turn' : children}</span>
    </li>
  )
}

/**
 * First answers from the viewer's own GPU while the server works (and a record of them after):
 * frame scan, mains hum, high tones and an overview spectrogram for videos; the JPEG ghost for
 * photos. Labelled as browser results; the server's findings remain the official record.
 */
export function QuickLook({ duration, open: startOpen = true }: { duration: number; open?: boolean }) {
  const checks = useGpuChecks()
  const gpuState = useGpu()
  const gpu = gpuState === 'loading' ? null : gpuState
  const pb = usePlayback()
  const rf = useRegionFocus()
  const [open, setOpen] = useState(startOpen)
  const spec = useGpuCanvas(gpu)
  const overview = checks.sound.overview

  useEffect(() => {
    if (gpu && spec.ctx && overview) drawTexture(gpu, spec.ctx, overview.texture)
  }, [gpu, spec.ctx, spec.size, overview])

  if (!checks.enabled) return null
  const video = checks.frames.status !== 'off' || checks.sound.status !== 'off'
  if (!video && checks.ghost.status === 'off') return null
  const when = (t: number) => (duration < 60 ? `${t.toFixed(2)} s` : mmss(t))
  const fr = checks.frames.result
  const fl = fr ? flickerVerdict(fr) : null
  const g = checks.ghost.result

  return (
    <section className="quick-look">
      <button className="ql-head" onClick={() => setOpen(!open)} aria-expanded={open}>
        <Icon name="microchip" /> <b>Quick look on your GPU</b>
        <span className="faint small">first answers from this browser while the server works; its findings remain the record</span>
        <Icon name={open ? 'chevron-up' : 'chevron-down'} className="ql-chev" />
      </button>
      {/* Kept mounted while collapsed: the spectrogram canvas must exist to be set up for WebGPU. */}
      {(
        <div className="ql-body" hidden={!open}>
          <ul className="ql-lines">
            {video && (
              <>
                <Line status={checks.frames.status === 'running' ? 'running' : checks.frames.status} icon="film" label="Frames">
                  {checks.frames.status === 'off' ? (
                    'long video: run the frame scan below when you want it'
                  ) : checks.frames.status === 'failed' ? (
                    `could not read the frames (${checks.frames.error})`
                  ) : fr ? (
                    <>
                      {fr.jumps.length ? (
                        <>
                          {fr.jumps.length} abrupt change{fr.jumps.length === 1 ? '' : 's'} at{' '}
                          {fr.jumps.slice(0, 4).map((k, i) => (
                            <button key={k} className="linkish" onClick={() => pb.seek(fr.times[k])}>
                              {i ? ', ' : ''}
                              {when(fr.times[k])}
                            </button>
                          ))}
                        </>
                      ) : (
                        'no abrupt changes'
                      )}
                      {' · '}
                      {fr.duplicates.length ? `${fr.duplicates.length} repeated frames` : 'no repeated frames'}
                      {' · '}
                      {fl?.grid ? `${fl.grid} Hz-grid lamp flicker (${fl.ratio.toFixed(0)}× noise)` : 'no lamp flicker'}
                    </>
                  ) : null}
                </Line>
                {checks.frames.status === 'running' && (
                  <div className="progress ql-progress">
                    <span style={{ width: `${checks.frames.progress * 100}%` }} />
                  </div>
                )}
                <Line status={checks.sound.status} icon="bolt" label="Mains hum">
                  {checks.sound.status === 'off' ? 'no soundtrack' : checks.sound.status === 'failed' ? `could not decode the sound (${checks.sound.error})` : checks.sound.hum ? `steady hum at ${checks.sound.hum.freq.toFixed(2)} Hz → ${checks.sound.hum.grid} Hz grid (${Math.round(checks.sound.hum.steady * 100)}% of the time)` : 'no steady mains hum'}
                </Line>
                <Line status={checks.sound.status} icon="wave-square" label="High tones">
                  {checks.sound.status === 'off' ? 'no soundtrack' : checks.sound.tone ? `steady tone at ${(checks.sound.tone.freq / 1000).toFixed(2)} kHz (often inaudible: a marker or planted signal?)` : 'none'}
                </Line>
              </>
            )}
            {checks.ghost.status !== 'off' && (
              <Line status={checks.ghost.status} icon="layer-group" label="JPEG ghost">
                {checks.ghost.status === 'failed' ? (
                  `could not read the picture (${checks.ghost.error})`
                ) : g?.dominant == null ? (
                  'no block carries a clear JPEG quality'
                ) : (
                  <>
                    saved at Q{g.dominant}
                    {g.regions.length ? (
                      <>
                        {'; '}
                        {g.regions.map((r, i) => (
                          <button key={i} className="linkish" onClick={() => rf.focus({ x: r.x, y: r.y, w: r.w, h: r.h }, `Saved at about Q${r.quality} (browser)`)}>
                            {i ? ', ' : ''}an area at Q{r.quality}
                          </button>
                        ))}{' '}
                        (shown as dashed boxes on the picture)
                      </>
                    ) : (
                      '; no area points to another quality'
                    )}
                  </>
                )}
              </Line>
            )}
          </ul>
          {overview && (
            <figure className="ql-spec">
              <canvas ref={spec.ref} />
              <figcaption className="faint small">Sound, 0–{Math.round(overview.rate / 2000)} kHz (bottom to top), computed in your browser</figcaption>
            </figure>
          )}
        </div>
      )}
    </section>
  )
}
