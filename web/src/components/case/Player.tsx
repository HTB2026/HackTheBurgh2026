import { useEffect, useMemo, useRef, useState } from 'react'
import type { Artifact, CaseInfo } from '../../lib/types'
import { useIsPlaying, usePlayback, usePlayhead } from '../../lib/playback'
import { useElementSize } from '../../lib/hooks'
import { mmss } from '../../lib/format'
import { capture } from '../../lib/pointer'
import { Icon } from '../ui/ui'
import { EnhanceModal, type EnhanceBox } from './EnhanceModal'
import { segmentsOf } from '../viz/Transcript'

interface Rect {
  x: number
  y: number
  w: number
  h: number
}

/** Where the picture sits inside an object-fit:contain element (letterboxing). */
function contentRect(boxW: number, boxH: number, vw: number, vh: number): Rect {
  if (!vw || !vh || !boxW || !boxH) return { x: 0, y: 0, w: boxW, h: boxH }
  const s = Math.min(boxW / vw, boxH / vh)
  return { x: (boxW - vw * s) / 2, y: (boxH - vh * s) / 2, w: vw * s, h: vh * s }
}

export function Player({
  info,
  mediaUrl,
  mediaReady,
  running,
  heatmaps,
  captions,
  height = 420,
  compact,
}: {
  info: CaseInfo | null
  mediaUrl: string | null
  mediaReady: boolean
  running: boolean
  heatmaps: Artifact[]
  captions: Artifact[]
  height?: number
  compact?: boolean
}) {
  const pb = usePlayback()
  const playing = useIsPlaying()
  const t = usePlayhead()
  const videoRef = useRef<HTMLVideoElement>(null)
  const [stageRef, stage] = useElementSize<HTMLDivElement>()
  const [natural, setNatural] = useState({ w: 0, h: 0 })
  const [rate, setRate] = useState(1)
  const [muted, setMuted] = useState(true)
  const [enhanceMode, setEnhanceMode] = useState(false)
  const [drag, setDrag] = useState<{ x0: number; y0: number; x1: number; y1: number } | null>(null)
  const [request, setRequest] = useState<EnhanceBox | null>(null)
  const [overlayOn, setOverlayOn] = useState(false)
  const [ccOn, setCcOn] = useState(true)

  const fps = info?.video?.fps || 30
  const vw = natural.w || info?.video?.width || 16
  const vh = natural.h || info?.video?.height || 9
  const rect = contentRect(stage.width, stage.height, vw, vh)
  const hasVideo = !!mediaUrl

  useEffect(() => {
    const v = videoRef.current
    if (!v) return
    pb.attach((time) => {
      v.currentTime = time
    })
    let raf = 0
    const loop = () => {
      pb.set({ time: v.currentTime })
      if (!v.paused) raf = requestAnimationFrame(loop)
    }
    const onPlay = () => {
      pb.set({ playing: true })
      cancelAnimationFrame(raf)
      raf = requestAnimationFrame(loop)
    }
    const onPause = () => {
      cancelAnimationFrame(raf)
      pb.set({ playing: false, time: v.currentTime })
    }
    const onSeeked = () => pb.set({ time: v.currentTime })
    const onMeta = () => {
      setNatural({ w: v.videoWidth, h: v.videoHeight })
      if (isFinite(v.duration) && v.duration > 0) pb.set({ duration: Math.max(pb.duration, v.duration) })
      // the player can remount when switching tabs: resume where we were
      if (pb.time > 0 && Math.abs(v.currentTime - pb.time) > 0.05) v.currentTime = pb.time
    }
    v.addEventListener('play', onPlay)
    v.addEventListener('pause', onPause)
    v.addEventListener('ended', onPause)
    v.addEventListener('seeked', onSeeked)
    v.addEventListener('timeupdate', onSeeked)
    v.addEventListener('loadedmetadata', onMeta)
    return () => {
      cancelAnimationFrame(raf)
      pb.attach(null)
      pb.set({ playing: false })
      v.removeEventListener('play', onPlay)
      v.removeEventListener('pause', onPause)
      v.removeEventListener('ended', onPause)
      v.removeEventListener('seeked', onSeeked)
      v.removeEventListener('timeupdate', onSeeked)
      v.removeEventListener('loadedmetadata', onMeta)
    }
  }, [pb, mediaUrl])

  useEffect(() => {
    if (videoRef.current) videoRef.current.playbackRate = rate
  }, [rate, mediaUrl])

  const toggle = () => {
    const v = videoRef.current
    if (!v) return
    if (v.paused) {
      setEnhanceMode(false)
      v.play().catch(() => undefined)
    } else v.pause()
  }
  const step = (frames: number) => {
    const v = videoRef.current
    if (!v) return
    v.pause()
    pb.seek(v.currentTime + frames / fps)
  }
  const startEnhance = () => {
    videoRef.current?.pause()
    setEnhanceMode((m) => !m)
    setDrag(null)
  }

  useEffect(() => {
    const on = (e: KeyboardEvent) => {
      const el = e.target as HTMLElement
      if (el && (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA' || el.tagName === 'SELECT' || el.isContentEditable)) return
      if (e.metaKey || e.ctrlKey || e.altKey) return
      if (e.key === ' ') {
        e.preventDefault()
        toggle()
      } else if (e.key === 'ArrowLeft') {
        e.preventDefault()
        step(e.shiftKey ? -10 : -1)
      } else if (e.key === 'ArrowRight') {
        e.preventDefault()
        step(e.shiftKey ? 10 : 1)
      } else if ((e.key === 'z' || e.key === 'Z') && hasVideo) startEnhance()
      else if (e.key === 'Escape') setEnhanceMode(false)
    }
    window.addEventListener('keydown', on)
    return () => window.removeEventListener('keydown', on)
  })

  const local = (e: React.PointerEvent) => {
    const b = (e.currentTarget as HTMLElement).getBoundingClientRect()
    return {
      x: Math.min(rect.x + rect.w, Math.max(rect.x, e.clientX - b.left)),
      y: Math.min(rect.y + rect.h, Math.max(rect.y, e.clientY - b.top)),
    }
  }
  const sel = drag && { x: Math.min(drag.x0, drag.x1), y: Math.min(drag.y0, drag.y1), w: Math.abs(drag.x1 - drag.x0), h: Math.abs(drag.y1 - drag.y0) }
  const finishDrag = () => {
    if (!sel || sel.w < 6 || sel.h < 6 || !rect.w || !rect.h) return setDrag(null)
    setRequest({
      x: (sel.x - rect.x) / rect.w,
      y: (sel.y - rect.y) / rect.h,
      w: sel.w / rect.w,
      h: sel.h / rect.h,
      time_s: videoRef.current?.currentTime ?? pb.time,
    })
    setDrag(null)
    setEnhanceMode(false)
  }

  const heat = useMemo(() => {
    if (!overlayOn) return null
    let best: Artifact | null = null
    let bestD = Infinity
    for (const a of heatmaps) {
      if (!a.url || a.data?.overlay === false) continue
      const at = a.data?.time_s ?? a.time_s
      const d = at == null ? 0.75 : Math.abs(at - t)
      if (d < bestD) {
        best = a
        bestD = d
      }
    }
    return bestD <= 1.5 ? best : null
  }, [heatmaps, t, overlayOn])

  const subtitle = useMemo(() => {
    if (!ccOn) return null
    for (const c of captions) for (const s of segmentsOf(c)) if (t >= s.start && t < s.end) return s.text
    return null
  }, [captions, t, ccOn])

  return (
    <div className={`player ${compact ? 'compact' : ''}`}>
      <div className="stage" ref={stageRef} style={{ height }}>
        {hasVideo ? (
          <video ref={videoRef} src={mediaUrl!} poster={info?.thumb_url ?? undefined} playsInline muted={muted} preload="auto" onClick={toggle} />
        ) : (
          info?.thumb_url && <img className="poster" src={info.thumb_url} alt="" />
        )}
        {heat && <img className="heat" src={heat.url!} alt="" style={{ left: rect.x, top: rect.y, width: rect.w, height: rect.h }} />}
        {subtitle && (
          <div className="subtitle" style={{ left: rect.x, width: rect.w, top: rect.y + rect.h - 56 }}>
            <span>{subtitle}</span>
          </div>
        )}
        <div className="stage-tag">
          {running && (
            <span className="live">
              <i className="pulse-dot" /> ANALYSING
            </span>
          )}
        </div>
        <div className="stage-tc mono">
          {mmss(t)} <span className="faint">/ {mmss(pb.duration || info?.duration_s)}</span>
          {!compact && hasVideo && <span className="faint"> · frame {Math.round(t * fps)}</span>}
        </div>
        {!hasVideo && <div className="stage-msg">{mediaReady ? 'This video cannot be played in the browser.' : 'Preparing a playable copy…'}</div>}
        {heat && <div className="stage-msg top">Overlay: {heat.title}</div>}

        {enhanceMode && (
          <div
            className="enhance-layer"
            onPointerDown={(e) => {
              capture(e)
              const p = local(e)
              setDrag({ x0: p.x, y0: p.y, x1: p.x, y1: p.y })
            }}
            onPointerMove={(e) => {
              if (!drag) return
              const p = local(e)
              setDrag({ ...drag, x1: p.x, y1: p.y })
            }}
            onPointerUp={finishDrag}
          >
            <div className="content-box" style={{ left: rect.x, top: rect.y, width: rect.w, height: rect.h }} />
            {sel && sel.w > 0 && (
              <div className="sel" style={{ left: sel.x, top: sel.y, width: sel.w, height: sel.h }}>
                <span className="mono">
                  {Math.round((sel.w / rect.w) * vw)}×{Math.round((sel.h / rect.h) * vh)} px
                </span>
              </div>
            )}
            {!sel && <div className="enhance-hint">Drag a box around the detail you want to see (Esc to cancel)</div>}
          </div>
        )}
      </div>

      <div className="controls">
        <button className="ctl" onClick={() => step(-1)} disabled={!hasVideo} data-tip="Back one frame (←)" aria-label="Back one frame">
          <Icon name="backward-step" />
        </button>
        <button className="ctl play" onClick={toggle} disabled={!hasVideo} data-tip="Play / pause (space)" aria-label={playing ? 'Pause' : 'Play'}>
          <Icon name={playing ? 'pause' : 'play'} />
        </button>
        <button className="ctl" onClick={() => step(1)} disabled={!hasVideo} data-tip="Forward one frame (→)" aria-label="Forward one frame">
          <Icon name="forward-step" />
        </button>
        {!compact &&
          [0.25, 0.5, 1].map((r) => (
            <button key={r} className={`ctl txt ${rate === r ? 'on' : ''}`} onClick={() => setRate(r)} disabled={!hasVideo} data-tip="Playback speed">
              {r}×
            </button>
          ))}
        <button className="ctl" onClick={() => setMuted((m) => !m)} disabled={!hasVideo || !info?.has_audio} data-tip={muted ? 'Turn sound on' : 'Mute'} aria-label="Sound">
          <Icon name={muted ? 'volume-xmark' : 'volume-high'} />
        </button>
        {captions.length > 0 && (
          <button className={`ctl ${ccOn ? 'on' : ''}`} onClick={() => setCcOn((c) => !c)} data-tip="Subtitles from the speech transcript" aria-label="Subtitles">
            <Icon name="closed-captioning" />
          </button>
        )}
        {heatmaps.length > 0 && (
          <button className={`ctl ${overlayOn ? 'on' : ''}`} onClick={() => setOverlayOn((o) => !o)} data-tip="Overlay the nearest tamper map on the video" aria-label="Tamper overlay">
            <Icon name="layer-group" />
          </button>
        )}
        <span className="grow" />
        <button className={`btn btn-red sm ${enhanceMode ? 'on' : ''}`} onClick={startEnhance} disabled={!hasVideo} data-tip="Stack nearby frames to sharpen a region (Z)">
          <Icon name="magnifying-glass-plus" /> {compact ? 'Zoom' : 'Zoom in on a detail'}
        </button>
      </div>

      {request && info && <EnhanceModal caseId={info.id} box={request} onClose={() => setRequest(null)} />}
    </div>
  )
}
