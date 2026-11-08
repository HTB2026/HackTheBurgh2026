import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import type { Artifact, Box, CaseInfo, Finding } from '../../lib/types'
import { boxesOf, boxOf } from '../../lib/types'
import { useElementSize } from '../../lib/hooks'
import { capture } from '../../lib/pointer'
import { useFocusedRegion, useRegionFocus } from '../../lib/region'
import { SEVERITY } from '../../lib/plain'
import { rankFindings } from '../../lib/facts'
import { Icon, SevChip } from '../ui/ui'
import { EnhanceModal, type EnhanceBox } from './EnhanceModal'
import { GpuLens, type GpuLensMode } from './GpuLens'
import { useGpu } from '../../lib/gpu'

interface View {
  z: number // zoom relative to "fit"
  tx: number // image top-left in stage pixels
  ty: number
}

type LensKey = 'none' | 'invert' | 'grey' | 'red' | 'green' | 'blue' | 'edges' | 'saturate' | 'levels' | `gpu-${GpuLensMode}`

/** Lenses computed on the GPU at full resolution (shown only when the browser has WebGPU). */
const GPU_LENSES: { key: LensKey; mode: GpuLensMode; label: string; tip: string; gain: [number, number, number] }[] = [
  { key: 'gpu-ela', mode: 'ela', label: 'Error level (ELA)', tip: 'Re-saves the picture as JPEG and shows how much each pixel changes. Areas pasted from a picture saved at a different quality glow differently from the rest.', gain: [5, 60, 20] },
  { key: 'gpu-noise', mode: 'noise', label: 'Noise residual', tip: 'The fine grain left after removing the picture itself. Pasted, cloned or smoothed areas have a different grain from the rest.', gain: [2, 40, 10] },
  { key: 'gpu-contrast', mode: 'contrast', label: 'Local contrast', tip: 'How much detail each small area holds. Retouched or blurred areas go dark.', gain: [1, 16, 4] },
  { key: 'gpu-sobel', mode: 'sobel', label: 'Edges', tip: 'Edge strength at full resolution: halos and double edges around pasted objects.', gain: [1, 16, 3] },
]

/** Formats browsers decode, so the GPU lenses can read the original file rather than the re-encoded viewing copy. */
const DECODABLE = new Set(['JPEG', 'JPG', 'MPO', 'PNG', 'WEBP', 'GIF', 'BMP'])

const LENSES: { key: LensKey; label: string; tip: string }[] = [
  { key: 'none', label: 'Normal', tip: 'The picture as stored' },
  { key: 'levels', label: 'Levels', tip: 'Stretch a narrow range of brightness to full range: detail in shadows or highlights' },
  { key: 'invert', label: 'Negative', tip: 'Faint dark marks on light backgrounds jump out' },
  { key: 'grey', label: 'Grey', tip: 'Luminance only: shapes and edges without colour distraction' },
  { key: 'saturate', label: 'Colour ×4', tip: 'Colour pushed hard: pasted objects with a different white balance stand out' },
  { key: 'red', label: 'Red', tip: 'Red channel only' },
  { key: 'green', label: 'Green', tip: 'Green channel only: the sharpest channel on most sensors' },
  { key: 'blue', label: 'Blue', tip: 'Blue channel only: the noisiest, where retouching shows first' },
  { key: 'edges', label: 'Edges', tip: 'Edge strength: halos and double edges around pasted objects' },
]

function lensFilter(lens: LensKey, levels: [number, number]): string {
  switch (lens) {
    case 'invert':
      return 'invert(1)'
    case 'grey':
      return 'grayscale(1)'
    case 'saturate':
      return 'saturate(4)'
    case 'red':
    case 'green':
    case 'blue':
    case 'edges':
      return `url(#lens-${lens})`
    case 'levels': {
      const [lo, hi] = levels
      const span = Math.max(0.02, hi - lo)
      const c = 1 + (2 * lo) / span
      const b = 1 / (span * c)
      return `brightness(${b.toFixed(3)}) contrast(${c.toFixed(3)})`
    }
    default:
      return 'none'
  }
}

/** SVG filters for channel isolation and edge detection; referenced by CSS `filter: url(#…)`. */
function LensDefs() {
  const only = (i: number) => {
    const row = ['0 0 0 0 0', '0 0 0 0 0', '0 0 0 0 0']
    const pick = ['1 0 0 0 0', '0 1 0 0 0', '0 0 1 0 0'][i]
    return `${row.map(() => pick).join(' ')} 0 0 0 1 0`
  }
  return (
    <svg width="0" height="0" style={{ position: 'absolute' }} aria-hidden>
      <filter id="lens-red" colorInterpolationFilters="sRGB">
        <feColorMatrix type="matrix" values={only(0)} />
      </filter>
      <filter id="lens-green" colorInterpolationFilters="sRGB">
        <feColorMatrix type="matrix" values={only(1)} />
      </filter>
      <filter id="lens-blue" colorInterpolationFilters="sRGB">
        <feColorMatrix type="matrix" values={only(2)} />
      </filter>
      <filter id="lens-edges" colorInterpolationFilters="sRGB">
        <feColorMatrix type="saturate" values="0" />
        <feConvolveMatrix order="3" kernelMatrix="-1 -1 -1 -1 8 -1 -1 -1 -1" divisor="1" bias="0" preserveAlpha="true" />
        <feComponentTransfer>
          <feFuncR type="linear" slope="3" />
          <feFuncG type="linear" slope="3" />
          <feFuncB type="linear" slope="3" />
        </feComponentTransfer>
      </filter>
    </svg>
  )
}

const BOX_COLOUR: Record<string, string> = { critical: '#ff4040', warning: '#fbbd08', notice: '#38d6ff', info: '#9a9a9a' }

export function PhotoStage({
  info,
  mediaUrl,
  running,
  overlays,
  findings,
  height = 560,
  onOpenFinding,
}: {
  info: CaseInfo
  mediaUrl: string | null
  running: boolean
  overlays: Artifact[]
  findings: Finding[]
  height?: number
  onOpenFinding: (id: string) => void
}) {
  const [stageRef, stage] = useElementSize<HTMLDivElement>()
  const iw = info.image?.width || 4
  const ih = info.image?.height || 3
  const s0 = stage.width && stage.height ? Math.min(stage.width / iw, stage.height / ih) : 1
  const fitView = useCallback((): View => ({ z: 1, tx: (stage.width - iw * s0) / 2, ty: (stage.height - ih * s0) / 2 }), [stage.width, stage.height, iw, ih, s0])
  // Keep the picture in the frame: centred on an axis where it fits, edges never inside the frame where it is larger.
  const clamp = useCallback(
    (v: View): View => {
      const axis = (t: number, shown: number, room: number) => (shown <= room ? (room - shown) / 2 : Math.min(0, Math.max(room - shown, t)))
      return { z: v.z, tx: axis(v.tx, iw * s0 * v.z, stage.width), ty: axis(v.ty, ih * s0 * v.z, stage.height) }
    },
    [stage.width, stage.height, iw, ih, s0],
  )
  const [view, setView] = useState<View>({ z: 1, tx: 0, ty: 0 })
  const [fitted, setFitted] = useState(false)
  const [drag, setDrag] = useState<{ x: number; y: number; tx: number; ty: number } | null>(null)
  const [cursor, setCursor] = useState<{ x: number; y: number } | null>(null)
  const [lens, setLens] = useState<LensKey>('none')
  const [levels, setLevels] = useState<[number, number]>([0, 1])
  const [overlayId, setOverlayId] = useState<string>('')
  const [opacity, setOpacity] = useState(0.7)
  const [blend, setBlend] = useState<'normal' | 'screen'>('normal')
  const [showBoxes, setShowBoxes] = useState(true)
  const [loupe, setLoupe] = useState(false)
  const [enhanceMode, setEnhanceMode] = useState(false)
  const [sel, setSel] = useState<{ x0: number; y0: number; x1: number; y1: number } | null>(null)
  const [request, setRequest] = useState<EnhanceBox | null>(null)
  const [hover, setHover] = useState<string | null>(null)
  const rf = useRegionFocus()
  const focused = useFocusedRegion()
  const lastFocus = useRef(0)
  const gpuState = useGpu()
  const gpu = gpuState === 'loading' ? null : gpuState
  const gpuLens = GPU_LENSES.find((l) => l.key === lens) ?? null
  const [elaQuality, setElaQuality] = useState(90)
  const [gains, setGains] = useState<Record<string, number>>({})
  const gain = gpuLens ? (gains[gpuLens.key] ?? gpuLens.gain[2]) : 1
  const [gpuStatus, setGpuStatus] = useState<{ loading: boolean; error: string | null }>({ loading: false, error: null })
  // Error level analysis needs the original bytes: the viewing copy has been re-encoded.
  const original = info.original_url && DECODABLE.has((info.image?.format ?? '').toUpperCase()) ? info.original_url : null
  const gpuSrc = original ?? mediaUrl
  const lensList = gpu ? [...LENSES.filter((l) => l.key !== 'edges'), ...GPU_LENSES] : LENSES

  useEffect(() => {
    if (!fitted && stage.width && stage.height) {
      setView(fitView())
      setFitted(true)
    }
  }, [fitted, stage.width, stage.height, fitView])

  // A resized frame (window, layout) must not leave the picture off-frame.
  useEffect(() => {
    if (fitted) setView((v) => clamp(v))
  }, [fitted, clamp])

  const scale = s0 * view.z // stage pixels per image pixel
  const frame = { left: view.tx, top: view.ty, width: iw * scale, height: ih * scale }

  const zoomAt = useCallback(
    (factor: number, cx?: number, cy?: number) => {
      setView((v) => {
        // Never smaller than "fit": zooming out further only adds black around the picture.
        const z = Math.min(60, Math.max(1, v.z * factor))
        const f = z / v.z
        const px = cx ?? stage.width / 2
        const py = cy ?? stage.height / 2
        return clamp({ z, tx: px - (px - v.tx) * f, ty: py - (py - v.ty) * f })
      })
    },
    [stage.width, stage.height, clamp],
  )

  const zoomToBox = useCallback(
    (b: Box) => {
      if (!stage.width || !stage.height) return
      const bw = b.w * iw * s0
      const bh = b.h * ih * s0
      const z = Math.min(40, Math.max(1, 0.6 * Math.min(stage.width / bw, stage.height / bh)))
      const cxImg = (b.x + b.w / 2) * iw * s0 * z
      const cyImg = (b.y + b.h / 2) * ih * s0 * z
      setView(clamp({ z, tx: stage.width / 2 - cxImg, ty: stage.height / 2 - cyImg }))
    },
    [stage.width, stage.height, iw, ih, s0, clamp],
  )

  // A finding asked to be shown: zoom to its box.
  useEffect(() => {
    if (focused.nonce === lastFocus.current) return
    lastFocus.current = focused.nonce
    if (focused.box) zoomToBox(focused.box)
  }, [focused.nonce, focused.box, zoomToBox])

  const reset = () => setView(fitView())
  const oneToOne = () => {
    // A picture smaller than the frame is already shown at least 1:1 at "fit".
    const z = Math.max(1, 1 / s0)
    const cx = stage.width / 2
    const cy = stage.height / 2
    setView((v) => clamp({ z, tx: cx - (cx - v.tx) * (z / v.z), ty: cy - (cy - v.ty) * (z / v.z) }))
  }

  const local = (e: React.PointerEvent | React.WheelEvent | React.MouseEvent) => {
    const r = (e.currentTarget as HTMLElement).getBoundingClientRect()
    return { x: e.clientX - r.left, y: e.clientY - r.top }
  }
  const toImage = (p: { x: number; y: number }) => ({ x: (p.x - view.tx) / scale, y: (p.y - view.ty) / scale })

  useEffect(() => {
    const on = (e: KeyboardEvent) => {
      const el = e.target as HTMLElement
      if (el && (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA' || el.tagName === 'SELECT' || el.isContentEditable)) return
      if (e.metaKey || e.ctrlKey || e.altKey) return
      if (e.key === '+' || e.key === '=') zoomAt(1.5)
      else if (e.key === '-') zoomAt(1 / 1.5)
      else if (e.key === '0') reset()
      else if (e.key === '1') oneToOne()
      else if (e.key === 'b' || e.key === 'B') setShowBoxes((b) => !b)
      else if (e.key === 'm' || e.key === 'M') setLoupe((l) => !l)
      else if (e.key === 'z' || e.key === 'Z') setEnhanceMode((m) => !m)
      else if (e.key === 'Escape') {
        setEnhanceMode(false)
        setSel(null)
      }
    }
    window.addEventListener('keydown', on)
    return () => window.removeEventListener('keydown', on)
  })

  const overlay = overlays.find((a) => a.id === overlayId) ?? null
  // One box per thing a finding points at: e.g. every face, not just the largest.
  const boxed = useMemo(() => rankFindings(findings).flatMap((f) => boxesOf(f).map((b) => ({ f, ...b }))), [findings])
  const cursorImg = cursor ? toImage(cursor) : null
  const inside = cursorImg && cursorImg.x >= 0 && cursorImg.y >= 0 && cursorImg.x <= iw && cursorImg.y <= ih
  const selRect = sel && { x: Math.min(sel.x0, sel.x1), y: Math.min(sel.y0, sel.y1), w: Math.abs(sel.x1 - sel.x0), h: Math.abs(sel.y1 - sel.y0) }

  const finishSelection = () => {
    if (!selRect || selRect.w < 6 || selRect.h < 6) return setSel(null)
    const a = toImage({ x: selRect.x, y: selRect.y })
    const b = toImage({ x: selRect.x + selRect.w, y: selRect.y + selRect.h })
    const x0 = Math.max(0, a.x / iw)
    const y0 = Math.max(0, a.y / ih)
    const x1 = Math.min(1, b.x / iw)
    const y1 = Math.min(1, b.y / ih)
    setSel(null)
    setEnhanceMode(false)
    if (x1 - x0 < 0.002 || y1 - y0 < 0.002) return
    setRequest({ x: x0, y: y0, w: x1 - x0, h: y1 - y0, time_s: 0 })
  }

  const pointerDown = (e: React.PointerEvent) => {
    capture(e)
    const p = local(e)
    if (enhanceMode) {
      setSel({ x0: p.x, y0: p.y, x1: p.x, y1: p.y })
      return
    }
    setDrag({ x: p.x, y: p.y, tx: view.tx, ty: view.ty })
  }
  const pointerMove = (e: React.PointerEvent) => {
    const p = local(e)
    setCursor(p)
    if (sel) setSel({ ...sel, x1: p.x, y1: p.y })
    else if (drag) setView((v) => clamp({ ...v, tx: drag.tx + (p.x - drag.x), ty: drag.ty + (p.y - drag.y) }))
  }
  const pointerUp = () => {
    if (sel) finishSelection()
    setDrag(null)
  }
  // Wheel zooms the picture and must not scroll the page. React's onWheel is passive
  // (preventDefault is ignored), so listen natively with passive: false.
  useEffect(() => {
    const el = stageRef.current
    if (!el) return
    const on = (e: WheelEvent) => {
      e.preventDefault()
      e.stopPropagation()
      const r = el.getBoundingClientRect()
      // Zoom in proportion to the scroll: a mouse notch (~100 px) is about 1.2x, while a
      // trackpad's stream of small deltas zooms smoothly instead of 1.2x per event.
      // Lines/pages (deltaMode 1/2, e.g. Firefox mouse wheels) are converted to pixels.
      const px = e.deltaY * (e.deltaMode === 1 ? 16 : e.deltaMode === 2 ? r.height : 1)
      // Pinch on a trackpad arrives as ctrl+wheel with small deltas: make it a bit livelier.
      const factor = Math.exp(-Math.max(-300, Math.min(300, px)) * (e.ctrlKey ? 0.01 : 0.0018))
      zoomAt(factor, e.clientX - r.left, e.clientY - r.top)
    }
    el.addEventListener('wheel', on, { passive: false })
    return () => el.removeEventListener('wheel', on)
  }, [stageRef, zoomAt])

  const pixelated = scale >= 3
  // Something to drag only once the picture is larger than the frame.
  const pannable = iw * scale > stage.width + 0.5 || ih * scale > stage.height + 0.5
  const loupeSize = 180
  const loupeZoom = 3

  return (
    <div className="photo" id="photo-stage">
      <LensDefs />
      <div
        className={`stage photo-canvas ${enhanceMode ? 'enhancing' : drag && pannable ? 'dragging' : pannable ? 'pannable' : ''}`}
        ref={stageRef}
        style={{ height }}
        onPointerDown={pointerDown}
        onPointerMove={pointerMove}
        onPointerUp={pointerUp}
        onPointerLeave={() => {
          setCursor(null)
          setDrag(null)
        }}
        onDoubleClick={(e) => {
          const p = local(e)
          zoomAt(2, p.x, p.y)
        }}
      >
        {gpu && gpuLens && gpuSrc && (
          <GpuLens
            gpu={gpu}
            src={gpuSrc}
            mode={gpuLens.mode}
            quality={elaQuality}
            gain={gain}
            view={{ tx: view.tx, ty: view.ty, scale }}
            iw={iw}
            ih={ih}
            pixelated={pixelated}
            onState={setGpuStatus}
          />
        )}
        <div className="photo-frame" style={frame}>
          {mediaUrl ? (
            <img
              className={`photo-img ${pixelated ? 'pixelated' : ''}`}
              src={mediaUrl}
              alt={info.filename}
              draggable={false}
              style={{ filter: lensFilter(lens, levels), visibility: gpuLens && !gpuStatus.error ? 'hidden' : undefined }}
            />
          ) : (
            info.thumb_url && <img className="photo-img" src={info.thumb_url} alt="" draggable={false} />
          )}
          {overlay?.url && (
            <img className={`photo-overlay ${pixelated ? 'pixelated' : ''}`} src={overlay.url} alt="" draggable={false} style={{ opacity, mixBlendMode: blend }} />
          )}
          {showBoxes &&
            boxed.map(({ f, box, label, tip }) => (
              <button
                key={`${f.id}:${label ?? ''}`}
                className={`photo-box sev-${f.severity} ${hover === f.id ? 'hover' : ''} ${focused.box === box ? 'focus' : ''}`}
                style={{ left: `${box.x * 100}%`, top: `${box.y * 100}%`, width: `${box.w * 100}%`, height: `${box.h * 100}%`, borderColor: BOX_COLOUR[f.severity] }}
                onPointerDown={(e) => e.stopPropagation()}
                onClick={(e) => {
                  e.stopPropagation()
                  rf.focus(box, f.title)
                  onOpenFinding(f.id)
                }}
                onMouseEnter={() => setHover(f.id)}
                onMouseLeave={() => setHover((h) => (h === f.id ? null : h))}
                data-tip={`${SEVERITY[f.severity]?.word ?? ''}: ${f.title}${tip ? ` (${tip})` : ''}`}
                aria-label={tip ? `${f.title}: ${tip}` : f.title}
              >
                <span className="photo-box-tag" style={{ background: BOX_COLOUR[f.severity] }}>
                  {f.id}
                  {label ? ` · ${label}` : ''}
                </span>
              </button>
            ))}
          {focused.box && (
            <div
              key={focused.nonce}
              className="photo-focus"
              style={{ left: `${focused.box.x * 100}%`, top: `${focused.box.y * 100}%`, width: `${focused.box.w * 100}%`, height: `${focused.box.h * 100}%` }}
            />
          )}
        </div>

        {loupe && cursor && inside && mediaUrl && (
          <div
            className="loupe"
            style={{
              left: cursor.x - loupeSize / 2,
              top: cursor.y - loupeSize / 2,
              width: loupeSize,
              height: loupeSize,
              backgroundImage: `url(${mediaUrl})`,
              backgroundSize: `${frame.width * loupeZoom}px ${frame.height * loupeZoom}px`,
              backgroundPosition: `${-(cursor.x - view.tx) * loupeZoom + loupeSize / 2}px ${-(cursor.y - view.ty) * loupeZoom + loupeSize / 2}px`,
              filter: lensFilter(lens, levels),
            }}
          />
        )}

        {enhanceMode && (
          <div className="enhance-layer passive">
            {selRect && selRect.w > 0 && (
              <div className="sel" style={{ left: selRect.x, top: selRect.y, width: selRect.w, height: selRect.h }}>
                <span className="mono">
                  {Math.round(selRect.w / scale)}×{Math.round(selRect.h / scale)} px
                </span>
              </div>
            )}
            {!selRect && <div className="enhance-hint">Drag a box around the detail you want to examine (Esc to cancel)</div>}
          </div>
        )}

        <div className="stage-tag">
          {running && (
            <span className="live">
              <i className="pulse-dot" /> ANALYSING
            </span>
          )}
          {overlay && <span>OVERLAY: {overlay.title.toUpperCase().slice(0, 48)}</span>}
          {lens !== 'none' && <span>LENS: {lensList.find((l) => l.key === lens)?.label.toUpperCase()}</span>}
          {gpuLens && (
            <span data-tip={original ? 'Computed on your GPU from the original file' : 'Computed on your GPU from the viewing copy (the original format cannot be decoded by the browser), so error levels reflect that copy'}>
              GPU · {gpuStatus.error ? `FAILED: ${gpuStatus.error.slice(0, 40)}` : gpuStatus.loading ? 'LOADING…' : original ? 'ORIGINAL FILE' : 'VIEWING COPY'}
            </span>
          )}
        </div>
        <div className="stage-tc mono">
          {Math.round(scale * 100)}%
          {inside && cursorImg ? (
            <span className="faint">
              {' '}
              · x {Math.round(cursorImg.x)} y {Math.round(cursorImg.y)}
            </span>
          ) : (
            <span className="faint"> · {iw}×{ih} px</span>
          )}
        </div>
        {!mediaUrl && <div className="stage-msg">Preparing a viewable copy…</div>}
        {hover && <div className="stage-msg top">{boxed.find((b) => b.f.id === hover)?.f.title}</div>}
      </div>

      <div className="controls photo-controls">
        <button className="ctl" onClick={() => zoomAt(1 / 1.5)} data-tip="Zoom out (−)" aria-label="Zoom out">
          <Icon name="minus" />
        </button>
        <button className="ctl" onClick={() => zoomAt(1.5)} data-tip="Zoom in (+)" aria-label="Zoom in">
          <Icon name="plus" />
        </button>
        <button className="ctl txt" onClick={reset} data-tip="Fit the whole picture (0)">
          Fit
        </button>
        <button className="ctl txt" onClick={oneToOne} data-tip="One screen pixel per picture pixel (1)">
          1:1
        </button>
        <span className="ctl-sep" />
        <select className="input ctl-select" value={lens} onChange={(e) => setLens(e.target.value as LensKey)} aria-label="Lens" data-tip={lensList.find((l) => l.key === lens)?.tip}>
          {gpu ? (
            <>
              <optgroup label="Picture">
                {lensList.filter((l) => !l.key.startsWith('gpu-')).map((l) => (
                  <option key={l.key} value={l.key}>
                    {l.label}
                  </option>
                ))}
              </optgroup>
              <optgroup label="Forensic (GPU)">
                {GPU_LENSES.map((l) => (
                  <option key={l.key} value={l.key}>
                    {l.label}
                  </option>
                ))}
              </optgroup>
            </>
          ) : (
            LENSES.map((l) => (
              <option key={l.key} value={l.key}>
                {l.label}
              </option>
            ))
          )}
        </select>
        {gpuLens?.mode === 'ela' && (
          <label className="lens-slider" data-tip="JPEG quality used for the re-save. Try the quality the file was likely saved at (often 85–95).">
            <span>Q {elaQuality}</span>
            <input type="range" min={50} max={98} step={1} value={elaQuality} onChange={(e) => setElaQuality(Number(e.target.value))} aria-label="Re-save quality" />
          </label>
        )}
        {gpuLens && (
          <label className="lens-slider" data-tip="Amplify the differences">
            <span>×{gain}</span>
            <input type="range" min={gpuLens.gain[0]} max={gpuLens.gain[1]} step={1} value={gain} onChange={(e) => setGains((g) => ({ ...g, [gpuLens.key]: Number(e.target.value) }))} aria-label="Gain" />
          </label>
        )}
        {lens === 'levels' && (
          <span className="levels" data-tip="Drag the two ends: everything darker than the left end goes black, everything brighter than the right end goes white">
            <input type="range" min={0} max={0.98} step={0.01} value={levels[0]} onChange={(e) => setLevels(([, hi]) => [Math.min(Number(e.target.value), hi - 0.02), hi])} aria-label="Black point" />
            <input type="range" min={0.02} max={1} step={0.01} value={levels[1]} onChange={(e) => setLevels(([lo]) => [lo, Math.max(Number(e.target.value), lo + 0.02)])} aria-label="White point" />
          </span>
        )}
        <span className="ctl-sep" />
        <select className="input ctl-select" value={overlayId} onChange={(e) => setOverlayId(e.target.value)} aria-label="Overlay" data-tip="Lay a forensic map over the picture">
          <option value="">No overlay</option>
          {overlays.map((a) => (
            <option key={a.id} value={a.id}>
              {a.title}
            </option>
          ))}
        </select>
        {overlay && (
          <>
            <input className="opacity" type="range" min={0} max={1} step={0.05} value={opacity} onChange={(e) => setOpacity(Number(e.target.value))} aria-label="Overlay opacity" data-tip="Overlay opacity" />
            <button className={`ctl txt ${blend === 'screen' ? 'on' : ''}`} onClick={() => setBlend((b) => (b === 'screen' ? 'normal' : 'screen'))} data-tip="Blend the overlay additively so the picture stays visible">
              Blend
            </button>
          </>
        )}
        <span className="grow" />
        <button className={`ctl ${showBoxes ? 'on' : ''}`} onClick={() => setShowBoxes((b) => !b)} data-tip={`${boxed.length} flagged regions (B)`} aria-label="Toggle regions">
          <Icon name="vector-square" /> <span className="n">{boxed.length}</span>
        </button>
        <button className={`ctl ${loupe ? 'on' : ''}`} onClick={() => setLoupe((l) => !l)} data-tip="Magnifier under the cursor (M)" aria-label="Magnifier">
          <Icon name="magnifying-glass" />
        </button>
        <button className={`btn btn-red sm ${enhanceMode ? 'on' : ''}`} onClick={() => setEnhanceMode((m) => !m)} disabled={!mediaUrl} data-tip="Enlarge a region at full resolution with eight forensic lenses (Z)">
          <Icon name="magnifying-glass-plus" /> Examine a detail
        </button>
      </div>

      {request && <EnhanceModal caseId={info.id} box={request} onClose={() => setRequest(null)} />}
    </div>
  )
}

/** The photo counterpart of "key moments": every finding that points at a region. */
export function RegionList({ findings, onOpenFinding }: { findings: Finding[]; onOpenFinding: (id: string) => void }) {
  const rf = useRegionFocus()
  const items = useMemo(() => rankFindings(findings).map((f) => ({ f, box: boxOf(f) })).filter((x) => x.box) as { f: Finding; box: Box }[], [findings])
  return (
    <div className="moments regions">
      <div className="moments-head">
        <span>Flagged regions</span>
        <span className="muted">{items.length ? 'Click a region to zoom the picture there' : 'No finding points at a particular part of the picture'}</span>
      </div>
      {items.length > 0 && (
        <ul className="region-list">
          {items.map(({ f, box }) => (
            <li key={f.id}>
              <button
                className={`region sev-${f.severity}`}
                onClick={() => rf.focus(box, f.title)}
                onDoubleClick={() => onOpenFinding(f.id)}
                data-tip={`${f.detail.slice(0, 160)}${f.detail.length > 160 ? '…' : ''}`}
              >
                <SevChip sev={f.severity} solid={false} />
                <span className="region-title">{f.title}</span>
                <span className="mono faint">
                  {Math.round(box.w * 100)}×{Math.round(box.h * 100)}% at {Math.round(box.x * 100)},{Math.round(box.y * 100)}
                </span>
                <span className="mono faint">{f.id}</span>
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}
