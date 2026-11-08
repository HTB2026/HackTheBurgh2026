import { useEffect, useMemo, useRef, useState } from 'react'
import { textureFromSource, uniformBuffer, useGpuCanvas, type Gpu } from '../../lib/gpu'
import { lookAt, multiply, perspective, type Vec3 } from '../../lib/mat4'
import { GLOBE, ARCS } from '../../lib/shaders/globe.wgsl'
import { radiusText, type MapPoint } from './MapView'

const EARTH_M = 6_371_000
const RAD = Math.PI / 180
const MAX_POINTS = 64

/** 0 GPS, 1 AI answer, 2 lead, 3 other measured (matches kindColour in the shader). */
const kindId = (k?: string) => (k === 'gps' ? 0 : k === 'ai' || k === 'inferred' ? 1 : k === 'ai-candidate' ? 2 : 3)

/** The place the globe flies to first: the most precise AI answer, else GPS, else the first point. */
export function globeTarget(points: MapPoint[]): number {
  const ai = points.map((p, i) => ({ p, i })).filter(({ p }) => kindId(p.kind) === 1)
  if (ai.length) return ai.sort((a, b) => (a.p.accuracy_m ?? Infinity) - (b.p.accuracy_m ?? Infinity))[0].i
  const gps = points.findIndex((p) => kindId(p.kind) === 0)
  return gps >= 0 ? gps : 0
}

interface Res {
  globe: GPURenderPipeline
  arcs: GPURenderPipeline
  uniforms: GPUBuffer
  sampler: GPUSampler
}
const cache = new WeakMap<GPUDevice, Res>()
function resources(gpu: Gpu): Res {
  const hit = cache.get(gpu.device)
  if (hit) return hit
  const { device, format } = gpu
  const depthStencil: GPUDepthStencilState = { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less' }
  const g = device.createShaderModule({ code: GLOBE, label: 'globe' })
  const a = device.createShaderModule({ code: ARCS, label: 'arcs' })
  const made = {
    globe: device.createRenderPipeline({
      layout: 'auto',
      vertex: { module: g, entryPoint: 'vs' },
      fragment: { module: g, entryPoint: 'fs', targets: [{ format }] },
      primitive: { topology: 'triangle-list' },
      depthStencil,
      label: 'globe',
    }),
    arcs: device.createRenderPipeline({
      layout: 'auto',
      vertex: { module: a, entryPoint: 'vs' },
      fragment: { module: a, entryPoint: 'fs', targets: [{ format }] },
      primitive: { topology: 'line-list' },
      depthStencil: { ...depthStencil, depthWriteEnabled: false },
      label: 'arcs',
    }),
    uniforms: uniformBuffer(device, 112),
    sampler: device.createSampler({ magFilter: 'linear', minFilter: 'linear', addressModeU: 'repeat' }),
  }
  cache.set(device, made)
  return made
}

/** World imagery: 4x4 zoom-2 tiles stitched into one Web Mercator image (Esri, else OpenStreetMap). */
async function earthImage(): Promise<OffscreenCanvas | null> {
  const sources = [
    (z: number, x: number, y: number) => `https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/${z}/${y}/${x}`,
    (z: number, x: number, y: number) => `https://tile.openstreetmap.org/${z}/${x}/${y}.png`,
  ]
  for (const url of sources) {
    try {
      const canvas = new OffscreenCanvas(1024, 1024)
      const g = canvas.getContext('2d')!
      await Promise.all(
        Array.from({ length: 16 }, async (_, i) => {
          const x = i % 4
          const y = Math.floor(i / 4)
          const r = await fetch(url(2, x, y), { mode: 'cors' })
          if (!r.ok) throw new Error(`tile ${r.status}`)
          g.drawImage(await createImageBitmap(await r.blob()), x * 256, y * 256)
        }),
      )
      return canvas
    } catch {
      /* try the next source */
    }
  }
  return null
}

let earthPromise: Promise<OffscreenCanvas | null> | null = null

interface Cam {
  lat: number
  lon: number
  dist: number
}

const slerpCam = (a: Cam, b: Cam, t: number): Cam => {
  // Shortest way round in longitude; zoom out mid-flight for long hops.
  let dLon = b.lon - a.lon
  if (dLon > Math.PI) dLon -= 2 * Math.PI
  if (dLon < -Math.PI) dLon += 2 * Math.PI
  const e = t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2
  const hop = Math.min(1, Math.hypot(b.lat - a.lat, dLon) / 1.2)
  return { lat: a.lat + (b.lat - a.lat) * e, lon: a.lon + dLon * e, dist: a.dist + (b.dist - a.dist) * e + hop * 1.2 * Math.sin(Math.PI * t) }
}

const camFor = (p: MapPoint): Cam => {
  const r = Math.min(Math.PI, (p.accuracy_m ?? 50_000) / EARTH_M)
  // The imagery is a world-scale texture: stop at a regional view, never street level.
  return { lat: p.lat * RAD, lon: p.lon * RAD, dist: 1 + Math.min(2.6, Math.max(0.42, r * 3.2)) }
}

/**
 * A 3D globe of the geolocation answer: candidate places, the verified answer and GPS as pins,
 * uncertainty as caps on the surface, arcs from each lead to the answer.
 */
export function Globe({ gpu, points, height = 420, focus }: { gpu: Gpu; points: MapPoint[]; height?: number; focus: { index: number; nonce: number } }) {
  const { ref, ctx, size } = useGpuCanvas(gpu)
  const pts = useMemo(() => points.filter((p) => isFinite(p?.lat) && isFinite(p?.lon)).slice(0, MAX_POINTS), [points])
  const [earth, setEarth] = useState<GPUTexture | null>(null)
  const [noImagery, setNoImagery] = useState(false)
  const cam = useRef<Cam>({ lat: 20 * RAD, lon: 0, dist: 3.4 })
  const flight = useRef<{ from: Cam; to: Cam; start: number; ms: number } | null>(null)
  const drag = useRef<{ x: number; y: number; cam: Cam } | null>(null)
  const depth = useRef<GPUTexture | null>(null)
  const raf = useRef(0)
  const bind = useRef<{ globe: GPUBindGroup; arcs: GPUBindGroup; arcCount: number } | null>(null)

  // Imagery, shared across globes.
  useEffect(() => {
    let live = true
    earthPromise ??= earthImage()
    earthPromise.then((img) => {
      if (!live) return
      if (!img) return setNoImagery(true)
      setEarth(textureFromSource(gpu.device, img).texture)
    })
    return () => {
      live = false
    }
  }, [gpu])

  // Point and arc buffers, rebuilt when the points or imagery change.
  useEffect(() => {
    const R = resources(gpu)
    const { device } = gpu
    const P = new Float32Array(Math.max(1, pts.length) * 4)
    pts.forEach((p, i) => P.set([p.lat * RAD, p.lon * RAD, Math.min(Math.PI, (p.accuracy_m ?? 0) / EARTH_M), kindId(p.kind)], i * 4))
    const target = pts[globeTarget(pts)]
    const arcs: number[] = []
    if (target) {
      for (const p of pts) {
        // No arc for the target itself, or for a guess too vague to have a real centre.
        if (p === target || (p.accuracy_m ?? 0) > 10_000_000) continue
        const k = kindId(p.kind)
        // Leads arc to the answer in grey; GPS joins the answer in yellow (as on the 2D map).
        const colour = k === 0 ? [1, 0.82, 0.25, 0.22] : k === 2 ? [0.7, 0.7, 0.75, 0.16] : [0.78, 0.36, 1, 0.16]
        arcs.push(p.lat * RAD, p.lon * RAD, target.lat * RAD, target.lon * RAD, ...colour)
      }
    }
    const A = new Float32Array(Math.max(8, arcs.length))
    A.set(arcs)
    const pBuf = device.createBuffer({ size: P.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST })
    const aBuf = device.createBuffer({ size: A.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST })
    device.queue.writeBuffer(pBuf, 0, P)
    device.queue.writeBuffer(aBuf, 0, A)
    const placeholder = earth ?? device.createTexture({ size: [1, 1], format: 'rgba8unorm', usage: GPUTextureUsage.TEXTURE_BINDING })
    bind.current = {
      globe: device.createBindGroup({
        layout: R.globe.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: R.uniforms } },
          { binding: 1, resource: { buffer: pBuf } },
          { binding: 2, resource: placeholder.createView() },
          { binding: 3, resource: R.sampler },
        ],
      }),
      arcs: device.createBindGroup({
        layout: R.arcs.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: R.uniforms } },
          { binding: 2, resource: { buffer: aBuf } },
        ],
      }),
      arcCount: arcs.length / 8,
    }
    return () => {
      pBuf.destroy()
      aBuf.destroy()
      if (!earth) placeholder.destroy()
    }
  }, [gpu, pts, earth])

  // Fly to the answer on open, and to a row when the table asks.
  useEffect(() => {
    const p = pts[focus.index >= 0 ? focus.index : globeTarget(pts)]
    if (!p) return
    flight.current = { from: { ...cam.current }, to: camFor(p), start: performance.now(), ms: focus.nonce === 0 ? 2600 : 1600 }
  }, [focus.index, focus.nonce, pts])

  // Render loop (the pins pulse and arcs flow); paused when the tab is hidden.
  useEffect(() => {
    if (!ctx || !size.width) return
    const R = resources(gpu)
    const { device } = gpu
    const frame = (now: number) => {
      raf.current = requestAnimationFrame(frame)
      if (document.hidden || !bind.current) return
      const f = flight.current
      if (f) {
        const t = Math.min(1, (now - f.start) / f.ms)
        cam.current = slerpCam(f.from, f.to, t)
        if (t >= 1) flight.current = null
      }
      const c = cam.current
      const eye: Vec3 = [Math.cos(c.lat) * Math.sin(c.lon) * c.dist, Math.sin(c.lat) * c.dist, Math.cos(c.lat) * Math.cos(c.lon) * c.dist]
      const up: Vec3 = Math.abs(c.lat) > 1.5 ? [0, 0, -1] : [0, 1, 0]
      const near = Math.max(0.001, (c.dist - 1) * 0.3)
      const vp = multiply(perspective(0.7, size.width / size.height, near, c.dist + 2), lookAt(eye, [0, 0, 0], up))
      const u = new Float32Array(28)
      u.set(vp, 0)
      u.set([...eye, c.dist], 16)
      u.set([pts.length, earth ? 1 : 0, now / 1000, 0], 20)
      device.queue.writeBuffer(R.uniforms, 0, u)
      if (!depth.current || depth.current.width !== size.width || depth.current.height !== size.height) {
        depth.current?.destroy()
        depth.current = device.createTexture({ size: [size.width, size.height], format: 'depth24plus', usage: GPUTextureUsage.RENDER_ATTACHMENT })
      }
      const enc = device.createCommandEncoder({ label: 'globe' })
      const pass = enc.beginRenderPass({
        colorAttachments: [{ view: ctx.getCurrentTexture().createView(), loadOp: 'clear', storeOp: 'store', clearValue: { r: 0.03, g: 0.035, b: 0.05, a: 1 } }],
        depthStencilAttachment: { view: depth.current.createView(), depthLoadOp: 'clear', depthClearValue: 1, depthStoreOp: 'store' },
      })
      pass.setPipeline(R.globe)
      pass.setBindGroup(0, bind.current.globe)
      pass.draw(192 * 96 * 6)
      if (bind.current.arcCount) {
        pass.setPipeline(R.arcs)
        pass.setBindGroup(0, bind.current.arcs)
        pass.draw(64 * 2, bind.current.arcCount)
      }
      pass.end()
      device.queue.submit([enc.finish()])
    }
    raf.current = requestAnimationFrame(frame)
    return () => cancelAnimationFrame(raf.current)
  }, [gpu, ctx, size, pts.length, earth])

  useEffect(() => () => depth.current?.destroy(), [])

  // Wheel zooms the globe, not the page.
  useEffect(() => {
    const el = ref.current
    if (!el) return
    const on = (e: WheelEvent) => {
      e.preventDefault()
      flight.current = null
      const px = e.deltaY * (e.deltaMode === 1 ? 16 : e.deltaMode === 2 ? 400 : 1)
      const c = cam.current
      cam.current = { ...c, dist: 1 + Math.min(5, Math.max(0.12, (c.dist - 1) * Math.exp(px * 0.002))) }
    }
    el.addEventListener('wheel', on, { passive: false })
    return () => el.removeEventListener('wheel', on)
  }, [ref])

  const target = pts[globeTarget(pts)]
  return (
    <div className="globe" style={{ height }}>
      <canvas
        ref={ref}
        className="globe-canvas"
        onPointerDown={(e) => {
          e.currentTarget.setPointerCapture(e.pointerId)
          flight.current = null
          drag.current = { x: e.clientX, y: e.clientY, cam: { ...cam.current } }
        }}
        onPointerMove={(e) => {
          const g = drag.current
          if (!g) return
          // Turn by an angle that matches the zoom, so the ground follows the pointer.
          const k = (g.cam.dist - 1) * 0.0026 + 0.0004
          cam.current = { ...g.cam, lon: g.cam.lon - (e.clientX - g.x) * k, lat: Math.max(-1.45, Math.min(1.45, g.cam.lat + (e.clientY - g.y) * k)) }
        }}
        onPointerUp={() => (drag.current = null)}
        onDoubleClick={() => target && (flight.current = { from: { ...cam.current }, to: camFor(target), start: performance.now(), ms: 1400 })}
      />
      <div className="globe-legend">
        <span>
          <i style={{ background: '#c95cff' }} /> answer {target ? radiusText(target.accuracy_m) : ''}
        </span>
        <span>
          <i style={{ background: '#bfbfbf' }} /> other places considered
        </span>
        <span>
          <i style={{ background: '#ff4040' }} /> GPS in the file
        </span>
        <span className="faint">drag to turn · scroll to zoom · double-click to return · click a row below to fly there</span>
        {noImagery && <span className="faint">(satellite imagery unavailable)</span>}
      </div>
    </div>
  )
}
