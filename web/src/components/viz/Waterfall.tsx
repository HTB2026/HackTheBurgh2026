import { useEffect, useRef, useState } from 'react'
import type { Artifact } from '../../lib/types'
import { textureFromUrl, uniformBuffer, useFrameRequest, useGpuCanvas, type Gpu } from '../../lib/gpu'
import { add, invert, lookAt, multiply, perspective, scale, sub, transformPoint, type Vec3 } from '../../lib/mat4'
import { usePlayback, usePlayhead } from '../../lib/playback'
import { WATERFALL } from '../../lib/shaders/waterfall.wgsl'
import { num } from '../../lib/format'

const GRID: [number, number] = [512, 256]
const WIDTH = 4
const DEPTH = 2.4
const AMP = 0.7
const HOME = { yaw: -0.55, pitch: 0.62, dist: 5.2 }

interface Res {
  pipeline: GPURenderPipeline
  uniforms: GPUBuffer
  sampler: GPUSampler
}
const cache = new WeakMap<GPUDevice, Res>()
function resources(gpu: Gpu): Res {
  const hit = cache.get(gpu.device)
  if (hit) return hit
  const module = gpu.device.createShaderModule({ code: WATERFALL, label: 'waterfall' })
  const made = {
    pipeline: gpu.device.createRenderPipeline({
      layout: 'auto',
      vertex: { module, entryPoint: 'vs' },
      fragment: { module, entryPoint: 'fs', targets: [{ format: gpu.format }] },
      primitive: { topology: 'triangle-list' },
      depthStencil: { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less' },
      label: 'waterfall',
    }),
    uniforms: uniformBuffer(gpu.device, 128),
    sampler: gpu.device.createSampler({ magFilter: 'linear', minFilter: 'linear' }),
  }
  cache.set(gpu.device, made)
  return made
}

/** The spectrogram as a lit 3D terrain: loudness is height. Drag to orbit, wheel to zoom. */
export function Waterfall({ gpu, a, height }: { gpu: Gpu; a: Artifact; height: number }) {
  const { ref, ctx, size } = useGpuCanvas(gpu)
  const pb = usePlayback()
  const playhead = usePlayhead()
  const d = a.data ?? {}
  const t0: number = d.t0 ?? 0
  const t1: number = d.t1 ?? (pb.duration || 1)
  const f0: number = d.f0 ?? 0
  const f1: number = d.f1 ?? 1
  const [cam, setCam] = useState(HOME)
  const [hover, setHover] = useState<{ u: number; v: number; x: number; y: number } | null>(null)
  const [tex, setTex] = useState<{ bind: GPUBindGroup; texture: GPUTexture } | null>(null)
  const [error, setError] = useState<string | null>(null)
  const depth = useRef<GPUTexture | null>(null)
  const drag = useRef<{ x: number; y: number; yaw: number; pitch: number; moved: boolean } | null>(null)
  const viewProj = useRef<Float32Array | null>(null)

  useEffect(() => {
    if (!a.url) return
    let live = true
    let made: GPUTexture | null = null
    textureFromUrl(gpu.device, a.url)
      .then(({ texture }) => {
        if (!live) return texture.destroy()
        made = texture
        const R = resources(gpu)
        const bind = gpu.device.createBindGroup({
          layout: R.pipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: texture.createView() },
            { binding: 1, resource: R.sampler },
            { binding: 2, resource: { buffer: R.uniforms } },
          ],
        })
        setTex({ bind, texture })
      })
      .catch((e) => live && setError(String(e?.message ?? e)))
    return () => {
      live = false
      made?.destroy()
      setTex(null)
    }
  }, [gpu, a.url])

  const eye = (): Vec3 => {
    const cp = Math.cos(cam.pitch)
    return [Math.sin(cam.yaw) * cp * cam.dist, Math.sin(cam.pitch) * cam.dist, Math.cos(cam.yaw) * cp * cam.dist]
  }

  const draw = () => {
    if (!ctx || !tex || !size.width) return
    const R = resources(gpu)
    const { device } = gpu
    if (!depth.current || depth.current.width !== size.width || depth.current.height !== size.height) {
      depth.current?.destroy()
      depth.current = device.createTexture({ size: [size.width, size.height], format: 'depth24plus', usage: GPUTextureUsage.RENDER_ATTACHMENT })
    }
    const vp = multiply(perspective(0.75, size.width / size.height, 0.05, 50), lookAt(eye(), [0, AMP * 0.25, 0]))
    viewProj.current = vp
    const playU = t1 > t0 && playhead >= t0 && playhead <= t1 ? (playhead - t0) / (t1 - t0) : -1
    const buf = new Float32Array(32)
    buf.set(vp, 0)
    buf.set([GRID[0], GRID[1], WIDTH, DEPTH, AMP, playU, hover ? hover.u : -1, hover ? hover.v : -1], 16)
    buf.set([-0.4, 0.9, 0.5, 0], 24)
    device.queue.writeBuffer(R.uniforms, 0, buf)
    const enc = device.createCommandEncoder({ label: 'waterfall' })
    const pass = enc.beginRenderPass({
      colorAttachments: [{ view: ctx.getCurrentTexture().createView(), loadOp: 'clear', storeOp: 'store', clearValue: { r: 0.055, g: 0.055, b: 0.055, a: 1 } }],
      depthStencilAttachment: { view: depth.current.createView(), depthLoadOp: 'clear', depthClearValue: 1, depthStoreOp: 'store' },
    })
    pass.setPipeline(R.pipeline)
    pass.setBindGroup(0, tex.bind)
    pass.draw((GRID[0] - 1) * (GRID[1] - 1) * 6)
    pass.end()
    device.queue.submit([enc.finish()])
  }
  const requestDraw = useFrameRequest(draw)
  useEffect(requestDraw, [ctx, tex, size, cam, playhead, hover]) // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => () => depth.current?.destroy(), [])

  // Wheel zooms the terrain, not the page (React's onWheel is passive).
  useEffect(() => {
    const el = ref.current
    if (!el) return
    const on = (e: WheelEvent) => {
      e.preventDefault()
      const px = e.deltaY * (e.deltaMode === 1 ? 16 : e.deltaMode === 2 ? 400 : 1)
      setCam((c) => ({ ...c, dist: Math.min(14, Math.max(1.6, c.dist * Math.exp(px * 0.0015))) }))
    }
    el.addEventListener('wheel', on, { passive: false })
    return () => el.removeEventListener('wheel', on)
  }, [ref])

  /** Where the pointer ray meets the floor (y = 0), as spectrogram coordinates. */
  const pick = (clientX: number, clientY: number) => {
    const el = ref.current
    const vp = viewProj.current
    if (!el || !vp) return null
    const r = el.getBoundingClientRect()
    const nx = ((clientX - r.left) / r.width) * 2 - 1
    const ny = 1 - ((clientY - r.top) / r.height) * 2
    const inv = invert(vp)
    const near = transformPoint(inv, [nx, ny, 0])
    const far = transformPoint(inv, [nx, ny, 1])
    const dir = sub(far, near)
    if (Math.abs(dir[1]) < 1e-6) return null
    const hit = add(near, scale(dir, -near[1] / dir[1]))
    const u = hit[0] / WIDTH + 0.5
    const v = hit[2] / DEPTH + 0.5
    if (u < 0 || u > 1 || v < 0 || v > 1) return null
    return { u, v, x: clientX - r.left, y: clientY - r.top }
  }

  const at = hover && { t: t0 + hover.u * (t1 - t0), f: f0 + (1 - hover.v) * (f1 - f0) }

  if (error) return <div className="chart-foot">3D view unavailable: {error}</div>
  return (
    <div className="waterfall" style={{ height }}>
      <canvas
        ref={ref}
        className="waterfall-canvas"
        onPointerDown={(e) => {
          e.currentTarget.setPointerCapture(e.pointerId)
          drag.current = { x: e.clientX, y: e.clientY, yaw: cam.yaw, pitch: cam.pitch, moved: false }
        }}
        onPointerMove={(e) => {
          const g = drag.current
          if (g) {
            const dx = e.clientX - g.x
            const dy = e.clientY - g.y
            if (Math.abs(dx) + Math.abs(dy) > 3) g.moved = true
            setCam((c) => ({ ...c, yaw: g.yaw - dx * 0.008, pitch: Math.min(1.45, Math.max(0.08, g.pitch + dy * 0.006)) }))
            setHover(null)
          } else setHover(pick(e.clientX, e.clientY))
        }}
        onPointerUp={(e) => {
          const g = drag.current
          drag.current = null
          if (g && !g.moved) {
            const p = pick(e.clientX, e.clientY)
            if (p) pb.seek(t0 + p.u * (t1 - t0))
          }
        }}
        onPointerLeave={() => setHover(null)}
        onDoubleClick={() => setCam(HOME)}
      />
      <div className="waterfall-axes">
        <span>time →</span>
        <span>pitch: low at the front, {num(f1)} Hz at the back</span>
        <span className="faint">drag to turn · scroll to zoom · double-click to reset</span>
      </div>
      {hover && at && (
        <div className="tooltip" style={{ left: Math.min(hover.x + 12, (ref.current?.clientWidth ?? 400) - 170), top: hover.y }}>
          <div>
            <b>{at.t.toFixed(2)} s</b> into the video
          </div>
          <div>
            pitch <b>{num(at.f)} Hz</b>
          </div>
          <div className="faint">Click to jump the video here</div>
        </div>
      )}
    </div>
  )
}
