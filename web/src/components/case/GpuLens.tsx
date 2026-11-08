import { useEffect, useRef, useState } from 'react'
import { textureFromUrl, uniformBuffer, useFrameRequest, useGpuCanvas, type Gpu } from '../../lib/gpu'
import { LENS_COMPUTE, LENS_DRAW } from '../../lib/shaders/lens.wgsl'

export type GpuLensMode = 'ela' | 'noise' | 'contrast' | 'sobel' | 'ghost'

const MODE_ID: Record<GpuLensMode, number> = { ela: 0, noise: 1, contrast: 2, sobel: 3, ghost: 4 }

/** JPEG ghost: candidate save qualities 30, 35, ..., 65, then every quality 70..100 (as in the shader). */
export const GHOST_QUALITIES = [...Array.from({ length: 8 }, (_, k) => 30 + 5 * k), ...Array.from({ length: 31 }, (_, k) => 70 + k)]
const HIST = (GHOST_QUALITIES.length + 1) * 4

interface Pipelines {
  ela: GPUComputePipeline
  filters: GPUComputePipeline
  ghost: GPUComputePipeline
  ghostPaint: GPUComputePipeline
  hist: GPUBuffer
  draw: GPURenderPipeline
  params: GPUBuffer
  view: GPUBuffer
  linear: GPUSampler
  nearest: GPUSampler
}

const cache = new WeakMap<GPUDevice, Pipelines>()

function pipelines(gpu: Gpu): Pipelines {
  const hit = cache.get(gpu.device)
  if (hit) return hit
  const { device } = gpu
  const compute = device.createShaderModule({ code: LENS_COMPUTE, label: 'lens compute' })
  const draw = device.createShaderModule({ code: LENS_DRAW, label: 'lens draw' })
  const made: Pipelines = {
    ela: device.createComputePipeline({ layout: 'auto', compute: { module: compute, entryPoint: 'ela' }, label: 'ela' }),
    filters: device.createComputePipeline({ layout: 'auto', compute: { module: compute, entryPoint: 'filters' }, label: 'filters' }),
    ghost: device.createComputePipeline({ layout: 'auto', compute: { module: compute, entryPoint: 'ghost' }, label: 'ghost' }),
    ghostPaint: device.createComputePipeline({ layout: 'auto', compute: { module: compute, entryPoint: 'ghostPaint' }, label: 'ghost paint' }),
    hist: device.createBuffer({ size: HIST, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST }),
    draw: device.createRenderPipeline({
      layout: 'auto',
      vertex: { module: draw, entryPoint: 'vs' },
      fragment: { module: draw, entryPoint: 'fs', targets: [{ format: gpu.format }] },
      primitive: { topology: 'triangle-list' },
      label: 'lens draw',
    }),
    params: uniformBuffer(device, 16),
    view: uniformBuffer(device, 48),
    linear: device.createSampler({ magFilter: 'linear', minFilter: 'linear' }),
    nearest: device.createSampler({ magFilter: 'nearest', minFilter: 'nearest' }),
  }
  cache.set(device, made)
  return made
}

/**
 * A forensic map of the picture computed on the GPU at full resolution, drawn through the
 * viewer's pan/zoom. It covers the stage; finding boxes and overlays stay on top of it.
 */
export function GpuLens({
  gpu,
  src,
  mode,
  quality,
  gain,
  view,
  iw,
  ih,
  pixelated,
  onState,
  onGhost,
}: {
  gpu: Gpu
  src: string
  mode: GpuLensMode
  quality: number
  gain: number
  view: { tx: number; ty: number; scale: number }
  iw: number
  ih: number
  pixelated: boolean
  onState?: (s: { loading: boolean; error: string | null }) => void
  /** JPEG ghost only: blocks per estimated quality (GHOST_QUALITIES order, then "none"). */
  onGhost?: (counts: number[]) => void
}) {
  const { ref, ctx, size } = useGpuCanvas(gpu)
  const [source, setSource] = useState<GPUTexture | null>(null)
  const result = useRef<{ texture: GPUTexture; bind: GPUBindGroup } | null>(null)
  const [resultVersion, setResultVersion] = useState(0)
  const stateCb = useRef(onState)
  stateCb.current = onState
  const ghostCb = useRef(onGhost)
  ghostCb.current = onGhost

  // Upload the picture once per source URL.
  useEffect(() => {
    let live = true
    let made: GPUTexture | null = null
    stateCb.current?.({ loading: true, error: null })
    textureFromUrl(gpu.device, src)
      .then(({ texture }) => {
        if (!live) return texture.destroy()
        made = texture
        setSource(texture)
        stateCb.current?.({ loading: false, error: null })
      })
      .catch((e) => live && stateCb.current?.({ loading: false, error: String(e?.message ?? e) }))
    return () => {
      live = false
      made?.destroy()
      setSource(null)
    }
  }, [gpu, src])

  // Compute the forensic map whenever the picture, lens or a slider changes.
  useEffect(() => {
    if (!source) return
    const P = pipelines(gpu)
    const { device } = gpu
    const w = source.width
    const h = source.height
    const texture = device.createTexture({
      size: [w, h],
      format: 'rgba8unorm',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
      label: `lens ${mode}`,
    })
    device.queue.writeBuffer(P.params, 0, new Uint32Array([MODE_ID[mode]]))
    device.queue.writeBuffer(P.params, 4, new Float32Array([quality, gain, 0]))
    const enc = device.createCommandEncoder({ label: 'lens compute' })
    const pass = enc.beginComputePass()
    let readback: GPUBuffer | null = null
    let blocks: GPUBuffer | null = null
    if (mode === 'ghost') {
      // Pass 1: estimate each block's save quality and count them; pass 2: paint against the dominant one.
      const bx = Math.ceil(w / 8)
      const by = Math.ceil(h / 8)
      blocks = device.createBuffer({ size: bx * by * 4, usage: GPUBufferUsage.STORAGE })
      device.queue.writeBuffer(P.hist, 0, new Uint32Array(HIST / 4))
      pass.setPipeline(P.ghost)
      pass.setBindGroup(0, device.createBindGroup({
        layout: P.ghost.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: source.createView() },
          { binding: 3, resource: { buffer: P.hist } },
          { binding: 4, resource: { buffer: blocks } },
        ],
      }))
      pass.dispatchWorkgroups(bx, by)
      pass.setPipeline(P.ghostPaint)
      pass.setBindGroup(0, device.createBindGroup({
        layout: P.ghostPaint.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: source.createView() },
          { binding: 1, resource: texture.createView() },
          { binding: 3, resource: { buffer: P.hist } },
          { binding: 4, resource: { buffer: blocks } },
        ],
      }))
      pass.dispatchWorkgroups(bx, by)
    } else {
      const pipeline = mode === 'ela' ? P.ela : P.filters
      pass.setPipeline(pipeline)
      pass.setBindGroup(0, device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: source.createView() },
          { binding: 1, resource: texture.createView() },
          { binding: 2, resource: { buffer: P.params } },
        ],
      }))
      pass.dispatchWorkgroups(Math.ceil(w / 8), Math.ceil(h / 8))
    }
    pass.end()
    if (mode === 'ghost') {
      readback = device.createBuffer({ size: HIST, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST })
      enc.copyBufferToBuffer(P.hist, 0, readback, 0, HIST)
    }
    device.queue.submit([enc.finish()])
    blocks?.destroy()
    if (readback) {
      const rb = readback
      rb.mapAsync(GPUMapMode.READ)
        .then(() => {
          ghostCb.current?.(Array.from(new Uint32Array(rb.getMappedRange().slice(0))))
          rb.unmap()
        })
        .catch(() => {})
        .finally(() => rb.destroy())
    }

    const bind = device.createBindGroup({
      layout: P.draw.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: texture.createView() },
        { binding: 1, resource: P.linear },
        { binding: 2, resource: P.nearest },
        { binding: 3, resource: { buffer: P.view } },
      ],
    })
    const previous = result.current
    result.current = { texture, bind }
    setResultVersion((v) => v + 1)
    previous?.texture.destroy()
  }, [gpu, source, mode, quality, gain])

  useEffect(
    () => () => {
      result.current?.texture.destroy()
      result.current = null
    },
    [],
  )

  // Draw (cheap): only the view transform changes while panning and zooming.
  const draw = () => {
    const r = result.current
    if (!ctx || !r || !size.width) return
    const P = pipelines(gpu)
    const { device } = gpu
    device.queue.writeBuffer(
      P.view,
      0,
      new Float32Array([size.width, size.height, view.tx, view.ty, iw, ih, view.scale, size.dpr, pixelated ? 1 : 0, 0]),
    )
    const enc = device.createCommandEncoder({ label: 'lens draw' })
    const pass = enc.beginRenderPass({
      colorAttachments: [{ view: ctx.getCurrentTexture().createView(), loadOp: 'clear', storeOp: 'store', clearValue: { r: 0, g: 0, b: 0, a: 0 } }],
    })
    pass.setPipeline(P.draw)
    pass.setBindGroup(0, r.bind)
    pass.draw(3)
    pass.end()
    device.queue.submit([enc.finish()])
  }
  const requestDraw = useFrameRequest(draw)
  useEffect(requestDraw, [ctx, size, view.tx, view.ty, view.scale, iw, ih, pixelated, resultVersion]) // eslint-disable-line react-hooks/exhaustive-deps

  return <canvas ref={ref} className="gpu-lens" aria-hidden />
}
