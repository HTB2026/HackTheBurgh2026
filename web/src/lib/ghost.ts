/**
 * JPEG ghost on the GPU: for every 8x8 block, the JPEG quality it was last saved at (see the
 * `ghost` / `ghostPaint` passes in shaders/lens.wgsl.ts). Used by the "JPEG ghost" lens and by
 * the automatic browser checks, which turn odd-quality blocks into regions.
 */
import { textureFromUrl, type Gpu } from './gpu'
import { LENS_COMPUTE } from './shaders/lens.wgsl'

/** Candidate save qualities 30, 35, ..., 65, then every quality 70..100 (as in the shader). */
export const GHOST_QUALITIES = [...Array.from({ length: 8 }, (_, k) => 30 + 5 * k), ...Array.from({ length: 31 }, (_, k) => 70 + k)]
const NONE = GHOST_QUALITIES.length
const HIST = (NONE + 1) * 4

export interface GhostRegion {
  x: number // fractions of the picture
  y: number
  w: number
  h: number
  quality: number // the most common estimate inside
  blocks: number
}

export interface GhostResult {
  counts: number[] // blocks per GHOST_QUALITIES entry, then "no clear answer"
  dominant: number | null // the picture's main save quality
  odd: number // textured blocks pointing to a clearly different quality
  regions: GhostRegion[] // connected areas of such blocks (6+ blocks)
}

const pipelines = new WeakMap<GPUDevice, { ghost: GPUComputePipeline; paint: GPUComputePipeline }>()

/**
 * Estimate every block's save quality. With `paintTarget`, also paint the colour map into it
 * (same size as `source`, rgba8unorm storage).
 */
export async function runGhost(gpu: Gpu, source: GPUTexture, paintTarget: GPUTexture | null = null): Promise<GhostResult> {
  const { device } = gpu
  let P = pipelines.get(device)
  if (!P) {
    const module = device.createShaderModule({ code: LENS_COMPUTE, label: 'ghost' })
    P = {
      ghost: device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'ghost' } }),
      paint: device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'ghostPaint' } }),
    }
    pipelines.set(device, P)
  }
  const w = source.width
  const h = source.height
  const bx = Math.ceil(w / 8)
  const by = Math.ceil(h / 8)
  const hist = device.createBuffer({ size: HIST, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST })
  const blocks = device.createBuffer({ size: bx * by * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC })
  const histRead = device.createBuffer({ size: HIST, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST })
  const blocksRead = device.createBuffer({ size: bx * by * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST })
  device.queue.writeBuffer(hist, 0, new Uint32Array(HIST / 4))

  const enc = device.createCommandEncoder({ label: 'ghost' })
  const pass = enc.beginComputePass()
  pass.setPipeline(P.ghost)
  pass.setBindGroup(0, device.createBindGroup({
    layout: P.ghost.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: source.createView() },
      { binding: 3, resource: { buffer: hist } },
      { binding: 4, resource: { buffer: blocks } },
    ],
  }))
  pass.dispatchWorkgroups(bx, by)
  if (paintTarget) {
    pass.setPipeline(P.paint)
    pass.setBindGroup(0, device.createBindGroup({
      layout: P.paint.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: source.createView() },
        { binding: 1, resource: paintTarget.createView() },
        { binding: 3, resource: { buffer: hist } },
        { binding: 4, resource: { buffer: blocks } },
      ],
    }))
    pass.dispatchWorkgroups(bx, by)
  }
  pass.end()
  enc.copyBufferToBuffer(hist, 0, histRead, 0, HIST)
  enc.copyBufferToBuffer(blocks, 0, blocksRead, 0, bx * by * 4)
  device.queue.submit([enc.finish()])
  try {
    await Promise.all([histRead.mapAsync(GPUMapMode.READ), blocksRead.mapAsync(GPUMapMode.READ)])
    const counts = Array.from(new Uint32Array(histRead.getMappedRange().slice(0)))
    const est = new Uint32Array(blocksRead.getMappedRange().slice(0))
    histRead.unmap()
    blocksRead.unmap()
    return summarise(counts, est, bx, by)
  } finally {
    for (const b of [hist, blocks, histRead, blocksRead]) b.destroy()
  }
}

/** Load a picture and run the ghost on it (no painting). */
export async function ghostFromUrl(gpu: Gpu, url: string): Promise<GhostResult> {
  const { texture } = await textureFromUrl(gpu.device, url)
  try {
    return await runGhost(gpu, texture)
  } finally {
    texture.destroy()
  }
}

function summarise(counts: number[], est: Uint32Array, bx: number, by: number): GhostResult {
  const known = counts.slice(0, NONE)
  const total = known.reduce((a, b) => a + b, 0)
  if (!total) return { counts, dominant: null, odd: 0, regions: [] }
  const top = known.indexOf(Math.max(...known))
  const dominant = GHOST_QUALITIES[top]
  const isOdd = (k: number) => k < NONE && Math.abs(GHOST_QUALITIES[k] - dominant) > 1
  let odd = 0
  for (let i = 0; i < est.length; i++) if (isOdd(est[i])) odd++

  // Connected areas of odd blocks (4-neighbours); small specks are noise.
  const seen = new Uint8Array(est.length)
  const regions: GhostRegion[] = []
  for (let start = 0; start < est.length; start++) {
    if (seen[start] || !isOdd(est[start])) continue
    const stack = [start]
    seen[start] = 1
    let x0 = bx, y0 = by, x1 = 0, y1 = 0, n = 0
    const q = new Map<number, number>()
    while (stack.length) {
      const i = stack.pop()!
      const x = i % bx
      const y = Math.floor(i / bx)
      n++
      x0 = Math.min(x0, x); y0 = Math.min(y0, y); x1 = Math.max(x1, x); y1 = Math.max(y1, y)
      q.set(est[i], (q.get(est[i]) ?? 0) + 1)
      for (const [dx, dy] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
        const nx = x + dx
        const ny = y + dy
        if (nx < 0 || ny < 0 || nx >= bx || ny >= by) continue
        const j = ny * bx + nx
        if (!seen[j] && isOdd(est[j])) {
          seen[j] = 1
          stack.push(j)
        }
      }
    }
    if (n < 6) continue
    const mode = [...q.entries()].sort((a, b) => b[1] - a[1])[0][0]
    regions.push({ x: x0 / bx, y: y0 / by, w: (x1 - x0 + 1) / bx, h: (y1 - y0 + 1) / by, quality: GHOST_QUALITIES[mode], blocks: n })
  }
  regions.sort((a, b) => b.blocks - a.blocks)
  return { counts, dominant, odd, regions: regions.slice(0, 8) }
}
