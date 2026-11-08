/**
 * Multi-frame "examine a detail" on the GPU: crop a region from N frames around a moment,
 * enlarge it, align each frame to the middle one (sub-pixel, by trying every shift), average.
 * Averaging aligned frames cancels sensor and compression noise, so text and plates read better.
 */
import { copyVideoFrame, type Gpu } from './gpu'
import { STACK } from './shaders/stack.wgsl'

export interface Region {
  x: number
  y: number
  w: number
  h: number
}

export interface StackResult {
  texture: GPUTexture // the stacked, enlarged region
  single: GPUTexture // the middle frame alone, enlarged the same way (for comparison)
  frames: number // frames actually combined
  skipped: number // frames left out because the region itself changed (motion)
  shifts: [number, number][] // per combined frame, in source pixels
}

const RADIUS = 8

/** A hidden video that can be stepped to exact times. */
export class FrameSource {
  video: HTMLVideoElement
  ready: Promise<void>
  constructor(url: string) {
    const v = document.createElement('video')
    v.muted = true
    v.playsInline = true
    v.preload = 'auto'
    v.src = url
    this.video = v
    this.ready = new Promise((resolve, reject) => {
      v.onloadeddata = () => resolve()
      v.onerror = () => reject(new Error('the video could not be decoded'))
    })
  }
  /** Seek and wait until that frame is decoded and presentable. */
  async at(t: number) {
    await this.ready
    const v = this.video
    const target = Math.max(0, Math.min(t, (v.duration || t) - 0.001))
    // Already there: no seek happens, so no new frame (and no callback) would come.
    if (Math.abs(v.currentTime - target) < 1e-4 && v.readyState >= 2) return
    await new Promise<void>((resolve) => {
      let finished = false
      const finish = () => {
        if (!finished) {
          finished = true
          resolve()
        }
      }
      const done = () => {
        v.removeEventListener('seeked', done)
        // requestVideoFrameCallback guarantees the new frame is the one we copy; the timeout
        // covers seeks that land on the frame already shown (no new frame is presented).
        if ('requestVideoFrameCallback' in v) v.requestVideoFrameCallback(finish)
        setTimeout(finish, 250)
      }
      v.addEventListener('seeked', done)
      v.currentTime = target
    })
  }
  dispose() {
    this.video.removeAttribute('src')
    this.video.load()
  }
}

export async function stackFrames(gpu: Gpu, src: FrameSource, region: Region, centre: number, fps: number, count: number, sharpen: number): Promise<StackResult> {
  const { device } = gpu
  await src.ready
  const vw = src.video.videoWidth
  const vh = src.video.videoHeight
  // Enlarge 4x, but keep the crop at most 768 px on its longer side.
  const pxW = region.w * vw
  const pxH = region.h * vh
  const scale = Math.max(1, Math.min(4, 768 / Math.max(pxW, pxH)))
  const ow = Math.max(16, Math.round(pxW * scale))
  const oh = Math.max(16, Math.round(pxH * scale))
  const n = ow * oh
  const side = 2 * RADIUS + 1

  const module = device.createShaderModule({ code: STACK, label: 'stack' })
  const pipe = (entryPoint: string) => device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint } })
  const P = { crop: pipe('crop'), align: pipe('alignShift'), accumulate: pipe('accumulate'), setRef: pipe('setRef'), resolve: pipe('resolve') }
  const frameTex = device.createTexture({ size: [vw, vh], format: 'rgba8unorm', usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT })
  const sampler = device.createSampler({ magFilter: 'linear', minFilter: 'linear' })
  const buf = (bytes: number, extra = 0) => device.createBuffer({ size: bytes, usage: GPUBufferUsage.STORAGE | extra })
  const cur = buf(n * 16)
  const ref = buf(n * 16)
  const acc = buf(n * 16)
  const score = buf(side * side * 4, GPUBufferUsage.COPY_SRC)
  const scoreRead = device.createBuffer({ size: side * side * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST })
  const uni = device.createBuffer({ size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST })
  const out = () => device.createTexture({ size: [ow, oh], format: 'rgba8unorm', usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING })
  const texture = out()
  const single = out()

  const writeUni = (shift: [number, number], frames: number, sh: number) => {
    const b = new ArrayBuffer(48)
    new Uint32Array(b, 0, 2).set([ow, oh])
    new Int32Array(b, 8, 2).set(shift)
    new Float32Array(b, 16, 4).set([region.x, region.y, region.w, region.h])
    new Int32Array(b, 32, 1).set([RADIUS])
    new Float32Array(b, 36, 3).set([frames, sh, 0])
    device.queue.writeBuffer(uni, 0, b)
  }
  const group = (p: GPUComputePipeline, entries: [number, GPUBindingResource][]) =>
    device.createBindGroup({ layout: p.getBindGroupLayout(0), entries: entries.map(([binding, resource]) => ({ binding, resource })) })
  const run = (p: GPUComputePipeline, entries: [number, GPUBindingResource][], x: number, y = 1) => {
    const enc = device.createCommandEncoder()
    const pass = enc.beginComputePass()
    pass.setPipeline(p)
    pass.setBindGroup(0, group(p, entries))
    pass.dispatchWorkgroups(x, y)
    pass.end()
    return enc
  }
  const U: [number, GPUBindingResource] = [0, { buffer: uni }]
  const gx = Math.ceil(ow / 16)
  const gy = Math.ceil(oh / 16)

  // Middle frame first: it is the reference everything is aligned to.
  const order = [0]
  for (let k = 1; order.length < count; k++) {
    order.push(k)
    if (order.length < count) order.push(-k)
  }
  const shifts: [number, number][] = []
  let used = 0
  let skipped = 0
  let bestErr = Infinity
  for (const k of order) {
    await src.at(centre + k / fps)
    copyVideoFrame(device, src.video, frameTex)
    writeUni([0, 0], 1, 0)
    device.queue.submit([run(P.crop, [U, [1, frameTex.createView()], [2, sampler], [3, { buffer: cur }]], gx, gy).finish()])
    if (k === 0) {
      device.queue.submit([run(P.setRef, [U, [3, { buffer: cur }], [4, { buffer: ref }], [5, { buffer: acc }]], gx, gy).finish()])
      // The single-frame view: resolve the reference alone, without sharpening.
      device.queue.submit([run(P.resolve, [U, [5, { buffer: acc }], [7, single.createView()]], gx, gy).finish()])
      shifts.push([0, 0])
      used = 1
      continue
    }
    const enc = run(P.align, [U, [3, { buffer: cur }], [4, { buffer: ref }], [6, { buffer: score }]], Math.ceil((side * side) / 64))
    enc.copyBufferToBuffer(score, 0, scoreRead, 0, side * side * 4)
    device.queue.submit([enc.finish()])
    await scoreRead.mapAsync(GPUMapMode.READ)
    const sc = new Float32Array(scoreRead.getMappedRange().slice(0))
    scoreRead.unmap()
    let best = 0
    for (let i = 1; i < sc.length; i++) if (sc[i] < sc[best]) best = i
    const shift: [number, number] = [(best % side) - RADIUS, Math.floor(best / side) - RADIUS]
    // Even aligned, a frame whose region changed (something moved) would smear the average: skip it.
    bestErr = Math.min(bestErr, sc[best])
    if (sc[best] > 2 * bestErr + 0.004) {
      skipped++
      continue
    }
    shifts.push([shift[0] / scale, shift[1] / scale])
    used++
    writeUni(shift, 1, 0)
    device.queue.submit([run(P.accumulate, [U, [3, { buffer: cur }], [5, { buffer: acc }]], gx, gy).finish()])
  }
  writeUni([0, 0], used, sharpen)
  device.queue.submit([run(P.resolve, [U, [5, { buffer: acc }], [7, texture.createView()]], gx, gy).finish()])
  await device.queue.onSubmittedWorkDone()
  for (const b of [cur, ref, acc, score, scoreRead, uni]) b.destroy()
  frameTex.destroy()
  return { texture, single, frames: used, skipped, shifts }
}
