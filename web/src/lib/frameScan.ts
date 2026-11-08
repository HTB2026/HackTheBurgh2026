/**
 * Read every frame of the video into the GPU (a hidden <video> plays it; each decoded frame is
 * copied to a texture) and measure, per frame, how much the picture changed, plus per-pixel
 * flicker at the frequencies where 50/60 Hz-grid lamp flicker lands after sampling.
 */
import type { Gpu } from './gpu'
import { FRAMES } from './shaders/frames.wgsl'

export interface ScanResult {
  source: 'original' | 'copy'
  times: number[] // media time of each frame read (s)
  change: number[] // mean |luma difference| from the previous frame read (0-255)
  duplicates: number[] // indices into times: identical to the frame before
  jumps: number[] // indices: abrupt changes (much larger than around them)
  expected: number // frames the file should have (fps x duration)
  /** Per-pixel flicker amplitude (luma units) at each grid's aliased frequency; null when that
   *  frequency lands too close to 0 Hz to measure at this frame rate. */
  flicker: {
    w: number
    h: number
    amp50: Float32Array | null
    amp60: Float32Array | null
    fa50: number
    fa60: number
    /** Same measurement at a frequency where no flicker lands: the noise floor. */
    control: Float32Array
    fc: number
  }
}

/** Where a lamp flickering at `f` Hz appears when sampled at `fps` frames per second. */
export const aliasOf = (f: number, fps: number) => Math.abs(f - fps * Math.round(f / fps))

export async function scanFrames(
  gpu: Gpu,
  url: string,
  originalUrl: string | null,
  fps: number,
  duration: number,
  opts: { rate: number; onProgress?: (fraction: number) => void; onChanges?: (texture: GPUTexture) => void; signal?: AbortSignal },
): Promise<ScanResult> {
  const { device } = gpu
  // Prefer the original file (exact repeated frames survive only there); the viewing copy is
  // a re-encode. Fall back when the browser can't play the original (e.g. HEVC on some systems).
  const open = (src: string) =>
    new Promise<HTMLVideoElement>((resolve, reject) => {
      const v = document.createElement('video')
      v.muted = true
      v.playsInline = true
      v.preload = 'auto'
      const timer = setTimeout(() => reject(new Error('timeout')), 8000)
      v.onloadeddata = () => (clearTimeout(timer), resolve(v))
      v.onerror = () => (clearTimeout(timer), reject(new Error('the video could not be decoded')))
      v.src = src
    })
  let video: HTMLVideoElement
  let source: 'original' | 'copy' = 'copy'
  try {
    if (!originalUrl) throw new Error('no original')
    video = await open(originalUrl)
    if (!video.videoWidth) throw new Error('no picture')
    source = 'original'
  } catch {
    video = await open(url)
  }
  const vw = video.videoWidth
  const vh = video.videoHeight
  const w = Math.min(480, vw)
  const h = Math.max(2, Math.round((vh * w) / vw))
  const n = w * h
  const maxFrames = Math.ceil(duration * fps * 1.2) + 30

  const module = device.createShaderModule({ code: FRAMES, label: 'frame scan' })
  const pipeline = device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'scanStep' } })
  const frameTex = device.createTexture({ size: [vw, vh], format: 'rgba8unorm', usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT })
  const changes = device.createTexture({ size: [w, h], format: 'rgba8unorm', usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING })
  const uni = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST })
  const prev = device.createBuffer({ size: n * 4, usage: GPUBufferUsage.STORAGE })
  const acc = device.createBuffer({ size: n * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC })
  const sumY = device.createBuffer({ size: n * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC })
  const accC = device.createBuffer({ size: n * 8, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC })
  const sums = device.createBuffer({ size: maxFrames * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC })
  const bind = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uni } },
      { binding: 1, resource: frameTex.createView() },
      { binding: 2, resource: { buffer: prev } },
      { binding: 3, resource: { buffer: acc } },
      { binding: 4, resource: { buffer: sums } },
      { binding: 5, resource: changes.createView() },
      { binding: 6, resource: { buffer: sumY } },
      { binding: 7, resource: { buffer: accC } },
    ],
  })
  opts.onChanges?.(changes)

  const fa50 = aliasOf(100, fps)
  const fa60 = aliasOf(120, fps)
  // Control: a frequency at least 1.5 Hz from both flicker aliases (and from 0 and Nyquist).
  const fc = [0.4637, 0.37, 0.29, 0.21].map((k) => k * fps).find((f) => Math.abs(f - fa50) > 1.5 && Math.abs(f - fa60) > 1.5 && f > 1) ?? 0.4637 * fps
  const times: number[] = []

  /** One compute step on the frame the video is showing now (at media time t). */
  const processFrame = (t: number) => {
    // The same frame can be reported twice (the first one, around play()): read it once.
    if (times.length && Math.abs(t - times[times.length - 1]) < 1e-4) return
    const k = times.length
    if (k >= maxFrames) return
    device.queue.copyExternalImageToTexture({ source: video }, { texture: frameTex }, [vw, vh])
    const b = new ArrayBuffer(32)
    new Uint32Array(b, 0, 4).set([w, h, k, k === 0 ? 1 : 0])
    new Float32Array(b, 16, 4).set([t, fa50, fa60, fc])
    device.queue.writeBuffer(uni, 0, b)
    const enc = device.createCommandEncoder({ label: 'frame' })
    const pass = enc.beginComputePass()
    pass.setPipeline(pipeline)
    pass.setBindGroup(0, bind)
    pass.dispatchWorkgroups(Math.ceil(w / 16), Math.ceil(h / 16))
    pass.end()
    device.queue.submit([enc.finish()])
    times.push(t)
    opts.onProgress?.(Math.min(1, t / Math.max(0.001, duration)))
  }

  // The first frame is already on screen before playback starts, so no frame callback reports
  // it: read it explicitly.
  processFrame(video.currentTime)

  await new Promise<void>((resolve, reject) => {
    const onFrame = (_now: number, meta: VideoFrameCallbackMetadata) => {
      if (opts.signal?.aborted) {
        video.pause()
        return reject(new DOMException('cancelled', 'AbortError'))
      }
      processFrame(meta.mediaTime)
      if (!video.ended) video.requestVideoFrameCallback(onFrame)
    }
    video.onended = () => resolve()
    video.onerror = () => reject(new Error('the video stopped decoding'))
    video.requestVideoFrameCallback(onFrame)
    video.playbackRate = opts.rate
    video.play().catch(reject)
  })

  // Read the per-frame sums and the flicker accumulators back.
  const sumsRead = device.createBuffer({ size: maxFrames * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST })
  const accRead = device.createBuffer({ size: n * 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST })
  const yRead = device.createBuffer({ size: n * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST })
  const cRead = device.createBuffer({ size: n * 8, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST })
  const enc = device.createCommandEncoder()
  enc.copyBufferToBuffer(sums, 0, sumsRead, 0, maxFrames * 4)
  enc.copyBufferToBuffer(acc, 0, accRead, 0, n * 16)
  enc.copyBufferToBuffer(sumY, 0, yRead, 0, n * 4)
  enc.copyBufferToBuffer(accC, 0, cRead, 0, n * 8)
  device.queue.submit([enc.finish()])
  await Promise.all([sumsRead.mapAsync(GPUMapMode.READ), accRead.mapAsync(GPUMapMode.READ), yRead.mapAsync(GPUMapMode.READ), cRead.mapAsync(GPUMapMode.READ)])
  const s = new Uint32Array(sumsRead.getMappedRange().slice(0))
  const a = new Float32Array(accRead.getMappedRange().slice(0))
  const ys = new Float32Array(yRead.getMappedRange().slice(0))
  const ac = new Float32Array(cRead.getMappedRange().slice(0))
  sumsRead.unmap()
  accRead.unmap()
  yRead.unmap()
  cRead.unmap()
  for (const b of [uni, prev, acc, sumY, accC, sums, sumsRead, accRead, yRead, cRead]) b.destroy()
  frameTex.destroy()
  video.removeAttribute('src')
  video.load()

  const count = times.length
  const change = Array.from({ length: count }, (_, k) => (k === 0 ? 0 : s[k] / 16 / n))
  // A repeated frame decodes to exactly the same pixels: zero change (near-static scenes still differ).
  const duplicates = change.map((_c, k) => (k > 0 && s[k] === 0 ? k : -1)).filter((k) => k >= 0)
  // Abrupt changes: far above the typical change around them. Repeated frames (zero change) are
  // left out of "typical", or a stretch of repeats would make every normal frame look abrupt.
  const jumps: number[] = []
  for (let k = 1; k < count; k++) {
    if (s[k] === 0) continue
    const around: number[] = []
    for (let j = Math.max(1, k - 15); j < Math.min(count, k + 16); j++) if (j !== k && s[j] !== 0) around.push(change[j])
    around.sort((x, y) => x - y)
    const med = around.length ? around[Math.floor(around.length / 2)] : 0
    if (change[k] > Math.max(6, 5 * med)) jumps.push(k)
  }
  // Remove each pixel's own brightness from the Fourier sums (it leaks in unless the clip holds
  // a whole number of flicker cycles): re -= mean * sum(cos), im -= mean * sum(sin).
  const amplitude = (fa: number, o: number, src: Float32Array = a, stride = 4) => {
    if (fa < 0.5 || count < 8) return null
    let c = 0
    let sn = 0
    for (const t of times) {
      c += Math.cos(2 * Math.PI * fa * t)
      sn += Math.sin(2 * Math.PI * fa * t)
    }
    const re = new Float32Array(n)
    const im = new Float32Array(n)
    for (let i = 0; i < n; i++) {
      const mean = ys[i] / count
      re[i] = src[i * stride + o] - mean * c
      im[i] = src[i * stride + o + 1] - mean * sn
    }
    // Rolling-shutter flicker forms horizontal bands: the pulse has the same phase along a row,
    // so averaging the sums along rows (and a few rows) keeps it while cancelling pixel noise.
    const rx = Math.max(4, Math.round(w / 30))
    const ry = 2
    const sm = (src: Float32Array) => boxBlur(boxBlur(src, w, h, rx, 0), w, h, 0, ry)
    const sre = sm(re)
    const sim = sm(im)
    const out = new Float32Array(n)
    for (let i = 0; i < n; i++) out[i] = (2 * Math.hypot(sre[i], sim[i])) / count
    return out
  }
  const amp50 = amplitude(fa50, 0)
  const amp60 = amplitude(fa60, 2)
  const control = amplitude(fc, 0, ac, 2)!
  return { source, times, change, duplicates, jumps, expected: Math.round(duration * fps), flicker: { w, h, amp50, amp60, fa50, fa60, control, fc } }
}

/** Box blur of radius rx (horizontal) or ry (vertical), edges clamped. */
function boxBlur(src: Float32Array, w: number, h: number, rx: number, ry: number): Float32Array {
  const out = new Float32Array(src.length)
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++) {
      let sum = 0
      let k = 0
      for (let d = -(rx || ry); d <= (rx || ry); d++) {
        const xx = rx ? x + d : x
        const yy = ry ? y + d : y
        if (xx < 0 || yy < 0 || xx >= w || yy >= h) continue
        sum += src[yy * w + xx]
        k++
      }
      out[y * w + x] = sum / k
    }
  return out
}
