/** Live spectrogram on the GPU: any band, any resolution, real decibels, and a peak trace. */
import type { Gpu } from './gpu'
import { decimate, type DecodedAudio } from './media'
import { STFT } from './shaders/stft.wgsl'

export interface SpecRequest {
  f0: number
  f1: number
  rows: number
  cols: number
  /** Window length in seconds; 0 = automatic (longer for narrow bands). */
  windowS: number
  /** Dynamic range shown, in dB below the loudest bin. */
  range: number
}

export interface SpecResult {
  texture: GPUTexture // cols x rows, rgba (magma), row 0 = f1
  peaks: Float32Array // strongest frequency per column (Hz)
  topDb: number
  windowS: number
  rate: number // sample rate after decimation
  t0: number // time of the first column's window centre
  t1: number
}

const pipelines = new WeakMap<GPUDevice, { dft: GPUComputePipeline; paint: GPUComputePipeline }>()
const decimated = new Map<string, Float32Array>()

export const autoWindow = (f0: number, f1: number, rows: number) => Math.min(6, Math.max(0.02, (0.2 * rows) / Math.max(1e-6, f1 - f0)))

export async function gpuSpectrogram(gpu: Gpu, audio: DecodedAudio, req: SpecRequest): Promise<SpecResult> {
  const { device } = gpu
  let pl = pipelines.get(device)
  if (!pl) {
    const module = device.createShaderModule({ code: STFT, label: 'stft' })
    pl = {
      dft: device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'dft' } }),
      paint: device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'paint' } }),
    }
    pipelines.set(device, pl)
  }
  // Decimate so the band's top is still below Nyquist (with margin): cheaper long windows.
  const factor = Math.max(1, Math.floor(audio.rate / (2.4 * Math.max(req.f1, 1))))
  const key = `${audio.samples.length}:${audio.rate}:${factor}`
  let x = decimated.get(key)
  if (!x) {
    x = decimate(audio.samples, factor)
    decimated.set(key, x)
  }
  const rate = audio.rate / factor
  const windowS = req.windowS > 0 ? req.windowS : autoWindow(req.f0, req.f1, req.rows)
  const win = Math.max(16, Math.min(x.length, Math.round(windowS * rate)))
  const cols = Math.max(2, Math.min(req.cols, x.length - win + 1))
  const hop = (x.length - win) / (cols - 1)
  let wsum = 0
  for (let k = 0; k < win; k++) wsum += 0.5 - 0.5 * Math.cos((2 * Math.PI * k) / (win - 1))

  const sig = device.createBuffer({ size: x.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST })
  device.queue.writeBuffer(sig, 0, x as Float32Array<ArrayBuffer>)
  const db = device.createBuffer({ size: cols * req.rows * 4, usage: GPUBufferUsage.STORAGE })
  const top = device.createBuffer({ size: 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC })
  device.queue.writeBuffer(top, 0, new Uint32Array([0]))
  const peak = device.createBuffer({ size: cols * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC })
  const peakRead = device.createBuffer({ size: cols * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST })
  const topRead = device.createBuffer({ size: 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST })
  const texture = device.createTexture({
    size: [cols, req.rows],
    format: 'rgba8unorm',
    usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
  })

  const params = (colStart: number) => {
    const b = new ArrayBuffer(48)
    new Uint32Array(b, 0, 6).set([x!.length, win, cols, req.rows, colStart, 0])
    new Float32Array(b, 24, 6).set([rate, hop, req.f0, req.f1, wsum, req.range])
    return b
  }
  const uniforms: GPUBuffer[] = []
  // Split the work so no single dispatch runs long (GPUs reset work that takes seconds).
  const perDispatch = Math.max(16, Math.floor(2e8 / (req.rows * win)))
  for (let c = 0; c < cols; c += perDispatch) {
    const uni = device.createBuffer({ size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST })
    uniforms.push(uni)
    device.queue.writeBuffer(uni, 0, params(c))
    const enc = device.createCommandEncoder({ label: 'stft' })
    const pass = enc.beginComputePass()
    pass.setPipeline(pl.dft)
    pass.setBindGroup(0, device.createBindGroup({
      layout: pl.dft.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uni } },
        { binding: 1, resource: { buffer: sig } },
        { binding: 2, resource: { buffer: db } },
        { binding: 3, resource: { buffer: top } },
      ],
    }))
    pass.dispatchWorkgroups(Math.ceil(Math.min(perDispatch, cols - c) / 16), Math.ceil(req.rows / 16))
    pass.end()
    device.queue.submit([enc.finish()])
    // Let the GPU breathe between chunks so the page stays responsive.
    if (c + perDispatch < cols) await device.queue.onSubmittedWorkDone()
  }
  const enc = device.createCommandEncoder({ label: 'stft paint' })
  const pass = enc.beginComputePass()
  pass.setPipeline(pl.paint)
  pass.setBindGroup(0, device.createBindGroup({
    layout: pl.paint.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniforms[0] } },
      { binding: 2, resource: { buffer: db } },
      { binding: 3, resource: { buffer: top } },
      { binding: 4, resource: texture.createView() },
      { binding: 5, resource: { buffer: peak } },
    ],
  }))
  pass.dispatchWorkgroups(Math.ceil(cols / 64))
  pass.end()
  enc.copyBufferToBuffer(peak, 0, peakRead, 0, cols * 4)
  enc.copyBufferToBuffer(top, 0, topRead, 0, 4)
  device.queue.submit([enc.finish()])
  await Promise.all([peakRead.mapAsync(GPUMapMode.READ), topRead.mapAsync(GPUMapMode.READ)])
  const peaks = new Float32Array(peakRead.getMappedRange().slice(0))
  const topDb = new Float32Array(topRead.getMappedRange().slice(0))[0] - 300
  peakRead.unmap()
  topRead.unmap()
  for (const b of [sig, db, top, peak, peakRead, topRead, ...uniforms]) b.destroy()
  const half = win / rate / 2
  return { texture, peaks, topDb, windowS: win / rate, rate, t0: half, t1: x.length / rate - half }
}
