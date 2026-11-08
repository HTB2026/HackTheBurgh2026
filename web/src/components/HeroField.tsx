import { useEffect, useRef } from 'react'
import { uniformBuffer, useGpu, useGpuCanvas } from '../lib/gpu'
import { PARTICLE_DRAW, PARTICLE_SIM } from '../lib/shaders/particles.wgsl'

const COUNT = 20_000
const ENV = 256

/** Resample a waveform's peaks to `n` values in 0..1 (light smoothing, normalised). */
function envelopeOf(peaks: number[] | null): Float32Array<ArrayBuffer> {
  const out = new Float32Array(ENV)
  if (!peaks?.length) return out
  for (let i = 0; i < ENV; i++) {
    const a = Math.floor((i / ENV) * peaks.length)
    const b = Math.max(a + 1, Math.floor(((i + 1) / ENV) * peaks.length))
    let m = 0
    for (let j = a; j < b; j++) m = Math.max(m, Math.abs(peaks[j] ?? 0))
    out[i] = m
  }
  // Scale to the 90th percentile, not the loudest peak, so a few loud beeps don't flatten the rest.
  const top = Math.max(1e-6, [...out].sort((a, b) => a - b)[Math.floor(ENV * 0.9)])
  for (let i = 0; i < ENV; i++) out[i] = Math.min(1, Math.pow(out[i] / top, 0.7))
  return out
}

/**
 * A field of glowing particles behind the homepage heading, shaped by the latest recording's
 * waveform. WebGPU only; renders nothing otherwise, or when the user prefers reduced motion.
 */
export function HeroField({ peaks }: { peaks: number[] | null }) {
  const gpuState = useGpu()
  const reduced = typeof matchMedia !== 'undefined' && matchMedia('(prefers-reduced-motion: reduce)').matches
  const gpu = gpuState === 'loading' || reduced ? null : gpuState
  const { ref, ctx, size } = useGpuCanvas(gpu)
  const mouse = useRef<[number, number]>([-1, 0])
  const envBuf = useRef<GPUBuffer | null>(null)
  const envCount = useRef(0)

  // Simulation state and pipelines.
  useEffect(() => {
    if (!gpu || !ctx || !size.width) return
    const { device, format } = gpu
    const simModule = device.createShaderModule({ code: PARTICLE_SIM, label: 'particles sim' })
    const drawModule = device.createShaderModule({ code: PARTICLE_DRAW, label: 'particles draw' })
    const simPipe = device.createComputePipeline({ layout: 'auto', compute: { module: simModule, entryPoint: 'simulate' } })
    const drawPipe = device.createRenderPipeline({
      layout: 'auto',
      vertex: { module: drawModule, entryPoint: 'vs' },
      fragment: {
        module: drawModule,
        entryPoint: 'fs',
        targets: [{ format, blend: { color: { srcFactor: 'one', dstFactor: 'one', operation: 'add' }, alpha: { srcFactor: 'one', dstFactor: 'one', operation: 'add' } } }],
      },
      primitive: { topology: 'triangle-list' },
    })
    const init = new Float32Array(COUNT * 4)
    for (let i = 0; i < COUNT; i++) init.set([Math.random(), (Math.random() * 2 - 1) * 0.6, 0, 0], i * 4)
    const parts = device.createBuffer({ size: init.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST })
    device.queue.writeBuffer(parts, 0, init)
    const uni = uniformBuffer(device, 32)
    envBuf.current ??= device.createBuffer({ size: ENV * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST })
    const simBind = device.createBindGroup({
      layout: simPipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uni } },
        { binding: 1, resource: { buffer: parts } },
        { binding: 2, resource: { buffer: envBuf.current } },
      ],
    })
    const drawBind = device.createBindGroup({
      layout: drawPipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uni } },
        { binding: 1, resource: { buffer: parts } },
      ],
    })

    let raf = 0
    let last = performance.now()
    const frame = (now: number) => {
      raf = requestAnimationFrame(frame)
      if (document.hidden) {
        last = now
        return
      }
      const dt = Math.min(0.05, (now - last) / 1000)
      last = now
      device.queue.writeBuffer(uni, 0, new Float32Array([dt, now / 1000, size.width / size.height, envCount.current, mouse.current[0], mouse.current[1], 2 / size.width, 2 / size.height]))
      const enc = device.createCommandEncoder({ label: 'particles' })
      const cpass = enc.beginComputePass()
      cpass.setPipeline(simPipe)
      cpass.setBindGroup(0, simBind)
      cpass.dispatchWorkgroups(Math.ceil(COUNT / 256))
      cpass.end()
      const rpass = enc.beginRenderPass({
        colorAttachments: [{ view: ctx.getCurrentTexture().createView(), loadOp: 'clear', storeOp: 'store', clearValue: { r: 0, g: 0, b: 0, a: 0 } }],
      })
      rpass.setPipeline(drawPipe)
      rpass.setBindGroup(0, drawBind)
      rpass.draw(6, COUNT)
      rpass.end()
      device.queue.submit([enc.finish()])
    }
    raf = requestAnimationFrame(frame)
    return () => {
      cancelAnimationFrame(raf)
      parts.destroy()
      uni.destroy()
    }
  }, [gpu, ctx, size])

  // The waveform that shapes the field (a synthetic one until a case has audio).
  useEffect(() => {
    if (!gpu || !envBuf.current) return
    const env = envelopeOf(peaks)
    envCount.current = peaks?.length ? ENV : 0
    gpu.device.queue.writeBuffer(envBuf.current, 0, env)
  }, [gpu, peaks, ctx, size.width])

  useEffect(() => () => envBuf.current?.destroy(), [])

  // Track the mouse over the whole intro (the canvas itself ignores pointer events).
  useEffect(() => {
    const el = ref.current?.parentElement
    if (!el || !gpu) return
    const move = (e: PointerEvent) => {
      const r = el.getBoundingClientRect()
      mouse.current = [(e.clientX - r.left) / r.width, 1 - ((e.clientY - r.top) / r.height) * 2]
    }
    const leave = () => (mouse.current = [-1, 0])
    el.addEventListener('pointermove', move)
    el.addEventListener('pointerleave', leave)
    return () => {
      el.removeEventListener('pointermove', move)
      el.removeEventListener('pointerleave', leave)
    }
  }, [ref, gpu])

  if (!gpu) return null
  return <canvas ref={ref} className="hero-field" aria-hidden />
}
