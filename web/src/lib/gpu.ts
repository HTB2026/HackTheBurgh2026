/**
 * Shared WebGPU access. Everything GPU in the app is optional: when the browser has no
 * WebGPU (or no adapter), `useGpu()` returns null and callers render today's 2D views.
 */
import { useEffect, useRef, useState } from 'react'

export interface Gpu {
  device: GPUDevice
  format: GPUTextureFormat
}

let pending: Promise<Gpu | null> | null = null

/** The one shared device, created on first use; null when WebGPU is unavailable. */
export function getGpu(): Promise<Gpu | null> {
  if (pending) return pending
  pending = (async () => {
    try {
      if (typeof navigator === 'undefined' || !navigator.gpu) return null
      const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' })
      if (!adapter) return null
      const device = await adapter.requestDevice({
        requiredLimits: {
          // Full-resolution photos: up to what the adapter allows (16384 on Apple GPUs).
          maxTextureDimension2D: adapter.limits.maxTextureDimension2D,
          maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
          maxBufferSize: adapter.limits.maxBufferSize,
        },
      })
      device.lost.then(() => {
        pending = null
      })
      return { device, format: navigator.gpu.getPreferredCanvasFormat() }
    } catch {
      return null
    }
  })()
  return pending
}

/** 'loading' until known, then the device or null (no WebGPU: show the 2D fallback). */
export function useGpu(): Gpu | null | 'loading' {
  const [gpu, setGpu] = useState<Gpu | null | 'loading'>('loading')
  useEffect(() => {
    let live = true
    getGpu().then((g) => live && setGpu(g))
    return () => {
      live = false
    }
  }, [])
  return gpu
}

/** Whether this browser can do WebGPU at all (synchronous hint for showing toggles). */
export const hasWebGpu = typeof navigator !== 'undefined' && 'gpu' in navigator

/**
 * A canvas configured for WebGPU that tracks its CSS size (device-pixel accurate).
 * Returns the context and the current pixel size; re-render when `size` changes.
 */
export function useGpuCanvas(gpu: Gpu | null | 'loading') {
  const ref = useRef<HTMLCanvasElement | null>(null)
  const [ctx, setCtx] = useState<GPUCanvasContext | null>(null)
  const [size, setSize] = useState({ width: 0, height: 0, dpr: 1 })

  useEffect(() => {
    const canvas = ref.current
    if (!canvas || !gpu || gpu === 'loading') return
    const context = canvas.getContext('webgpu')
    if (!context) return
    context.configure({ device: gpu.device, format: gpu.format, alphaMode: 'premultiplied' })
    setCtx(context)
    const ro = new ResizeObserver(([entry]) => {
      const dpr = Math.min(window.devicePixelRatio || 1, 2)
      const box = entry.contentRect
      const width = Math.max(1, Math.round(box.width * dpr))
      const height = Math.max(1, Math.round(box.height * dpr))
      if (canvas.width !== width || canvas.height !== height) {
        canvas.width = width
        canvas.height = height
      }
      setSize((s) => (s.width === width && s.height === height && s.dpr === dpr ? s : { width, height, dpr }))
    })
    ro.observe(canvas)
    return () => {
      ro.disconnect()
      context.unconfigure()
      setCtx(null)
    }
  }, [gpu])

  return { ref, ctx, size }
}

/** Fetch an image and upload it as an rgba8unorm texture (sampled + copy source). */
export async function textureFromUrl(device: GPUDevice, url: string, maxSide = 16384): Promise<{ texture: GPUTexture; width: number; height: number }> {
  const response = await fetch(url, { credentials: 'same-origin' })
  if (!response.ok) throw new Error(`image ${response.status}`)
  let bitmap = await createImageBitmap(await response.blob(), { colorSpaceConversion: 'none', premultiplyAlpha: 'none' })
  const limit = Math.min(maxSide, device.limits.maxTextureDimension2D)
  if (bitmap.width > limit || bitmap.height > limit) {
    const s = limit / Math.max(bitmap.width, bitmap.height)
    const scaled = await createImageBitmap(bitmap, { resizeWidth: Math.floor(bitmap.width * s), resizeHeight: Math.floor(bitmap.height * s), resizeQuality: 'high' })
    bitmap.close()
    bitmap = scaled
  }
  return { ...textureFromSource(device, bitmap), width: bitmap.width, height: bitmap.height }
}

/** Upload an ImageBitmap or canvas as a texture. */
export function textureFromSource(device: GPUDevice, source: ImageBitmap | HTMLCanvasElement | OffscreenCanvas): { texture: GPUTexture } {
  const texture = device.createTexture({
    size: [source.width, source.height],
    format: 'rgba8unorm',
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
  })
  device.queue.copyExternalImageToTexture({ source }, { texture }, [source.width, source.height])
  return { texture }
}

export function uniformBuffer(device: GPUDevice, bytes: number): GPUBuffer {
  return device.createBuffer({ size: Math.ceil(bytes / 16) * 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST })
}

/** Run `draw` at most once per animation frame however often it is requested. */
export function useFrameRequest(draw: () => void) {
  const queued = useRef(0)
  const latest = useRef(draw)
  latest.current = draw
  useEffect(() => () => cancelAnimationFrame(queued.current), [])
  return () => {
    if (queued.current) return
    queued.current = requestAnimationFrame(() => {
      queued.current = 0
      latest.current()
    })
  }
}
