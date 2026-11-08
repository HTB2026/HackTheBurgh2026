/**
 * "Find copies of this region": normalised cross-correlation of a selected patch against every
 * position of the picture, on the GPU. Works on a scaled-down greyscale copy sized so the patch
 * is 12-40 px across, which keeps the search to a fraction of a second.
 */
import type { Gpu } from './gpu'

export interface Box {
  x: number
  y: number
  w: number
  h: number
}
export interface CopyMatch extends Box {
  score: number
}

const NCC = /* wgsl */ `
struct P { size: vec2u, tsize: vec2u, sigmaT: f32, n: f32, _pad: vec2f };
@group(0) @binding(0) var<uniform> prm: P;
@group(0) @binding(1) var<storage, read> img: array<f32>;
@group(0) @binding(2) var<storage, read> tpl: array<f32>;   // patch minus its mean
@group(0) @binding(3) var<storage, read_write> score: array<f32>;

@compute @workgroup_size(16, 16)
fn ncc(@builtin(global_invocation_id) gid: vec3u) {
  let maxXY = prm.size - prm.tsize;
  if (gid.x > maxXY.x || gid.y > maxXY.y) { return; }
  var s = 0.0;
  var s2 = 0.0;
  var st = 0.0;
  for (var y = 0u; y < prm.tsize.y; y++) {
    let row = (gid.y + y) * prm.size.x + gid.x;
    let trow = y * prm.tsize.x;
    for (var x = 0u; x < prm.tsize.x; x++) {
      let v = img[row + x];
      s += v;
      s2 += v * v;
      st += v * tpl[trow + x];
    }
  }
  let mean = s / prm.n;
  let sigma = sqrt(max(0.0, s2 / prm.n - mean * mean));
  // Flat windows (sky, walls) can't match a textured patch.
  var r = 0.0;
  if (sigma > 1.5) { r = st / (prm.n * sigma * prm.sigmaT); }
  score[gid.y * (maxXY.x + 1u) + gid.x] = r;
}
`

/**
 * Gaussian blur (sigma 1 px), in place. A copy rarely lands on the same sub-pixel phase as the
 * selection once the picture is scaled down; a slight blur keeps fine texture matching (on the
 * demo copy-move: 0.87 -> 0.95) while unrelated areas stay far below the threshold.
 */
function blur(a: Float32Array, w: number, h: number) {
  const k = [0.0545, 0.2442, 0.4026, 0.2442, 0.0545]
  const tmp = new Float32Array(a.length)
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++) {
      let s = 0
      for (let i = -2; i <= 2; i++) s += k[i + 2] * a[y * w + Math.min(w - 1, Math.max(0, x + i))]
      tmp[y * w + x] = s
    }
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++) {
      let s = 0
      for (let i = -2; i <= 2; i++) s += k[i + 2] * tmp[Math.min(h - 1, Math.max(0, y + i)) * w + x]
      a[y * w + x] = s
    }
}

interface Working {
  key: string
  w: number
  h: number
  luma: Float32Array
  buf: GPUBuffer
}
let working: Working | null = null
let pipeline: { device: GPUDevice; p: GPUComputePipeline } | null = null

async function greyscale(gpu: Gpu, src: string, scale: number, fullW: number, fullH: number): Promise<Working> {
  const w = Math.max(16, Math.round(fullW * scale))
  const h = Math.max(16, Math.round(fullH * scale))
  const key = `${src}@${w}x${h}`
  if (working?.key === key) return working
  const blob = await (await fetch(src, { credentials: 'same-origin' })).blob()
  const bmp = await createImageBitmap(blob, { resizeWidth: w, resizeHeight: h, resizeQuality: 'high' })
  const canvas = new OffscreenCanvas(w, h)
  const g = canvas.getContext('2d', { willReadFrequently: true })!
  g.drawImage(bmp, 0, 0)
  bmp.close()
  const rgba = g.getImageData(0, 0, w, h).data
  const luma = new Float32Array(w * h)
  for (let i = 0; i < w * h; i++) luma[i] = 0.299 * rgba[i * 4] + 0.587 * rgba[i * 4 + 1] + 0.114 * rgba[i * 4 + 2]
  blur(luma, w, h)
  const buf = gpu.device.createBuffer({ size: luma.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST })
  gpu.device.queue.writeBuffer(buf, 0, luma)
  working?.buf.destroy()
  working = { key, w, h, luma, buf }
  return working
}

/**
 * Find places in the picture that look like `box` (fractions 0..1 of the picture).
 * Returns up to `limit` matches (score = correlation, 1 = identical), best first,
 * excluding the selected region itself.
 */
export async function findCopies(gpu: Gpu, src: string, fullW: number, fullH: number, box: Box, limit = 6, minScore = 0.8): Promise<{ matches: CopyMatch[]; best: number }> {
  const { device } = gpu
  const side = Math.max(box.w * fullW, box.h * fullH)
  // Scale so the patch is 12-40 px, but the working picture is at most 2048 px.
  let scale = Math.min(1, Math.max(12, Math.min(40, side)) / side)
  scale = Math.min(scale, 2048 / Math.max(fullW, fullH))
  const W = await greyscale(gpu, src, scale, fullW, fullH)
  const pw = Math.max(4, Math.round(box.w * W.w))
  const ph = Math.max(4, Math.round(box.h * W.h))
  const px = Math.min(W.w - pw, Math.max(0, Math.round(box.x * W.w)))
  const py = Math.min(W.h - ph, Math.max(0, Math.round(box.y * W.h)))
  const n = pw * ph
  const tpl = new Float32Array(n)
  let mean = 0
  for (let y = 0; y < ph; y++) for (let x = 0; x < pw; x++) mean += tpl[y * pw + x] = W.luma[(py + y) * W.w + px + x]
  mean /= n
  let varSum = 0
  for (let i = 0; i < n; i++) {
    tpl[i] -= mean
    varSum += tpl[i] * tpl[i]
  }
  const sigmaT = Math.sqrt(varSum / n)
  if (sigmaT < 3) throw new Error('This area is too plain to match: pick a region with some detail.')

  if (pipeline?.device !== device) {
    const module = device.createShaderModule({ code: NCC, label: 'find copies' })
    pipeline = { device, p: device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'ncc' } }) }
  }
  const outW = W.w - pw + 1
  const outH = W.h - ph + 1
  const prm = new ArrayBuffer(32)
  new Uint32Array(prm, 0, 4).set([W.w, W.h, pw, ph])
  new Float32Array(prm, 16, 2).set([sigmaT, n])
  const uni = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST })
  const tplBuf = device.createBuffer({ size: tpl.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST })
  const out = device.createBuffer({ size: outW * outH * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC })
  const read = device.createBuffer({ size: outW * outH * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST })
  device.queue.writeBuffer(uni, 0, prm)
  device.queue.writeBuffer(tplBuf, 0, tpl)
  const bind = device.createBindGroup({
    layout: pipeline.p.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uni } },
      { binding: 1, resource: { buffer: W.buf } },
      { binding: 2, resource: { buffer: tplBuf } },
      { binding: 3, resource: { buffer: out } },
    ],
  })
  const enc = device.createCommandEncoder({ label: 'find copies' })
  const pass = enc.beginComputePass()
  pass.setPipeline(pipeline.p)
  pass.setBindGroup(0, bind)
  pass.dispatchWorkgroups(Math.ceil(outW / 16), Math.ceil(outH / 16))
  pass.end()
  enc.copyBufferToBuffer(out, 0, read, 0, outW * outH * 4)
  device.queue.submit([enc.finish()])
  try {
    await read.mapAsync(GPUMapMode.READ)
    const score = new Float32Array(read.getMappedRange().slice(0))
    read.unmap()
    // The best score anywhere away from the selection, to report when nothing passes minScore.
    let best = -1
    for (let i = 0; i < score.length; i++) {
      const x = i % outW
      const y = Math.floor(i / outW)
      if (score[i] > best && (Math.abs(x - px) >= pw * 0.6 || Math.abs(y - py) >= ph * 0.6)) best = score[i]
    }
    return { matches: peaks(score, outW, pw, ph, px, py, W.w, W.h, limit, minScore), best }
  } finally {
    for (const b of [uni, tplBuf, out, read]) b.destroy()
  }
}

/** Best non-overlapping peaks above minScore, away from the source patch. */
function peaks(score: Float32Array, outW: number, pw: number, ph: number, px: number, py: number, W: number, H: number, limit: number, minScore: number): CopyMatch[] {
  const candidates: number[] = []
  for (let i = 0; i < score.length; i++) if (score[i] >= minScore) candidates.push(i)
  candidates.sort((a, b) => score[b] - score[a])
  const found: { x: number; y: number; s: number }[] = []
  const overlaps = (ax: number, ay: number, bx: number, by: number) => Math.abs(ax - bx) < pw * 0.6 && Math.abs(ay - by) < ph * 0.6
  for (const i of candidates.slice(0, 20000)) {
    const x = i % outW
    const y = Math.floor(i / outW)
    if (overlaps(x, y, px, py)) continue
    if (found.some((f) => overlaps(x, y, f.x, f.y))) continue
    found.push({ x, y, s: score[i] })
    if (found.length >= limit) break
  }
  return found.map((f) => ({ x: f.x / W, y: f.y / H, w: pw / W, h: ph / H, score: Math.min(1, f.s) }))
}
