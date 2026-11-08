/** WGSL for the GPU forensic lenses in the photo viewer (see components/case/GpuLens.tsx). */

/**
 * Compute passes: read the full-resolution picture, write a forensic map of the same size.
 * mode 0 = error level (JPEG re-encode per 8x8 tile), 1 = noise residual,
 * 2 = local contrast, 3 = Sobel edges.
 */
export const LENS_COMPUTE = /* wgsl */ `
struct Params { mode: u32, quality: f32, gain: f32, _pad: f32 };

@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var dst: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(2) var<uniform> P: Params;

const PI = 3.14159265358979;

// Standard JPEG quantisation tables (ITU T.81 Annex K), row = vertical frequency.
const QY = array<f32, 64>(
  16, 11, 10, 16, 24, 40, 51, 61,   12, 12, 14, 19, 26, 58, 60, 55,
  14, 13, 16, 24, 40, 57, 69, 56,   14, 17, 22, 29, 51, 87, 80, 62,
  18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
  49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99);
const QC = array<f32, 64>(
  17, 18, 24, 47, 99, 99, 99, 99,   18, 21, 26, 66, 99, 99, 99, 99,
  24, 26, 56, 99, 99, 99, 99, 99,   47, 66, 99, 99, 99, 99, 99, 99,
  99, 99, 99, 99, 99, 99, 99, 99,   99, 99, 99, 99, 99, 99, 99, 99,
  99, 99, 99, 99, 99, 99, 99, 99,   99, 99, 99, 99, 99, 99, 99, 99);

fn rgb255(p: vec2i) -> vec3f {
  let size = vec2i(textureDimensions(src));
  let q = clamp(p, vec2i(0), size - 1);
  return round(textureLoad(src, q, 0).rgb * 255.0);
}

fn luma(c: vec3f) -> f32 { return dot(c, vec3f(0.299, 0.587, 0.114)); }

// Inferno-like palette (polynomial fit), t in 0..1.
fn inferno(t0: f32) -> vec3f {
  let t = clamp(t0, 0.0, 1.0);
  let c0 = vec3f(0.0002189, 0.0016510, -0.0194809);
  let c1 = vec3f(0.1065134, 0.5639564, 3.9327123);
  let c2 = vec3f(11.6024930, -3.9728537, -15.9423940);
  let c3 = vec3f(-41.7039961, 17.4363988, 44.3541452);
  let c4 = vec3f(77.1629356, -33.4023589, -81.8073092);
  let c5 = vec3f(-71.3194283, 32.6260642, 73.2095198);
  let c6 = vec3f(25.1311262, -12.2426690, -23.0703250);
  return clamp(c0 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * (c5 + t * c6))))), vec3f(0.0), vec3f(1.0));
}

fn qscale(q: f32) -> f32 {
  let quality = clamp(q, 1.0, 100.0);
  return select(200.0 - 2.0 * quality, 5000.0 / quality, quality < 50.0);
}

fn quant(t: f32, s: f32) -> f32 { return clamp(floor((t * s + 50.0) / 100.0), 1.0, 255.0); }

fn cu(u: u32) -> f32 { return select(1.0, 0.70710678, u == 0u); }

var<workgroup> tile: array<vec3f, 64>;
var<workgroup> coef: array<vec3f, 64>;

// Error level: re-encode each 8x8 tile as JPEG at the chosen quality (4:4:4) and show
// how much each pixel changes. Areas saved at a different quality than the rest stand res.
@compute @workgroup_size(8, 8)
fn ela(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  let p = vec2i(wg.xy * 8u + lid.xy);
  let i = lid.y * 8u + lid.x;
  let rgb = rgb255(p);
  // RGB -> YCbCr, level-shifted by 128 as JPEG does.
  let y = 0.299 * rgb.r + 0.587 * rgb.g + 0.114 * rgb.b - 128.0;
  let cb = -0.168736 * rgb.r - 0.331264 * rgb.g + 0.5 * rgb.b;
  let cr = 0.5 * rgb.r - 0.418688 * rgb.g - 0.081312 * rgb.b;
  tile[i] = vec3f(y, cb, cr);
  workgroupBarrier();

  // Forward DCT for coefficient (u, v) = (lid.x, lid.y), then quantise.
  let u = lid.x;
  let v = lid.y;
  var sum = vec3f(0.0);
  for (var yy = 0u; yy < 8u; yy++) {
    let cy = cos((2.0 * f32(yy) + 1.0) * f32(v) * PI / 16.0);
    for (var xx = 0u; xx < 8u; xx++) {
      sum += tile[yy * 8u + xx] * (cos((2.0 * f32(xx) + 1.0) * f32(u) * PI / 16.0) * cy);
    }
  }
  let F = 0.25 * cu(u) * cu(v) * sum;
  let s = qscale(P.quality);
  let qy = quant(QY[i], s);
  let qc = quant(QC[i], s);
  coef[i] = vec3f(round(F.x / qy) * qy, round(F.y / qc) * qc, round(F.z / qc) * qc);
  workgroupBarrier();

  // Inverse DCT back to pixel (x, y) = lid.
  var g = vec3f(0.0);
  for (var vv = 0u; vv < 8u; vv++) {
    let cy = cu(vv) * cos((2.0 * f32(lid.y) + 1.0) * f32(vv) * PI / 16.0);
    for (var uu = 0u; uu < 8u; uu++) {
      g += coef[vv * 8u + uu] * (cu(uu) * cos((2.0 * f32(lid.x) + 1.0) * f32(uu) * PI / 16.0) * cy);
    }
  }
  g *= 0.25;
  let Y = g.x + 128.0;
  let back = clamp(round(vec3f(Y + 1.402 * g.z, Y - 0.344136 * g.y - 0.714136 * g.z, Y + 1.772 * g.y)), vec3f(0.0), vec3f(255.0));
  let d = abs(back - rgb);
  let e = max(d.r, max(d.g, d.b)) * P.gain / 255.0;

  let size = vec2i(textureDimensions(src));
  if (p.x < size.x && p.y < size.y) {
    textureStore(dst, p, vec4f(inferno(e), 1.0));
  }
}

fn median9(a: array<f32, 9>) -> f32 {
  var v = a;
  // Partial selection sort: the 5th smallest is the median.
  for (var i = 0u; i < 5u; i++) {
    var m = i;
    for (var j = i + 1u; j < 9u; j++) { if (v[j] < v[m]) { m = j; } }
    let t = v[i]; v[i] = v[m]; v[m] = t;
  }
  return v[4];
}

@compute @workgroup_size(8, 8)
fn filters(@builtin(global_invocation_id) gid: vec3u) {
  let size = vec2i(textureDimensions(src));
  let p = vec2i(gid.xy);
  if (p.x >= size.x || p.y >= size.y) { return; }
  var res = vec3f(0.0);

  if (P.mode == 1u) {
    // Noise residual: luminance minus its 3x3 median, signed around mid-grey.
    var n: array<f32, 9>;
    var k = 0u;
    for (var dy = -1; dy <= 1; dy++) {
      for (var dx = -1; dx <= 1; dx++) { n[k] = luma(rgb255(p + vec2i(dx, dy))); k++; }
    }
    let r = n[4] - median9(n);
    res = vec3f(clamp(0.5 + r * P.gain / 255.0, 0.0, 1.0));
  } else if (P.mode == 2u) {
    // Local contrast: standard deviation of luminance in a 7x7 window.
    var s = 0.0;
    var s2 = 0.0;
    for (var dy = -3; dy <= 3; dy++) {
      for (var dx = -3; dx <= 3; dx++) {
        let l = luma(rgb255(p + vec2i(dx, dy)));
        s += l;
        s2 += l * l;
      }
    }
    let mean = s / 49.0;
    let sd = sqrt(max(0.0, s2 / 49.0 - mean * mean));
    res = inferno(sd * P.gain / 128.0);
  } else {
    // Sobel edge magnitude on luminance.
    var l: array<f32, 9>;
    var k = 0u;
    for (var dy = -1; dy <= 1; dy++) {
      for (var dx = -1; dx <= 1; dx++) { l[k] = luma(rgb255(p + vec2i(dx, dy))); k++; }
    }
    let gx = (l[2] + 2.0 * l[5] + l[8]) - (l[0] + 2.0 * l[3] + l[6]);
    let gy = (l[6] + 2.0 * l[7] + l[8]) - (l[0] + 2.0 * l[1] + l[2]);
    res = vec3f(clamp(length(vec2f(gx, gy)) * P.gain / 1020.0, 0.0, 1.0));
  }
  textureStore(dst, p, vec4f(res, 1.0));
}

// ---- JPEG ghost: the quality each 8x8 block was last saved at -------------------------
// A block decoded from a JPEG saved at quality q0 has luma DCT coefficients that sit on q0's
// quantisation grid. For each candidate quality: actual requantisation error divided by what an
// unrelated block would give (min(F^2, Q^2/12) per coefficient). The best-fitting quality, if it
// fits well, is the block's estimate. Pass 1 (ghost) estimates and counts; pass 2 (ghostPaint)
// finds the picture's dominant quality from the counts and colours the blocks that disagree.
@group(0) @binding(3) var<storage, read_write> hist: array<atomic<u32>>;   // GHOST_LEVELS + "none"
@group(0) @binding(4) var<storage, read_write> blockQ: array<u32>;         // estimate per block

const GHOST_LEVELS = 39u;     // 30, 35, ..., 65, then every quality 70..100
var<workgroup> fy: array<f32, 64>;
var<workgroup> ratio: array<f32, 64>;
var<workgroup> blockEnergy: f32;
var<workgroup> clipped: atomic<u32>;

fn ghostQuality(k: u32) -> f32 { return select(70.0 + f32(k - 8u), 30.0 + 5.0 * f32(k), k < 8u); }

// Hue by quality: red (30) through yellow and green to blue (100).
fn ghostColour(q: f32) -> vec3f {
  let h = clamp((q - 30.0) / 70.0, 0.0, 1.0) * 4.0;
  return clamp(vec3f(abs(h - 3.0) - 1.0, 2.0 - abs(h - 2.0), 2.0 - abs(h - 4.0)), vec3f(0.0), vec3f(1.0)) * 0.85 + 0.15;
}

@compute @workgroup_size(8, 8)
fn ghost(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) lid: vec3u, @builtin(num_workgroups) nwg: vec3u) {
  let p = vec2i(wg.xy * 8u + lid.xy);
  let i = lid.y * 8u + lid.x;
  let px = rgb255(p);
  tile[i] = vec3f(luma(px) - 128.0, 0.0, 0.0);
  // Pixels clipped at 0 or 255 have been pushed off the JPEG grid: such blocks are not evidence.
  if (i == 0u) { atomicStore(&clipped, 0u); }
  workgroupBarrier();
  if (min(px.r, min(px.g, px.b)) <= 1.0 || max(px.r, max(px.g, px.b)) >= 254.0) { atomicOr(&clipped, 1u); }
  workgroupBarrier();

  var sum = 0.0;
  for (var yy = 0u; yy < 8u; yy++) {
    let cy = cos((2.0 * f32(yy) + 1.0) * f32(lid.y) * PI / 16.0);
    for (var xx = 0u; xx < 8u; xx++) {
      sum += tile[yy * 8u + xx].x * cos((2.0 * f32(xx) + 1.0) * f32(lid.x) * PI / 16.0) * cy;
    }
  }
  fy[i] = 0.25 * cu(lid.x) * cu(lid.y) * sum;
  workgroupBarrier();

  if (i < GHOST_LEVELS) {
    let s = qscale(ghostQuality(i));
    var err = 0.0;
    var rnd = 0.0;
    for (var c = 0u; c < 64u; c++) {
      let q = quant(QY[c], s);
      let f = fy[c];
      let d = f - round(f / q) * q;
      err += d * d;
      rnd += min(f * f, q * q / 12.0);
    }
    ratio[i] = err / (rnd + 1e-3);
  }
  if (i == 63u) {
    var ac = 0.0;
    for (var c = 1u; c < 64u; c++) { ac += fy[c] * fy[c]; }
    blockEnergy = ac;
  }
  workgroupBarrier();

  if (i == 0u) {
    // Best fit, if it fits well; near-flat blocks carry no evidence either way.
    var k = GHOST_LEVELS;
    if (blockEnergy > 50.0 && atomicLoad(&clipped) == 0u) {
      var bestK = 0u;
      for (var j = 1u; j < GHOST_LEVELS; j++) { if (ratio[j] < ratio[bestK]) { bestK = j; } }
      if (ratio[bestK] < 0.35) { k = bestK; }
    }
    let size = vec2u(textureDimensions(src));
    if (wg.x * 8u < size.x && wg.y * 8u < size.y) {
      blockQ[wg.y * nwg.x + wg.x] = k;
      atomicAdd(&hist[k], 1u);
    }
  }
}

@compute @workgroup_size(8, 8)
fn ghostPaint(@builtin(global_invocation_id) gid: vec3u) {
  let size = vec2u(textureDimensions(src));
  if (gid.x >= size.x || gid.y >= size.y) { return; }
  var dom = 0u;
  var domN = 0u;
  for (var k = 0u; k < GHOST_LEVELS; k++) {
    let n = atomicLoad(&hist[k]);
    if (n > domN) { domN = n; dom = k; }
  }
  let blocksX = (size.x + 7u) / 8u;
  let k = blockQ[(gid.y / 8u) * blocksX + gid.x / 8u];
  let y = luma(rgb255(vec2i(gid.xy))) / 255.0;
  var c = vec3f(y * 0.28);
  if (k < GHOST_LEVELS) {
    let q = ghostQuality(k);
    if (abs(q - ghostQuality(dom)) <= 1.0) {
      c = vec3f(y * 0.5);            // agrees with the rest of the picture
    } else {
      c = mix(vec3f(y * 0.5), ghostColour(q), 0.85);   // a different save quality
    }
  }
  textureStore(dst, vec2i(gid.xy), vec4f(c, 1.0));
}
`

/** Render pass: draw the forensic map through the viewer's pan/zoom onto a stage-sized canvas. */
export const LENS_DRAW = /* wgsl */ `
struct View {
  stage: vec2f,   // canvas size in device pixels
  t: vec2f,       // picture top-left in CSS pixels
  img: vec2f,     // picture size in picture pixels (as the viewer counts them)
  scale: f32,     // CSS pixels per picture pixel
  dpr: f32,
  nearest: f32,   // 1 = show picture pixels as blocks (zoomed in)
  _pad: f32,
};
@group(0) @binding(0) var lensMap: texture_2d<f32>;
@group(0) @binding(1) var sampLinear: sampler;
@group(0) @binding(2) var sampNearest: sampler;
@group(0) @binding(3) var<uniform> V: View;

@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
  let xy = vec2f(f32((i << 1u) & 2u), f32(i & 2u)) * 2.0 - 1.0;
  return vec4f(xy, 0.0, 1.0);
}

fn tap(uv: vec2f) -> vec3f {
  if (V.nearest > 0.5) { return textureSampleLevel(lensMap, sampNearest, uv, 0.0).rgb; }
  return textureSampleLevel(lensMap, sampLinear, uv, 0.0).rgb;
}

@fragment
fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let css = pos.xy / V.dpr;
  let ip = (css - V.t) / V.scale;          // picture pixels
  let uv = ip / V.img;
  if (uv.x < 0.0 || uv.y < 0.0 || uv.x > 1.0 || uv.y > 1.0) { return vec4f(0.0); }
  if (V.scale >= 1.0) { return vec4f(tap(uv), 1.0); }
  // Zoomed res: one screen pixel covers several picture pixels. Take the brightest of
  // four taps so small hot spots (a pasted patch) don't vanish when downsampled.
  let f = 0.25 / (V.scale * V.img);
  let c = max(max(tap(uv + vec2f(-f.x, -f.y)), tap(uv + vec2f(f.x, -f.y))), max(tap(uv + vec2f(-f.x, f.y)), tap(uv + vec2f(f.x, f.y))));
  return vec4f(c, 1.0);
}
`
