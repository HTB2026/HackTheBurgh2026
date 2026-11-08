/** WGSL for the live GPU spectrogram (components/viz/LiveSpectrogram.tsx). */
export const STFT = /* wgsl */ `
struct P {
  n: u32,          // samples
  win: u32,        // window length (samples)
  cols: u32,
  rows: u32,
  colStart: u32,   // first column of this dispatch (work is split to stay short)
  _pad: u32,
  rate: f32,       // sample rate after decimation
  hop: f32,        // samples between window starts
  f0: f32,
  f1: f32,
  wsum: f32,       // sum of the Hann window, for amplitude normalisation
  range: f32,      // dB shown below the loudest bin
};
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> db: array<f32>;      // rows x cols, row 0 = f1 (top)
@group(0) @binding(3) var<storage, read_write> topBits: atomic<u32>;     // loudest dB + 300, as float bits

const PI = 3.14159265358979;

fn rowFreq(row: u32) -> f32 {
  return p.f1 - (f32(row) + 0.5) / f32(p.rows) * (p.f1 - p.f0);
}

// One (column, row): Hann-windowed DFT at exactly this row's frequency.
@compute @workgroup_size(16, 16)
fn dft(@builtin(global_invocation_id) gid: vec3u) {
  let col = gid.x + p.colStart;
  let row = gid.y;
  if (col >= p.cols || row >= p.rows) { return; }
  let start = u32(f32(col) * p.hop);
  let f = rowFreq(row) / p.rate;              // cycles per sample
  var re = 0.0;
  var im = 0.0;
  for (var k = 0u; k < p.win; k++) {
    let i = start + k;
    if (i >= p.n) { break; }
    let w = 0.5 - 0.5 * cos(2.0 * PI * f32(k) / f32(p.win - 1u));
    let ph = 2.0 * PI * fract(f * f32(k));    // fract keeps the angle precise for long windows
    let v = x[i] * w;
    re += v * cos(ph);
    im -= v * sin(ph);
  }
  let amp = 2.0 * sqrt(re * re + im * im) / p.wsum;
  let d = 20.0 * log(max(amp, 1e-9)) / log(10.0);
  db[row * p.cols + col] = d;
  atomicMax(&topBits, bitcast<u32>(d + 300.0));
}

@group(0) @binding(4) var outTex: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(5) var<storage, read_write> peak: array<f32>;   // per column: peak frequency (Hz)

fn magma(t0: f32) -> vec3f {
  let t = clamp(t0, 0.0, 1.0);
  let c0 = vec3f(-0.002136485, -0.000749655, -0.005386127);
  let c1 = vec3f(0.2516605, 0.6775232, 2.494026);
  let c2 = vec3f(8.353717, -3.577719, 0.3144679);
  let c3 = vec3f(-27.66874, 14.26473, -13.64921);
  let c4 = vec3f(52.17614, -27.94361, 12.94416);
  let c5 = vec3f(-50.76852, 29.04658, 4.23415);
  let c6 = vec3f(18.65570, -11.48977, -5.601961);
  return clamp(c0 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * (c5 + t * c6))))), vec3f(0.0), vec3f(1.0));
}

// Colour every bin relative to the loudest one, and find each column's strongest frequency.
@compute @workgroup_size(64)
fn paint(@builtin(global_invocation_id) gid: vec3u) {
  let col = gid.x;
  if (col >= p.cols) { return; }
  let topDb = bitcast<f32>(atomicLoad(&topBits)) - 300.0;
  var best = 0u;
  var bestDb = -1e9;
  for (var row = 0u; row < p.rows; row++) {
    let d = db[row * p.cols + col];
    textureStore(outTex, vec2u(col, row), vec4f(magma((d - (topDb - p.range)) / p.range), 1.0));
    if (d > bestDb) { bestDb = d; best = row; }
  }
  // Parabolic interpolation between neighbouring rows for a sub-row peak estimate.
  var off = 0.0;
  if (best > 0u && best + 1u < p.rows) {
    let a = db[(best - 1u) * p.cols + col];
    let c = db[(best + 1u) * p.cols + col];
    let den = a - 2.0 * bestDb + c;
    if (abs(den) > 1e-6) { off = clamp(0.5 * (a - c) / den, -0.5, 0.5); }
  }
  peak[col] = p.f1 - (f32(best) + off + 0.5) / f32(p.rows) * (p.f1 - p.f0);
}
`

/** Draw the spectrogram texture to a canvas (stretched to fit). */
export const STFT_DRAW = /* wgsl */ `
@group(0) @binding(0) var img: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;
struct O { @builtin(position) pos: vec4f, @location(0) uv: vec2f };
@vertex fn vs(@builtin(vertex_index) i: u32) -> O {
  let xy = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
  var o: O;
  o.pos = vec4f(xy * 2.0 - 1.0, 0.0, 1.0);
  o.uv = vec2f(xy.x, 1.0 - xy.y);
  return o;
}
@fragment fn fs(o: O) -> @location(0) vec4f { return textureSampleLevel(img, samp, o.uv, 0.0); }
`
