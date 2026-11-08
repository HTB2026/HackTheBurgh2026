/** WGSL for the video frame scan (lib/frameScan.ts): change curve, duplicate frames, flicker map. */
export const FRAMES = /* wgsl */ `
struct P {
  size: vec2u,      // working size (pixels)
  frame: u32,       // index of this frame in the scan
  first: u32,       // 1 for the first frame (nothing to compare with)
  t: f32,           // frame time (s)
  fa50: f32,        // where 100 Hz (50 Hz grid) lamp flicker lands after sampling at the frame rate
  fa60: f32,        // ... and 120 Hz (60 Hz grid)
  fc: f32,          // a control frequency where no flicker lands (noise reference)
};
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var frame: texture_2d<f32>;
@group(0) @binding(2) var<storage, read_write> prev: array<f32>;          // last frame's luma
@group(0) @binding(3) var<storage, read_write> acc: array<vec4f>;         // re50, im50, re60, im60 per pixel
@group(0) @binding(4) var<storage, read_write> sums: array<atomic<u32>>;  // per frame: sum |diff| x 16
@group(0) @binding(5) var changes: texture_storage_2d<rgba8unorm, write>; // this frame's change map
@group(0) @binding(6) var<storage, read_write> sumY: array<f32>;          // per pixel: sum of luma
@group(0) @binding(7) var<storage, read_write> accC: array<vec2f>;        // re, im at the control frequency

const PI = 3.14159265358979;

@compute @workgroup_size(16, 16)
fn scanStep(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= p.size.x || gid.y >= p.size.y) { return; }
  // Average the frame pixels that fall on this working pixel (2x2 taps around its centre).
  let src = vec2f(textureDimensions(frame));
  let s = src / vec2f(p.size);
  let c = (vec2f(gid.xy) + 0.5) * s;
  var rgb = vec3f(0.0);
  for (var dy = 0; dy < 2; dy++) {
    for (var dx = 0; dx < 2; dx++) {
      let q = clamp(vec2i(c + (vec2f(f32(dx), f32(dy)) - 0.5) * s * 0.5), vec2i(0), vec2i(src) - 1);
      rgb += textureLoad(frame, q, 0).rgb;
    }
  }
  let y = dot(rgb / 4.0, vec3f(0.299, 0.587, 0.114)) * 255.0;
  let i = gid.y * p.size.x + gid.x;

  var d = 0.0;
  if (p.first == 0u) { d = abs(y - prev[i]); }
  prev[i] = y;
  atomicAdd(&sums[p.frame], u32(d * 16.0));
  textureStore(changes, vec2i(gid.xy), vec4f(min(1.0, d / 40.0), min(1.0, d / 120.0), 0.15 * min(1.0, d / 10.0), 1.0));

  // Single-frequency Fourier sums at the two aliased flicker frequencies (real frame times).
  let a50 = 2.0 * PI * p.fa50 * p.t;
  let a60 = 2.0 * PI * p.fa60 * p.t;
  acc[i] += vec4f(y * cos(a50), y * sin(a50), y * cos(a60), y * sin(a60));
  sumY[i] += y;
  let ac = 2.0 * PI * p.fc * p.t;
  accC[i] += vec2f(y * cos(ac), y * sin(ac));
}
`
