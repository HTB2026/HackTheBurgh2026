/** WGSL for multi-frame stacking of a video region (lib/stackFrames.ts). */
export const STACK = /* wgsl */ `
struct P {
  osize: vec2u,     // enlarged crop size (pixels)
  shift: vec2i,     // integer shift applied when accumulating
  region: vec4f,    // x, y, w, h as fractions of the frame
  radius: i32,      // alignment search radius (enlarged pixels)
  frames: f32,      // frames accumulated so far (for resolve)
  sharpen: f32,
  _pad: f32,
};
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var frame: texture_2d<f32>;
@group(0) @binding(2) var samp: sampler;
@group(0) @binding(3) var<storage, read_write> cur: array<vec4f>;     // this frame's enlarged crop
@group(0) @binding(4) var<storage, read_write> refBuf: array<vec4f>;     // the reference frame's crop
@group(0) @binding(5) var<storage, read_write> acc: array<vec4f>;     // running sum (aligned)
@group(0) @binding(6) var<storage, read_write> score: array<f32>;     // per candidate shift: mean |diff|
@group(0) @binding(7) var outTex: texture_storage_2d<rgba8unorm, write>;

@compute @workgroup_size(16, 16)
fn crop(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= p.osize.x || gid.y >= p.osize.y) { return; }
  let uv = p.region.xy + (vec2f(gid.xy) + 0.5) / vec2f(p.osize) * p.region.zw;
  cur[gid.y * p.osize.x + gid.x] = vec4f(textureSampleLevel(frame, samp, uv, 0.0).rgb, 1.0);
}

// One invocation per candidate shift: how different is this frame, shifted, from the reference?
@compute @workgroup_size(64)
fn alignShift(@builtin(global_invocation_id) gid: vec3u) {
  let side = u32(2 * p.radius + 1);
  if (gid.x >= side * side) { return; }
  let dx = i32(gid.x % side) - p.radius;
  let dy = i32(gid.x / side) - p.radius;
  let w = i32(p.osize.x);
  let h = i32(p.osize.y);
  var sum = 0.0;
  var n = 0.0;
  // Compare on a sparse grid inside the margin, enough to rank shifts.
  for (var y = p.radius; y < h - p.radius; y += 2) {
    for (var x = p.radius; x < w - p.radius; x += 2) {
      let a = refBuf[y * w + x].rgb;
      let b = cur[(y + dy) * w + (x + dx)].rgb;
      sum += dot(abs(a - b), vec3f(0.299, 0.587, 0.114));
      n += 1.0;
    }
  }
  score[gid.x] = sum / max(n, 1.0);
}

@compute @workgroup_size(16, 16)
fn accumulate(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= p.osize.x || gid.y >= p.osize.y) { return; }
  let q = clamp(vec2i(gid.xy) + p.shift, vec2i(0), vec2i(p.osize) - 1);
  acc[gid.y * p.osize.x + gid.x] += cur[u32(q.y) * p.osize.x + u32(q.x)];
}

@compute @workgroup_size(16, 16)
fn setRef(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= p.osize.x || gid.y >= p.osize.y) { return; }
  let i = gid.y * p.osize.x + gid.x;
  refBuf[i] = cur[i];
  acc[i] = cur[i];
}

fn avg(x: i32, y: i32) -> vec3f {
  let q = clamp(vec2i(x, y), vec2i(0), vec2i(p.osize) - 1);
  return acc[u32(q.y) * p.osize.x + u32(q.x)].rgb / p.frames;
}

// Average, then a light unsharp mask (the stack is softer than one frame, but cleaner).
@compute @workgroup_size(16, 16)
fn resolve(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= p.osize.x || gid.y >= p.osize.y) { return; }
  let x = i32(gid.x);
  let y = i32(gid.y);
  let c = avg(x, y);
  let blur = (avg(x - 2, y) + avg(x + 2, y) + avg(x, y - 2) + avg(x, y + 2)) * 0.25;
  let v = clamp(c + p.sharpen * (c - blur), vec3f(0.0), vec3f(1.0));
  textureStore(outTex, vec2i(gid.xy), vec4f(v, 1.0));
}
`
