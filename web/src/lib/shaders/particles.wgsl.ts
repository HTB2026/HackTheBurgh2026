/** WGSL for the homepage particle field (components/HeroField.tsx). */
const COMMON = /* wgsl */ `
struct Sim {
  dt: f32,
  time: f32,
  aspect: f32,        // canvas width / height
  envCount: f32,
  mouse: vec2f,       // x 0..1, y -1..1 (y up); x < 0 when the mouse is away
  pixel: vec2f,       // size of one device pixel in clip units
};
struct P { v: vec4f };  // x (0..1), y (-1..1), vx, vy
@group(0) @binding(0) var<uniform> sim: Sim;
`

export const PARTICLE_SIM = /* wgsl */ `
${COMMON}
@group(0) @binding(1) var<storage, read_write> parts: array<P>;
@group(0) @binding(2) var<storage, read> env: array<f32>;

fn hash(n: u32) -> f32 {
  var x = n * 747796405u + 2891336453u;
  x = ((x >> ((x >> 28u) + 4u)) ^ x) * 277803737u;
  return f32((x >> 22u) ^ x) / 4294967295.0;
}

// The waveform's loudness at x (0..1), smoothed between samples.
fn envelope(x: f32) -> f32 {
  let n = u32(sim.envCount);
  if (n == 0u) { return 0.4 + 0.25 * sin(x * 18.0 + sim.time * 0.7) * sin(x * 5.0); }
  let f = clamp(x, 0.0, 1.0) * f32(n - 1u);
  let i = u32(f);
  let j = min(i + 1u, n - 1u);
  return mix(env[i], env[j], fract(f));
}

@compute @workgroup_size(256)
fn simulate(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= arrayLength(&parts)) { return; }
  var p = parts[i].v;
  let seed = i * 9781u + u32(sim.time * 60.0);
  let speed = 0.025 + 0.05 * hash(i * 3u);

  // Drift left to right; re-enter at the left edge at a fresh height.
  p.x += speed * sim.dt;
  if (p.x > 1.0) { p.x -= 1.0; p.y = (hash(seed) * 2.0 - 1.0) * 0.9; p.w = 0.0; }

  // Pull into the band |y| <= loudness, so the cloud takes the waveform's shape.
  let h = 0.06 + 0.88 * envelope(p.x);
  let over = abs(p.y) - h;
  if (over > 0.0) { p.w -= sign(p.y) * over * 9.0 * sim.dt; }
  p.w += (hash(seed + 7u) - 0.5) * 0.9 * sim.dt;
  p.w += sin(p.x * 25.0 + sim.time * 1.3 + f32(i % 13u)) * 0.06 * sim.dt;

  // The mouse pushes particles away.
  if (sim.mouse.x >= 0.0) {
    let d = vec2f((p.x - sim.mouse.x) * sim.aspect * 0.5, p.y - sim.mouse.y);
    let r = length(d);
    if (r < 0.35 && r > 1e-4) {
      let push = (0.35 - r) * 6.0 * sim.dt;
      p.y += d.y / r * push;
      p.x += d.x / r * push / (sim.aspect * 0.5);
    }
  }
  p.w *= exp(-2.2 * sim.dt);
  p.y = clamp(p.y + p.w * sim.dt * 4.0, -1.0, 1.0);
  parts[i].v = p;
}
`

export const PARTICLE_DRAW = /* wgsl */ `
${COMMON}
@group(0) @binding(1) var<storage, read> parts: array<P>;

struct Out { @builtin(position) pos: vec4f, @location(0) uv: vec2f, @location(1) heat: f32 };

@vertex
fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> Out {
  let p = parts[ii].v;
  var corner = array<vec2f, 6>(vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0), vec2f(-1.0, 1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0));
  let c = corner[vi];
  var o: Out;
  let centre = vec2f(p.x * 2.0 - 1.0, p.y * 0.92);
  o.pos = vec4f(centre + c * sim.pixel * 1.6, 0.0, 1.0);
  o.uv = c;
  o.heat = clamp(abs(p.w) * 6.0 + 0.25, 0.0, 1.0);
  return o;
}

@fragment
fn fs(i: Out) -> @location(0) vec4f {
  let a = (1.0 - smoothstep(0.2, 1.0, length(i.uv))) * 0.55;
  // Brand red, warming to amber where particles move fast.
  let c = mix(vec3f(0.85, 0.16, 0.16), vec3f(1.0, 0.62, 0.3), i.heat);
  return vec4f(c * a, a);
}
`
