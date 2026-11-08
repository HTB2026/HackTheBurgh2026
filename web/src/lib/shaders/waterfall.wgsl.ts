/** WGSL for the 3D spectrogram terrain (components/viz/Waterfall.tsx). */
export const WATERFALL = /* wgsl */ `
struct U {
  viewProj: mat4x4f,
  grid: vec2f,        // vertices along time, along frequency
  width: f32,         // world size along time
  depth: f32,         // world size along frequency
  amp: f32,           // world height of the loudest bin
  playU: f32,         // playhead position 0..1 (negative: hidden)
  hoverU: f32,        // hover crosshair (negative: hidden)
  hoverV: f32,
  light: vec4f,
};
@group(0) @binding(0) var spec: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var<uniform> u: U;

struct Out {
  @builtin(position) pos: vec4f,
  @location(0) uv: vec2f,
  @location(1) normal: vec3f,
  @location(2) height: f32,
};

// The spectrogram uses the magma palette, whose lightness rises steadily: lightness ~ loudness.
fn level(uv: vec2f) -> f32 {
  let c = textureSampleLevel(spec, samp, uv, 0.0).rgb;
  return pow(clamp(dot(c, vec3f(0.2126, 0.7152, 0.0722)) * 1.15, 0.0, 1.0), 1.4);
}

fn world(uv: vec2f) -> vec3f {
  // x: time left to right; z: pitch, low frequencies (bottom of the image) nearest the viewer.
  return vec3f((uv.x - 0.5) * u.width, level(uv) * u.amp, (uv.y - 0.5) * u.depth);
}

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> Out {
  let nx = u32(u.grid.x);
  let quad = vi / 6u;
  let corner = vi % 6u;
  let qx = quad % (nx - 1u);
  let qy = quad / (nx - 1u);
  var off = array<vec2u, 6>(vec2u(0u, 0u), vec2u(1u, 0u), vec2u(0u, 1u), vec2u(0u, 1u), vec2u(1u, 0u), vec2u(1u, 1u));
  let g = vec2f(f32(qx + off[corner].x), f32(qy + off[corner].y));
  let uv = g / (u.grid - 1.0);
  let p = world(uv);
  let e = 1.0 / (u.grid - 1.0);
  let dx = world(uv + vec2f(e.x, 0.0)) - world(uv - vec2f(e.x, 0.0));
  let dz = world(uv + vec2f(0.0, e.y)) - world(uv - vec2f(0.0, e.y));
  var o: Out;
  o.pos = u.viewProj * vec4f(p, 1.0);
  o.uv = uv;
  o.normal = normalize(cross(dz, dx));
  o.height = level(uv);
  return o;
}

@fragment
fn fs(i: Out) -> @location(0) vec4f {
  let base = textureSampleLevel(spec, samp, i.uv, 0.0).rgb;
  let n = normalize(i.normal);
  let diffuse = max(dot(n, normalize(u.light.xyz)), 0.0);
  var c = base * (0.35 + 0.75 * diffuse) + vec3f(0.05) * pow(i.height, 3.0);
  // Faint grid every tenth of the range, so the shape reads in 3D.
  let gl = abs(fract(i.uv * 10.0 + 0.5) - 0.5) / fwidth(i.uv * 10.0);
  c += vec3f(0.06) * (1.0 - clamp(min(gl.x, gl.y), 0.0, 1.0));
  if (u.playU >= 0.0) {
    let d = abs(i.uv.x - u.playU) / fwidth(i.uv.x);
    c = mix(c, vec3f(1.0, 0.25, 0.25), 0.85 * (1.0 - clamp(d - 1.0, 0.0, 1.0)));
  }
  if (u.hoverU >= 0.0) {
    let dx = abs(i.uv.x - u.hoverU) / fwidth(i.uv.x);
    let dz = abs(i.uv.y - u.hoverV) / fwidth(i.uv.y);
    c = mix(c, vec3f(0.85), 0.6 * (1.0 - clamp(min(dx, dz) - 0.5, 0.0, 1.0)));
  }
  return vec4f(c, 1.0);
}
`
