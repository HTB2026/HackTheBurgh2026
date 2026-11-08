/** WGSL for the geolocation globe (components/viz/Globe.tsx). */

const COMMON = /* wgsl */ `
struct U {
  viewProj: mat4x4f,
  eye: vec4f,          // camera position (w: distance from centre)
  info: vec4f,         // x: number of points, y: has texture, z: time (s)
};
struct Pt { v: vec4f };  // lat (rad), lon (rad), radius (rad on the sphere), kind
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read> pts: array<Pt>;

const PI = 3.14159265358979;

fn onSphere(lat: f32, lon: f32) -> vec3f {
  return vec3f(cos(lat) * sin(lon), sin(lat), cos(lat) * cos(lon));
}

// kind: 0 = GPS (red), 1 = AI answer (purple), 2 = lead (grey), 3 = other measured (red)
fn kindColour(k: f32) -> vec3f {
  if (k < 0.5) { return vec3f(1.0, 0.25, 0.25); }
  if (k < 1.5) { return vec3f(0.78, 0.36, 1.0); }
  if (k < 2.5) { return vec3f(0.75, 0.75, 0.75); }
  return vec3f(1.0, 0.35, 0.3);
}
`

export const GLOBE = /* wgsl */ `
${COMMON}
@group(0) @binding(2) var earth: texture_2d<f32>;
@group(0) @binding(3) var samp: sampler;

struct Out { @builtin(position) pos: vec4f, @location(0) p: vec3f };

const SEG_LON = 192u;
const SEG_LAT = 96u;

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> Out {
  let quad = vi / 6u;
  let corner = vi % 6u;
  var off = array<vec2u, 6>(vec2u(0u, 0u), vec2u(1u, 0u), vec2u(0u, 1u), vec2u(0u, 1u), vec2u(1u, 0u), vec2u(1u, 1u));
  let g = vec2u(quad % SEG_LON, quad / SEG_LON) + off[corner];
  let lon = f32(g.x) / f32(SEG_LON) * 2.0 * PI - PI;
  let lat = f32(g.y) / f32(SEG_LAT) * PI - PI / 2.0;
  let p = onSphere(lat, lon);
  var o: Out;
  o.pos = u.viewProj * vec4f(p, 1.0);
  o.p = p;
  return o;
}

@fragment
fn fs(i: Out) -> @location(0) vec4f {
  let n = normalize(i.p);
  let lat = asin(clamp(n.y, -1.0, 1.0));
  let lon = atan2(n.x, n.z);
  var c = vec3f(0.07, 0.08, 0.11);
  if (u.info.y > 0.5 && abs(lat) < 1.4844) {
    // Web Mercator tiles: latitude is not linear in v.
    let uvx = (lon + PI) / (2.0 * PI);
    let uvy = 0.5 - log(tan(PI / 4.0 + lat / 2.0)) / (2.0 * PI);
    c = textureSampleLevel(earth, samp, vec2f(uvx, uvy), 0.0).rgb;
  }
  // Graticule every 30 degrees, faint.
  let gl = vec2f(lat, lon) * (6.0 / PI);
  let gw = abs(fract(gl + 0.5) - 0.5) / fwidth(gl);
  c = mix(c, vec3f(0.5, 0.55, 0.65), 0.18 * (1.0 - clamp(min(gw.x, gw.y), 0.0, 1.0)));

  // Light from over the viewer's shoulder, plus a cool rim.
  let view = normalize(u.eye.xyz - i.p);
  let light = normalize(u.eye.xyz + vec3f(1.5, 2.0, 0.0));
  c *= 0.45 + 0.75 * max(dot(n, light), 0.0);
  c += vec3f(0.12, 0.25, 0.5) * pow(1.0 - max(dot(n, view), 0.0), 3.0);

  // Uncertainty caps and pins, drawn on the surface.
  let count = u32(u.info.x);
  // Pin size follows the height above the surface, so pins stay the same size on screen.
  let pinR = 0.018 * (u.eye.w - 1.0) + 0.00004;
  for (var k = 0u; k < count; k++) {
    let q = pts[k].v;
    let centre = onSphere(q.x, q.y);
    // Angle between the two points; asin of the chord stays precise for tiny angles (acos does not).
    let d = 2.0 * asin(clamp(length(n - centre) * 0.5, 0.0, 1.0));
    let col = kindColour(q.w);
    let r = max(q.z, 1e-6);
    let fw = max(fwidth(d), 1e-5);
    if (q.z > pinR * 1.5) {
      let inside = 1.0 - smoothstep(r - fw, r + fw, d);
      c = mix(c, col, 0.16 * inside);
      let ring = 1.0 - clamp(abs(d - r) / (1.6 * fw), 0.0, 1.0);
      let dash = select(1.0, step(0.5, fract(atan2(dot(cross(centre, n), vec3f(0.0, 1.0, 0.0)), dot(n, vec3f(0.0, 1.0, 0.0))) * 12.0)), q.w > 0.5 && q.w < 1.5);
      c = mix(c, col, 0.85 * ring * dash);
    }
    // A guess vaguer than ~10,000 km has no meaningful centre: show only its tint, no pin.
    if (q.z > 1.5) { continue; }
    let pulse = 0.5 + 0.5 * sin(u.info.z * 3.0 + f32(k));
    let pin = 1.0 - smoothstep(pinR * 0.55, pinR, d);
    let glow = (1.0 - smoothstep(pinR, pinR * (2.2 + pulse), d)) * 0.5;
    c = mix(c, col, glow);
    c = mix(c, mix(col, vec3f(1.0), 0.35), pin);
  }
  return vec4f(c, 1.0);
}
`

/** Great-circle arcs lifted off the surface, as line segments. */
export const ARCS = /* wgsl */ `
${COMMON}
struct Arc { v: vec4f, colour: vec4f };  // from lat/lon, to lat/lon; rgb + lift
@group(0) @binding(2) var<storage, read> arcs: array<Arc>;

struct Out { @builtin(position) pos: vec4f, @location(0) colour: vec4f };
const SEGS = 64u;

fn slerp(a: vec3f, b: vec3f, t: f32) -> vec3f {
  let w = acos(clamp(dot(a, b), -1.0, 1.0));
  if (w < 1e-4) { return a; }
  return (sin((1.0 - t) * w) * a + sin(t * w) * b) / sin(w);
}

@vertex
fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> Out {
  let a = arcs[ii];
  let seg = vi / 2u + (vi % 2u);
  let t = f32(seg) / f32(SEGS);
  let p0 = onSphere(a.v.x, a.v.y);
  let p1 = onSphere(a.v.z, a.v.w);
  let span = acos(clamp(dot(p0, p1), -1.0, 1.0));
  let lift = 1.0 + 0.004 + a.colour.w * span * sin(PI * t);
  let p = slerp(p0, p1, t) * lift;
  var o: Out;
  o.pos = u.viewProj * vec4f(p, 1.0);
  let flow = 0.8 + 0.2 * sin(t * 18.0 - u.info.z * 4.0);
  o.colour = vec4f(a.colour.rgb * flow, 1.0);
  return o;
}

@fragment
fn fs(i: Out) -> @location(0) vec4f { return i.colour; }
`
