/** Minimal column-major 4×4 matrix maths for the WebGPU views (matches WGSL mat4x4f). */
export type Mat4 = Float32Array
export type Vec3 = [number, number, number]

export function perspective(fovY: number, aspect: number, near: number, far: number): Mat4 {
  const f = 1 / Math.tan(fovY / 2)
  const nf = 1 / (near - far)
  // WebGPU clip space: z in [0, 1].
  return new Float32Array([f / aspect, 0, 0, 0, 0, f, 0, 0, 0, 0, far * nf, -1, 0, 0, far * near * nf, 0])
}

export function lookAt(eye: Vec3, target: Vec3, up: Vec3 = [0, 1, 0]): Mat4 {
  const z = norm(sub(eye, target))
  const x = norm(cross(up, z))
  const y = cross(z, x)
  return new Float32Array([x[0], y[0], z[0], 0, x[1], y[1], z[1], 0, x[2], y[2], z[2], 0, -dot(x, eye), -dot(y, eye), -dot(z, eye), 1])
}

export function multiply(a: Mat4, b: Mat4): Mat4 {
  const out = new Float32Array(16)
  for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++) {
    let s = 0
    for (let k = 0; k < 4; k++) s += a[k * 4 + r] * b[c * 4 + k]
    out[c * 4 + r] = s
  }
  return out
}

/** Inverse of a general 4×4 matrix (for picking rays). */
export function invert(m: Mat4): Mat4 {
  const inv = new Float32Array(16)
  const [a00, a01, a02, a03, a10, a11, a12, a13, a20, a21, a22, a23, a30, a31, a32, a33] = m
  const b00 = a00 * a11 - a01 * a10, b01 = a00 * a12 - a02 * a10, b02 = a00 * a13 - a03 * a10
  const b03 = a01 * a12 - a02 * a11, b04 = a01 * a13 - a03 * a11, b05 = a02 * a13 - a03 * a12
  const b06 = a20 * a31 - a21 * a30, b07 = a20 * a32 - a22 * a30, b08 = a20 * a33 - a23 * a30
  const b09 = a21 * a32 - a22 * a31, b10 = a21 * a33 - a23 * a31, b11 = a22 * a33 - a23 * a32
  const det = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06 || 1e-12
  const d = 1 / det
  inv[0] = (a11 * b11 - a12 * b10 + a13 * b09) * d
  inv[1] = (a02 * b10 - a01 * b11 - a03 * b09) * d
  inv[2] = (a31 * b05 - a32 * b04 + a33 * b03) * d
  inv[3] = (a22 * b04 - a21 * b05 - a23 * b03) * d
  inv[4] = (a12 * b08 - a10 * b11 - a13 * b07) * d
  inv[5] = (a00 * b11 - a02 * b08 + a03 * b07) * d
  inv[6] = (a32 * b02 - a30 * b05 - a33 * b01) * d
  inv[7] = (a20 * b05 - a22 * b02 + a23 * b01) * d
  inv[8] = (a10 * b10 - a11 * b08 + a13 * b06) * d
  inv[9] = (a01 * b08 - a00 * b10 - a03 * b06) * d
  inv[10] = (a30 * b04 - a31 * b02 + a33 * b00) * d
  inv[11] = (a21 * b02 - a20 * b04 - a23 * b00) * d
  inv[12] = (a11 * b07 - a10 * b09 - a12 * b06) * d
  inv[13] = (a00 * b09 - a01 * b07 + a02 * b06) * d
  inv[14] = (a31 * b01 - a30 * b03 - a32 * b00) * d
  inv[15] = (a20 * b03 - a21 * b01 + a22 * b00) * d
  return inv
}

export function transformPoint(m: Mat4, p: Vec3): Vec3 {
  const x = m[0] * p[0] + m[4] * p[1] + m[8] * p[2] + m[12]
  const y = m[1] * p[0] + m[5] * p[1] + m[9] * p[2] + m[13]
  const z = m[2] * p[0] + m[6] * p[1] + m[10] * p[2] + m[14]
  const w = m[3] * p[0] + m[7] * p[1] + m[11] * p[2] + m[15] || 1
  return [x / w, y / w, z / w]
}

export const sub = (a: Vec3, b: Vec3): Vec3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
export const add = (a: Vec3, b: Vec3): Vec3 => [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
export const scale = (a: Vec3, s: number): Vec3 => [a[0] * s, a[1] * s, a[2] * s]
export const dot = (a: Vec3, b: Vec3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
export const cross = (a: Vec3, b: Vec3): Vec3 => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
export const norm = (a: Vec3): Vec3 => {
  const l = Math.hypot(a[0], a[1], a[2]) || 1
  return [a[0] / l, a[1] / l, a[2] / l]
}
