export function timecode(t: number | null | undefined, fps = 30): string {
  if (t == null || !isFinite(t)) return '--:--:--'
  const sign = t < 0 ? '-' : ''
  t = Math.abs(t)
  const m = Math.floor(t / 60)
  const s = Math.floor(t % 60)
  const f = Math.floor((t - Math.floor(t)) * fps)
  return `${sign}${pad(m)}:${pad(s)}:${pad(f)}`
}

export function seconds(t: number | null | undefined, digits = 2): string {
  if (t == null || !isFinite(t)) return '—'
  return `${t.toFixed(digits)}s`
}

export function pad(n: number, width = 2): string {
  return String(n).padStart(width, '0')
}

export function bytes(n: number | null | undefined): string {
  if (!n) return '—'
  const units = ['B', 'KB', 'MB', 'GB']
  let i = 0
  while (n >= 1024 && i < units.length - 1) {
    n /= 1024
    i++
  }
  return `${n.toFixed(n < 10 && i > 0 ? 1 : 0)} ${units[i]}`
}

export function clockUTC(d = new Date()): string {
  return `${pad(d.getUTCHours())}:${pad(d.getUTCMinutes())}:${pad(d.getUTCSeconds())}Z`
}

export function dateShort(unix: number): string {
  const d = new Date(unix * 1000)
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}`
}

/** Compact number for axis ticks and readouts. */
export function num(v: number, unit?: string): string {
  if (!isFinite(v)) return '—'
  const a = Math.abs(v)
  let s: string
  if (a >= 1e6) s = (v / 1e6).toFixed(a >= 1e7 ? 0 : 1) + 'M'
  else if (a >= 1e4) s = (v / 1e3).toFixed(a >= 1e5 ? 0 : 1) + 'k'
  else if (a >= 100) s = v.toFixed(0)
  else if (a >= 10) s = v.toFixed(1)
  else if (a >= 1) s = v.toFixed(2)
  else if (a === 0) s = '0'
  else s = v.toPrecision(2)
  return unit ? `${s} ${unit}` : s
}

/** "Nice" tick values covering [lo, hi]. */
export function niceTicks(lo: number, hi: number, count = 5): number[] {
  if (!isFinite(lo) || !isFinite(hi)) return []
  if (lo === hi) {
    lo -= 1
    hi += 1
  }
  const span = hi - lo
  const raw = span / Math.max(1, count)
  const mag = Math.pow(10, Math.floor(Math.log10(raw)))
  const norm = raw / mag
  const step = (norm >= 5 ? 10 : norm >= 2 ? 5 : norm >= 1 ? 2 : 1) * mag
  const start = Math.ceil(lo / step) * step
  const out: number[] = []
  for (let v = start; v <= hi + step * 1e-9; v += step) out.push(Math.abs(v) < step * 1e-9 ? 0 : v)
  return out
}

/** Format a set of evenly spaced ticks with just enough decimals to tell them apart. */
export function tickFormatter(ticks: number[]): (v: number) => string {
  const step = ticks.length > 1 ? Math.abs(ticks[1] - ticks[0]) : 0
  const maxAbs = Math.max(0, ...ticks.map(Math.abs))
  if (!step || maxAbs >= 1e4) return (v) => num(v)
  const decimals = Math.max(0, Math.min(6, -Math.floor(Math.log10(step) + 1e-9)))
  return (v) => v.toFixed(decimals)
}

export const SEVERITY_ORDER: Record<string, number> = { critical: 0, warning: 1, notice: 2, info: 3 }

/** Friendly clock time: 0:20, 1:05:09. */
export function mmss(t: number | null | undefined): string {
  if (t == null || !isFinite(t)) return ''
  const s = Math.max(0, Math.round(t))
  const h = Math.floor(s / 3600)
  const m = Math.floor((s % 3600) / 60)
  const sec = s % 60
  return h ? `${h}:${pad(m)}:${pad(sec)}` : `${m}:${pad(sec)}`
}

/** Remove "[F012]" citations for plain-language display. */
export const stripCites = (s: string) => s.replace(/\s*\[F\d+\](\s*\[F\d+\])*/g, '').trim()

/** "22 Jul 2024, 12:14 UTC" from an ISO string; null-safe. */
export function niceDate(iso: string | null | undefined, withTime = true): string {
  if (!iso) return ''
  const d = new Date(iso)
  if (isNaN(d.getTime())) return String(iso)
  const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
  const base = `${d.getUTCDate()} ${months[d.getUTCMonth()]} ${d.getUTCFullYear()}`
  return withTime ? `${base}, ${pad(d.getUTCHours())}:${pad(d.getUTCMinutes())} UTC` : base
}

/** "3 min ago" style relative time from unix seconds. */
export function ago(unix: number | null | undefined): string {
  if (!unix) return ''
  const s = Date.now() / 1000 - unix
  if (s < 60) return 'just now'
  if (s < 3600) return `${Math.floor(s / 60)} min ago`
  if (s < 86400) return `${Math.floor(s / 3600)} h ago`
  return `${Math.floor(s / 86400)} days ago`
}
