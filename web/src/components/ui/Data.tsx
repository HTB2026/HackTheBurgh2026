import { useState, type ReactNode } from 'react'
import { DATA_LABEL, unitFor } from '../../lib/plain'
import { bytes, mmss, niceDate } from '../../lib/format'
import { usePlayback } from '../../lib/playback'
import { CopyButton } from './ui'
import { googleMapsUrl } from '../../lib/geo'

const ISO = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}/

export function humanKey(k: string): string {
  if (DATA_LABEL[k]) return DATA_LABEL[k]
  const s = k.replace(/_(hz|db|ms|s|m|km|h)$/, '').replace(/_/g, ' ')
  return s.charAt(0).toUpperCase() + s.slice(1)
}

function fmtNumber(v: number, unit: string, key: string): string {
  if (key === 'presence' || key === 'coherence') return `${(v * 100).toFixed(v >= 0.995 ? 0 : 1)}%`
  if (unit === 's' && key.endsWith('time_s')) return `${mmss(v)} (${v.toFixed(2)} s)`
  const abs = Math.abs(v)
  const digits = Number.isInteger(v) ? 0 : abs >= 1000 ? 0 : abs >= 100 ? 1 : abs >= 1 ? 2 : 4
  const n = v.toLocaleString('en-GB', { maximumFractionDigits: digits, minimumFractionDigits: 0 })
  if (unit === 'Hz' && abs >= 1000) return `${(v / 1000).toLocaleString('en-GB', { maximumFractionDigits: 2 })} kHz`
  if (unit === 'm' && abs >= 10000) return `${(v / 1000).toLocaleString('en-GB', { maximumFractionDigits: 1 })} km`
  if (unit === 'h' && key.includes('offset')) return `UTC${v >= 0 ? '+' : ''}${v}`
  return unit ? `${n}${unit === '°' ? '°' : ` ${unit}`}` : n
}

/** Format a single value for humans. */
export function Value({ k, v, depth = 0 }: { k: string; v: unknown; depth?: number }): ReactNode {
  const pb = usePlayback()
  if (v === null || v === undefined || v === '') return <span className="faint">none</span>
  if ((k === 'bit_rate' || k.endsWith('_bit_rate')) && (typeof v === 'number' || (typeof v === 'string' && /^\d+$/.test(v))))
    return <span className="mono">{Math.round(Number(v) / 1000).toLocaleString('en-GB')} kbit/s</span>
  if ((k === 'size_bytes' || k === 'size') && typeof v === 'number')
    return (
      <span className="mono">
        {bytes(v)} <span className="faint">({v.toLocaleString('en-GB')} bytes)</span>
      </span>
    )
  if (k === 'sample_rate' && (typeof v === 'number' || /^\d+$/.test(String(v)))) return <span className="mono">{Number(v).toLocaleString('en-GB')} Hz</span>
  if (typeof v === 'boolean') return <span className={v ? 'yes' : 'no'}>{v ? 'Yes' : 'No'}</span>
  if (typeof v === 'number') {
    if (k.endsWith('time_s') || k === 'start_s')
      return (
        <button className="link mono" onClick={() => pb.seek(v)} data-tip="Jump the video here">
          {fmtNumber(v, unitFor(k), k)}
        </button>
      )
    return <span className="mono">{fmtNumber(v, unitFor(k), k)}</span>
  }
  if (typeof v === 'string') {
    if (ISO.test(v))
      return (
        <span>
          {niceDate(v)} <span className="faint mono">{v}</span>
        </span>
      )
    return <span className="pre">{v}</span>
  }
  if (Array.isArray(v)) {
    if (!v.length) return <span className="faint">none</span>
    if (v.every((x) => x === null || typeof x !== 'object'))
      return (
        <span className="mono">
          {k === 'box' || k === 'offset' ? v.map((x) => (typeof x === 'number' ? Math.round(x) : String(x))).join(', ') : v.map((x) => (typeof x === 'number' ? fmtNumber(x, unitFor(k), k) : String(x))).join(' · ')}
        </span>
      )
    if (v.every((x) => Array.isArray(x)))
      return (
        <ol className="compact-list mono">
          {(v as unknown[][]).slice(0, 50).map((row, i) => (
            <li key={i}>{row.map((x) => (typeof x === 'number' ? +x.toFixed(3) : String(x))).join(', ')}</li>
          ))}
          {v.length > 50 && <li className="faint">… {v.length - 50} more</li>}
        </ol>
      )
    if (v.every((x) => x && typeof x === 'object' && !Array.isArray(x))) return <ObjTable rows={v as Record<string, unknown>[]} />
    return <JsonTree value={v} name={k} />
  }
  if (typeof v === 'object') return depth > 2 ? <JsonTree value={v} name={k} /> : <DataView data={v as Record<string, unknown>} depth={depth + 1} />
  return <span>{String(v)}</span>
}

function ObjTable({ rows }: { rows: Record<string, unknown>[] }) {
  const cols = [...new Set(rows.flatMap((r) => Object.keys(r)))]
  return (
    <div className="obj-table">
      <table className="dtable">
        <thead>
          <tr>
            {cols.map((c) => (
              <th key={c}>{humanKey(c)}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.slice(0, 100).map((r, i) => (
            <tr key={i}>
              {cols.map((c) => (
                <td key={c}>
                  <Value k={c} v={r[c]} depth={3} />
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

/** Key/value table of a finding's data object. Never hides anything. */
export function DataView({ data, depth = 0 }: { data: Record<string, unknown> | null | undefined; depth?: number }) {
  if (!data || typeof data !== 'object') return null
  const keys = Object.keys(data)
  if (!keys.length) return <span className="faint">No extra measurements were recorded for this finding.</span>
  const hasCoords = typeof data.lat === 'number' && typeof data.lon === 'number'
  const latlon = hasCoords ? [data.lat as number, data.lon as number] : typeof data.latitude === 'number' && typeof data.longitude === 'number' ? [data.latitude as number, data.longitude as number] : null
  return (
    <dl className={`kv ${depth ? 'nested' : ''}`}>
      {latlon && (
        <div>
          <dt>Coordinates</dt>
          <dd>
            <span className="mono">
              {latlon[0].toFixed(5)}, {latlon[1].toFixed(5)}
            </span>{' '}
            <CopyButton text={`${latlon[0]}, ${latlon[1]}`} label="Copy coordinates" />{' '}
            <a className="link" href={googleMapsUrl(latlon[0], latlon[1], typeof data.radius_km === 'number' ? data.radius_km * 1000 : typeof data.accuracy_m === 'number' ? data.accuracy_m : null)} target="_blank" rel="noreferrer">
              Open in Google Maps
            </a>
          </dd>
        </div>
      )}
      {keys.map((k) => (
        <div key={k}>
          <dt data-tip={k}>{humanKey(k)}</dt>
          <dd>
            <Value k={k} v={data[k]} depth={depth} />
          </dd>
        </div>
      ))}
    </dl>
  )
}

function preview(v: unknown): string {
  if (Array.isArray(v)) return `[${v.length}]`
  if (v && typeof v === 'object') return `{${Object.keys(v).length}}`
  if (typeof v === 'string') return JSON.stringify(v.length > 60 ? v.slice(0, 57) + '…' : v)
  return String(v)
}

/** Collapsible JSON tree for raw dumps (ffprobe, unknown kinds). */
export function JsonTree({ value, name = 'root', open = false }: { value: unknown; name?: string; open?: boolean }) {
  const [isOpen, setOpen] = useState(open)
  const isObj = value !== null && typeof value === 'object'
  if (!isObj)
    return (
      <div className="jt-leaf">
        <span className="jt-key">{name}</span>: <span className={`jt-v jt-${typeof value}`}>{JSON.stringify(value)}</span>
      </div>
    )
  const entries = Array.isArray(value) ? value.map((v, i) => [String(i), v] as const) : Object.entries(value as Record<string, unknown>)
  return (
    <div className="jt-node">
      <button className="jt-toggle" onClick={() => setOpen((o) => !o)} aria-expanded={isOpen}>
        <i className={`fa-solid fa-caret-${isOpen ? 'down' : 'right'}`} aria-hidden />
        <span className="jt-key">{name}</span> <span className="faint">{preview(value)}</span>
      </button>
      {isOpen && (
        <div className="jt-children">
          {entries.map(([k, v]) => (
            <JsonTree key={k} name={k} value={v} />
          ))}
        </div>
      )}
    </div>
  )
}
