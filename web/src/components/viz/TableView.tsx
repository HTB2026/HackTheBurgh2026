import type { Artifact } from '../../lib/types'
import { usePlayback } from '../../lib/playback'
import { mmss } from '../../lib/format'

const isNum = (v: unknown) => typeof v === 'number' || (typeof v === 'string' && /^-?\d+(\.\d+)?$/.test(v))

export function TableView({ a, max = 400 }: { a: Artifact; max?: number }) {
  const pb = usePlayback()
  const rows: unknown[][] = a.data?.rows ?? []
  const columns: string[] | undefined = a.data?.columns
  if (!rows.length) return <div className="faint small">Nothing was recorded here.</div>
  const timeCol = columns?.findIndex((c) => /\(s\)|time|start/i.test(c)) ?? -1
  return (
    <div className="table-scroll">
      <table className="dtable">
        {columns && (
          <thead>
            <tr>
              {columns.map((c, i) => (
                <th key={i} className={rows.every((r) => r[i] == null || isNum(r[i])) ? 'num' : ''}>
                  {c}
                </th>
              ))}
            </tr>
          </thead>
        )}
        <tbody>
          {rows.slice(0, max).map((r, i) => (
            <tr key={i}>
              {(Array.isArray(r) ? r : [r]).map((v, j) => {
                if (j === timeCol && typeof v === 'number')
                  return (
                    <td key={j} className="num">
                      <button className="link mono" onClick={() => pb.seek(v)} data-tip="Jump the video here">
                        {mmss(v)} <span className="faint">({v})</span>
                      </button>
                    </td>
                  )
                const text = v == null || v === '' ? '—' : typeof v === 'object' ? JSON.stringify(v) : String(v)
                return (
                  <td key={j} className={`${isNum(v) ? 'num mono' : ''} ${!columns && j === 0 ? 'key' : ''} ${v == null || v === '' ? 'faint' : ''}`}>
                    {text}
                  </td>
                )
              })}
            </tr>
          ))}
        </tbody>
      </table>
      {rows.length > max && <div className="faint small">Showing {max} of {rows.length} rows.</div>}
    </div>
  )
}
