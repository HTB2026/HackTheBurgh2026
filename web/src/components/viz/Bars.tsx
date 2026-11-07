import type { Artifact } from '../../lib/types'
import { num } from '../../lib/format'
import { SERIES } from './SeriesChart'

export function Bars({ a }: { a: Artifact }) {
  const labels: string[] = a.data?.labels ?? []
  const values: number[] = a.data?.values ?? []
  const unit: string | undefined = a.data?.unit ?? a.data?.y_unit
  if (!labels.length) return <div className="faint small">No data.</div>
  const max = Math.max(0, ...values.map((v) => (isFinite(v) ? v : 0)))
  const min = Math.min(0, ...values.map((v) => (isFinite(v) ? v : 0)))
  const span = max - min || 1
  const zero = (-min / span) * 100
  return (
    <div className="bars">
      {labels.map((l, i) => {
        const v = values[i] ?? 0
        const w = (Math.abs(v) / span) * 100
        return (
          <div className="bar-row" key={i} data-tip={`${l}: ${num(v, unit)}`}>
            <span className="bar-label">{l}</span>
            <span className="bar-track">
              <i
                style={{
                  left: v >= 0 ? `${zero}%` : `${zero - w}%`,
                  width: `${w}%`,
                  background: SERIES[0],
                  borderRadius: v >= 0 ? '0 4px 4px 0' : '4px 0 0 4px',
                }}
              />
            </span>
            <span className="bar-val mono">{num(v, unit)}</span>
          </div>
        )
      })}
    </div>
  )
}
