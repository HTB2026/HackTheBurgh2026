import type { CSSProperties } from 'react'
import type { Effort } from '../lib/types'
import { Icon } from './ui/ui'

export const EFFORTS: Effort[] = ['low', 'medium', 'high']

export const EFFORT_LABEL: Record<Effort, string> = { low: 'Quick', medium: 'Standard', high: 'Deep' }

/** Rough wall-clock time for a short clip, shown next to each choice. */
export const EFFORT_TIME: Record<Effort, string> = { low: '~1 min', medium: '~3 min', high: '5–10 min' }

export const EFFORT_HINT: Record<Effort, string> = {
  low: 'Fewer frames and shorter AI reasoning: a fast first look. Geolocation is rougher and the pulse check is skipped.',
  medium: 'Every check, with geolocation verified against Wikipedia photos and satellite imagery.',
  high: 'Everything in Standard, plus: three independent geolocation passes, a clue inventory for places without text, ground-level photos near each candidate, OpenStreetMap features, the terrain skyline, the sun, stars and weather, and a zoom-in round on the best region.',
}

export const effortName = (e: Effort) => `${EFFORT_LABEL[e]} (${EFFORT_TIME[e]})`

/**
 * Quick / Standard / Deep as a slider, for the upload area. A native range
 * input underneath (keyboard and screen readers work); the labels are buttons.
 */
export function EffortSlider({ value, onChange, disabled }: { value: Effort; onChange: (e: Effort) => void; disabled?: boolean }) {
  const at = EFFORTS.indexOf(value)
  return (
    <div className={`effort lvl-${value}`}>
      <span className="ai-choice-label">
        <Icon name="gauge-high" /> Effort
      </span>
      <div className="effort-slider" style={{ '--pos': at / (EFFORTS.length - 1) } as CSSProperties}>
        <input
          type="range"
          min={0}
          max={EFFORTS.length - 1}
          step={1}
          value={at}
          disabled={disabled}
          aria-label="Effort"
          aria-valuetext={`${EFFORT_LABEL[value]}, ${EFFORT_TIME[value]}`}
          onChange={(e) => onChange(EFFORTS[Number(e.target.value)])}
        />
        <div className="effort-stops">
          {EFFORTS.map((e) => (
            <button key={e} type="button" className={e === value ? 'on' : ''} disabled={disabled} onClick={() => onChange(e)} tabIndex={-1}>
              <b>{EFFORT_LABEL[e]}</b>
              <small>{EFFORT_TIME[e]}</small>
            </button>
          ))}
        </div>
      </div>
      <small className="faint effort-hint">{EFFORT_HINT[value]}</small>
    </div>
  )
}

/** A compact dropdown for the case header: the effort used when the case is run again. */
export function EffortSelect({ value, onChange }: { value: Effort; onChange: (e: Effort) => void }) {
  return (
    <select
      className="input ai-picker sm"
      value={value}
      onChange={(e) => onChange(e.target.value as Effort)}
      aria-label="Effort"
      data-tip="Effort used when this case is run again"
    >
      {EFFORTS.map((e) => (
        <option key={e} value={e} title={EFFORT_HINT[e]}>
          {effortName(e)}
        </option>
      ))}
    </select>
  )
}

/** The effort the home page remembers between uploads. */
export function storedEffort(): Effort {
  try {
    const v = localStorage.getItem('earwitness.effort')
    return v === 'low' || v === 'high' ? v : 'medium'
  } catch {
    return 'medium'
  }
}
