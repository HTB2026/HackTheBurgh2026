import { useState } from 'react'
import { useGpu } from '../../lib/gpu'
import { MapView, type MapPoint, type MapZone } from '../viz/MapView'
import { Globe } from '../viz/Globe'

/**
 * The Leaflet map, with a "Map | Globe" switch when the browser has WebGPU.
 * `focus` (a point index plus a nonce that changes per request) flies the globe to that point.
 */
export function MapOrGlobe({ points, zones, height, focus }: { points: MapPoint[]; zones?: MapZone[]; height: number; focus?: { index: number; nonce: number } }) {
  const gpuState = useGpu()
  const gpu = gpuState === 'loading' ? null : gpuState
  const [view, setView] = useState<'map' | 'globe'>('map')
  return (
    <div className="map-switchable">
      {gpu && (
        <div className="tabs map-mode" role="radiogroup" aria-label="Map view">
          {(['map', 'globe'] as const).map((m) => (
            <button
              key={m}
              type="button"
              role="radio"
              aria-checked={view === m}
              className={view === m ? 'active' : ''}
              onClick={() => setView(m)}
              data-tip={m === 'globe' ? '3D globe: flies to the answer, shows every place considered (WebGPU)' : 'Street and satellite map'}
            >
              {m === 'map' ? 'Map' : 'Globe'}
            </button>
          ))}
        </div>
      )}
      {gpu && view === 'globe' ? <Globe gpu={gpu} points={points} height={height} focus={focus ?? { index: -1, nonce: 0 }} rowsFly={!!focus} /> : <MapView points={points} zones={zones} height={height} />}
    </div>
  )
}
