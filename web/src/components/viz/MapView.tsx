import { useEffect, useRef, useState } from 'react'
import L from 'leaflet'
import { mockName } from '../../lib/mock'
import { mockMapLayer } from '../../lib/mockMap'

export interface MapPoint {
  lat: number
  lon: number
  label?: string
  accuracy_m?: number
  kind?: string
  /** compass direction the camera pointed, degrees from north (still photos) */
  heading_deg?: number | null
}

/** A wedge of `spread` degrees either side of `heading`, `length_m` long, as lat/lon vertices. */
function wedge(lat: number, lon: number, heading: number, length_m: number, spread = 25, steps = 12): [number, number][] {
  const out: [number, number][] = [[lat, lon]]
  const dLat = length_m / 111_320
  const dLon = length_m / (111_320 * Math.max(0.2, Math.cos((lat * Math.PI) / 180)))
  for (let i = 0; i <= steps; i++) {
    const a = ((heading - spread + (2 * spread * i) / steps) * Math.PI) / 180
    out.push([lat + dLat * Math.cos(a), lon + dLon * Math.sin(a)])
  }
  return out
}

const esc = (s: string) => s.replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]!)

export const isLead = (p: MapPoint) => p.kind === 'ai-candidate'
export const isAI = (p: MapPoint) => p.kind === 'ai' || p.kind === 'inferred'

/** "±35 m" / "±1.2 km" / "±350 km". */
export function radiusText(m: number | undefined): string {
  if (!m) return ''
  if (m < 1000) return `±${Math.round(m)} m`
  const km = m / 1000
  return `±${km < 10 ? km.toFixed(1) : Math.round(km).toLocaleString('en-GB')} km`
}

export function distanceText(km: number): string {
  if (km < 1) return `${Math.round(km * 1000)} m`
  return `${km < 10 ? km.toFixed(1) : Math.round(km).toLocaleString('en-GB')} km`
}

/**
 * Dark street map (OpenStreetMap, darkened in CSS) with a Satellite option (Esri).
 * Red pins: measured GPS. Purple, dashed: AI estimates. A line joins GPS and AI estimate.
 */
export function MapView({ points, height = 360 }: { points: MapPoint[]; height?: number }) {
  const el = useRef<HTMLDivElement>(null)
  const map = useRef<L.Map | null>(null)
  const layer = useRef<L.LayerGroup | null>(null)
  const [offline, setOffline] = useState(false)
  const pts = points.filter((p) => isFinite(p?.lat) && isFinite(p?.lon))

  useEffect(() => {
    if (!el.current) return
    const m = L.map(el.current, { zoomControl: true, attributionControl: true, worldCopyJump: true, scrollWheelZoom: false })
    const preview = !!mockName()
    const street = preview ? mockMapLayer() : L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
      attribution: '&copy; OpenStreetMap contributors',
      maxZoom: 19,
      className: 'tiles-dark',
    })
    const satellite = preview ? mockMapLayer(true) : L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
      attribution: 'Tiles &copy; Esri &mdash; Source: Esri, Maxar, Earthstar Geographics, and the GIS User Community',
      maxZoom: 19,
    })
    street.addTo(m)
    L.control.layers({ 'Street map': street, Satellite: satellite }, undefined, { position: 'topright', collapsed: false }).addTo(m)

    // Offline / blocked tiles: show a plain grid and a note instead of a broken map.
    let loaded = 0
    let failed = 0
    const check = () => setOffline(failed >= 3 && loaded === 0)
    for (const t of [street, satellite]) {
      t.on('tileload', () => {
        loaded++
        check()
      })
      t.on('tileerror', () => {
        failed++
        check()
      })
    }
    m.on('baselayerchange', () => {
      loaded = 0
      failed = 0
      setOffline(false)
    })

    map.current = m
    layer.current = L.layerGroup().addTo(m)
    const ro = new ResizeObserver(() => m.invalidateSize())
    ro.observe(el.current)
    return () => {
      ro.disconnect()
      m.remove()
      map.current = null
    }
  }, [])

  const key = JSON.stringify(pts)
  useEffect(() => {
    const m = map.current
    const g = layer.current
    if (!m || !g) return
    g.clearLayers()
    if (!pts.length) {
      m.setView([30, 0], 2)
      return
    }
    const bounds = L.latLngBounds([])
    for (const p of pts) {
      const ll = L.latLng(p.lat, p.lon)
      if (isLead(p)) {
        // other candidate places: small grey dots, no circle
        L.circleMarker(ll, { radius: 5, color: '#9a9a9a', weight: 1.5, fillColor: '#6b6b6b', fillOpacity: 0.9 })
          .bindTooltip(esc(p.label ?? 'Lead'), { direction: 'top' })
          .addTo(g)
        bounds.extend(ll)
        continue
      }
      const ai = isAI(p)
      const color = ai ? '#c77dea' : '#ff4040'
      if (p.accuracy_m) {
        L.circle(ll, { radius: p.accuracy_m, color, weight: 1.5, fillColor: color, fillOpacity: 0.1, dashArray: ai ? '6 6' : undefined }).addTo(g)
        // very large AI radii would zoom out to a continent; only fit to them when there is nothing better
        if (!ai || pts.length === 1 || p.accuracy_m < 50_000) bounds.extend(ll.toBounds(p.accuracy_m * 2))
      }
      bounds.extend(ll)
      if (typeof p.heading_deg === 'number' && isFinite(p.heading_deg) && !ai) {
        const length = Math.max(60, Math.min(400, (p.accuracy_m ?? 30) * 4))
        L.polygon(wedge(p.lat, p.lon, p.heading_deg, length), { color: '#ff4040', weight: 1, fillColor: '#ff4040', fillOpacity: 0.25 })
          .bindTooltip(`Camera pointed ${Math.round(p.heading_deg)}°: the scene lies this way`, { direction: 'top' })
          .addTo(g)
      }
      L.marker(ll, {
        icon: L.divIcon({ className: '', html: `<div class="map-pin ${ai ? 'ai' : ''}"></div>`, iconSize: [18, 18], iconAnchor: [9, 9] }),
      })
        .bindPopup(
          `<b>${esc(p.label ?? 'Point')}</b><br/>${p.lat.toFixed(5)}, ${p.lon.toFixed(5)}${p.accuracy_m ? `<br/>within ${radiusText(p.accuracy_m).slice(1)}` : ''}<br/>${ai ? 'AI estimate' : 'Measured'}`,
        )
        .bindTooltip(esc(p.label ?? ''), { direction: 'top', offset: [0, -10] })
        .addTo(g)
    }
    // "How wrong was the AI": a line from each measured point to each AI estimate.
    const measured = pts.filter((p) => !isAI(p) && !isLead(p))
    const guesses = pts.filter(isAI)
    for (const a of measured)
      for (const b of guesses) {
        const km = m.distance([a.lat, a.lon], [b.lat, b.lon]) / 1000
        L.polyline(
          [
            [a.lat, a.lon],
            [b.lat, b.lon],
          ],
          { color: '#fbbd08', weight: 2, dashArray: '2 6', opacity: 0.9 },
        )
          .bindTooltip(`GPS ↔ AI estimate: ${distanceText(km)}`, { permanent: true, direction: 'center', className: 'dist-label' })
          .addTo(g)
      }
    m.fitBounds(bounds.pad(0.25), { maxZoom: 16 })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key])

  return (
    <div className={`map-wrap ${offline ? 'offline' : ''}`} style={{ height }}>
      <div className="map" ref={el} style={{ height }} />
      {offline && (
        <div className="map-offline">
          <i className="fa-solid fa-wifi" aria-hidden /> Map tiles unavailable offline; coordinates below are still exact.
        </div>
      )}
    </div>
  )
}
