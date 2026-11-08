/** Estimates this broad do not identify a location, as in the case brief. */
export function hasLocation(lat: unknown, lon: unknown, radiusKm?: unknown): boolean {
  return typeof lat === 'number' && Number.isFinite(lat) && Math.abs(lat) <= 90 &&
    typeof lon === 'number' && Number.isFinite(lon) && Math.abs(lon) <= 180 &&
    (radiusKm == null || (typeof radiusKm === 'number' && Number.isFinite(radiusKm) && radiusKm >= 0 && radiusKm < 2500))
}

/**
 * Google Maps link: a pin on the exact coordinates (so Maps shows them), zoomed to fit the
 * uncertainty circle for wide estimates and close in for exact points.
 */
export function googleMapsUrl(lat: number, lon: number, radiusM?: number | null): string {
  const ll = `${lat.toFixed(6)},${lon.toFixed(6)}`
  const zoom = !radiusM || radiusM < 500 ? 17 : Math.max(3, Math.min(17, Math.round(14 - Math.log2(radiusM / 1000))))
  return `https://www.google.com/maps/place/${ll}/@${ll},${zoom}z`
}
