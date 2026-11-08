/** Estimates this broad do not identify a location, as in the case brief. */
export function hasLocation(lat: unknown, lon: unknown, radiusKm?: unknown): boolean {
  return typeof lat === 'number' && Number.isFinite(lat) && Math.abs(lat) <= 90 &&
    typeof lon === 'number' && Number.isFinite(lon) && Math.abs(lon) <= 180 &&
    (radiusKm == null || (typeof radiusKm === 'number' && Number.isFinite(radiusKm) && radiusKm >= 0 && radiusKm < 2500))
}
