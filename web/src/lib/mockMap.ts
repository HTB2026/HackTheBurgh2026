import L from 'leaflet'

/** Coordinate-correct grid tiles for offline UI exploration; never represented as real imagery. */
export function mockMapLayer(satellite = false): L.GridLayer {
  class Grid extends L.GridLayer {
    createTile(coords: L.Coords) {
      const canvas = document.createElement('canvas')
      canvas.width = 256; canvas.height = 256
      const ctx = canvas.getContext('2d')!
      ctx.fillStyle = satellite ? '#19292b' : '#202630'; ctx.fillRect(0, 0, 256, 256)
      ctx.strokeStyle = satellite ? '#345344' : '#3d4655'; ctx.lineWidth = 1
      for (let i = 0; i <= 256; i += 32) { ctx.beginPath(); ctx.moveTo(i, 0); ctx.lineTo(i, 256); ctx.moveTo(0, i); ctx.lineTo(256, i); ctx.stroke() }
      const n = 2 ** coords.z
      const lon = (coords.x + 0.5) / n * 360 - 180
      const lat = Math.atan(Math.sinh(Math.PI * (1 - 2 * (coords.y + 0.5) / n))) * 180 / Math.PI
      ctx.fillStyle = '#9faeb9'; ctx.font = '12px monospace'
      ctx.fillText(`${lat.toFixed(3)}°, ${lon.toFixed(3)}°`, 14, 118)
      ctx.font = '10px sans-serif'; ctx.fillStyle = '#738b91'
      ctx.fillText(satellite ? 'SATELLITE PREVIEW · NO IMAGERY' : 'LOCAL PREVIEW · COORDINATE GRID', 14, 138)
      return canvas
    }
  }
  return new Grid({ attribution: 'Synthetic preview grid · no map service contacted', maxZoom: 19 })
}
