import { useEffect, useState } from 'react'
import { api } from './api'
import type { Health } from './types'

let cached: Health | null = null
let pending: Promise<Health> | null = null

/** /api/health, fetched once per page load and shared. */
export function useHealth(): { health: Health | null; offline: boolean } {
  const [health, setHealth] = useState<Health | null>(cached)
  const [offline, setOffline] = useState(false)
  useEffect(() => {
    if (cached) return
    pending ??= api.health()
    pending
      .then((h) => {
        cached = h
        setHealth(h)
      })
      .catch(() => {
        pending = null
        setOffline(true)
      })
  }, [])
  return { health, offline }
}
