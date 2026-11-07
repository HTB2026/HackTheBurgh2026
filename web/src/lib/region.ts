import { createContext, useContext, useSyncExternalStore } from 'react'
import type { Box } from './types'

/**
 * Shared "region of interest" for still pictures, the photo counterpart of the
 * playhead: findings point at a box, the viewer highlights and zooms to it.
 */
export class RegionFocus {
  box: Box | null = null
  label = ''
  /** bumps on every focus so the viewer re-animates even for the same box */
  nonce = 0
  private listeners = new Set<() => void>()

  subscribe = (fn: () => void) => {
    this.listeners.add(fn)
    return () => this.listeners.delete(fn)
  }
  focus = (box: Box | null, label = '') => {
    this.box = box
    this.label = label
    this.nonce++
    this.listeners.forEach((l) => l())
    if (box) document.getElementById('photo-stage')?.scrollIntoView({ behavior: 'smooth', block: 'nearest' })
  }
}

export const RegionContext = createContext<RegionFocus>(new RegionFocus())
export const useRegionFocus = () => useContext(RegionContext)

export function useFocusedRegion(): { box: Box | null; label: string; nonce: number } {
  const rf = useRegionFocus()
  return useSyncExternalStore(rf.subscribe, () => ({ box: rf.box, label: rf.label, nonce: rf.nonce }), () => ({ box: rf.box, label: rf.label, nonce: rf.nonce }))
}

/** Context for which kind of evidence the page shows, so shared widgets pick the right words. */
export const MediaKindContext = createContext<'video' | 'image' | 'audio'>('video')
export const useMediaKind = () => useContext(MediaKindContext)
