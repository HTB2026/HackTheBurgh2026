import { createContext, useContext, useSyncExternalStore } from 'react'
import type { Box } from './types'

/**
 * Shared "region of interest" for still pictures, the photo counterpart of the
 * playhead: findings point at a box, the viewer highlights and zooms to it.
 */
export interface FocusSnapshot {
  box: Box | null
  label: string
  /** bumps on every focus so the viewer re-animates even for the same box */
  nonce: number
}

export class RegionFocus {
  /** one immutable snapshot per focus: useSyncExternalStore needs a stable reference */
  snapshot: FocusSnapshot = { box: null, label: '', nonce: 0 }
  private listeners = new Set<() => void>()

  get box() {
    return this.snapshot.box
  }
  subscribe = (fn: () => void) => {
    this.listeners.add(fn)
    return () => this.listeners.delete(fn)
  }
  focus = (box: Box | null, label = '') => {
    this.snapshot = { box, label, nonce: this.snapshot.nonce + 1 }
    this.listeners.forEach((l) => l())
    if (box) document.getElementById('photo-stage')?.scrollIntoView({ behavior: 'smooth', block: 'nearest' })
  }
}

export const RegionContext = createContext<RegionFocus>(new RegionFocus())
export const useRegionFocus = () => useContext(RegionContext)

export function useFocusedRegion(): FocusSnapshot {
  const rf = useRegionFocus()
  return useSyncExternalStore(rf.subscribe, () => rf.snapshot, () => rf.snapshot)
}

/** Context for which kind of evidence the page shows, so shared widgets pick the right words. */
export const MediaKindContext = createContext<'video' | 'image' | 'audio'>('video')
export const useMediaKind = () => useContext(MediaKindContext)
