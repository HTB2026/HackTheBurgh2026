import { createContext, useContext, useSyncExternalStore } from 'react'

/**
 * Shared playhead. The video element writes currentTime here (rAF-throttled);
 * the timeline and charts subscribe. Seeks go back through `seek`.
 */
export class Playback {
  time = 0
  duration = 0
  playing = false
  private listeners = new Set<() => void>()
  private seeker: ((t: number) => void) | null = null

  subscribe = (fn: () => void) => {
    this.listeners.add(fn)
    return () => this.listeners.delete(fn)
  }
  set(partial: Partial<Pick<Playback, 'time' | 'duration' | 'playing'>>) {
    Object.assign(this, partial)
    this.listeners.forEach((l) => l())
  }
  attach(seeker: ((t: number) => void) | null) {
    this.seeker = seeker
  }
  seek = (t: number) => {
    const clamped = Math.max(0, this.duration ? Math.min(this.duration, t) : t)
    this.set({ time: clamped })
    this.seeker?.(clamped)
  }
}

export const PlaybackContext = createContext<Playback>(new Playback())
export const usePlayback = () => useContext(PlaybackContext)

export function usePlayhead(): number {
  const pb = usePlayback()
  return useSyncExternalStore(pb.subscribe, () => pb.time)
}

export function useIsPlaying(): boolean {
  const pb = usePlayback()
  return useSyncExternalStore(pb.subscribe, () => pb.playing)
}
