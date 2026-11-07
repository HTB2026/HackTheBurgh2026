import { useEffect, useLayoutEffect, useRef, useState } from 'react'

export function useElementSize<T extends HTMLElement>() {
  const ref = useRef<T | null>(null)
  const [size, setSize] = useState({ width: 0, height: 0 })
  useLayoutEffect(() => {
    const el = ref.current
    if (!el) return
    const ro = new ResizeObserver(([entry]) => {
      const r = entry.contentRect
      setSize((s) => (Math.abs(s.width - r.width) < 0.5 && Math.abs(s.height - r.height) < 0.5 ? s : { width: r.width, height: r.height }))
    })
    ro.observe(el)
    return () => ro.disconnect()
  }, [])
  return [ref, size] as const
}

/** Re-render every `ms` (live clocks, elapsed counters). */
export function useTick(ms = 1000, enabled = true) {
  const [, set] = useState(0)
  useEffect(() => {
    if (!enabled) return
    const id = setInterval(() => set((n) => n + 1), ms)
    return () => clearInterval(id)
  }, [ms, enabled])
}

/** Reveal `text` character by character when `animate` is true. */
export function useTypewriter(text: string, animate: boolean, cps = 90) {
  const [n, setN] = useState(animate ? 0 : text.length)
  useEffect(() => {
    if (!animate || matchMedia('(prefers-reduced-motion: reduce)').matches) {
      setN(text.length)
      return
    }
    setN(0)
    const start = performance.now()
    let raf = 0
    const step = () => {
      const k = Math.min(text.length, Math.floor(((performance.now() - start) / 1000) * cps))
      setN(k)
      if (k < text.length) raf = requestAnimationFrame(step)
    }
    raf = requestAnimationFrame(step)
    return () => cancelAnimationFrame(raf)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [text, animate])
  return text.slice(0, n)
}
