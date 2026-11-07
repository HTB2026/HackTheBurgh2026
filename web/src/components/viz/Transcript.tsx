import { useEffect, useRef } from 'react'
import type { Artifact } from '../../lib/types'
import { usePlayback, usePlayhead } from '../../lib/playback'
import { mmss } from '../../lib/format'

export interface Segment {
  start: number
  end: number
  text: string
}

export function segmentsOf(a: Artifact): Segment[] {
  return ((a.data?.segments ?? []) as Segment[]).filter((s) => s && typeof s.start === 'number')
}

export function Transcript({ a }: { a: Artifact }) {
  const segs = segmentsOf(a)
  const t = usePlayhead()
  const pb = usePlayback()
  const list = useRef<HTMLOListElement>(null)
  const active = segs.findIndex((s) => t >= s.start && t < s.end)
  useEffect(() => {
    const box = list.current
    const el = active >= 0 ? (box?.children[active] as HTMLElement | undefined) : undefined
    if (!box || !el) return
    if (el.offsetTop < box.scrollTop || el.offsetTop + el.offsetHeight > box.scrollTop + box.clientHeight) box.scrollTop = el.offsetTop - box.clientHeight / 3
  }, [active])
  if (!segs.length) return <div className="faint small">No speech was found.</div>
  return (
    <ol className="transcript" ref={list}>
      {segs.map((s, i) => (
        <li key={i} className={i === active ? 'on' : ''} onClick={() => pb.seek(s.start + 0.01)}>
          <span className="mono t">{mmss(s.start)}</span>
          <span>{s.text}</span>
        </li>
      ))}
    </ol>
  )
}
