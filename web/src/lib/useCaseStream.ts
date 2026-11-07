import { useEffect, useReducer, useState } from 'react'
import { emptyState, reduce } from './state'
import type { CaseEvent } from './types'
import { loadMock, mockName } from './mock'

export type Conn = 'connecting' | 'live' | 'replay' | 'closed' | 'mock'

/** Subscribe to a case's event stream (WebSocket, or a recorded mock in dev). */
export function useCaseStream(caseId: string, nonce = 0) {
  const [state, dispatch] = useReducer(reduce, undefined, emptyState)
  const [conn, setConn] = useState<Conn>('connecting')

  useEffect(() => {
    dispatch({ type: 'reset' })
    let cancelled = false
    const mock = mockName()

    if (mock) {
      setConn('mock')
      const timers: number[] = []
      loadMock(mock).then((events) => {
        if (cancelled) return
        // Compress real timing into a watchable replay: 40..260 ms per event.
        let t = 300
        for (let i = 0; i < events.length; i++) {
          const ev = events[i]
          const gap = i === 0 ? 0 : Math.min(260, Math.max(40, (ev.ts - events[i - 1].ts) * 300))
          t += ev.type === 'module' && ev.status !== 'running' ? gap : gap + 60
          timers.push(window.setTimeout(() => dispatch({ type: 'events', events: [ev] }), t))
        }
      })
      return () => {
        cancelled = true
        timers.forEach(clearTimeout)
      }
    }

    let socket: WebSocket | null = null
    let attempt = 0
    let retry: number | undefined
    let buffer: CaseEvent[] = []
    let flush: number | undefined
    let connectedBefore = false

    const connect = () => {
      setConn('connecting')
      const proto = location.protocol === 'https:' ? 'wss' : 'ws'
      socket = new WebSocket(`${proto}://${location.host}/ws/cases/${encodeURIComponent(caseId)}`)
      socket.onopen = () => {
        attempt = 0
        // The server replays the whole history on every connect. On a reconnect, start clean in case a
        // rerun happened meanwhile; on the first connect keep what the HTTP fetch already painted.
        if (connectedBefore) dispatch({ type: 'reset' })
        connectedBefore = true
        setConn('live')
      }
      socket.onmessage = (msg) => {
        let parsed: CaseEvent
        try {
          parsed = JSON.parse(msg.data)
        } catch {
          return
        }
        // Server-side rerun: start the case over without reconnecting.
        if ((parsed as { type: string }).type === 'reset') {
          buffer = []
          dispatch({ type: 'reset' })
          return
        }
        buffer.push(parsed)
        // Batch the replay burst. A timer, not requestAnimationFrame: rAF never fires in
        // background tabs or headless browsers, which left the page stuck at "0 of 0 checks".
        if (flush === undefined)
          flush = window.setTimeout(() => {
            flush = undefined
            const events = buffer
            buffer = []
            dispatch({ type: 'events', events })
          }, 30)
      }
      socket.onclose = () => {
        socket = null
        if (cancelled) return
        setConn('closed')
        const delay = Math.min(10000, 500 * 2 ** attempt++)
        retry = window.setTimeout(connect, delay)
      }
      socket.onerror = () => socket?.close()
    }
    connect()

    // Also fetch the history over plain HTTP: it paints faster than the socket replay and works where
    // WebSockets are slow or blocked. Events carry seq numbers, so the two sources never double-count.
    fetch(`/api/cases/${encodeURIComponent(caseId)}`)
      .then((r) => (r.ok ? r.json() : null))
      .then((body: { events?: CaseEvent[] } | null) => {
        if (!cancelled && body?.events?.length) dispatch({ type: 'events', events: body.events })
      })
      .catch(() => undefined)

    return () => {
      cancelled = true
      if (retry) clearTimeout(retry)
      if (flush !== undefined) clearTimeout(flush)
      socket?.close()
    }
  }, [caseId, nonce])

  return { state, conn }
}
