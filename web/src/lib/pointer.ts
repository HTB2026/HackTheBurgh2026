/** Capture the pointer for drags; tolerate synthetic events with unknown ids. */
export function capture(e: { currentTarget: EventTarget; pointerId: number }) {
  try {
    ;(e.currentTarget as HTMLElement).setPointerCapture(e.pointerId)
  } catch {
    /* not a real active pointer */
  }
}
