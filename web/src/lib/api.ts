import type { CaseSummary, EnhanceResult, Health } from './types'

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`
    try {
      const body = await res.json()
      if (body?.detail) detail = typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail)
    } catch {
      /* not json */
    }
    throw new Error(detail)
  }
  return res.json() as Promise<T>
}

export const api = {
  health: () => fetch('/api/health').then((r) => json<Health>(r)),
  cases: () => fetch('/api/cases').then((r) => json<CaseSummary[]>(r)),
  fromPath: (path: string, options: Record<string, unknown> = {}) =>
    fetch('/api/cases/from-path', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ path, options }),
    }).then((r) => json<CaseSummary>(r)),
  remove: (id: string) => fetch(`/api/cases/${id}`, { method: 'DELETE' }).then((r) => json<{ deleted: boolean }>(r)),
  rerun: (id: string) => fetch(`/api/cases/${id}/rerun`, { method: 'POST' }).then((r) => json<CaseSummary>(r)),
  enhance: (id: string, body: { time_s?: number; x: number; y: number; w: number; h: number; frames?: number }) =>
    fetch(`/api/cases/${id}/enhance`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify(body),
    }).then((r) => json<EnhanceResult>(r)),
  reportUrl: (id: string) => `/api/cases/${id}/report`,
}

/** Multipart upload with progress (fetch has no upload progress). */
export function uploadEvidence(
  file: File,
  approxTime: string | null,
  onProgress: (fraction: number) => void,
): Promise<CaseSummary> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest()
    const q = approxTime ? `?approx_time=${encodeURIComponent(approxTime)}` : ''
    xhr.open('POST', `/api/cases${q}`)
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) onProgress(e.loaded / e.total)
    }
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          resolve(JSON.parse(xhr.responseText))
        } catch (err) {
          reject(err)
        }
      } else {
        reject(new Error(`upload failed: ${xhr.status} ${xhr.responseText.slice(0, 200)}`))
      }
    }
    xhr.onerror = () => reject(new Error('upload failed: network error'))
    const form = new FormData()
    form.append('file', file)
    xhr.send(form)
  })
}
