import { authHeaders } from './auth'
import type { AiProvider, CaseEvent, CaseSummary, EnhanceResult, Health } from './types'
import { isMock, mockCases, mockEnhance, mockEvents, mockFromPath, mockHealth, mockRemove, mockRerun, mockUpload } from './mockBackend'

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
  health: () => isMock() ? mockHealth() : fetch('/api/health').then((r) => json<Health>(r)),
  cases: () => isMock() ? mockCases() : fetch('/api/cases').then((r) => json<CaseSummary[]>(r)),
  events: (id: string) => isMock() ? mockEvents(id) : fetch(`/api/cases/${encodeURIComponent(id)}`).then((r) => json<{ events: CaseEvent[] }>(r)),
  // Writes need an Auth0 login; authHeaders() starts one when logged out.
  fromPath: async (path: string, options: Record<string, unknown> = {}) =>
    isMock() ? mockFromPath(path) : fetch('/api/cases/from-path', {
      method: 'POST',
      headers: { 'content-type': 'application/json', ...(await authHeaders()) },
      body: JSON.stringify({ path, options }),
    }).then((r) => json<CaseSummary>(r)),
  remove: async (id: string) =>
    isMock() ? mockRemove(id) : fetch(`/api/cases/${id}`, { method: 'DELETE', headers: await authHeaders() }).then((r) => json<{ deleted: boolean }>(r)),
  rerun: async (id: string, aiProvider?: AiProvider | null) =>
    isMock()
      ? mockRerun(id)
      : fetch(`/api/cases/${id}/rerun${aiProvider ? `?ai_provider=${aiProvider}` : ''}`, {
          method: 'POST',
          headers: await authHeaders(),
        }).then((r) => json<CaseSummary>(r)),
  enhance: (id: string, body: { time_s?: number; x: number; y: number; w: number; h: number; frames?: number }) =>
    isMock() ? mockEnhance(id, body) : fetch(`/api/cases/${id}/enhance`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify(body),
    }).then((r) => json<EnhanceResult>(r)),
}

/** Multipart upload with progress (fetch has no upload progress). */
export async function uploadEvidence(
  file: File,
  approxTime: string | null,
  aiProvider: AiProvider | null,
  onProgress: (fraction: number) => void,
): Promise<CaseSummary> {
  if (isMock()) return mockUpload(file, onProgress)
  const headers = await authHeaders()
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest()
    const params = new URLSearchParams()
    if (approxTime) params.set('approx_time', approxTime)
    if (aiProvider) params.set('ai_provider', aiProvider)
    const q = params.size ? `?${params}` : ''
    xhr.open('POST', `/api/cases${q}`)
    for (const [k, v] of Object.entries(headers)) xhr.setRequestHeader(k, v)
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
