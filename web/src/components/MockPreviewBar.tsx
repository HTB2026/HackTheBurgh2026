import { useState } from 'react'
import { MOCK_CASES } from '../lib/mockShowcase'
import { Icon } from './ui/ui'
import '../styles/mock.css'

const labels = { planted: 'Video · all signals', photo: 'Photo · edited', generated: 'Photo · AI-generated', clean: 'Video · no edits' }

export function MockPreviewBar({ caseId }: { caseId: string | null }) {
  const [expanded, setExpanded] = useState(false)
  return (
    <aside className="mock-preview" aria-label="Frontend preview controls">
      <div className="wrap mock-preview-inner">
        <div className="mock-preview-label"><b><Icon name="flask" /> UI preview</b><span>Synthetic findings · files stay in this browser · no backend needed</span></div>
        <div className="mock-preview-actions">
          <label className="mock-select">Example
            <select aria-label="Preview example" value={MOCK_CASES.find((n) => caseId === `mock-${n}`) ?? ''} onChange={(e) => { location.hash = `#/case/mock-${e.target.value}` }}>
              <option value="" disabled>Select an example</option>
              {MOCK_CASES.map((n) => <option key={n} value={n}>{labels[n]}</option>)}
            </select>
          </label>
          <a className="btn ghost sm" href="#/">Case gallery</a>
          {caseId && <>
            <button className="btn ghost sm" onClick={() => { const next = !expanded; setExpanded(next); window.dispatchEvent(new CustomEvent('mock-sections', { detail: next })) }}>{expanded ? 'Close technical details' : 'Open technical details'}</button>
            <button className="btn ghost sm" onClick={() => window.dispatchEvent(new Event('mock-replay'))}><Icon name="rotate-right" /> Replay analysis</button>
          </>}
          <button className="btn ghost sm" onClick={() => location.reload()} title="Restore deleted examples and clear uploaded files">Reset preview</button>
        </div>
      </div>
    </aside>
  )
}
