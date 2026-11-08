import { useCallback, useEffect, useState } from 'react'
import { Header } from './components/Header'
import { Landing } from './components/Landing'
import { HowItWorks } from './components/HowItWorks'
import { CasePage, type SectionKey } from './components/case/CasePage'
import { TipLayer } from './components/ui/ui'
import { loadMock, mockName } from './lib/mock'
import { MockPreviewBar } from './components/MockPreviewBar'

type Route = { page: 'home' } | { page: 'examine' } | { page: 'how' } | { page: 'case'; id: string; section?: string }

function parseHash(): Route {
  const h = location.hash
  const m = h.match(/^#\/case\/([^/?#]+)(?:\/([a-z]+))?/)
  if (m) return { page: 'case', id: decodeURIComponent(m[1]), section: m[2] }
  if (h.startsWith('#/examine')) return { page: 'examine' }
  if (h.startsWith('#/how')) return { page: 'how' }
  return { page: 'home' }
}

export function App() {
  const [route, setRoute] = useState<Route>(parseHash)

  useEffect(() => {
    const on = () => setRoute(parseHash())
    window.addEventListener('hashchange', on)
    return () => window.removeEventListener('hashchange', on)
  }, [])

  // scroll to the top on page changes (not on tab changes within a case)
  const pageKey = route.page === 'case' ? `case:${route.id}` : route.page
  useEffect(() => {
    window.scrollTo(0, 0)
    if (route.page !== 'case') document.title = 'EARWITNESS'
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pageKey])

  useEffect(() => {
    const mock = mockName()
    if (mock && !location.hash)
      loadMock(mock).then((evs) => {
        const first = evs[0]
        if (first?.type === 'case' && !location.hash) location.hash = `#/case/${first.case.id}`
      })
  }, [])

  const caseId = route.page === 'case' ? route.id : null
  const onSection = useCallback(
    (s: SectionKey | null) => {
      if (!caseId) return
      history.replaceState(null, '', `#/case/${caseId}${s ? `/${s}` : ''}`)
      setRoute({ page: 'case', id: caseId, section: s ?? undefined })
    },
    [caseId],
  )

  return (
    <>
      <Header route={route.page === 'case' ? 'home' : route.page} />
      {mockName() && <MockPreviewBar caseId={caseId} />}
      {route.page === 'case' ? (
        <CasePage key={route.id} caseId={route.id} section={route.section} onSection={onSection} />
      ) : route.page === 'how' ? (
        <HowItWorks />
      ) : (
        <Landing examine={route.page === 'examine'} />
      )}
      <footer className="foot">
        <div className="wrap">
          <span>EARWITNESS · video and photo forensics</span>
          <span className="faint">Findings are leads, not proof. Labels marked AI come from a model and can be wrong.</span>
        </div>
      </footer>
      <TipLayer />
    </>
  )
}
