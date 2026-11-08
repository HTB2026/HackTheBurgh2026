import { useAuth } from '../lib/auth'
import { useHealth } from '../lib/health'
import { Icon } from './ui/ui'

export function Wordmark() {
  return (
    <a className="logo" href="#/" aria-label="EARWITNESS home">
      <svg width="26" height="26" viewBox="0 0 32 32" aria-hidden>
        <path d="M2 16 C8 6 24 6 30 16 C24 26 8 26 2 16 Z" fill="none" stroke="#ededed" strokeWidth="2" />
        <path d="M9 16 h3 l2 -5 l3 10 l2 -5 h4" fill="none" stroke="#db2828" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
      <span>
        EAR<b>WITNESS</b>
      </span>
    </a>
  )
}

export function Header({ route }: { route: string }) {
  const { health, offline } = useHealth()
  const auth = useAuth()
  const nav = [
    { href: '#/', key: 'home', label: 'Cases', icon: 'folder-open', accent: 'blue' },
    { href: '#/examine', key: 'examine', label: 'Locate', icon: 'location-crosshairs', accent: 'red' },
  ]
  return (
    <header className="top">
      <div className="wrap">
        <Wordmark />
        <nav className="nav">
          {nav.map((n) => (
            <a key={n.key} href={n.href} className={`c-${n.accent} ${route === n.key ? 'active' : ''}`}>
              <Icon name={n.icon} />
              {n.label}
            </a>
          ))}
        </nav>
        <div className="top-right">
          {offline ? (
            <span className="status-pill c-red" data-tip="The analysis server is not reachable. Start it with: uv run earwitness serve">
              <i className="status-dot" /> Server offline
            </span>
          ) : (
            health && route !== 'case' && (
              <span
                className={`status-pill ${health.openai ? 'c-purple' : 'c-grey'}`}
                data-tip={
                  health.openai
                    ? 'AI checks are on: speech, the AI eyewitness and location-from-pixels run, and their results are labelled AI.'
                    : 'AI checks are off. Every result comes from measured checks only.'
                }
              >
                <Icon name="robot" /> AI {health.openai ? 'on' : 'off'}
              </span>
            )
          )}
          {health && route !== 'case' && (
            <span className="status-pill c-green" data-tip={`${health.modules.length} checks run on every video, ${health.image_modules?.length ?? 0} on every photo`}>
              <i className="status-dot on" /> {health.modules.length + (health.image_modules?.length ?? 0)} checks ready
            </span>
          )}
          {auth.enabled &&
            !auth.loading &&
            (auth.user ? (
              <button
                className="status-pill c-blue auth-btn"
                onClick={auth.logout}
                data-tip={`Logged in as ${auth.user.email ?? auth.user.name}. Click to log out.`}
              >
                <Icon name="right-from-bracket" /> {auth.user.name ?? auth.user.email ?? 'Log out'}
              </button>
            ) : (
              <button
                className="status-pill c-blue auth-btn"
                onClick={auth.login}
                data-tip="Log in to upload, re-run or delete cases. Viewing needs no login."
              >
                <Icon name="right-to-bracket" /> Log in
              </button>
            ))}
        </div>
      </div>
    </header>
  )
}
