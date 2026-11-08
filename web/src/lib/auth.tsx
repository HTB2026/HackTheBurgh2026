import { Auth0Provider, useAuth0 } from '@auth0/auth0-react'
import { createContext, useContext, useEffect, useState, type ReactNode } from 'react'
import type { AuthConfig, Owner } from './types'
import { mockName } from './mock'

/** What the header needs; `enabled` is false when the server has no Auth0 config. */
export interface AuthState {
  enabled: boolean
  loading: boolean
  user: { name?: string; email?: string; picture?: string } | null
  /** The server's view of the user (from /api/me): the id cases record, and admin rights. */
  me: (Owner & { admin: boolean }) | null
  login: () => void
  logout: () => void
}

const disabled: AuthState = { enabled: false, loading: false, user: null, me: null, login: () => {}, logout: () => {} }
const AuthContext = createContext<AuthState>(disabled)

export const useAuth = () => useContext(AuthContext)

/**
 * Whether the current visitor may rerun or delete a case with this owner.
 * 'login' means: maybe, once logged in (the click starts the login).
 * The server decides for real; this only keeps buttons honest.
 */
export function canChange(auth: AuthState, owner: Owner | null | undefined): true | false | 'login' {
  if (!auth.enabled) return true
  if (!auth.user) return 'login'
  if (!owner) return true
  if (!auth.me) return true
  return auth.me.admin || auth.me.id === owner.id
}

/** "Alice", or a neutral phrase when the name was not recorded. */
export const ownerName = (o: Owner | null | undefined) => o?.name ?? 'a signed-in user'

// api.ts asks this for a token before any write (upload, rerun, delete).
let tokenSource: (() => Promise<string>) | null = null

export class LoginRedirect extends Error {
  constructor() {
    super('Taking you to the login page…')
  }
}

/** Authorization header for a write; starts a login (and throws) when logged out. */
export async function authHeaders(): Promise<Record<string, string>> {
  if (!tokenSource) return {}
  return { Authorization: `Bearer ${await tokenSource()}` }
}

const returnTo = () => location.hash || '#/'

function Bridge({ children }: { children: ReactNode }) {
  const { isAuthenticated, isLoading, user, loginWithRedirect, logout, getAccessTokenSilently } = useAuth0()

  useEffect(() => {
    tokenSource = async () => {
      if (!isAuthenticated) {
        await loginWithRedirect({ appState: { returnTo: returnTo() } })
        throw new LoginRedirect()
      }
      const token = await getAccessTokenSilently()
      if (!token) throw new Error('no access token from Auth0')
      return token
    }
    return () => {
      tokenSource = null
    }
  }, [isAuthenticated, loginWithRedirect, getAccessTokenSilently])

  // Ask the server who we are: it hashes the id the same way cases record it.
  const [me, setMe] = useState<AuthState['me']>(null)
  useEffect(() => {
    if (!isAuthenticated) {
      setMe(null)
      return
    }
    let live = true
    getAccessTokenSilently()
      .then((token) => fetch('/api/me', { headers: { Authorization: `Bearer ${token}` } }))
      .then((r) => (r.ok ? r.json() : null))
      .then((m) => live && setMe(m))
      .catch(() => live && setMe(null))
    return () => {
      live = false
    }
  }, [isAuthenticated, user?.sub, getAccessTokenSilently])

  const state: AuthState = {
    enabled: true,
    loading: isLoading,
    user: isAuthenticated && user ? { name: user.name, email: user.email, picture: user.picture } : null,
    me,
    login: () => void loginWithRedirect({ appState: { returnTo: returnTo() } }),
    logout: () => void logout({ logoutParams: { returnTo: location.origin + location.pathname } }),
  }
  return <AuthContext.Provider value={state}>{children}</AuthContext.Provider>
}

/** Wraps the app in Auth0 when the server publishes a login config; otherwise a no-op. */
export function AuthProvider({ config, children }: { config: AuthConfig | null | undefined; children: ReactNode }) {
  if (mockName()) return <MockAuthProvider>{children}</MockAuthProvider>
  if (!config) return <AuthContext.Provider value={disabled}>{children}</AuthContext.Provider>
  return (
    <Auth0Provider
      domain={config.domain}
      clientId={config.client_id}
      authorizationParams={{ redirect_uri: location.origin + location.pathname, audience: config.audience }}
      // Survives reloads without third-party cookies (Safari, Brave).
      cacheLocation="localstorage"
      useRefreshTokens
      onRedirectCallback={(appState) => {
        // The app routes on the hash; put it back and drop ?code=&state=.
        history.replaceState(null, '', location.pathname + (appState?.returnTo ?? '#/'))
        window.dispatchEvent(new HashChangeEvent('hashchange'))
      }}
    >
      <Bridge>{children}</Bridge>
    </Auth0Provider>
  )
}

function MockAuthProvider({ children }: { children: ReactNode }) {
  const [loggedIn, setLoggedIn] = useState(true)
  return (
    <AuthContext.Provider
      value={{
        enabled: true,
        loading: false,
        user: loggedIn ? { name: 'Preview user', email: 'preview@localhost' } : null,
        me: loggedIn ? { id: 'preview', name: 'Preview user', admin: false } : null,
        login: () => setLoggedIn(true),
        logout: () => setLoggedIn(false),
      }}
    >
      {children}
    </AuthContext.Provider>
  )
}
