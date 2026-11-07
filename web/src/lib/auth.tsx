import { Auth0Provider, useAuth0 } from '@auth0/auth0-react'
import { createContext, useContext, useEffect, type ReactNode } from 'react'
import type { AuthConfig } from './types'

/** What the header needs; `enabled` is false when the server has no Auth0 config. */
export interface AuthState {
  enabled: boolean
  loading: boolean
  user: { name?: string; email?: string; picture?: string } | null
  login: () => void
  logout: () => void
}

const disabled: AuthState = { enabled: false, loading: false, user: null, login: () => {}, logout: () => {} }
const AuthContext = createContext<AuthState>(disabled)

export const useAuth = () => useContext(AuthContext)

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

  const state: AuthState = {
    enabled: true,
    loading: isLoading,
    user: isAuthenticated && user ? { name: user.name, email: user.email, picture: user.picture } : null,
    login: () => void loginWithRedirect({ appState: { returnTo: returnTo() } }),
    logout: () => void logout({ logoutParams: { returnTo: location.origin + location.pathname } }),
  }
  return <AuthContext.Provider value={state}>{children}</AuthContext.Provider>
}

/** Wraps the app in Auth0 when the server publishes a login config; otherwise a no-op. */
export function AuthProvider({ config, children }: { config: AuthConfig | null | undefined; children: ReactNode }) {
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
