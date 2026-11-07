import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import '@fontsource/dm-sans/400.css'
import '@fontsource/dm-sans/500.css'
import '@fontsource/dm-sans/600.css'
import '@fontsource/dm-sans/700.css'
import '@fontsource/jetbrains-mono/400.css'
import '@fontsource/jetbrains-mono/600.css'
import '@fortawesome/fontawesome-free/css/fontawesome.min.css'
import '@fortawesome/fontawesome-free/css/solid.min.css'
import 'leaflet/dist/leaflet.css'
import './styles/tokens.css'
import './styles/base.css'
import './styles/home.css'
import './styles/case.css'
import './styles/reveal.css'
import { App } from './App'
import { api } from './lib/api'
import { AuthProvider } from './lib/auth'

// Login settings come from the server, so changing them needs no rebuild.
api
  .health()
  .then((h) => h.auth ?? null)
  .catch(() => null)
  .then((auth) =>
    createRoot(document.getElementById('root')!).render(
      <StrictMode>
        <AuthProvider config={auth}>
          <App />
        </AuthProvider>
      </StrictMode>,
    ),
  )
