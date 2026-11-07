import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// EARWITNESS_API lets you point the dev proxy at another backend port,
// e.g. EARWITNESS_API=http://127.0.0.1:8766 npm run dev
// eslint-disable-next-line @typescript-eslint/no-explicit-any
const target: string = (globalThis as any).process?.env?.EARWITNESS_API ?? 'http://127.0.0.1:8765'

export default defineConfig({
  plugins: [react()],
  base: '/',
  server: {
    proxy: {
      '/api': { target, changeOrigin: true },
      '/ws': { target: target.replace(/^http/, 'ws'), ws: true, changeOrigin: true },
    },
  },
  build: {
    outDir: 'dist',
    assetsDir: 'assets',
    emptyOutDir: true,
    chunkSizeWarningLimit: 1200,
  },
})
