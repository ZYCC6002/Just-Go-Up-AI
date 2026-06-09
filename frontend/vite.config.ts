import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  define: {
    // plotly.js references Node's `global`; polyfill for browser builds
    global: 'globalThis',
  },
  build: {
    // 1. Empties the target backend folders before building fresh copies
    emptyOutDir: true,
    // 2. Compiles static assets (js, css, images) into backend/static/
    outDir: '../backend/static',
    // 3. Keeps the compiled files clean without nested asset subfolders
    assetsDir: '.',
  },
  server: {
    // Proxy /api requests to FastAPI during local development.
    // In production, FastAPI serves both the API and the built frontend.
    proxy: {
      '/api': 'http://localhost:7860',
    },
  },
})