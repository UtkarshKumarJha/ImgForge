import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  // fixed ports so another local Vite app (default 5173) cannot push us onto one the API's CORS rejects
  server: { port: 5180, strictPort: true },
  preview: { port: 5180, strictPort: true },
})
