import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { resolve } from 'path'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  build: {
    rollupOptions: {
      input: {
        main: resolve(__dirname, 'index.html'),
        'content-script': resolve(__dirname, 'src/content.tsx')
      },
      output: {
        entryFileNames: '[name].js'
      }
    }
  }
})
