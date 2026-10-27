import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
const backend = 'http://127.0.0.1:8510'
const localProxy = {
  target: backend,
  changeOrigin: true,
  configure(proxy) {
    proxy.on('proxyReq', (request) => {
      request.setHeader('Origin', backend)
      request.setHeader('Referer', `${backend}/`)
    })
  },
}
export default defineConfig({ plugins:[vue()], server:{ proxy:{ '/api': localProxy, '/media': localProxy } } })
