var _a;
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
/**
 * Development proxy: the browser only ever talks to the Vite origin, and Vite
 * forwards /api to the Flask backend. That keeps the frontend free of any absolute
 * host, so the same build works locally, behind the preview proxy and in production
 * where Flask serves the built assets itself.
 */
export default defineConfig({
    plugins: [react()],
    server: {
        host: '0.0.0.0',
        port: 5173,
        strictPort: false,
        // allow the sandbox preview host and any host while developing behind a proxy
        allowedHosts: true,
        proxy: {
            '/api': {
                target: (_a = process.env.VITE_API_TARGET) !== null && _a !== void 0 ? _a : 'http://127.0.0.1:5000',
                changeOrigin: true,
                secure: false,
            },
        },
    },
    build: {
        outDir: 'dist',
        sourcemap: false,
        chunkSizeWarningLimit: 1200,
    },
});
