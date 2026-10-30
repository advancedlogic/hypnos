import { fileURLToPath } from 'node:url';
import { defineConfig } from 'vite';
import { svelte } from '@sveltejs/vite-plugin-svelte';

const sdk = fileURLToPath(new URL('../sdk/javascript/src', import.meta.url));

// The SDK is compiled from its TypeScript sources, so it needs no separate build step.
export default defineConfig({
  plugins: [svelte()],
  resolve: {
    alias: { '@hypnos/client': `${sdk}/index.ts` },
  },
  clearScreen: false,
  server: {
    port: 1420,
    strictPort: true,
    fs: { allow: ['.', sdk] },
  },
  envPrefix: ['VITE_', 'TAURI_ENV_'],
  build: {
    target: 'es2022',
    outDir: 'dist',
    emptyOutDir: true,
  },
});
