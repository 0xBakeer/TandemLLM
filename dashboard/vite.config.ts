import { defineConfig } from 'vite';
import type { Plugin } from 'vite';

// The mock harness is a dev/preview middleware only. It is loaded through a dynamic
// import inside the plugin hooks, so nothing from `mock/` is ever part of `dist/` (the build
// test in tests/build.test.ts checks that).
function mockPlugin(): Plugin {
  return {
    name: 'qse-mock-harness',
    async configureServer(server) {
      const { createMockMiddleware } = await import('./mock/server.ts');
      server.middlewares.use(createMockMiddleware());
    },
    async configurePreviewServer(server) {
      const { createMockMiddleware } = await import('./mock/server.ts');
      server.middlewares.use(createMockMiddleware());
    },
  };
}

export default defineConfig(({ mode }) => ({
  // Served by the engine at /dashboard/ — relative asset paths keep the build path-agnostic.
  base: './',
  plugins: mode === 'mock' ? [mockPlugin()] : [],
  define: {
    __GRAFANA_URL__: JSON.stringify(process.env.VITE_GRAFANA_URL ?? ''),
    __DASH_VERSION__: JSON.stringify(process.env.npm_package_version ?? '0.0.0'),
  },
  build: {
    outDir: 'dist',
    emptyOutDir: true,
    target: 'es2022',
    sourcemap: false,
    assetsInlineLimit: 0,
    rollupOptions: {
      output: {
        entryFileNames: 'assets/app-[hash].js',
        chunkFileNames: 'assets/chunk-[hash].js',
        assetFileNames: 'assets/[name]-[hash][extname]',
      },
    },
  },
  server: { port: 5173, strictPort: true },
  preview: { port: 4173, strictPort: true },
  test: {
    include: ['tests/**/*.test.ts'],
    environment: 'node',
  },
}));
