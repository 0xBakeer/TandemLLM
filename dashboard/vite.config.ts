import { defineConfig } from 'vite';
import { minifySync, type Plugin } from 'vite';
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';

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

// VIS-109: dist/sw.js from pwa/sw.js, its cache named by a hash of the build's file names and its
// precache list the entry's HTML, JS and CSS, the manifest and the icon (the rest of the build is
// cached as the page loads it)
function serviceWorker(): Plugin {
  return {
    name: 'qse-service-worker',
    apply: 'build',
    generateBundle(_opts, bundle) {
      const names = Object.keys(bundle).sort();
      const build = createHash('sha256').update(names.join('\n')).digest('hex').slice(0, 12);
      const entry = Object.values(bundle).find((f) => f.type === 'chunk' && f.isEntry);
      const css = names.filter((n) => n.startsWith('assets/index-') && n.endsWith('.css'));
      const shell = ['./', './index.html', './manifest.webmanifest', './favicon.svg', ...(entry ? ['./' + entry.fileName] : []), ...css.map((n) => './' + n)];
      const src = readFileSync(new URL('./pwa/sw.js', import.meta.url), 'utf8').replace("'__BUILD__'", JSON.stringify(build)).replace('__SHELL__', JSON.stringify(shell));
      this.emitFile({ type: 'asset', fileName: 'sw.js', source: minifySync('sw.js', src).code });
    },
  };
}

export default defineConfig(({ mode }) => ({
  // Served by the engine at /dashboard/ — relative asset paths keep the build path-agnostic.
  base: './',
  plugins: mode === 'mock' ? [mockPlugin(), serviceWorker()] : [serviceWorker()],
  define: {
    __GRAFANA_URL__: JSON.stringify(process.env.VITE_GRAFANA_URL ?? ''),
    __DASH_VERSION__: JSON.stringify(process.env.npm_package_version ?? '0.0.0'),
  },
  build: {
    outDir: 'dist',
    emptyOutDir: true,
    target: 'es2022',
    sourcemap: false,
    // every browser the es2022 target reaches preloads modules itself; the polyfill was 0.2 KB gzip
    modulePreload: { polyfill: false },
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
