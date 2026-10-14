// The dashboard's service worker (VIS-109, plan note Chat 8). The build writes it to dist/sw.js
// with BUILD (a hash of the build's file names) and SHELL (the entry's HTML, JS and CSS, the
// manifest and the icon) filled in. It caches the app shell of the current build only: the
// pages and the hashed assets. It never caches an API answer, a conversation or an attachment:
// anything under /v1/ goes to the network untouched, and a navigation asks the network first, so
// a new build is never hidden behind an old one. The cache is named by the build and an old
// build's cache goes when the new worker activates; the page offers a reload when one waits.
// OPS-131: a navigation that the reverse proxy answers 502, 503 or 504 (the engine, which serves the
// dashboard, restarts) gets the cached shell, as a network error does; the page then asks
// `./sw-away` and, when the worker served it so, says the engine restarts and reloads once /health
// answers (src/lib/pwa.ts).
const BUILD = '__BUILD__';
const SHELL = __SHELL__;
const CACHE = 'tandem-shell-' + BUILD;
const AWAY = [502, 503, 504];
// the last navigation this worker answered from the cache: {status (0: network error), at (ms)}
let away = null;

function shell(status) {
  return caches.match('./index.html', { ignoreSearch: true }).then((r) => {
    if (!r) return null;
    away = { status, at: Date.now() };
    return r;
  });
}

self.addEventListener('install', (e) => {
  e.waitUntil(caches.open(CACHE).then((c) => c.addAll(SHELL)));
});

self.addEventListener('activate', (e) => {
  e.waitUntil(
    caches
      .keys()
      .then((keys) => Promise.all(keys.filter((k) => k.startsWith('tandem-shell-') && k !== CACHE).map((k) => caches.delete(k))))
      .then(() => self.clients.claim()),
  );
});

self.addEventListener('message', (e) => {
  if (e.data === 'skip-waiting') self.skipWaiting();
});

self.addEventListener('fetch', (e) => {
  const req = e.request;
  const url = new URL(req.url);
  if (req.method !== 'GET' || url.origin !== self.location.origin || url.pathname.includes('/v1/')) return;
  const scope = new URL(self.registration.scope).pathname;
  if (!url.pathname.startsWith(scope)) return;
  if (url.pathname === scope + 'sw-away') {
    // what the page asks right after it loaded: was it the cached shell, and why
    const body = JSON.stringify(away ? { away: true, status: away.status, at: away.at } : { away: false });
    away = null;
    e.respondWith(new Response(body, { headers: { 'Content-Type': 'application/json', 'Cache-Control': 'no-store' } }));
    return;
  }
  if (req.mode === 'navigate') {
    e.respondWith(
      fetch(req).then(
        (res) => (AWAY.includes(res.status) ? shell(res.status).then((r) => r || res) : res),
        () => shell(0).then((r) => r || Response.error()),
      ),
    );
    return;
  }
  if (!url.pathname.startsWith(scope + 'assets/')) return;
  // a hashed asset of this build: from the cache, else the network (then kept)
  e.respondWith(
    caches.open(CACHE).then((c) =>
      c.match(req).then(
        (hit) =>
          hit ||
          fetch(req).then((res) => {
            if (res.ok) c.put(req, res.clone());
            return res;
          }),
      ),
    ),
  );
});

// a notification's click opens its chat in the dashboard's window (or a new one)
self.addEventListener('notificationclick', (e) => {
  e.notification.close();
  const url = new URL((e.notification.data && e.notification.data.url) || './#/chat', self.registration.scope).href;
  e.waitUntil(
    self.clients.matchAll({ type: 'window', includeUncontrolled: true }).then((wins) => {
      for (const w of wins) {
        if (w.url.startsWith(self.registration.scope) && 'focus' in w) {
          w.postMessage({ type: 'open', url });
          return w.focus();
        }
      }
      return self.clients.openWindow(url);
    }),
  );
});
