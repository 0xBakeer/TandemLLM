// VIS-109: the service worker (dist/sw.js, written by the build) in a secure context only, and the
// reload offer when a new build's worker waits. A plain-http install gets no worker and works as a
// normal page. A notification's click (the worker posts `open`) moves this window to its chat.

export function registerWorker(onUpdate: (apply: () => void) => void): void {
  if (!import.meta.env.PROD || !window.isSecureContext || !('serviceWorker' in navigator)) return;
  const sw = navigator.serviceWorker;
  let reloading = false;
  sw.addEventListener('controllerchange', () => {
    if (!reloading) return;
    location.reload();
  });
  sw.addEventListener('message', (e: MessageEvent<{ type?: string; url?: string }>) => {
    if (e.data?.type === 'open' && e.data.url) location.href = e.data.url;
  });
  const offer = (w: ServiceWorker) =>
    onUpdate(() => {
      reloading = true;
      w.postMessage('skip-waiting');
    });
  void sw.register('./sw.js').then((reg) => {
    if (reg.waiting && sw.controller) offer(reg.waiting);
    reg.addEventListener('updatefound', () => {
      const w = reg.installing;
      // an update replaces the worker that controls the page now; the first install has none (its
      // `installed` event can arrive after it claimed the page, which offered a reload for nothing)
      const replaces = !!sw.controller;
      w?.addEventListener('statechange', () => {
        if (w.state === 'installed' && replaces) offer(w);
      });
    });
    // a tab left open checks for a new build now and then
    setInterval(() => void reg.update().catch(() => undefined), 30 * 60_000);
  });
}
