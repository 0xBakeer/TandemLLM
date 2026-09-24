// Hash routing: #/usage?client=k:…&range=90d. Reload keeps the tab and its filters.

export type View = 'usage' | 'performance' | 'dev' | 'system';
export const VIEWS: View[] = ['usage', 'performance', 'dev', 'system'];

export interface Route {
  view: View;
  params: URLSearchParams;
}

export function parseHash(hash = location.hash): Route {
  const h = hash.replace(/^#\/?/, '');
  const [path, query = ''] = h.split('?');
  const view = (VIEWS as string[]).includes(path) ? (path as View) : 'usage';
  return { view, params: new URLSearchParams(query) };
}

export function hashFor(view: View, params?: URLSearchParams | Record<string, string | null | undefined>): string {
  let q: URLSearchParams;
  if (params instanceof URLSearchParams) q = params;
  else {
    q = new URLSearchParams();
    for (const [k, v] of Object.entries(params ?? {})) if (v != null && v !== '') q.set(k, v);
  }
  const s = q.toString();
  return `#/${view}${s ? '?' + s : ''}`;
}

export function navigate(view: View, params?: URLSearchParams | Record<string, string | null | undefined>): void {
  const next = hashFor(view, params);
  if (location.hash !== next) location.hash = next;
}

/** Update the current view's params without changing the view. */
export function setParams(patch: Record<string, string | null | undefined>): void {
  const r = parseHash();
  for (const [k, v] of Object.entries(patch)) {
    if (v == null || v === '') r.params.delete(k);
    else r.params.set(k, v);
  }
  const next = hashFor(r.view, r.params);
  if (location.hash !== next) history.replaceState(null, '', next);
  window.dispatchEvent(new HashChangeEvent('hashchange'));
}

export function onRoute(fn: (r: Route) => void): () => void {
  const h = () => fn(parseHash());
  window.addEventListener('hashchange', h);
  return () => window.removeEventListener('hashchange', h);
}
