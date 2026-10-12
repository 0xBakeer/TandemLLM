// Hash routing: #/usage?client=k:…&range=90d. Reload keeps the tab and its filters. The setup guide
// has a path below its view: #/setup/<model>/<step> (VIS-78); the Chat tab too: #/chat/<id>,
// #/chat/settings/<section> (VIS-99).

export type View = 'chat' | 'usage' | 'performance' | 'dev' | 'playground' | 'conductor' | 'backends' | 'system' | 'setup';
export const VIEWS: View[] = ['chat', 'usage', 'performance', 'dev', 'playground', 'conductor', 'backends', 'system', 'setup'];

export interface Route {
  view: View;
  params: URLSearchParams;
  /** the segments after the view (#/setup/qwen3.8-27b/weights -> ['qwen3.8-27b', 'weights']) */
  path: string[];
}

export function parseHash(hash = location.hash): Route {
  const h = hash.replace(/^#\/?/, '');
  const [full, query = ''] = h.split('?');
  const [first, ...rest] = full.split('/');
  const known = (VIEWS as string[]).includes(first);
  const view = known ? (first as View) : 'usage';
  const path = known && (view === 'setup' || view === 'chat') ? rest.filter(Boolean).map((x) => decodeURIComponent(x)) : [];
  return { view, params: new URLSearchParams(query), path };
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
  const next = hashFor(r.view, r.params).replace(`#/${r.view}`, `#/${[r.view, ...r.path].join('/')}`);
  if (location.hash !== next) history.replaceState(null, '', next);
  window.dispatchEvent(new HashChangeEvent('hashchange'));
}

export function onRoute(fn: (r: Route) => void): () => void {
  const h = () => fn(parseHash());
  window.addEventListener('hashchange', h);
  return () => window.removeEventListener('hashchange', h);
}
