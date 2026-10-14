// Web search (VIS-157, chatd SRV-469): the words for the providers, the settings form's draft and
// its checks (the server checks again), a chat's effective provider, and a web_search card's
// results. No DOM here; settings.ts, thread.ts and composer.ts draw with these.

import type { Capabilities, ChatSettings, SafeSearch, TimeRange, ToolCallPart, WebSearchOverride, WebSearchPatch, WebSearchProvider, WebSearchResult, WebSearchSettings } from './types';

export const PROVIDERS: WebSearchProvider[] = ['off', 'duckduckgo', 'searxng'];
export const SAFE_SEARCH: SafeSearch[] = ['off', 'moderate', 'strict'];
export const TIME_RANGES: TimeRange[] = ['', 'day', 'week', 'month', 'year'];
export const MAX_RESULTS = 20;
/** the languages offered in the select; the server takes any code such as de or de-DE */
export const LANGUAGES: { id: string; label: string }[] = [
  { id: '', label: 'Any' },
  { id: 'en-US', label: 'English (US)' },
  { id: 'en-GB', label: 'English (UK)' },
  { id: 'de-DE', label: 'German' },
  { id: 'fr-FR', label: 'French' },
  { id: 'es-ES', label: 'Spanish' },
  { id: 'ar-SA', label: 'Arabic' },
];

export function providerWord(p: WebSearchProvider | null | undefined): string {
  return p === 'duckduckgo' ? 'DuckDuckGo' : p === 'searxng' ? 'SearXNG' : 'off';
}

export function timeWord(t: TimeRange): string {
  return t === '' ? 'any time' : `past ${t}`;
}

/** what the form edits: the stored settings plus the auth value typed (empty: keep the stored) */
export interface WebSearchDraft extends Omit<WebSearchSettings, 'searxng_auth_set'> {
  searxng_auth_value: string;
  /** the user asked to remove the stored auth value */
  clear_auth: boolean;
}

export function draftOf(s: WebSearchSettings): WebSearchDraft {
  const { searxng_auth_set: _set, ...rest } = s;
  return { ...rest, searxng_auth_value: '', clear_auth: false };
}

/** the fields that differ from the stored settings, for PUT /web-search */
export function draftPatch(stored: WebSearchSettings, d: WebSearchDraft): WebSearchPatch {
  const out: WebSearchPatch = {};
  const keys = ['provider', 'searxng_url', 'searxng_auth_header', 'max_results', 'safe_search', 'language', 'time_range', 'timeout_s'] as const;
  for (const k of keys) if (d[k] !== stored[k]) (out as Record<string, unknown>)[k] = d[k];
  if (d.searxng_auth_value) out.searxng_auth_value = d.searxng_auth_value;
  else if (d.clear_auth && stored.searxng_auth_set) out.searxng_auth_value = null;
  return out;
}

/** the draft's problems by field, in the server's words; empty: it can be saved */
export function draftErrors(d: WebSearchDraft, authStored = false): Record<string, string> {
  const e: Record<string, string> = {};
  if (!Number.isInteger(d.max_results) || d.max_results < 1 || d.max_results > MAX_RESULTS) e.max_results = `1 to ${MAX_RESULTS}`;
  if (!Number.isInteger(d.timeout_s) || d.timeout_s < 2 || d.timeout_s > 30) e.timeout_s = '2 to 30 seconds';
  if (d.language && !/^[a-z]{2,3}(-[A-Z]{2})?$/.test(d.language)) e.language = 'a code such as de or de-DE';
  const url = d.searxng_url.trim();
  if (url) {
    let u: URL | null = null;
    try {
      u = new URL(url);
    } catch {
      u = null;
    }
    if (!u || !/^https?:$/.test(u.protocol) || !u.hostname) e.searxng_url = 'an http or https URL';
    else if (u.username || u.password) e.searxng_url = 'no user or password in the URL; use the auth header';
    else if (u.search || u.hash) e.searxng_url = "the instance's base URL, without a query";
  } else if (d.provider === 'searxng') e.searxng_url = "SearXNG needs the instance's URL";
  const name = d.searxng_auth_header.trim();
  if (name && !/^[A-Za-z0-9-]{1,64}$/.test(name)) e.searxng_auth_header = 'a header name such as Authorization or X-API-Key';
  if ((d.searxng_auth_value || (authStored && !d.clear_auth)) && !name) e.searxng_auth_header = 'name the header the auth value goes in';
  if (/[\r\n]/.test(d.searxng_auth_value)) e.searxng_auth_value = 'one line';
  return e;
}

/** the provider a chat searches with: its own choice, else the global one */
export function chatProvider(caps: Pick<Capabilities, 'web_search'> | null | undefined, s: Pick<ChatSettings, 'web_search'> | null | undefined): WebSearchProvider | null {
  const own = s?.web_search?.provider;
  if (own) return own;
  return caps?.web_search?.provider ?? null;
}

/** "This chat" select: '' follows the default, else a provider */
export function overrideOf(s: Pick<ChatSettings, 'web_search'> | null | undefined): '' | WebSearchProvider {
  return s?.web_search?.provider ?? '';
}

export function overridePatch(cur: WebSearchOverride | null | undefined, provider: '' | WebSearchProvider): WebSearchOverride | null {
  const rest = { ...(cur ?? {}) };
  delete rest.provider;
  const next: WebSearchOverride = provider ? { ...rest, provider } : rest;
  return Object.keys(next).length ? next : null;
}

/** a web_search card's results (none for another tool, a running or a failed call) */
export function searchResults(t: Pick<ToolCallPart, 'name' | 'output'>): WebSearchResult[] | null {
  if (t.name !== 'web_search' || !t.output || t.output.is_error) return null;
  const r = t.output.data?.results;
  return Array.isArray(r) ? r.filter((x) => x && typeof x.url === 'string' && /^https?:\/\//.test(x.url)) : null;
}

/** the card's word for a finished search */
export function resultsWord(n: number): string {
  return n === 0 ? 'no results' : `${n} result${n === 1 ? '' : 's'}`;
}

export function hostOf(url: string): string {
  try {
    return new URL(url).hostname.replace(/^www\./, '');
  } catch {
    return url;
  }
}
