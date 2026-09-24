// Small shared templates: panel chrome, skeletons, error/empty states, stat tiles, icons.

import { html, nothing, svg, type TemplateResult } from 'lit';
import type { Loadable } from './base';

export function icon(name: string): TemplateResult {
  const p = ICONS[name] ?? '';
  return svg`<svg class="icon" viewBox="0 0 20 20" aria-hidden="true" width="18" height="18">${p}</svg>`;
}

const ICONS: Record<string, TemplateResult> = {
  usage: svg`<rect x="2" y="3" width="3.5" height="3.5" rx=".8" fill="currentColor"/><rect x="7" y="3" width="3.5" height="3.5" rx=".8" fill="currentColor" opacity=".55"/><rect x="12" y="3" width="3.5" height="3.5" rx=".8" fill="currentColor"/><rect x="2" y="8" width="3.5" height="3.5" rx=".8" fill="currentColor" opacity=".35"/><rect x="7" y="8" width="3.5" height="3.5" rx=".8" fill="currentColor"/><rect x="12" y="8" width="3.5" height="3.5" rx=".8" fill="currentColor" opacity=".7"/><rect x="2" y="13" width="3.5" height="3.5" rx=".8" fill="currentColor" opacity=".8"/><rect x="7" y="13" width="3.5" height="3.5" rx=".8" fill="currentColor" opacity=".3"/><rect x="12" y="13" width="3.5" height="3.5" rx=".8" fill="currentColor"/>`,
  performance: svg`<path d="M2.5 14.5 7 8.5l3.5 3.5 6.5-8" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/><path d="M2.5 17.5h15" stroke="currentColor" stroke-width="1.2" opacity=".5"/>`,
  dev: svg`<path d="m6.5 6-4 4 4 4M13.5 6l4 4-4 4" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/><path d="M11.5 3.5 8.5 16.5" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" opacity=".6"/>`,
  system: svg`<rect x="2.5" y="3.5" width="15" height="9" rx="1.5" fill="none" stroke="currentColor" stroke-width="1.6"/><path d="M6 16.5h8M10 12.5v4" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/><circle cx="6" cy="8" r="1.2" fill="currentColor"/><path d="M9 8h5.5" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" opacity=".6"/>`,
  sun: svg`<circle cx="10" cy="10" r="3.2" fill="none" stroke="currentColor" stroke-width="1.6"/><path d="M10 2.5v2M10 15.5v2M2.5 10h2M15.5 10h2M4.7 4.7l1.4 1.4M13.9 13.9l1.4 1.4M4.7 15.3l1.4-1.4M13.9 6.1l1.4-1.4" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/>`,
  moon: svg`<path d="M15.5 12.4A6.5 6.5 0 0 1 7.6 4.5a6.5 6.5 0 1 0 7.9 7.9Z" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round"/>`,
  out: svg`<path d="M12 3.5h3.5a1 1 0 0 1 1 1v11a1 1 0 0 1-1 1H12M3.5 10h9M9.5 6.5 13 10l-3.5 3.5" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/>`,
  copy: svg`<rect x="7" y="7" width="10" height="10" rx="1.5" fill="none" stroke="currentColor" stroke-width="1.6"/><path d="M13 7V4.5a1.5 1.5 0 0 0-1.5-1.5h-7A1.5 1.5 0 0 0 3 4.5v7A1.5 1.5 0 0 0 4.5 13H7" fill="none" stroke="currentColor" stroke-width="1.6"/>`,
  x: svg`<path d="m5 5 10 10M15 5 5 15" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/>`,
  external: svg`<path d="M11 3.5h5.5V9M16.5 3.5 9 11M14.5 11.5v4a1 1 0 0 1-1 1h-9a1 1 0 0 1-1-1v-9a1 1 0 0 1 1-1h4" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/>`,
  play: svg`<path d="M6 4.5v11l9-5.5z" fill="currentColor"/>`,
  pause: svg`<rect x="5" y="4.5" width="3.5" height="11" rx=".8" fill="currentColor"/><rect x="11.5" y="4.5" width="3.5" height="11" rx=".8" fill="currentColor"/>`,
  down: svg`<path d="M10 3.5v12M5.5 11 10 15.5 14.5 11" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>`,
  trash: svg`<path d="M4 5.5h12M8 5.5V4a1 1 0 0 1 1-1h2a1 1 0 0 1 1 1v1.5M6 5.5l.7 10a1 1 0 0 0 1 .9h4.6a1 1 0 0 0 1-.9l.7-10" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/>`,
  warn: svg`<path d="M10 3 2.5 16h15L10 3Z" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round"/><path d="M10 8v3.5" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/><circle cx="10" cy="13.8" r="1" fill="currentColor"/>`,
  check: svg`<path d="m4 10.5 4 4 8-9" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>`,
  wrap: svg`<path d="M3 5h14M3 10h9.5a2.5 2.5 0 0 1 0 5H10M3 15h4" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/><path d="m11.5 13 -1.8 2 1.8 2" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/>`,
  search: svg`<circle cx="8.5" cy="8.5" r="5" fill="none" stroke="currentColor" stroke-width="1.7"/><path d="m12.5 12.5 4.5 4.5" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/>`,
  stop: svg`<rect x="5" y="5" width="10" height="10" rx="1.5" fill="currentColor"/>`,
  send: svg`<path d="M3 10 17 3.5 13.5 17 10 11.5z" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round"/>`,
  refresh: svg`<path d="M16 10a6 6 0 1 1-1.8-4.3" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round"/><path d="M16.5 3v4h-4" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"/>`,
};

export function skeleton(lines = 3, cls = ''): TemplateResult {
  return html`<div class="skeleton ${cls}" aria-busy="true" aria-live="polite">
    ${Array.from({ length: lines }, (_, i) => html`<div class="skeleton-line" style="width:${[92, 68, 80, 55, 74][i % 5]}%"></div>`)}
  </div>`;
}

export function errorState(endpoint: string, message: string, retry?: () => void): TemplateResult {
  return html`<div class="state state-error" role="alert">
    <div class="state-title">${icon('warn')} <span>Could not load <code>${endpoint || 'the engine'}</code></span></div>
    <div class="state-body">${message}</div>
    ${retry ? html`<button class="btn" @click=${retry}>${icon('refresh')} Try again</button>` : nothing}
  </div>`;
}

export function emptyState(title: string, body?: string): TemplateResult {
  return html`<div class="state state-empty">
    <div class="state-title">${title}</div>
    ${body ? html`<div class="state-body">${body}</div>` : nothing}
  </div>`;
}

/** Renders a loadable with the panel's own content function. */
export function whenReady<T>(l: Loadable<T>, ready: (d: T) => TemplateResult, opts: { lines?: number; retry?: () => void; emptyTitle?: string } = {}): TemplateResult {
  switch (l.state) {
    case 'loading':
      return skeleton(opts.lines ?? 3);
    case 'error':
      return errorState(l.endpoint, l.message, opts.retry);
    case 'empty':
      return emptyState(opts.emptyTitle ?? 'Nothing here yet', l.note);
    case 'ready':
      return ready(l.data);
  }
}

export function panel(title: string | TemplateResult, body: TemplateResult | typeof nothing, opts: { sub?: string | TemplateResult; tools?: TemplateResult; cls?: string; id?: string } = {}): TemplateResult {
  return html`<section class="panel ${opts.cls ?? ''}" id=${opts.id ?? nothing}>
    <header class="panel-head">
      <div class="panel-titles">
        <h2 class="panel-title">${title}</h2>
        ${opts.sub ? html`<p class="panel-sub">${opts.sub}</p>` : nothing}
      </div>
      ${opts.tools ? html`<div class="panel-tools">${opts.tools}</div>` : nothing}
    </header>
    <div class="panel-body">${body}</div>
  </section>`;
}

export interface StatTileProps {
  label: string;
  value: string;
  unit?: string;
  title?: string; // exact value tooltip
  sub?: TemplateResult | string;
  warn?: boolean;
  na?: boolean;
}

export function statTile(p: StatTileProps): TemplateResult {
  return html`<div class="stat ${p.warn ? 'is-warn' : ''} ${p.na ? 'is-na' : ''}" title=${p.title ?? nothing}>
    <div class="stat-label">${p.label}</div>
    <div class="stat-value"><span class="num">${p.value}</span>${p.unit ? html`<span class="stat-unit">${p.unit}</span>` : nothing}</div>
    ${p.sub ? html`<div class="stat-sub">${p.sub}</div>` : nothing}
  </div>`;
}

export function copyButton(text: string, label = 'Copy'): TemplateResult {
  return html`<button
    class="btn btn-ghost btn-sm"
    aria-label=${label}
    title=${label}
    @click=${async (e: Event) => {
      const b = e.currentTarget as HTMLButtonElement;
      try {
        await navigator.clipboard.writeText(text);
        b.classList.add('is-done');
        setTimeout(() => b.classList.remove('is-done'), 1200);
      } catch {
        /* clipboard blocked */
      }
    }}
  >
    ${icon('copy')}<span class="copy-done">${icon('check')}</span>
  </button>`;
}
