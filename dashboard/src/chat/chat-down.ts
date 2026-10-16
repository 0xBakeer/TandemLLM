// <qse-chat-down>: the Chat tab when the chat API is not there (VIS-179). The probe's state
// (backend.ts classifyProbe) as a panel: what is wrong in one line, the server's own message, the
// way out as a command to copy, "Check again", and a probe of its own every 5 s growing to 30 s
// while the page is visible. downBanner() is the slim line above the thread when a live tab loses
// chatd. Every string the panel shows is in downText(), one place.

import { html, nothing, type TemplateResult } from 'lit';
import { customElement, property } from 'lit/decorators.js';
import { LightElement } from '../ui/base';
import { copyButton } from '../ui/bits';
import type { ChatDown } from './backend';
import '../setup/figure';
import '../styles/chat-down.css';

export interface DownText {
  title: string;
  lead: string;
  /** the way out, to run on the box */
  command?: string;
  /** a second way, in words */
  alt?: string;
  /** a link in the app instead of a command */
  link?: { href: string; label: string };
  button: 'Check again' | 'Retry';
}

/** What the panel says for a state. Chosen by the cause alone; the server's message is shown
 *  beside it as written. */
export function downText(d: ChatDown): DownText {
  switch (d.cause) {
    case 'chatd_down':
      return { title: 'The chat service is not running.', lead: 'The engine server answers, but chatd, the service that keeps the chats, does not. It starts with the engine; to start it now, run on the box:', command: '~/TandemLLM/bin/tandem chat start', alt: 'Installed in another folder: <folder>/bin/tandem chat start. In a checkout: bash ops/chatd.sh start.', button: 'Check again' };
    case 'chatd_refused':
      return { title: "chatd is running but does not accept this server's key.", lead: 'The engine server and chatd read different secrets. Restart both so they read the same secrets.env:', command: '~/TandemLLM/bin/tandem chat restart && ~/TandemLLM/bin/tandem restart', button: 'Check again' };
    case 'no_chat_routes':
      return { title: 'This engine build has no Chat tab server (it needs 0.8.0 or newer).', lead: 'Update the box: run the installer again with the same --dir.', button: 'Check again' };
    case 'setup_mode':
      return { title: 'Finish the setup first.', lead: 'The box is in setup mode. The Chat tab starts with the engine once the Setup Guide is done.', link: { href: '#/setup', label: 'Open the Setup Guide' }, button: 'Check again' };
    case 'not_an_engine':
      return { title: 'This address does not serve the chat API.', lead: "The chat routes answered with a page, not JSON: a static server or a proxy's own page is in front of the engine. Use the dashboard on the engine server's own port.", button: 'Check again' };
    case 'offline':
      return { title: 'Cannot reach the engine server.', lead: 'It may be restarting. This page tries again by itself; on the box, tandem status shows the engine.', command: '~/TandemLLM/bin/tandem status', button: 'Retry' };
    default:
      return { title: `The chat server answered HTTP ${d.status ?? '?'}.`, lead: 'This page tries again by itself; on the box, tandem status shows the engine and chatd.', command: '~/TandemLLM/bin/tandem status', button: 'Retry' };
  }
}

export const PROBE_FIRST_MS = 5_000;
export const PROBE_MAX_MS = 30_000;

/** The wait before the n-th probe again (0 = the first): 5 s, 10 s, 20 s, then 30 s. */
export function probeDelay(n: number): number {
  return Math.min(PROBE_MAX_MS, PROBE_FIRST_MS * 2 ** n);
}

export interface PageVisibility {
  hidden(): boolean;
  on(f: () => void): void;
  off(f: () => void): void;
}

const documentVisibility: PageVisibility = {
  hidden: () => document.visibilityState === 'hidden',
  on: (f) => document.addEventListener('visibilitychange', f),
  off: (f) => document.removeEventListener('visibilitychange', f),
};

/** Probes on a schedule until `check` says the chat API is back: probeDelay() apart, paused while
 *  the page is hidden, and at once on now() ("Check again"). */
export class Reprobe {
  /** when the last probe ended, and when the next one is due (null: none due) */
  lastAt: number | null = null;
  nextAt: number | null = null;
  busy = false;
  private n = 0;
  private timer: ReturnType<typeof setTimeout> | null = null;
  private running = false;

  constructor(
    private check: () => Promise<boolean>,
    private changed: () => void = () => undefined,
    private page: PageVisibility = documentVisibility,
  ) {}

  start(): void {
    if (this.running) return;
    this.running = true;
    this.n = 0;
    this.page.on(this.onVisibility);
    this.schedule();
  }

  stop(): void {
    this.running = false;
    this.clear();
    this.page.off(this.onVisibility);
  }

  async now(): Promise<void> {
    if (!this.running || this.busy) return;
    this.clear();
    this.busy = true;
    this.changed();
    let ok = false;
    try {
      ok = await this.check();
    } catch {
      ok = false;
    }
    this.busy = false;
    this.lastAt = Date.now();
    if (ok) this.stop();
    else {
      this.n += 1;
      this.schedule();
    }
    this.changed();
  }

  private schedule(): void {
    this.clear();
    if (!this.running || this.page.hidden()) return;
    const ms = probeDelay(this.n);
    this.nextAt = Date.now() + ms;
    this.timer = setTimeout(() => void this.now(), ms);
  }

  private clear(): void {
    if (this.timer) clearTimeout(this.timer);
    this.timer = null;
    this.nextAt = null;
  }

  private onVisibility = (): void => {
    if (this.page.hidden()) this.clear();
    else if (this.running && !this.busy) void this.now();
  };
}

function clock(t: number): string {
  return new Date(t).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
}

@customElement('qse-chat-down')
export class QseChatDown extends LightElement {
  @property({ attribute: false }) down: ChatDown | null = null;
  @property({ attribute: false }) probe: Reprobe | null = null;
  /** re-rendered by the view on every probe and every second */
  @property({ type: Number }) now = Date.now();

  render() {
    const d = this.down;
    if (!d) return nothing;
    const t = downText(d);
    const p = this.probe;
    const next = p?.nextAt ? Math.max(0, Math.ceil((p.nextAt - this.now) / 1000)) : null;
    return html`<div class="ch-down" role="status" aria-live="polite" data-cause=${d.cause}>
      <qse-wren-figure class="ch-down-fig" state="idle" .size=${72}></qse-wren-figure>
      <h1 class="ch-down-title">${t.title}</h1>
      ${d.message ? html`<p class="ch-down-said muted">The server says: <q>${d.message}</q>${d.status ? html` <span class="mono small">(HTTP ${d.status})</span>` : nothing}</p>` : nothing}
      <p class="ch-down-lead">${t.lead}</p>
      ${t.command ? html`<div class="ch-down-cmd"><pre><code>${t.command}</code></pre>${copyButton(t.command, 'Copy the command')}</div>` : nothing}
      ${t.alt ? html`<p class="ch-down-alt small muted">${t.alt}</p>` : nothing}
      ${t.link ? html`<a class="btn ch-down-link" href=${t.link.href}>${t.link.label}</a>` : nothing}
      <div class="ch-down-actions">
        <button type="button" class="btn btn-primary ch-down-check" ?disabled=${!!p?.busy} @click=${() => void p?.now()}>${p?.busy ? 'Checking…' : t.button}</button>
        <span class="small muted ch-down-when">${p?.lastAt ? `Last checked ${clock(p.lastAt)}` : 'Not checked again yet'}${next != null ? `, next in ${next} s` : ''}</span>
      </div>
    </div>`;
  }
}

/** The slim line above the thread while a live tab has lost chatd. */
export function downBanner(): TemplateResult {
  return html`<div class="ch-down-banner" role="status" aria-live="polite">The chat service stopped. Your chats are safe. Retrying.</div>`;
}
