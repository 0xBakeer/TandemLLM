// <qse-installed-weights>: the weights on this box and how to remove them (VIS-215, on SRV-560).
// GET /v1/setup/weights lists each profile's folder with its size; the running one has a badge and
// no Remove (it goes after a switch, the review step's checkbox). Remove asks first, with the GB it
// frees and the line that downloads the weights again, then POST /v1/setup/remove.

import { html, nothing } from 'lit';
import { customElement, state } from 'lit/decorators.js';
import type { SetupInstalledWeights } from '../api/types';
import { describeError } from '../api/client';
import { LightElement } from '../ui/base';
import { icon } from '../ui/bits';
import { setupApi } from './api';
import { S } from './strings';

type Row = SetupInstalledWeights['weights'][number];

/** VIS-215: the confirmation's words for a row (pure; tests/setup-remove.test.ts) */
export function removeQuestion(r: Row): { title: string; body: string; again: string } {
  return {
    title: S.removeTitle(r.label, r.gb),
    body: r.shared_with.length ? S.removeShared(r.shared_with) : S.removeBody(r.model_name),
    again: `bash ops/use-profile.sh ${r.profile} --fetch-only`,
  };
}

/** VIS-215: one row a folder (two profiles that share one are one row), the running first */
export function rows(w: SetupInstalledWeights | null): Row[] {
  if (!w) return [];
  const seen = new Set<string>();
  const out: Row[] = [];
  for (const r of [...w.weights].sort((a, b) => Number(b.active) - Number(a.active) || b.gb - a.gb)) {
    if (seen.has(r.removes)) continue;
    seen.add(r.removes);
    out.push(r);
  }
  return out;
}

@customElement('qse-installed-weights')
export class QseInstalledWeights extends LightElement {
  @state() private data: SetupInstalledWeights | null = null;
  @state() private error: string | null = null;
  @state() private asking: string | null = null;
  @state() private busy = false;
  @state() private done: string | null = null;

  connectedCallback(): void {
    super.connectedCallback();
    void this.load();
  }

  private async load(): Promise<void> {
    try {
      this.data = await setupApi.weights();
      this.error = null;
    } catch (e) {
      this.error = describeError(e).message;
    }
  }

  private async removeRow(r: Row): Promise<void> {
    this.busy = true;
    try {
      const res = await setupApi.remove([r.profile]);
      this.done = S.removed(r.label, res.freed_gb, res.log ?? 'run/weights-removed.txt');
      this.asking = null;
      this.dispatchEvent(new CustomEvent('weights-removed', { detail: res, bubbles: true }));
      await this.load();
    } catch (e) {
      this.error = describeError(e).message;
    } finally {
      this.busy = false;
    }
  }

  render() {
    const list = rows(this.data);
    if (!this.data && !this.error) return nothing;
    return html`<section class="iw" aria-labelledby="iw-h" data-installed-weights>
      <header class="iw-head">
        <h2 class="iw-title" id="iw-h">${S.installedTitle}</h2>
        ${this.data?.free_gb != null ? html`<span class="muted small num">${S.diskFree(this.data.free_gb)}</span>` : nothing}
      </header>
      ${this.error ? html`<p class="rv-warn" role="alert">${icon('warn')} ${this.error}</p>` : nothing}
      ${this.done ? html`<p class="iw-done" role="status" data-removed>${icon('check')} ${this.done}</p>` : nothing}
      ${list.length === 0 && this.data ? html`<p class="muted small">${S.installedNone}</p>` : nothing}
      <ul class="iw-list">
        ${list.map((r) => {
          const q = removeQuestion(r);
          return html`<li class="iw-row ${r.active ? 'is-active' : ''}" data-installed=${r.profile}>
            <div class="iw-main">
              <span class="iw-name">${r.label}</span>
              <span class="muted small">${r.model_name}${r.shared_with.length ? ` · ${S.sharedWith(r.shared_with)}` : ''}</span>
            </div>
            <span class="iw-gb num">${r.gb.toFixed(1)} GB</span>
            ${r.active
              ? html`<span class="sbadge sbadge-now" title=${S.runningNoRemove}>${S.activeNow}</span>`
              : html`<button class="btn btn-sm btn-ghost" data-remove=${r.profile} ?disabled=${!r.removable || this.busy} @click=${() => (this.asking = r.profile)}>${icon('trash')} ${S.removeGo}</button>`}
            ${this.asking === r.profile
              ? html`<div class="iw-ask" role="alertdialog" aria-label=${q.title} data-remove-confirm>
                  <p class="iw-ask-t">${q.title}</p>
                  <p class="small">${q.body}</p>
                  <p class="small muted">${S.removeAgain} <code>${q.again}</code></p>
                  <div class="inst-actions">
                    <button class="btn btn-sm btn-danger" ?disabled=${this.busy} data-remove-yes @click=${() => void this.removeRow(r)}>${this.busy ? S.removing : S.removeYes}</button>
                    <button class="btn btn-sm" ?disabled=${this.busy} @click=${() => (this.asking = null)}>${S.cancel}</button>
                  </div>
                </div>`
              : nothing}
          </li>`;
        })}
      </ul>
    </section>`;
  }
}
