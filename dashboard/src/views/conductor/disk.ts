// Conductor · Memory · Disk cache (C1): the engine's disk tier on the box's NVMe. The tiles say what
// it holds against its budget, the free space it leaves above the 50 GB floor, how often it served a
// prefill it could have shortened and what that read; the table lists the most-hit entries (the
// prefix length, what it is -- an opencode head, a package, a repeated prefix -- its hits, what a hit
// reads, and a short preview for opencode heads and packages only, never a user's text); the event
// list says what it wrote, dropped and evicted. Below, its settings (on or off, the budget in GB with
// the disk's free space beside it, how long an unused entry is kept, after how many prefills a
// prefix is written, the free space it keeps), a versioned settings document like the transcripts',
// applied from the next request; and the clear button. Behind the admin token, like every memory
// route. On a phone the tiles stack and the table is cards.

import { html, nothing, type TemplateResult } from 'lit';
import { customElement, state } from 'lit/decorators.js';
import { describeError } from '../../api/client';
import { conductor, type DiskCacheSettings, type DiskEntry, type DiskMode, type MemoryDisk, type SettingsDoc } from '../../api/conductor';
import { setIn } from '../../lib/conductor';
import { bytes, dateTimeShort, exact, pct } from '../../lib/format';
import { DISK_FLOOR_GB, DISK_KIND_WORDS, diskEventLine, diskHitWords, diskReach, diskSettingErrors, diskSettings } from '../../lib/memory';
import { LightElement } from '../../ui/base';
import { emptyState, icon, panel, skeleton, statTile } from '../../ui/bits';
import { confirmDialog, DocEditor, errorsList, modeSeg, saveBar, withToken } from './shared';

const when = (s: number) => dateTimeShort(new Date(s * 1000).toISOString());
const counts = (o: Record<string, number> | undefined) =>
  Object.entries(o ?? {})
    .filter(([, v]) => v)
    .map(([k, v]) => `${exact(v)} ${k}`)
    .join(', ') || 'none';

@customElement('qse-cond-disk')
export class QseCondDisk extends LightElement {
  @state() private disk: MemoryDisk | null = null;
  @state() private error: string | null = null;
  @state() private busy = false;
  @state() private cleared: string | null = null;
  private settings = new DocEditor<SettingsDoc>(this, 'settings');

  connectedCallback(): void {
    super.connectedCallback();
    void this.load();
    void this.settings.load();
    this.settings.onSaved = () => void this.load();
  }

  private async load() {
    try {
      this.disk = await withToken(() => conductor.memoryDisk(), 'The disk tier view');
      this.error = null;
    } catch (e) {
      this.error = describeError(e).message;
    }
  }

  private async clear() {
    const d = this.disk;
    if (!d) return;
    const ok = await confirmDialog(
      'Clear the disk cache?',
      [`Every entry goes: ${exact(d.entries ?? 0)} prefixes in ${exact(d.chunks ?? 0)} chunk files, ${bytes(d.bytes ?? 0)}, and the counts that decide what is written next.`, 'Nothing a request needs is lost: the next prefill of each prefix runs cold, as without the tier, and writes it again.'],
      'Clear',
    );
    if (!ok) return;
    this.busy = true;
    try {
      const r = await withToken(() => conductor.memoryDiskClear());
      this.cleared = `cleared ${exact(r.entries)} entries, ${bytes(r.bytes)}`;
      this.disk = r.disk;
    } catch (e) {
      this.error = describeError(e).message;
    } finally {
      this.busy = false;
    }
  }

  render() {
    if (this.error && !this.disk) return html`<div class="state state-error" role="alert"><div class="state-title">${icon('warn')} ${this.error}</div></div>`;
    if (!this.disk) return skeleton(5);
    const d = this.disk;
    return html`${d.running ? this.renderRunning(d) : this.renderOff(d)}${this.renderSettings(d)}`;
  }

  private renderOff(d: MemoryDisk): TemplateResult {
    return panel(
      'Disk cache',
      emptyState(
        'The disk tier does not run on this server',
        html`It starts with the engine when <code>QSE_DISK_CACHE</code> in serve.env names its directory (the flag is <code class="mono">${d.flag}</code> now)${d.free_bytes != null ? html`; the NVMe has <span class="num">${bytes(d.free_bytes, 0)}</span> free` : nothing}. Its settings below apply once it runs.`,
      ),
      { id: 'cond-disk', sub: 'prompt-prefix state on the NVMe, under the caches in RAM' },
    );
  }

  private renderRunning(d: MemoryDisk): TemplateResult {
    const used = d.bytes ?? 0;
    const budget = d.budget ?? 0;
    const share = budget ? used / budget : 0;
    const free = d.free_bytes ?? null;
    const lowFree = free != null && d.floor != null && free < d.floor + 10e9;
    const tiles = html`<div class="stats disk-stats" data-disk-tiles>
      ${statTile({ label: 'on disk', value: bytes(used, 1), sub: html`of <span class="num">${bytes(budget, 0)}</span> · ${exact(d.entries ?? 0)} prefix${d.entries === 1 ? '' : 'es'}, ${exact(d.chunks ?? 0)} chunk files`, warn: share > 0.95 })}
      ${statTile({ label: 'NVMe free', value: free != null ? bytes(free, 0) : '—', sub: html`never under <span class="num">${bytes(d.floor ?? DISK_FLOOR_GB * 1e9, 0)}</span>`, warn: lowFree })}
      ${statTile({ label: 'hit rate', value: d.hit_rate != null ? pct(d.hit_rate, 0) : '—', sub: diskHitWords(d), na: d.hit_rate == null })}
      ${statTile({ label: 'restored', value: exact(d.tokens_restored ?? 0), unit: 'tokens', sub: d.last_restore_ms != null ? html`last <span class="num">${exact(Math.round(d.last_restore_ms))}</span> ms${d.last_read_gbs != null ? html` at <span class="num">${d.last_read_gbs.toFixed(1)}</span> GB/s` : nothing}` : 'no restore yet' })}
    </div>`;
    const meter = html`<div class="meter-block disk-meter">
      <div class="meter-head"><span>budget</span><span class="num">${bytes(used, 1)} of ${bytes(budget, 0)} · ${pct(share, 0)}</span></div>
      <div class="meter" role="img" aria-label=${`disk cache ${pct(share, 0)} of its budget`}><span class="meter-fill ${share > 0.95 ? 'is-warn' : ''}" style="width:${Math.min(100, share * 100).toFixed(1)}%"></span></div>
    </div>`;
    const rows = d.top ?? [];
    const body = html`${tiles}${meter}
      <p class="small muted disk-line" data-disk-counters>
        ${d.on ? html`<span class="badge badge-on">on</span>` : html`<span class="badge">off (settings)</span>`}
        written: <span class="num">${exact(d.writes_chunk ?? 0)}</span> chunks, <span class="num">${exact(d.writes_anchor ?? 0)}</span> anchors, <span class="num">${bytes(d.bytes_written ?? 0, 1)}</span> ·
        evicted: ${counts(d.evicted)} · dropped: ${counts(d.dropped)}${d.failed_restores ? html` · <span class="warn-ink">${exact(d.failed_restores)} failed restore${d.failed_restores === 1 ? '' : 's'}</span>` : nothing}
        · staging buffer <span class="num">${bytes(d.staging_bytes || d.staging_planned || 0, 2)}</span>${d.staging_bytes ? '' : ' (when first used)'} · <span class="mono">${d.root ?? ''}</span>
      </p>
      ${rows.length ? this.renderEntries(rows) : emptyState('Nothing on disk yet', `A prefix is written after ${d.settings.admit_after === 1 ? 'its first prefill' : `${d.settings.admit_after} prefills`} of at least ${exact(d.min_tokens ?? 4096)} tokens, and the first long request of an opencode session is written where its user message starts.`)}
      ${this.renderEvents(d)}`;
    return panel('Disk cache', body, {
      id: 'cond-disk',
      sub: 'prompt-prefix state on the NVMe, under the caches in RAM; a hit is read only past the rows already in place',
      tools: html`${this.cleared ? html`<span class="small muted" data-disk-cleared>${this.cleared}</span>` : nothing}
        <button class="btn btn-sm disk-clear" ?disabled=${this.busy || !(d.entries || d.chunks)} @click=${() => this.clear()} data-disk-clear>${icon('x')} Clear disk cache</button>
        <button class="btn btn-sm" @click=${() => this.load()} aria-label="reload the disk cache">${icon('refresh')}</button>`,
    });
  }

  private renderEntries(rows: DiskEntry[]): TemplateResult {
    const what = (r: DiskEntry) =>
      html`<span class="badge disk-kind disk-kind-${r.kind}">${DISK_KIND_WORDS[r.kind]}</span>${r.label ? html` <span class="mono small">${r.label}</span>` : nothing}${r.complete ? nothing : html` <span class="badge badge-warn" title="a chunk below it is not on disk: it is not read, and goes first">incomplete</span>`}`;
    const preview = (r: DiskEntry) => (r.preview ? html`<div class="small muted disk-preview" data-disk-preview>${r.preview}</div>` : nothing);
    return html`<h3 class="disk-sub">Most-hit prefixes</h3>
      <div class="table-scroll mem-table"><table class="table" data-disk-entries>
        <thead><tr><th class="num-col">tokens</th><th>what</th><th class="num-col">hits</th><th>last used</th><th class="num-col">a hit reads</th></tr></thead>
        <tbody>${rows.map(
          (r) => html`<tr data-disk-row=${`${r.kind}-${r.tokens}`}><td class="num num-col">${exact(r.tokens)}</td><td class="cell-wrap">${what(r)}${preview(r)}</td>
            <td class="num num-col">${exact(r.hits)}</td><td class="num">${when(r.used)}</td><td class="num num-col">${bytes(r.reads_bytes, 1)}</td></tr>`,
        )}</tbody></table></div>
      <div class="mem-cards" role="list">${rows.map(
        (r) => html`<div class="mem-card disk-card" role="listitem" data-disk-card=${`${r.kind}-${r.tokens}`}>
          <div class="mem-card-head"><span class="num">${exact(r.tokens)} tokens</span><span class="num small">${exact(r.hits)} hit${r.hits === 1 ? '' : 's'}</span></div>
          <div>${what(r)}</div>${preview(r)}
          <div class="small muted">last used ${when(r.used)} · a hit reads ${bytes(r.reads_bytes, 1)}</div>
        </div>`,
      )}</div>`;
  }

  private renderEvents(d: MemoryDisk): TemplateResult | typeof nothing {
    const evs = (d.events ?? []).slice(-10).reverse();
    if (!evs.length) return nothing;
    return html`<details class="sec" style="margin-top:12px" data-disk-events>
      <summary>Last events <span class="sec-sub">what it read, wrote, dropped and evicted (the journal keeps them, kind memory)</span></summary>
      <ul class="disk-events">${evs.map((e) => html`<li class="small"><span class="num muted">${when(e.ts)}</span> <span class="disk-ev disk-ev-${e.kind}">${diskEventLine(e)}</span></li>`)}</ul>
    </details>`;
  }

  private renderSettings(d: MemoryDisk): TemplateResult | typeof nothing {
    const ed = this.settings;
    if (!ed.draft) return ed.loadError ? html`<p class="err-ink small">${ed.loadError}</p>` : nothing;
    const s = diskSettings(ed.draft);
    const upd = (k: keyof DiskCacheSettings, v: unknown) => ed.set(setIn(ed.draft!, ['memory', 'disk_cache'], { ...s, [k]: v }));
    const num = (k: 'budget_gb' | 'ttl_days' | 'admit_after' | 'min_free_gb', label: string, min: number, max: number, step = 1) =>
      html`<label class="field"><span>${label}</span><input type="number" inputmode="decimal" min=${min} max=${max} step=${step} .value=${String(s[k])} data-disk-setting=${k} @input=${(e: Event) => upd(k, Number((e.target as HTMLInputElement).value))} /></label>`;
    const reach = diskReach(s, d);
    const local = diskSettingErrors(s);
    return html`<details class="sec" style="margin-top:16px" data-sec="disk" open>
      <summary>Disk cache settings <span class="sec-sub">on or off, its size on the NVMe, how long it keeps an entry, what it writes</span></summary>
      <div class="sec-body">
        <div class="toolbar" style="margin:0" data-disk-mode><span class="muted small">disk cache</span>${modeSeg(['on', 'off'] as DiskMode[], s.mode, (v) => upd('mode', v), { small: true, label: 'disk cache mode' })}<span class="small">${s.mode === 'on' ? 'prefixes are read from and written to the disk' : 'nothing is read or written; the files stay'}</span></div>
        <div class="form-grid">
          ${num('budget_gb', 'cache budget (GB)', 1, 4000)}
          ${num('ttl_days', 'drop an entry unused for (days)', 1, 90)}
          ${num('admit_after', 'write a prefix after it was prefilled (times)', 1, 10)}
          ${num('min_free_gb', `keep free on the NVMe (GB, at least ${DISK_FLOOR_GB})`, DISK_FLOOR_GB, 4000)}
        </div>
        <p class="small" data-disk-free>
          ${d.free_bytes != null ? html`The NVMe has <span class="num">${bytes(d.free_bytes, 0)}</span> free${d.bytes ? html`, and the cache holds <span class="num">${bytes(d.bytes, 1)}</span>` : nothing}. ` : nothing}
          ${reach.short ? html`<span class="warn-ink" data-disk-short>${icon('warn')} At most <span class="num">${reach.usable_gb!.toFixed(0)} GB</span> of the ${s.budget_gb} GB fit above the ${Math.max(DISK_FLOOR_GB, s.min_free_gb)} GB it keeps free; the tier stops there.</span>` : nothing}
        </p>
        <p class="muted small">Least valuable first when it is full: hits, times the tokens an entry saves per byte it holds, halved for every day unused. The first long request of an opencode session is written whatever the count, at the lines below where its user message starts. Applied from the next request; the free space it keeps is never under ${DISK_FLOOR_GB} GB, whatever is set here.</p>
        ${local.length ? html`<ul class="errors" role="alert" data-disk-errors>${local.map((e) => html`<li>${e}</li>`)}</ul>` : nothing}
        ${errorsList(ed)}
        ${saveBar(ed, 'the settings')}
      </div>
    </details>`;
  }
}
