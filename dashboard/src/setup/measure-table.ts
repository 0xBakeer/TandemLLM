// <qse-measure-table>: the registry's measured table (VIS-78, step Model). Sort by any column,
// greedy or sampled, 512 or 128 tokens, the everyday tasks or the long-context ones; the selected
// profile's row is marked; rows from another build (NVFP4 on 0.4.8) sit apart under their label.
// Each speed cell carries a bar against the column's best so the sizes compare at a glance.

import { html, nothing, type TemplateResult } from 'lit';
import { customElement, property, state } from 'lit/decorators.js';
import type { SetupMeasureCell, SetupMeasureMode, SetupMeasureRow, SetupMeasureTask, SetupModel } from '../api/types';
import { LightElement } from '../ui/base';
import { S } from './strings';

type Mode = 'greedy' | 'sampled';
type Len = '512' | '128';
type Group = 'everyday' | 'long';
type Col = 'profile' | SetupMeasureTask | 'ttft8' | 'ttft32' | 'kl' | 'lossless';

export interface TableView {
  mode: Mode;
  len: Len;
  group: Group;
  sort: Col;
  dir: 'asc' | 'desc';
}

export function modeKey(v: Pick<TableView, 'mode' | 'len' | 'group'>): SetupMeasureMode {
  if (v.group === 'long') return v.mode === 'greedy' ? 'long_greedy' : 'long_sampled';
  return `${v.mode}_${v.len}` as SetupMeasureMode;
}

export function columns(group: Group): Col[] {
  return group === 'everyday' ? ['profile', 'prose', 'chat', 'code', 'ttft8', 'kl', 'lossless'] : ['profile', 'long32k', 'long64k', 'ttft32', 'kl'];
}

const COL_LABEL: Record<Col, string> = {
  profile: S.colWeights,
  prose: S.colProse,
  chat: S.colChat,
  code: S.colCode,
  long32k: S.colLong32,
  long64k: S.colLong64,
  ttft8: S.colTtft8,
  ttft32: S.colTtft32,
  kl: S.colKl,
  lossless: S.colLossless,
};

/** The value a column sorts and shows for a row (null: not measured). */
export function cellValue(m: SetupModel, r: SetupMeasureRow, col: Col, v: TableView): number | null {
  const cells = r.cells[modeKey(v)] ?? {};
  switch (col) {
    case 'profile': {
      const p = m.profiles.find((x) => x.id === r.profile);
      return p?.bpw ?? p?.weights_gib ?? null;
    }
    case 'ttft8':
      return r.ttft_s['8192'] ?? null;
    case 'ttft32':
      return r.ttft_s['32768'] ?? null;
    case 'kl':
      return m.profiles.find((x) => x.id === r.profile)?.quality.kl_vs_bf16 ?? null;
    case 'lossless':
      return r.lossless ? r.lossless.identical : null;
    default:
      return (cells as Partial<Record<SetupMeasureTask, SetupMeasureCell>>)[col]?.tok_s ?? null;
  }
}

/** Comparable rows sorted; rows from another build stay below, in registry order. */
export function sortRows(m: SetupModel, v: TableView): { main: SetupMeasureRow[]; apart: SetupMeasureRow[] } {
  const rows = m.measurements.rows;
  const main = rows.filter((r) => r.comparable);
  const apart = rows.filter((r) => !r.comparable);
  const k = (r: SetupMeasureRow) => cellValue(m, r, v.sort, v);
  main.sort((a, b) => {
    const x = k(a);
    const y = k(b);
    if (x == null && y == null) return 0;
    if (x == null) return 1; // unmeasured last either way
    if (y == null) return -1;
    return v.dir === 'asc' ? x - y : y - x;
  });
  return { main, apart };
}

const SPEED: Col[] = ['prose', 'chat', 'code', 'long32k', 'long64k'];

@customElement('qse-measure-table')
export class QseMeasureTable extends LightElement {
  @property({ attribute: false }) model: SetupModel | null = null;
  @property({ attribute: false }) selected: string | null = null;
  @state() view: TableView = { mode: 'greedy', len: '512', group: 'everyday', sort: 'code', dir: 'desc' };

  private set(p: Partial<TableView>): void {
    const next = { ...this.view, ...p };
    if (p.group && !columns(p.group).includes(next.sort)) next.sort = p.group === 'long' ? 'long32k' : 'code';
    this.view = next;
  }

  private sortBy(c: Col): void {
    if (this.view.sort === c) this.set({ dir: this.view.dir === 'asc' ? 'desc' : 'asc' });
    // speed: fastest first; time, KL: smallest first
    else this.set({ sort: c, dir: SPEED.includes(c) || c === 'lossless' ? 'desc' : 'asc' });
  }

  render() {
    const m = this.model;
    if (!m) return nothing;
    const v = this.view;
    const cols = columns(v.group);
    const { main, apart } = sortRows(m, v);
    const max: Partial<Record<Col, number>> = {};
    for (const c of SPEED) max[c] = Math.max(0, ...main.map((r) => cellValue(m, r, c, v) ?? 0));
    const seg = <T extends string>(cur: T, opts: [T, string][], set: (x: T) => void, label: string) =>
      html`<div class="seg" role="radiogroup" aria-label=${label}>${opts.map(([val, text]) => html`<button class="seg-btn ${cur === val ? 'is-on' : ''}" role="radio" aria-checked=${cur === val} data-val=${val} @click=${() => set(val)}>${text}</button>`)}</div>`;
    return html`<div class="mt">
      <div class="mt-controls">
        ${seg(v.mode, [['greedy', S.modeGreedy], ['sampled', S.modeSampled]], (x) => this.set({ mode: x }), 'Decoding')}
        ${v.group === 'everyday' ? seg(v.len, [['512', S.tokens512], ['128', S.tokens128]], (x) => this.set({ len: x }), 'Answer length') : nothing}
        ${seg(v.group, [['everyday', S.groupEveryday], ['long', S.groupLong]], (x) => this.set({ group: x }), 'Tasks')}
      </div>
      <div class="table-scroll mt-scroll">
        <table class="table mt-table" data-mode=${modeKey(v)}>
          <thead>
            <tr>
              ${cols.map((c) => {
                const on = v.sort === c;
                const unit = SPEED.includes(c) ? S.tokPerS : c === 'ttft8' || c === 'ttft32' ? S.seconds : '';
                return html`<th scope="col" class=${c === 'profile' ? 'mt-name' : 'num-col'} aria-sort=${on ? (v.dir === 'asc' ? 'ascending' : 'descending') : 'none'}>
                  <button class="mt-sort ${on ? 'is-on' : ''}" data-col=${c} @click=${() => this.sortBy(c)} aria-label=${S.sortBy(COL_LABEL[c])}>
                    <span>${COL_LABEL[c]}</span>${unit ? html`<span class="mt-unit">${unit}</span>` : nothing}<span class="mt-arrow" aria-hidden="true">${on ? (v.dir === 'asc' ? '▴' : '▾') : ''}</span>
                  </button>
                </th>`;
              })}
            </tr>
          </thead>
          <tbody>
            ${main.map((r) => this.row(m, r, cols, v, max))}
            ${apart.length
              ? html`<tr class="mt-apart-head"><td colspan=${cols.length}>${S.notComparable}</td></tr>
                  ${apart.map((r) => this.row(m, r, cols, v, max, true))}`
              : nothing}
          </tbody>
        </table>
      </div>
      <div class="mt-foot muted small">${S.measuredOn(m.measurements.build, m.measurements.date)} <details class="mt-method"><summary>${S.howMeasured}</summary><p>${m.measurements.method}</p></details></div>
    </div>`;
  }

  private row(m: SetupModel, r: SetupMeasureRow, cols: Col[], v: TableView, max: Partial<Record<Col, number>>, apart = false): TemplateResult {
    const p = m.profiles.find((x) => x.id === r.profile);
    const sel = r.profile === this.selected;
    return html`<tr class="mt-row ${sel ? 'is-selected' : ''} ${apart ? 'is-apart' : ''}" data-profile=${r.profile} aria-current=${sel ? 'true' : nothing}>
      ${cols.map((c) => {
        if (c === 'profile')
          return html`<th scope="row" class="mt-name">
            <span class="mt-label">${p?.label ?? r.profile}</span>
            ${sel ? html`<span class="mt-mine">${S.selectedRow}</span>` : nothing}
            ${apart ? html`<span class="mt-note" title=${r.note ?? ''}>${S.notComparable}</span>` : nothing}
          </th>`;
        const val = cellValue(m, r, c, v);
        if (val == null) return html`<td class="num-col mt-na">–</td>`;
        if (c === 'kl') return html`<td class="num-col num">${val.toFixed(4)}</td>`;
        if (c === 'lossless') {
          const l = r.lossless!;
          return html`<td class="num-col num" title=${S.losslessTitle(l.identical, l.near_tie)}>${l.identical} + ${l.near_tie}</td>`;
        }
        if (c === 'ttft8' || c === 'ttft32') return html`<td class="num-col num">${val.toFixed(1)}</td>`;
        const w = max[c] ? (val / max[c]!) * 100 : 0;
        return html`<td class="num-col num mt-speed"><span class="mt-bar" style="--w:${w.toFixed(1)}" aria-hidden="true"></span><span class="mt-val">${val.toFixed(1)}</span></td>`;
      })}
    </tr>`;
  }
}
