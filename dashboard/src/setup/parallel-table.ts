// <qse-parallel-table>: the weight cards' speed rows (SPD-245, model.yaml `parallel`): decode alone, the
// first token of a cold 2k / 8k / 32k prompt, and 1, 2, 4 and 8 requests at once in total tok/s, each
// row with its build, commit, date and setup. A level is shown only with its proof (N decoded at once),
// which the registry already refuses to load without.

import { html, nothing, type TemplateResult } from 'lit';
import { customElement, property } from 'lit/decorators.js';
import type { SetupModel, SetupParallelDepth, SetupParallelLevel, SetupParallelRow } from '../api/types';
import { LightElement } from '../ui/base';
import { S } from './strings';

export const LEVELS: SetupParallelLevel[] = ['1', '2', '4', '8'];
export const DEPTHS: SetupParallelDepth[] = ['2048', '8192', '32768'];

/** the rows of a model, in the order of its weight cards (own profiles first, then community packs) */
export function parallelRows(m: Pick<SetupModel, 'profiles'> & Partial<Pick<SetupModel, 'parallel'>>): SetupParallelRow[] {
  const rows = m.parallel?.rows ?? [];
  const order = (r: SetupParallelRow) => {
    const i = m.profiles.findIndex((p) => p.id === r.profile);
    return i < 0 ? m.profiles.length : i;
  };
  return [...rows].sort((a, b) => order(a) - order(b) || a.profile.localeCompare(b.profile));
}

/** the row of one weight card, or null */
export function parallelFor(m: Partial<Pick<SetupModel, 'parallel'>> | null | undefined, profile: string): SetupParallelRow | null {
  return m?.parallel?.rows.find((r) => r.profile === profile) ?? null;
}

/** the weight card's name of a row: its profile's label, a community pack's name, else the id */
export function rowLabel(m: Pick<SetupModel, 'profiles'> & Partial<Pick<SetupModel, 'community'>>, r: SetupParallelRow): string {
  const p = m.profiles.find((x) => x.id === r.profile);
  if (p) return p.label;
  const c = (m.community ?? []).find((e) => e.profile === r.profile || `community/${e.id}` === r.profile);
  return c?.name ?? r.profile;
}

/** a level's wall rate (first tokens included), or null when the row has no such level */
export function levelRate(r: SetupParallelRow, n: SetupParallelLevel): number | null {
  const lv = r.levels[n];
  return lv && lv.at_once === Number(n) ? lv.wall_tok_s : null;
}

/** one line of the setup a row ran under */
export function setupLine(r: SetupParallelRow): string {
  const s = r.setup;
  const sm = s.sampling;
  return S.parallelSetup(s.lanes, s.window, s.max_tokens, `T ${sm.temperature}, top-k ${sm.top_k}, top-p ${sm.top_p}`,
    s.suffix_store, s.response_cache, r.rounds, s.prompts.set);
}

const f1 = (x: number | null | undefined) => (x == null ? '–' : x.toFixed(1));
const f2 = (x: number | null | undefined) => (x == null ? '–' : x.toFixed(2));

@customElement('qse-parallel-table')
export class QseParallelTable extends LightElement {
  @property({ attribute: false }) model: SetupModel | null = null;
  @property({ attribute: false }) selected: string | null = null;

  render() {
    const m = this.model;
    const rows = m ? parallelRows(m) : [];
    if (!m || !rows.length) return nothing;
    return html`<div class="mt pt">
      <h3 class="pt-title">${S.parallelTitle}</h3>
      <div class="table-scroll mt-scroll">
        <table class="table mt-table pt-table">
          <thead>
            <tr>
              <th scope="col" class="mt-name" rowspan="2">${S.colWeights}</th>
              <th scope="colgroup" colspan="2">${S.parallelDecode}<span class="mt-unit">${S.tokPerS}</span></th>
              <th scope="colgroup" colspan=${DEPTHS.length}>${S.parallelPrefill}<span class="mt-unit">${S.seconds}</span></th>
              <th scope="colgroup" colspan=${LEVELS.length}>${S.parallelAtOnce}<span class="mt-unit">${S.tokPerS}</span></th>
            </tr>
            <tr>
              <th scope="col" class="num-col">${S.colCode}</th>
              <th scope="col" class="num-col">${S.colProse}</th>
              ${DEPTHS.map((d) => html`<th scope="col" class="num-col">${Number(d) / 1024}k</th>`)}
              ${LEVELS.map((n) => html`<th scope="col" class="num-col">${n}</th>`)}
            </tr>
          </thead>
          <tbody>${rows.map((r) => this.row(m, r))}</tbody>
        </table>
      </div>
      <div class="mt-foot muted small">${S.parallelFoot}
        <details class="mt-method"><summary>${S.howMeasured}</summary><p>${m.parallel?.method ?? ''}</p>
          <ul class="pt-setups">${rows.map((r) => html`<li data-setup=${r.profile}><strong>${rowLabel(m, r)}</strong>: ${S.parallelBuild(r.build, r.commit, r.date)} ${setupLine(r)}${r.note ? html` ${r.note}` : nothing}</li>`)}</ul>
        </details>
      </div>
    </div>`;
  }

  private row(m: SetupModel, r: SetupParallelRow): TemplateResult {
    const sel = r.profile === this.selected;
    return html`<tr class="mt-row ${sel ? 'is-selected' : ''}" data-parallel=${r.profile} aria-current=${sel ? 'true' : nothing}>
      <th scope="row" class="mt-name"><span class="mt-label">${rowLabel(m, r)}</span>${sel ? html`<span class="mt-mine">${S.selectedRow}</span>` : nothing}</th>
      <td class="num-col num">${f1(r.decode?.code)}</td>
      <td class="num-col num">${f1(r.decode?.prose)}</td>
      ${DEPTHS.map((d) => {
        const p = r.prefill?.[d];
        return p ? html`<td class="num-col num" title=${S.parallelPrefillTitle(p.prompt_tokens, p.tok_s)}>${f2(p.ttft_s)}</td>` : html`<td class="num-col mt-na">–</td>`;
      })}
      ${LEVELS.map((n) => {
        const lv = r.levels[n];
        const v = levelRate(r, n);
        return v == null || !lv ? html`<td class="num-col mt-na">–</td>` : html`<td class="num-col num" title=${S.parallelLevelTitle(Number(n), lv.decode_tok_s, lv.ttft_s)}>${f1(v)}</td>`;
      })}
    </tr>`;
  }
}

/** the weight card's line: 1 and 8 requests at once, or nothing when the card has no row */
export function cardLine(m: Partial<Pick<SetupModel, 'parallel'>> | null | undefined, profile: string): string | null {
  const r = parallelFor(m, profile);
  if (!r) return null;
  const one = levelRate(r, '1');
  const top = (['8', '4', '2'] as const).find((n) => levelRate(r, n) != null);
  return top ? S.parallelCard(one, Number(top), levelRate(r, top)) : one != null ? S.parallelCard(one, 1, one) : null;
}

declare global {
  interface HTMLElementTagNameMap {
    'qse-parallel-table': QseParallelTable;
  }
}
