// The measured table (VIS-78) on the NemotronH entries (ENG-361): one comparable row a measured weight set,
// labelled as its weight card, the selected profile's row marked "your choice"; the modes these builds did
// not measure give no value (the page's "–").
import { describe, expect, it } from 'vitest';
import bolt from '../mock/setup/nemotron-3.5-bolt-74b-a9b.json';
import superVl from '../mock/setup/nemotron-3.5-super-vl-120b-a12b.json';
import type { SetupModel } from '../src/api/types';
import { cellValue, QseMeasureTable, sortRows, type TableView } from '../src/setup/measure-table';
import { S } from '../src/setup/strings';

const VIEW: TableView = { mode: 'greedy', len: '512', group: 'everyday', sort: 'code', dir: 'desc' };

/** A template's text with its values in place (nested templates and lists flattened). */
function text(v: unknown): string {
  if (Array.isArray(v)) return v.map(text).join('');
  if (v && typeof v === 'object' && 'strings' in v && 'values' in v) {
    const t = v as { strings: readonly string[]; values: unknown[] };
    return t.strings.map((s, i) => s + (i < t.values.length ? text(t.values[i]) : '')).join('');
  }
  return typeof v === 'string' || typeof v === 'number' ? String(v) : '';
}

function rendered(m: SetupModel, selected: string): string {
  const el = Object.create(QseMeasureTable.prototype) as QseMeasureTable;
  Object.defineProperty(el, 'model', { value: m });
  Object.defineProperty(el, 'selected', { value: selected });
  Object.defineProperty(el, 'view', { value: VIEW });
  return text(el.render());
}

describe.each([
  ['Bolt', bolt as unknown as SetupModel],
  ['Super VL', superVl as unknown as SetupModel],
])('%s', (_name, m) => {
  it('every row is a profile of the weight cards, measured on the table build, prose, chat and code at greedy 512', () => {
    const { main, apart } = sortRows(m, VIEW);
    expect(main.length).toBeGreaterThan(1);
    expect(apart).toEqual([]);
    for (const r of main) {
      expect(m.profiles.map((p) => p.id)).toContain(r.profile);
      expect(r.build).toBe(m.measurements.build);
      for (const t of ['prose', 'chat', 'code'] as const) expect(cellValue(m, r, t, VIEW)).toBeGreaterThan(0);
    }
    // sorted by code, fastest first
    const code = main.map((r) => cellValue(m, r, 'code', VIEW)!);
    expect(code).toEqual([...code].sort((a, b) => b - a));
  });

  it('sampled, 128 tokens and long context are not measured: no value, no bar', () => {
    for (const v of [{ ...VIEW, mode: 'sampled' as const }, { ...VIEW, len: '128' as const }, { ...VIEW, group: 'long' as const, sort: 'long32k' as const }])
      for (const r of m.measurements.rows) for (const t of ['prose', 'chat', 'code', 'long32k', 'long64k'] as const) expect(cellValue(m, r, t, v)).toBeNull();
  });

  it('the rows carry the weight cards\' labels and the selected profile is "your choice", on the recommended and on another', () => {
    const rec = m.profiles.find((p) => p.recommended)!;
    const other = m.measurements.rows.find((r) => r.profile !== rec.id)!.profile;
    for (const sel of [rec.id, other]) {
      const html = rendered(m, sel);
      for (const r of m.measurements.rows) expect(html).toContain(m.profiles.find((p) => p.id === r.profile)!.label);
      const mine = html.split('<tr class="mt-row').slice(1).filter((tr) => tr.includes(S.selectedRow));
      expect(mine).toHaveLength(1);
      expect(mine[0]).toContain(`data-profile=${sel} `);
      expect(mine[0]).toContain(m.profiles.find((p) => p.id === sel)!.label);
      expect(html).toContain(S.measuredOn(m.measurements.build, m.measurements.date));
    }
  });
});
