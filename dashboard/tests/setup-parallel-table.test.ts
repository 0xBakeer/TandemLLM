// The weight cards' speed rows (SPD-245, model.yaml `parallel`): the table lists every row under its weight
// card's name (a community pack by its name), decode, the first token at 2k / 8k / 32k and 1 / 2 / 4 / 8 at
// once; the selected row is "your choice"; a level whose proof is not N shows no number; the weight card's
// line names one request alone and the most at once; a model without rows renders nothing.
import { describe, expect, it } from 'vitest';
import m27 from '../mock/setup/qwen3.8-27b.json';
import type { SetupModel, SetupParallelRow } from '../src/api/types';
import { cardLine, levelRate, parallelFor, parallelRows, QseParallelTable, rowLabel } from '../src/setup/parallel-table';
import { S } from '../src/setup/strings';

function text(v: unknown): string {
  if (Array.isArray(v)) return v.map(text).join('');
  if (v && typeof v === 'object' && 'strings' in v && 'values' in v) {
    const t = v as { strings: readonly string[]; values: unknown[] };
    return t.strings.map((s, i) => s + (i < t.values.length ? text(t.values[i]) : '')).join('');
  }
  return typeof v === 'string' || typeof v === 'number' ? String(v) : '';
}

function row(profile: string, over: Partial<SetupParallelRow> = {}): SetupParallelRow {
  return {
    profile,
    build: '0.14.2',
    commit: '345ab0fa',
    date: '2026-10-26',
    raw: `models/qwen3.8-27b/measurements/parallel-2026-10-26-${profile.replace('community/', '')}.json`,
    setup: {
      env: 'live8-cold', lanes: 8, window: 262144, max_tokens: 512, suffix_store: false, response_cache: false, thinking: false,
      sampling: { temperature: 1, top_k: 20, top_p: 0.95, min_p: 0, seed: 'one per request' },
      prompts: { set: 'card-v1', sha256: '0123456789abcdef', prompt_tokens: { code: 246, prose: 190 } },
    },
    rounds: 3,
    decode: { code: 71.2, prose: 36.4 },
    prefill: { '2048': { ttft_s: 1.61, tok_s: 1270, prompt_tokens: 2045 }, '8192': { ttft_s: 5.87, tok_s: 1395, prompt_tokens: 8190 }, '32768': { ttft_s: 25.1, tok_s: 1303, prompt_tokens: 32760 } },
    levels: {
      '1': { wall_tok_s: 39.1, decode_tok_s: 40.2, ttft_s: 0.3, at_once: 1, rounds: 3 },
      '2': { wall_tok_s: 65.3, decode_tok_s: 68.0, ttft_s: 0.5, at_once: 2, rounds: 3 },
      '4': { wall_tok_s: 99.4, decode_tok_s: 104.0, ttft_s: 0.9, at_once: 4, rounds: 3 },
      '8': { wall_tok_s: 151.0, decode_tok_s: 160.0, ttft_s: 1.6, at_once: 8, rounds: 3 },
    },
    ...over,
  };
}

const TC = 'community/thinkingcap-27b-kearuga-exl3-4.00';

function model(rows: SetupParallelRow[]): SetupModel {
  const base = m27 as unknown as SetupModel;
  return {
    ...base,
    parallel: { method: 'tools/parallel_card.py', rows },
    community: [{ id: 'thinkingcap-27b-kearuga-exl3-4.00', profile: TC, name: 'ThinkingCap Qwen3.8-27B Kearuga' } as never],
  };
}

function rendered(m: SetupModel, selected: string | null): string {
  const el = Object.create(QseParallelTable.prototype) as QseParallelTable;
  Object.defineProperty(el, 'model', { value: m });
  Object.defineProperty(el, 'selected', { value: selected });
  return text(el.render());
}

describe('the weight cards\' speed rows', () => {
  const m = model([row(TC), row('exl3-6.0', { levels: { '1': { wall_tok_s: 27.0, decode_tok_s: 27.5, ttft_s: 0.3, at_once: 1, rounds: 3 } } }), row('exl3-4.0')]);

  it('orders the rows as the weight cards: own profiles first, then the community packs', () => {
    expect(parallelRows(m).map((r) => r.profile)).toEqual(['exl3-4.0', 'exl3-6.0', TC]);
  });

  it('names each row as its weight card, a community pack by its name', () => {
    expect(rowLabel(m, parallelFor(m, 'exl3-4.0')!)).toBe('EXL3 4.0 bpw');
    expect(rowLabel(m, parallelFor(m, TC)!)).toBe('ThinkingCap Qwen3.8-27B Kearuga');
  });

  it('renders decode, the three prefill lengths and the four levels, and marks the selected row', () => {
    const html = rendered(m, 'exl3-4.0');
    for (const s of [S.parallelTitle, S.parallelDecode, S.parallelPrefill, S.parallelAtOnce, '71.2', '36.4', '1.61', '5.87', '25.10', '39.1', '65.3', '99.4', '151.0', '2k', '8k', '32k'])
      expect(html).toContain(s);
    const mine = html.split('<tr class="mt-row').slice(1).filter((tr) => tr.includes(S.selectedRow));
    expect(mine).toHaveLength(1);
    expect(mine[0]).toContain('EXL3 4.0 bpw');
    // the setup of every row is named: build, commit, date, lanes, sampling, store, cache
    expect(html).toContain('build 0.14.2 (345ab0fa), 2026-10-26;');
    expect(html).toContain('8 lanes of 262,144 tokens, 512 tokens a request, T 1, top-k 20, top-p 0.95, thinking off, suffix store off, response cache off, 3 rounds a level');
  });

  it('a level that did not decode N at once shows no number', () => {
    const r = row('exl3-4.0', { levels: { ...row('exl3-4.0').levels, '8': { wall_tok_s: 120, decode_tok_s: 130, ttft_s: 2, at_once: 7, rounds: 3 } } });
    expect(levelRate(r, '8')).toBeNull();
    expect(levelRate(r, '4')).toBe(99.4);
  });

  it('the weight card\'s line: alone and the most at once; one request only; no row, no line', () => {
    expect(cardLine(m, 'exl3-4.0')).toBe('39.1 alone, 151.0 with 8 at once');
    expect(cardLine(m, 'exl3-6.0')).toBe('27.0 tok/s alone');
    expect(cardLine(m, 'exl3-2.5')).toBeNull();
    expect(cardLine(null, 'exl3-4.0')).toBeNull();
  });

  it('a model without rows renders nothing', () => {
    expect(rendered({ ...(m27 as unknown as SetupModel), parallel: undefined }, null)).toBe('');
    expect(rendered(model([]), null)).toBe('');
  });
});
