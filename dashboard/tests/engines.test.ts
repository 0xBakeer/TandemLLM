// P2, P3: the vLLM and SGLang backends' figures. The proxy's own page over two readings 15 s apart
// (tests/fixtures/proxy/proxy-engines-{a,b}.txt, written by tools/proxy_fixture_pages.py from pages
// shaped as vLLM v0.30.0 and SGLang v0.5.21 send them) must give the figures proxy/engines.py
// derives from the engines' own pages (proxy-engines-figures.json): the Performance view and the
// System view then show the same numbers. Plus the card's words for what is not reported.

import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';
import { backendEngines, backendFigures, parsePrometheus } from '../src/lib/prom';
import { engineCard, flagNote, renderEngines } from '../src/views/engines';
import type { EngineFigures } from '../src/api/types';

const FIX = join(__dirname, '..', '..', 'tests', 'fixtures', 'proxy');
const text = (n: string) => readFileSync(join(FIX, n), 'utf8');
const A = parsePrometheus(text('proxy-engines-a.txt'), 1_000_000);
const B = parsePrometheus(text('proxy-engines-b.txt'), 1_015_000);
const PY = JSON.parse(text('proxy-engines-figures.json')) as { figures: Record<string, EngineFigures> };

// a template's text without a DOM: its static parts and its values, tags dropped
function flat(t: unknown): string {
  if (t == null || typeof t === 'symbol' || typeof t === 'boolean') return '';
  if (Array.isArray(t)) return t.map(flat).join(' ');
  if (typeof t === 'object' && 'strings' in (t as object)) {
    const r = t as { strings: readonly string[]; values: unknown[] };
    return r.strings.map((x, i) => x + (i < r.values.length ? flat(r.values[i]) : '')).join('');
  }
  return String(t);
}
const html = (t: unknown) => flat(t).replace(/<[^>]*>/g, ' ').replace(/\s+/g, ' ');

describe('backend engines (P2, P3)', () => {
  it('finds the backends and their kinds on the page', () => {
    expect(backendEngines(B)).toEqual([
      { id: 'sglang1', kind: 'sglang', version: '0.5.21' },
      { id: 'vllm1', kind: 'vllm', version: '0.30.0' },
    ]);
    expect(backendEngines(null)).toEqual([]);
  });

  for (const [id, kind] of [
    ['vllm1', 'vllm'],
    ['sglang1', 'sglang'],
  ] as const) {
    it(`derives the same figures as the server for ${id}`, () => {
      const ts = backendFigures(A, B, id, kind);
      const py = PY.figures[id];
      for (const k of Object.keys(py) as (keyof EngineFigures)[]) {
        expect([k, ts[k]]).toEqual([k, py[k]]);
      }
      expect(ts.window_s).toBe(15);
    });
  }

  it('reads the hand-counted vLLM window', () => {
    const f = backendFigures(A, B, 'vllm1', 'vllm');
    expect(f.prefix_hit_rate).toBe(0.8); // hits +800 over queries +1000
    expect(f.decode_tps).toBe(40);
    expect(f.spec).toEqual({ source: 'counters', acceptance: 0.75, tokens_per_step: 3.25, per_position: [0.95, 0.75, 0.55], drafts: 200 });
  });

  it('gives no window figure from one scrape', () => {
    const f = backendFigures(null, B, 'vllm1', 'vllm');
    expect(f.running).toBe(1);
    expect(f.decode_tps).toBeNull();
    expect(f.prefix_hit_rate).toBeNull();
    expect(f.spec?.acceptance).toBeNull();
  });

  it('says what an engine does not report', () => {
    const sg = backendFigures(A, B, 'sglang1', 'sglang');
    const t = html(engineCard({ id: 'sglang1', label: 'SGLang box', kind: 'sglang', figures: sg }));
    expect(t).toContain('prefill not reported');
    expect(t).toContain('not reported by this backend');
    expect(t).toContain('prefix-cache hits');
    const off = html(engineCard({ id: 's2', label: null, kind: 'sglang', figures: null, engine: { layer: 'L1', metrics: 'off', figures: null } }));
    expect(off).toContain('--enable-metrics');
    const v = { ...backendFigures(A, B, 'vllm1', 'vllm'), spec: null };
    expect(html(engineCard({ id: 'v', label: 'v', kind: 'vllm', figures: v }))).toContain('speculation off');
    expect(flagNote('vllm', null)).toContain('--enable-prompt-tokens-details');
    expect(flagNote('generic', null)).toBeNull();
  });

  it('shows no panel without a vLLM or SGLang backend', () => {
    expect(renderEngines([{ id: 'ds', label: 'DeepSeek', kind: 'deepseek', figures: null }], '')).toBe(renderEngines([], ''));
  });
});
