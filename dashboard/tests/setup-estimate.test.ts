// VIS-78 (QA VIS-79): the page's copy of the memory formula against the server's fixture.
// models/estimate-cases.json is the source (server/setup/estimate.py writes it with
// `python tests/test_setup_estimate.py --write`), read here from the repository itself, so the two
// can never drift. Every `expect` is matched (numbers within 0.01, the rest equal, labels and notes
// included) and every `error` case refused with its status and code.
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';
import type { SetupModel } from '../src/api/types';
import { estimate, SelectionError } from '../src/setup/estimate';

const path = fileURLToPath(new URL('../../models/estimate-cases.json', import.meta.url));
const fixture = JSON.parse(readFileSync(path, 'utf8')) as {
  model: SetupModel;
  /** SRV-337: other models' formula payloads, by id (a case's selection names its model) */
  models?: Record<string, SetupModel>;
  cases: { name: string; selection: Record<string, unknown>; expect?: Record<string, unknown>; error?: { status: number; code: string } }[];
};

function close(actual: unknown, want: unknown, at: string): void {
  if (typeof want === 'number' && typeof actual === 'number') {
    expect(Math.abs(actual - want), `${at}: ${actual} vs ${want}`).toBeLessThanOrEqual(0.0100001);
    return;
  }
  if (Array.isArray(want)) {
    expect(Array.isArray(actual), at).toBe(true);
    expect((actual as unknown[]).length, `${at}.length`).toBe(want.length);
    want.forEach((w, i) => close((actual as unknown[])[i], w, `${at}[${i}]`));
    return;
  }
  if (want && typeof want === 'object') {
    expect(Object.keys(actual as object).sort(), `${at} keys`).toEqual(Object.keys(want).sort());
    for (const [k, w] of Object.entries(want)) close((actual as Record<string, unknown>)[k], w, `${at}.${k}`);
    return;
  }
  expect(actual, at).toEqual(want);
}

describe('the estimate fixture (models/estimate-cases.json)', () => {
  it('has cases of both kinds', () => {
    expect(fixture.cases.filter((c) => c.expect).length).toBeGreaterThan(10);
    expect(fixture.cases.filter((c) => c.error).length).toBeGreaterThan(0);
  });
  const modelOf = (sel: Record<string, unknown>) => fixture.models?.[sel.model as string] ?? fixture.model;
  for (const c of fixture.cases) {
    it(c.name, () => {
      if (c.error) {
        let err: unknown = null;
        try {
          estimate(modelOf(c.selection), c.selection as never);
        } catch (e) {
          err = e;
        }
        expect(err, 'refused').toBeInstanceOf(SelectionError);
        expect({ status: (err as SelectionError).status, code: (err as SelectionError).code }).toEqual(c.error);
        return;
      }
      close(estimate(modelOf(c.selection), c.selection as never), c.expect, 'estimate');
    });
  }
  it("refuses a window above a profile's largest, and caps the context at it", () => {
    const fn = structuredClone(fixture.models?.['qwen3.8-flash-next']) as SetupModel;
    expect(fn, 'the Flash-Next payload').toBeTruthy();
    const p = fn.profiles.find((x) => x.id === 'flashnext-3.05');
    if (!p) throw new Error('flashnext-3.05 missing');
    p.max_window = 131_072;
    let err: unknown = null;
    try {
      estimate(fn, { model: fn.id, profile: 'flashnext-3.05', window: 262_144 } as never);
    } catch (e) {
      err = e;
    }
    expect(err).toBeInstanceOf(SelectionError);
    expect((err as SelectionError).status).toBe(422);
    expect(estimate(fn, { model: fn.id, profile: 'flashnext-3.05', window: 131_072, slots: 1 } as never).context_tokens_fit).toBe(131_072);
  });
});

describe('#104: the engine family caps parallel chats', () => {
  it('every model takes the engine limit, Ornith too since its side slots (SPD-217)', async () => {
    const { slotsRange } = await import('../src/setup/slots');
    const { ENGINE_MAX_SLOTS, slotsMax } = await import('../src/setup/fields');
    const orn = fixture.models?.['ornith-1.5-122b'] as SetupModel;
    expect(slotsRange(orn).max).toBe(ENGINE_MAX_SLOTS);
    expect(orn.parallelism.slots.engine_max_reason).toBeNull();
    expect(() => estimate(orn, { model: orn.id, slots: ENGINE_MAX_SLOTS + 1 } as never)).toThrow(SelectionError);
    for (const m of [fixture.model, ...Object.values(fixture.models ?? {})]) {
      expect(slotsMax(m)).toBe(ENGINE_MAX_SLOTS);
    }
  });
});
