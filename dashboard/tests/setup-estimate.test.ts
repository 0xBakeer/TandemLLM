// VIS-78 (QA VIS-79): the page's copy of the memory formula against the server's fixture.
// models/estimate-cases.json is the source (server/setup/estimate.py writes it with
// `python tests/test_setup_estimate.py --write`); tests/fixtures/setup-estimate-cases.json is a copy
// for a dashboard checkout without the registry. Every `expect` is matched (numbers within 0.01,
// the rest equal, labels and notes included) and every `error` case refused with its status and code.
import { existsSync, readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';
import type { SetupModel } from '../src/api/types';
import { estimate, SelectionError } from '../src/setup/estimate';

const repo = fileURLToPath(new URL('../../models/estimate-cases.json', import.meta.url));
const copy = fileURLToPath(new URL('./fixtures/setup-estimate-cases.json', import.meta.url));
const path = existsSync(repo) ? repo : copy;
const fixture = JSON.parse(readFileSync(path, 'utf8')) as {
  model: SetupModel;
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

describe(`the estimate fixture (${path.endsWith('models/estimate-cases.json') ? 'models/' : 'the dashboard copy'})`, () => {
  it('has cases of both kinds', () => {
    expect(fixture.cases.filter((c) => c.expect).length).toBeGreaterThan(10);
    expect(fixture.cases.filter((c) => c.error).length).toBeGreaterThan(0);
  });
  for (const c of fixture.cases) {
    it(c.name, () => {
      if (c.error) {
        let err: unknown = null;
        try {
          estimate(fixture.model, c.selection as never);
        } catch (e) {
          err = e;
        }
        expect(err, 'refused').toBeInstanceOf(SelectionError);
        expect({ status: (err as SelectionError).status, code: (err as SelectionError).code }).toEqual(c.error);
        return;
      }
      close(estimate(fixture.model, c.selection as never), c.expect, 'estimate');
    });
  }
});
