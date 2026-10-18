// inflightOnce (deep audit 0.11.3): a refresh racing itself used to hit /api/seed-prices
// twice — the early-return checked the result, never the flight in the air.
import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { inflightOnce } from '../src/lib/inflight';

describe('inflightOnce', () => {
  it('runs the getter once for calls made while it flies', async () => {
    let runs = 0;
    let release: (() => void) | null = null;
    const gate = new Promise<void>((res) => (release = res));
    const once = inflightOnce(async () => {
      runs += 1;
      await gate;
      return runs;
    });
    const a = once();
    const b = once();
    expect(runs).toBe(1);
    release!();
    expect(await a).toBe(1);
    expect(await b).toBe(1);
    expect(await once()).toBe(1);
    expect(runs).toBe(1);
  });
  it('keeps a rejected flight from re-firing either (the holder decides to retry, not the flight)', async () => {
    let runs = 0;
    const once = inflightOnce(async () => {
      runs += 1;
      throw new Error('down');
    });
    await expect(once()).rejects.toThrow('down');
    await expect(once()).rejects.toThrow('down');
    expect(runs).toBe(1);
  });
});

describe('the seed loads through it', () => {
  const src = readFileSync(new URL('../src/views/backends.ts', import.meta.url), 'utf8');
  it('a second loadSeed shares the flight in the air', () => {
    expect(src).toContain('inflightOnce');
    const load = src.slice(src.indexOf('private async loadSeed'), src.indexOf('private async loadSeed') + 500);
    expect(load).toContain('await this.seedFlight()');
    expect(load).toContain('if (this.seed) return;');
  });
});
