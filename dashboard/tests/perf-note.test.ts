// SRV-221: the Engine page's subtitle names the engine's real sequence count.
import { describe, expect, it } from 'vitest';
import { perfEngineNote } from '../src/lib/perf-note';

describe('the Performance > Engine subtitle', () => {
  it('says one request at a time only with one sequence', () => {
    expect(perfEngineNote(1)).toBe('the engine, one request at a time');
  });
  it('names two and four sequences, never "one request at a time"', () => {
    expect(perfEngineNote(2)).toBe('the engine, 2 requests at once');
    expect(perfEngineNote(4)).toBe('the engine, 4 requests at once');
    expect(perfEngineNote(2)).not.toMatch(/one request/);
  });
  it('claims no number when the count is unknown', () => {
    for (const n of [null, undefined, 0, Number.NaN]) {
      expect(perfEngineNote(n)).not.toMatch(/one request|at once/);
    }
  });
});
