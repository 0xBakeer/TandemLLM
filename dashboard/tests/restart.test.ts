// VIS-209: the restart card's model (src/lib/restart.ts): the step, the load's phases, the time and the
// fallback with nothing to ask; the known restart that survives a refresh; the status sources.
import { describe, expect, it } from 'vitest';
import type { SetupJob, SetupJobStep, SetupJobStepId } from '../src/api/types';
import { cardModel, fetchStatus, knownJob, rememberJob, restarting, type SetupStatus, statusUrls } from '../src/lib/restart';

const IDS: SetupJobStepId[] = ['check', 'unload_assistant', 'fetch', 'preflight', 'write_config', 'switch', 'smoke', 'rollback'];
function job(states: Partial<Record<SetupJobStepId, SetupJobStep['state']>>, state: SetupJob['state'] = 'running'): SetupJob {
  return {
    id: '0123456789abcdef', kind: 'apply', state, created_at: 1760678000, ended_at: null, profile: 'exl3-6.0',
    steps: IDS.map((id) => ({ id, label: id === 'switch' ? 'Restart the engine with the new setup' : id, state: states[id] ?? 'pending', started_at: id === 'switch' ? 1760678100 : null, ended_at: null, detail: null })),
    message: null, changes: [], rolled_back: false, failed_step: null, url: '',
    summary: { model: 'Qwen3.8-27B', profile: 'exl3-6.0', profile_label: 'EXL3 6.0', tools: [], slots: 1, window: 262144, side_window: null },
  };
}
const RUNNING = job({ check: 'done', unload_assistant: 'done', fetch: 'skipped', preflight: 'done', write_config: 'done', switch: 'running' });
const STATUS: SetupStatus = {
  engine: { state: 'starting', typical_start_s: 240, left_s: 120 },
  job: null,
  load: { phases: [{ id: 'weights', label: 'Loading the weights', done: true }, { id: 'drafters', label: 'Loading the drafters', done: true }, { id: 'warmup', label: 'Warming up', done: false }, { id: 'graphs', label: 'Capturing the verify graphs', done: false }, { id: 'ready', label: 'Opening the port', done: false }], current: 'warmup', weights_detail: null },
};

class Mem implements Storage {
  m = new Map<string, string>();
  get length() { return this.m.size; }
  clear() { this.m.clear(); }
  getItem(k: string) { return this.m.get(k) ?? null; }
  key(i: number) { return [...this.m.keys()][i] ?? null; }
  removeItem(k: string) { this.m.delete(k); }
  setItem(k: string, v: string) { this.m.set(k, v); }
}

describe('VIS-209: one restart status card', () => {
  it('the engine restarts during an install: step, phase, time left', () => {
    const t0 = 1760678100_000;
    const m = cardModel(RUNNING, STATUS, t0 + 120_000, t0);
    expect(m.title).toBe('Restarting the engine with Qwen3.8-27B');
    expect(m.step).toBe('Step 6 of 7: Restart the engine with the new setup');
    expect(m.phases.find((p) => p.current)?.id).toBe('warmup');
    expect(m.elapsed).toBe('2 min');
    expect(m.left).toBe('about 2 min left');
    expect(m.progress).toBeCloseTo(0.45, 5); // the time (2 of 4 min) leads the phases (2 of 5)
    expect(m.blind).toBe(false);
    expect(m.late).toBe(false);
  });

  it('nothing answers: the card counts the time against a typical start, then says it is late', () => {
    const t0 = 1_000_000;
    const m = cardModel(RUNNING, null, t0 + 60_000, t0);
    expect(m.blind).toBe(true);
    expect(m.phases).toEqual([]);
    expect(m.left).toBe('about 4 min left');
    expect(m.progress).toBeCloseTo((60 / 300) * 0.9, 5);
    expect(cardModel(RUNNING, null, t0 + 1300_000, t0).late).toBe(true);
  });

  it('a rollback says it goes back', () => {
    const j = job({ switch: 'failed', rollback: 'running' });
    expect(cardModel(j, null, 0, 0).title).toBe('Going back to the previous setup');
    expect(cardModel(j, null, 0, 0).step).toBe('Step 8 of 8: rollback');
  });

  it('which jobs are in their restart', () => {
    expect(restarting(RUNNING)).toBe(true);
    expect(restarting(job({ fetch: 'running' }))).toBe(false);
    expect(restarting(job({ switch: 'done' }, 'done'))).toBe(false);
  });

  it('a refresh keeps the known restart; a final job or an old one is none', () => {
    const s = new Mem();
    const real = globalThis.localStorage;
    Object.defineProperty(globalThis, 'localStorage', { value: s, configurable: true });
    Object.defineProperty(globalThis, 'window', { value: { localStorage: s }, configurable: true, writable: true });
    try {
      rememberJob(RUNNING, 1_000);
      expect(knownJob(2_000, s)).toBe('0123456789abcdef');
      expect(knownJob(1_000 + 3 * 3600_000, s)).toBeNull();
      rememberJob({ ...RUNNING, state: 'done' }, 3_000);
      expect(knownJob(4_000, s)).toBeNull();
      s.setItem('tandem.statusPort', '8009');
      expect(statusUrls('/x', { protocol: 'http:', hostname: 'box.lan' }, s)).toEqual(['/x/v1/setup/status', 'http://box.lan:8009/v1/setup/status']);
    } finally {
      Object.defineProperty(globalThis, 'localStorage', { value: real, configurable: true });
    }
  });

  it('the status comes from the first source with its shape', async () => {
    const calls: string[] = [];
    const f = (async (u: string) => {
      calls.push(u);
      if (u === 'a') throw new TypeError('network');
      if (u === 'b') return new Response('{"error": {}}', { status: 503 });
      return new Response(JSON.stringify(STATUS), { status: 200 });
    }) as unknown as typeof fetch;
    expect(await fetchStatus(['a', 'b', 'c'], f)).toEqual(STATUS);
    expect(calls).toEqual(['a', 'b', 'c']);
    expect(await fetchStatus(['a', 'b'], f)).toBeNull();
  });
});
