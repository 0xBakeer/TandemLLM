// OPS: the poller's generations: a refresh during a flight is never swallowed,
// the timer chain never overlaps, and a stop() during a flight schedules nothing.
import { beforeEach, afterEach, describe, expect, it, vi } from 'vitest';

// lib/poll reads document.visibilityState and the two listeners; the node test
// env has none, so the whole surface the module touches is stubbed.
let hidden = false;
const listeners = new Set<() => void>();
(globalThis as Record<string, unknown>).document = {
  get visibilityState() { return hidden ? 'hidden' : 'visible'; },
  addEventListener: (_: string, fn: () => void) => { listeners.add(fn); },
  removeEventListener: (_: string, fn: () => void) => { listeners.delete(fn); },
};

const { poll } = await import('../src/lib/poll');

const tick = () => vi.advanceTimersByTimeAsync(0);   // real microtask flush; setTimeout itself is faked

describe('poll', () => {
  beforeEach(() => { vi.useFakeTimers(); hidden = false; });
  afterEach(() => { vi.useRealTimers(); listeners.clear(); });

  it('re-runs fn when refresh lands during a flight, and the older run sees itself superseded', async () => {
    let release: () => void = () => {};
    const gates: Array<() => void> = [];
    const currentPerRun: Array<() => boolean> = [];
    const p = poll(async (isCurrent) => {
      currentPerRun.push(isCurrent);
      await new Promise<void>((res) => { gates.push(res); });
    }, 1000);
    p.start();
    await tick();
    expect(gates.length).toBe(1);
    void p.refresh();
    await tick();
    expect(gates.length).toBe(2);          // a refresh during a flight re-runs fn
    gates.forEach((g) => g());
    await tick();
    expect(currentPerRun[0]()).toBe(false); // the first run is the previous params
    expect(currentPerRun[1]()).toBe(true);
    p.stop();
  });

  it('the timer chain never has two of its own runs at once', async () => {
    // the schedule is chain-shaped: the next timer arms only after the previous
    // run has finished, so no two chain runs can coexist (start()'s initial run
    // is a separate flight — the generation counter covers its supersession).
    let inFlight = 0;
    let overlapped = false;
    let chainStarts = 0;
    const p = poll(async () => {
      inFlight += 1;
      if (inFlight > 1) overlapped = true;
      await new Promise<void>((res) => setTimeout(res, 5));   // outlasts nothing but a refresh
      inFlight -= 1;
      chainStarts += 1;
    }, 20);
    p.start();
    await vi.advanceTimersByTimeAsync(300);                   // several intervals' worth
    expect(chainStarts).toBeGreaterThanOrEqual(3);            // the chain kept running
    expect(overlapped).toBe(false);
    p.stop();
  });

  it('a stop() during a flight schedules nothing further', async () => {
    let release: () => void = () => {};
    let runs = 0;
    const p = poll(async () => {
      runs += 1;
      await new Promise<void>((res) => { release = res; });
    }, 20);
    p.start();
    await tick();
    p.stop();
    release();
    await vi.advanceTimersByTimeAsync(500);
    expect(runs).toBe(1);                   // the chain stopped with the poller
    void tick;
  });
});
