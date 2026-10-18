// Visibility-aware polling: stops while the tab is hidden (no background load on the engine),
// runs once immediately when it becomes visible again.
//
// A refresh must never be swallowed while an older fetch is still in flight — param changes call
// refresh() and the new response has to replace the old one. So instead of a single-flight guard,
// every run claims a generation and fn receives an isCurrent() check: a response that lands after
// a newer run started belongs to the previous params and must be discarded by its own fn.
// Zero-arg fns stay valid (TS drops unused params); they simply keep last-to-finish semantics.

export type PollFn = (isCurrent: () => boolean) => Promise<void>;

export interface Poller {
  start(): void;
  stop(): void;
  refresh(): Promise<void>;
  readonly running: boolean;
}

export function poll(fn: PollFn, intervalMs: number): Poller {
  let timer: ReturnType<typeof setTimeout> | null = null;
  let running = false;
  let gen = 0;

  const run = async () => {
    const my = ++gen;
    await fn(() => gen === my).catch(() => undefined);
  };
  const schedule = () => {
    if (!running) return;
    timer = setTimeout(async () => {
      if (document.visibilityState === 'visible') await run();
      schedule();
    }, intervalMs);
  };
  const onVis = () => {
    if (document.visibilityState === 'visible' && running) void run();
  };
  return {
    get running() {
      return running;
    },
    start() {
      if (running) return;
      running = true;
      document.addEventListener('visibilitychange', onVis);
      void run();
      schedule();
    },
    stop() {
      running = false;
      if (timer) clearTimeout(timer);
      timer = null;
      document.removeEventListener('visibilitychange', onVis);
    },
    refresh: run,
  };
}
