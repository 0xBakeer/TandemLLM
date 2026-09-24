// Visibility-aware polling: stops while the tab is hidden (no background load on the engine),
// runs once immediately when it becomes visible again.

export interface Poller {
  start(): void;
  stop(): void;
  refresh(): Promise<void>;
  readonly running: boolean;
}

export function poll(fn: () => Promise<void>, intervalMs: number): Poller {
  let timer: ReturnType<typeof setTimeout> | null = null;
  let running = false;
  let inflight: Promise<void> | null = null;

  const run = async () => {
    if (inflight) return inflight;
    inflight = fn().catch(() => undefined).finally(() => (inflight = null));
    return inflight;
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
