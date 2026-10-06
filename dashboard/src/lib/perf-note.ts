// The Performance view's Engine page subtitle (SRV-221): the engine runs QSE_PARALLEL sequences
// (PL1), so "one request at a time" is said only when it is true. The count comes from
// /v1/dashboard/parallel's `engine_slots.n` (or /v1/config's `engine.parallel.n`); when it is
// unknown (an older server, the answer not in yet) the subtitle claims no number.

export function perfEngineNote(n: number | null | undefined): string {
  if (n == null || !Number.isFinite(n) || n < 1) return 'the engine: its live requests, its speed, its history';
  if (n === 1) return 'the engine, one request at a time';
  return `the engine, ${n} requests at once`;
}
