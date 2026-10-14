// The install page's job, folded from GET /v1/setup/jobs/{id} (pure; tests/setup-install.test.ts;
// the element is install.ts).

import type { SetupJob, SetupJobEvent } from '../api/types';

/** Fold one job event into the job. A final job stays final
 *  (VIS-149): a job or step event that would take it back to running is not applied. */
export function applyJobEvent(job: SetupJob | null, ev: SetupJobEvent): SetupJob | null {
  if (ev.type === 'job') return isFinal(job) && !isFinal(ev.job) ? job : ev.job;
  if (!job) return job;
  if (ev.type === 'step') {
    if (isFinal(job)) return job;
    const i = job.steps.findIndex((s) => s.id === ev.step.id);
    const steps = i >= 0 ? job.steps.map((s, j) => (j === i ? ev.step : s)) : [...job.steps, ev.step];
    return { ...job, steps };
  }
  if (ev.type === 'done') return { ...job, state: ev.state, message: ev.message, ended_at: job.ended_at ?? Date.now() / 1000 };
  return job;
}

export function isFinal(job: SetupJob | null): boolean {
  return !!job && (job.state === 'done' || job.state === 'failed' || job.state === 'rolled_back');
}

/** the log lines the page keeps (the runner's job keeps them all in events.jsonl) */
export const LOG_KEEP = 300;

export interface JobLog {
  lines: { line: string; ts: number }[];
  /** lines dropped from the top to stay at LOG_KEEP */
  dropped: number;
}

/** VIS-151: one more log line, the oldest dropped past `keep` and counted. */
export function appendLog(log: JobLog, line: { line: string; ts: number }, keep = LOG_KEEP): JobLog {
  const lines = [...log.lines, line];
  const over = Math.max(0, lines.length - keep);
  return { lines: over ? lines.slice(over) : lines, dropped: log.dropped + over };
}

/** VIS-151: the stream is done with once the server's `done` event arrived (it comes after the
 *  log and the final job), or once a final job's stream ended (a server that sends no `done`). */
export function streamFinished(ev: SetupJobEvent | null, job: SetupJob | null): boolean {
  return ev ? ev.type === 'done' : isFinal(job);
}

/** What the page does on a stream state: sign in, open, lost (404: the job's folder is gone,
 *  VIS-151), close (a final job's stream ended), away (the engine restarts), else nothing. */
export function streamAction(state: string, detail: string | undefined, job: SetupJob | null): 'signin' | 'open' | 'lost' | 'close' | 'away' | null {
  if (state === 'unauthorized') return 'signin';
  if (state === 'open') return 'open';
  if (state === 'error' && detail && /HTTP 404/.test(detail)) return 'lost';
  if (state === 'reconnecting' && streamFinished(null, job)) return 'close';
  if ((state === 'reconnecting' || state === 'error') && !isFinal(job)) return 'away';
  return null;
}
