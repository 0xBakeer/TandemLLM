// The install page's job, folded from GET /v1/setup/jobs/{id} (pure; tests/setup-install.test.ts;
// the element is install.ts).

import type { SetupEnvChange, SetupJob, SetupJobEvent } from '../api/types';

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
  if (ev.type === 'done')
    return {
      ...job,
      state: ev.state,
      message: ev.message,
      ended_at: job.ended_at ?? Date.now() / 1000,
      ...(ev.warnings ? { warnings: ev.warnings } : {}),
      ...(ev.notes ? { notes: ev.notes } : {}),
    };
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

/** VIS-155: one changed setting as words, old -> new; a key the new setup drops says so and what
 *  the engine then runs with (`unset`: the engine's own default). */
export function changeText(c: SetupEnvChange, unset = 'unset'): { from: string; to: string } {
  const to = c.next ?? unset;
  return { from: c.current ?? unset, to: c.removed ? `removed, ${to === unset ? "the engine's default" : to}` : to };
}

/** VIS-153: a done job's message without the tool warnings the server appends to it (the page
 *  lists `warnings` on their own). */
export function doneMessage(job: SetupJob): string {
  const msg = job.message ?? '';
  const tail = (job.warnings ?? []).join(' ');
  return tail && msg.endsWith(tail) ? msg.slice(0, msg.length - tail.length).trim() : msg;
}

/** VIS-211: the page dedupes log lines by their id only. The job snapshot of a catch-up and the step
 *  and done events are the job's state, idempotent: a server before 0.11.3 gave the snapshot the id
 *  of the log line before it, and a page that dropped it as a repeat held no job and spun over a
 *  finished install. */
export function dedupes(ev: SetupJobEvent): boolean {
  return ev.type === 'log';
}

/** VIS-211: a state event that arrives while the page holds no job: read the job by id instead of
 *  dropping it (and every later one with it). */
export function needsJob(job: SetupJob | null, ev: SetupJobEvent): boolean {
  return !job && ev.type !== 'log' && ev.type !== 'job';
}

/** VIS-211: how long without a job in hand before the page says it cannot read the install */
export const STALE_MS = 30_000;

/** VIS-211: what a finished job set up, as the finished screen's lines (a job before 0.11.3 has no
 *  summary: its profile alone). */
export function finishedLines(job: SetupJob, tok: (n: number) => string): { title: string; lines: string[] } {
  const s = job.summary;
  const model = s?.model ?? job.model ?? 'The new setup';
  const lines: string[] = [];
  lines.push(`Weights: ${s?.profile_label ?? job.profile}`);
  if (s?.tools?.length) lines.push(`Tools: ${s.tools.join(', ')}`);
  if (s?.slots != null && s.window != null)
    lines.push(
      s.slots > 1 && s.side_window != null
        ? `${s.slots} parallel chats: ${tok(s.window)} tokens for the first, ${tok(s.side_window)} for each of the other ${s.slots - 1}`
        : `${s.slots} chat at a time, ${tok(s.window)} tokens`,
    );
  if (job.ended_at && job.created_at) lines.push(`Took ${durationText(job.ended_at - job.created_at)}`);
  if (job.removed?.removed.length) lines.push(`The old weights were removed: ${job.removed.freed_gb.toFixed(1)} GB freed`);
  return { title: job.kind === 'reset' ? 'The shipped configuration is ready' : `${model} is ready`, lines };
}

export function durationText(s: number): string {
  if (s < 90) return `${Math.max(1, Math.round(s))} s`;
  const m = Math.floor(s / 60);
  const r = Math.round(s - m * 60);
  return r ? `${m} min ${r} s` : `${m} min`;
}

/** VIS-213: copy `text` to the clipboard: the Clipboard API where the page may use it (https, or a
 *  secure context), else a hidden text box and execCommand (plain http on a phone); false when
 *  neither worked (the page then shows the text to select by hand). */
export async function copyText(text: string, nav: Pick<Navigator, 'clipboard'> | null = typeof navigator === 'undefined' ? null : navigator, doc: Document | null = typeof document === 'undefined' ? null : document): Promise<boolean> {
  try {
    if (nav?.clipboard?.writeText) {
      await nav.clipboard.writeText(text);
      return true;
    }
  } catch {
    /* not allowed here: the fallback */
  }
  if (!doc) return false;
  const ta = doc.createElement('textarea');
  ta.value = text;
  ta.setAttribute('readonly', '');
  ta.style.position = 'fixed';
  ta.style.opacity = '0';
  doc.body.appendChild(ta);
  ta.select();
  let ok = false;
  try {
    ok = doc.execCommand('copy');
  } catch {
    ok = false;
  }
  ta.remove();
  return ok;
}
