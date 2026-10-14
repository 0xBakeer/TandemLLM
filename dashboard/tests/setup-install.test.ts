// The install page's job stream (VIS-149, VIS-151): their Gherkin as the test matrix. The server's
// half (the order it sends the job in) is tests/test_setup_job_stream.py.
import { describe, expect, it } from 'vitest';
import type { SetupJob, SetupJobEvent, SetupJobStep, SetupJobStepId } from '../src/api/types';
import { ResumableStream, type SseEvent } from '../src/lib/sse';
import { appendLog, applyJobEvent, isFinal, type JobLog, LOG_KEEP, streamAction, streamFinished } from '../src/setup/job-view';

const IDS: SetupJobStepId[] = ['check', 'unload_assistant', 'fetch', 'write_config', 'switch', 'smoke', 'rollback'];

function step(id: SetupJobStepId, state: SetupJobStep['state']): SetupJobStep {
  return { id, label: id, state, started_at: null, ended_at: null, detail: null };
}

function job(state: SetupJob['state'], states: Partial<Record<SetupJobStepId, SetupJobStep['state']>> = {}): SetupJob {
  return {
    id: '1e1ecbede39315c9',
    kind: 'apply',
    state,
    created_at: 1760460797,
    ended_at: null,
    profile: 'exl3-6.0',
    steps: IDS.map((i) => step(i, states[i] ?? 'pending')),
    message: null,
    changes: [],
    rolled_back: false,
    failed_step: null,
    url: 'http://<this box>:8000/dashboard/',
  };
}

const DONE = job('done', { check: 'done', unload_assistant: 'done', fetch: 'done', write_config: 'done', switch: 'done', smoke: 'done', rollback: 'skipped' });

/** a fetch whose body hands over `chunks` (each one read() of the stream) and then ends */
function fetchOf(chunks: string[]): typeof fetch {
  return (async () => {
    const enc = new TextEncoder();
    const body = new ReadableStream<Uint8Array>({
      start(c) {
        for (const ch of chunks) c.enqueue(enc.encode(ch));
        c.close();
      },
    });
    return new Response(body, { status: 200, headers: { 'Content-Type': 'text/event-stream' } });
  }) as typeof fetch;
}

const frame = (id: number | null, ev: SetupJobEvent) => `${id != null ? `id: ${id}\n` : ''}data: ${JSON.stringify(ev)}\n\n`;

/** the page's own loop (install.ts onEvent): fold, close on a final job */
function follow(chunks: string[]): Promise<{ job: SetupJob | null; seen: SseEvent[]; log: number }> {
  return new Promise((resolve) => {
    let j: SetupJob | null = null;
    let log = 0;
    const seen: SseEvent[] = [];
    const s: ResumableStream = new ResumableStream({
      url: () => '/v1/setup/jobs/1e1ecbede39315c9',
      fetchImpl: fetchOf(chunks),
      minBackoffMs: 5,
      maxBackoffMs: 5,
      onEvent: (ev) => {
        seen.push(ev);
        const d = JSON.parse(ev.data) as SetupJobEvent;
        if (d.type === 'log') log++;
        else j = applyJobEvent(j, d);
        if (isFinal(j)) s.close();
      },
      onState: (st) => {
        if (st === 'reconnecting') s.close();
        else if (st === 'closed') resolve({ job: j, seen, log });
      },
    });
    s.start();
  });
}

describe('VIS-149: the install page shows the job as it is on the box', () => {
  it('Reload after the job finished', async () => {
    // the server's order (setup_api._catch_up): the log on disk, the job now, its done event
    const lines = Array.from({ length: 236 }, (_, i) => frame(i + 6, { type: 'log', line: `line ${i}`, ts: i }));
    const all = lines.join('') + frame(257, { type: 'job', job: DONE }) + frame(258, { type: 'done', state: 'done', message: 'exl3-6.0 is live on port 8000.' });
    // whatever the network's split, the page ends on the final state
    for (const cut of [1, 100, 9596, all.length - 120, all.length - 1]) {
      const r = await follow([all.slice(0, cut), all.slice(cut)]);
      expect(r.job?.state).toBe('done');
      expect(r.job!.steps.filter((s) => s.state === 'running' || s.state === 'pending')).toEqual([]);
    }
  });

  it('Reload during the download', async () => {
    const running = job('running', { check: 'done', unload_assistant: 'done', fetch: 'running' });
    const chunks = [Array.from({ length: 145 }, (_, i) => frame(i + 6, { type: 'log', line: `[fetch] ${i}`, ts: i })).join('') + frame(150, { type: 'job', job: running })];
    const r = await follow(chunks);
    expect(r.job?.state).toBe('running');
    const st = Object.fromEntries(r.job!.steps.map((s) => [s.id, s.state]));
    expect(st).toMatchObject({ check: 'done', unload_assistant: 'done', fetch: 'running', switch: 'pending' });
    expect(r.log).toBe(145);
  });

  it('Events after close are ignored', async () => {
    // one chunk: the final job, then the old server's replay of the job at its start and more
    const old = job('running');
    const chunk =
      frame(null, { type: 'job', job: DONE }) +
      frame(1, { type: 'job', job: old }) +
      frame(2, { type: 'step', step: step('fetch', 'running') }) +
      frame(3, { type: 'log', line: '[fetch] base-fp8', ts: 1 });
    const r = await follow([chunk]);
    expect(r.seen).toHaveLength(1);
    expect(r.job?.state).toBe('done');
  });

  it('A final state never regresses', () => {
    let j: SetupJob | null = DONE;
    j = applyJobEvent(j, { type: 'job', job: job('running') });
    expect(j?.state).toBe('done');
    j = applyJobEvent(j, { type: 'step', step: step('fetch', 'running') });
    expect(j!.steps.find((s) => s.id === 'fetch')!.state).toBe('done');
    // a final job may still be replaced by another final one (rolled back after failed)
    expect(applyJobEvent(job('failed'), { type: 'job', job: job('rolled_back') })?.state).toBe('rolled_back');
    // and a running job still folds its steps
    const r = applyJobEvent(job('running'), { type: 'step', step: step('check', 'done') });
    expect(r!.steps.find((s) => s.id === 'check')!.state).toBe('done');
  });

  it('a later start() ends the earlier run: one stream dispatches at a time', async () => {
    const seen: string[] = [];
    let n = 0;
    const s = new ResumableStream({
      url: () => '/x',
      fetchImpl: (async () => {
        n++;
        const me = n;
        const body = new ReadableStream<Uint8Array>({
          async start(c) {
            await new Promise((r) => setTimeout(r, me === 1 ? 30 : 0));
            c.enqueue(new TextEncoder().encode(`data: run${me}\n\n`));
            c.close();
          },
        });
        return new Response(body, { status: 200 });
      }) as typeof fetch,
      minBackoffMs: 1000,
      onEvent: (ev) => seen.push(ev.data),
    });
    s.start();
    await new Promise((r) => setTimeout(r, 5));
    s.start();
    await new Promise((r) => setTimeout(r, 60));
    s.close();
    expect(seen).toEqual(['run2']);
  });
});

// VIS-151: the log of a finished install stays readable (the Gherkin of VIS-151)

/** the page's fold (install.ts onEvent) over the server's frames, cut into `chunks` */
function page(chunks: string[]): Promise<{ job: SetupJob | null; log: JobLog; lost: boolean }> {
  return new Promise((resolve) => {
    let j: SetupJob | null = null;
    let log: JobLog = { lines: [], dropped: 0 };
    let lost = false;
    const seen = new Set<string>();
    const s: ResumableStream = new ResumableStream({
      url: () => '/v1/setup/jobs/1e1ecbede39315c9',
      fetchImpl: chunks.length ? fetchOf(chunks) : ((async () => new Response('{"error":{"code":"not_found"}}', { status: 404 })) as typeof fetch),
      minBackoffMs: 5,
      maxBackoffMs: 5,
      onEvent: (ev) => {
        if (ev.id) {
          if (seen.has(ev.id)) return;
          seen.add(ev.id);
        }
        const d = JSON.parse(ev.data) as SetupJobEvent;
        if (d.type === 'log') {
          log = appendLog(log, { line: d.line, ts: d.ts });
          return;
        }
        j = applyJobEvent(j, d);
        if (streamFinished(d, j)) s.close();
      },
      onState: (st, detail) => {
        const act = streamAction(st, detail, j);
        if (act === 'lost') lost = true;
        if (act === 'lost' || act === 'close' || act === 'away') s.close();
        if (st === 'closed') resolve({ job: j, log, lost });
      },
    });
    s.start();
  });
}

const finished = (n: number) =>
  Array.from({ length: n }, (_, i) => frame(i + 6, { type: 'log', line: `line ${i}`, ts: i })).join('') +
  frame(n + 21, { type: 'job', job: DONE }) +
  frame(n + 22, { type: 'done', state: 'done', message: 'exl3-6.0 is live on port 8000.' });

describe('VIS-151: the log of a finished install stays readable', () => {
  it('Reload a finished job', async () => {
    const all = finished(236);
    // the 18:56 reload: frame 0 alone in the first chunk; and other splits
    for (const cut of [all.indexOf('\n\n') + 2, 9596, all.length - 1]) {
      const r = await page([all.slice(0, cut), all.slice(cut)]);
      expect(r.job?.state).toBe('done');
      expect(r.log.lines).toHaveLength(236);
      expect(r.log.lines[0].line).toBe('line 0');
      expect(r.log.lines[235].line).toBe('line 235');
    }
  });

  it('Live job turns final', async () => {
    const running = job('running', { check: 'done', unload_assistant: 'done', fetch: 'running' });
    const first = Array.from({ length: 120 }, (_, i) => frame(i + 6, { type: 'log', line: `line ${i}`, ts: i })).join('') + frame(130, { type: 'job', job: running });
    const later =
      Array.from({ length: 30 }, (_, i) => frame(131 + i, { type: 'log', line: `later ${i}`, ts: i })).join('') +
      frame(161, { type: 'step', step: step('smoke', 'done') }) +
      frame(162, { type: 'done', state: 'done', message: 'exl3-6.0 is live on port 8000.' });
    const r = await page([first, later]);
    expect(r.job?.state).toBe('done');
    expect(r.log.lines).toHaveLength(150);
    expect(r.log.lines[119].line).toBe('line 119');
    expect(r.log.lines[149].line).toBe('later 29');
  });

  it('Job folder gone', async () => {
    const r = await page([]);
    expect(r.lost).toBe(true);
    expect(r.log.lines).toHaveLength(0);
  });

  it('a log past the page\'s 300 lines says how many were dropped from the top', async () => {
    const r = await page([finished(412)]);
    expect(r.log.lines).toHaveLength(LOG_KEEP);
    expect(r.log.dropped).toBe(112);
    expect(r.log.lines[0].line).toBe('line 112');
    expect(appendLog({ lines: [], dropped: 0 }, { line: 'a', ts: 0 }, 1)).toEqual({ lines: [{ line: 'a', ts: 0 }], dropped: 0 });
  });

  it('the stream of a final job ends at its done event, not at the final job before it', () => {
    expect(streamFinished({ type: 'job', job: DONE }, DONE)).toBe(false);
    expect(streamFinished({ type: 'done', state: 'done', message: null }, DONE)).toBe(true);
    // a server that sends no done event: the final job's stream ending closes it
    expect(streamAction('reconnecting', undefined, DONE)).toBe('close');
    expect(streamAction('reconnecting', undefined, job('running'))).toBe('away');
    expect(streamAction('error', 'HTTP 404', DONE)).toBe('lost');
  });
});
