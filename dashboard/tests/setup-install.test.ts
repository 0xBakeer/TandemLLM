// The install page's job stream (VIS-149, VIS-151), its change list (VIS-155) and the tools that
// did not load (VIS-153), the true end of a job (VIS-211): their Gherkin as the test matrix. The server's
// half (the order it sends the job in) is tests/test_setup_job_stream.py.
import { describe, expect, it } from 'vitest';
import type { SetupEnvChange, SetupJob, SetupJobEvent, SetupJobStep, SetupJobStepId } from '../src/api/types';
import { ResumableStream, type SseEvent } from '../src/lib/sse';
import { appendLog, applyJobEvent, changeText, copyText, dedupes, doneMessage, durationText, finishedLines, isFinal, type JobLog, LOG_KEEP, needsJob, streamAction, streamFinished } from '../src/setup/job-view';

const IDS: SetupJobStepId[] = ['check', 'unload_assistant', 'fetch', 'preflight', 'write_config', 'switch', 'smoke', 'rollback'];

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

const DONE = job('done', { check: 'done', unload_assistant: 'done', fetch: 'done', preflight: 'done', write_config: 'done', switch: 'done', smoke: 'done', rollback: 'skipped' });

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

// VIS-155: the change list shows every effective change (the plan's half: tests/test_setup_change_list.py)
describe('VIS-155: the change list shows every effective change', () => {
  // the 16 changes of the LFM2.5 lfm25-exl3-2.5 to Qwen3.8-27B exl3-6.0 switch
  const kept: [string, string | null, string][] = [
    ['QSE_PARALLEL', '3', '4'], ['MAX_LEN', '131072', '262144'], ['QSE_SIDE_MAX_LEN', '131072', '98304'], ['QSE_KV_POOL', 'off', 'on'],
    ['QSE_PARALLEL_FLOOR_GB', '15', '10'], ['QSE_SIDE_GRAPHS', null, 'off'], ['QSE_SIDE_SLOT_CONST_GB', null, '1.5'], ['QSE_KV_HARD_FLOOR_GB', null, '10'],
    ['QSE_BARRIER_IDLE_OWNER', null, 'on'], ['QWEN38_SUFFIX_STORE', '', '/x/suffix'], ['QWEN38_SHADOW_MODEL', null, 'qwen3.5-0.8b'], ['VISION', null, 'on'], ['SERVED_MODEL', 'qwen38-spark-engine', 'TandemLLM'],
  ];
  const changes: SetupEnvChange[] = [
    ...kept.map(([key, current, next]) => ({ key, current, next, from: 'parallelism', removed: false })),
    { key: 'TEMPERATURE', current: '0.1', next: null, from: 'run/setup.env', removed: true },
    { key: 'TOP_K', current: '50', next: null, from: 'run/setup.env', removed: true },
    { key: 'REP_PENALTY', current: '1.05', next: null, from: 'run/setup.env', removed: true },
  ];

  it('Switching from LFM2.5 to Qwen3.8', () => {
    expect(changeText(changes[13])).toEqual({ from: '0.1', to: "removed, the engine's default" });
    expect(changeText({ key: 'X', current: '1', next: '2', from: 'run/setup.env', removed: true })).toEqual({ from: '1', to: 'removed, 2' });
  });

  it('The job page lists the changes', () => {
    const lines = changes.map((c) => changeText(c));
    expect(lines).toHaveLength(16);
    expect(lines[0]).toEqual({ from: '3', to: '4' });
    expect(lines[5]).toEqual({ from: 'unset', to: 'off' });
    expect(lines.filter((l) => l.to.startsWith('removed'))).toHaveLength(3);
    // a plan from a server without `removed` reads as before
    expect(changeText({ key: 'PORT', current: '8000', next: '8124', from: 'advanced.port' })).toEqual({ from: '8000', to: '8124' });
  });
});

// VIS-153: a tool that fails after the restart (the runner's half: tests/test_setup_tool_models.py)
describe('VIS-153: selected tools have their weights', () => {
  it('A component fails after the restart', () => {
    const w = 'Shadow reader did not load (FileNotFoundError: no config.json under /x/qwen3.5-0.8b).';
    const j: SetupJob = { ...DONE, message: `exl3-6.0 is live on port 8000. ${w}`, warnings: [w], notes: [] };
    expect(doneMessage(j)).toBe('exl3-6.0 is live on port 8000.');
    // a job from a server without warnings reads as before
    expect(doneMessage({ ...DONE, message: 'exl3-6.0 is live on port 8000.' })).toBe('exl3-6.0 is live on port 8000.');
    // the job events carry them through to the final job: the snapshot of a reload, and the done
    // event of a page that followed the job live
    expect(applyJobEvent(job('running'), { type: 'job', job: j })?.warnings).toEqual([w]);
    const live = applyJobEvent(job('running'), { type: 'done', state: 'done', message: j.message, warnings: [w], notes: ['Laya decision models is not loaded yet.'] });
    expect(live?.warnings).toEqual([w]);
    expect(live?.notes).toEqual(['Laya decision models is not loaded yet.']);
    expect(doneMessage(live!)).toBe('exl3-6.0 is live on port 8000.');
  });
});

/** the page's loop with its dedupe: `logsOnly` is VIS-211's rule, false the rule before it (every id) */
function followDeduped(chunks: string[], logsOnly: boolean): Promise<SetupJob | null> {
  return new Promise((resolve) => {
    let j: SetupJob | null = null;
    const seen = new Set<string>();
    const s: ResumableStream = new ResumableStream({
      url: () => '/v1/setup/jobs/1e1ecbede39315c9',
      fetchImpl: fetchOf(chunks),
      minBackoffMs: 5,
      maxBackoffMs: 5,
      onEvent: (ev) => {
        const d = JSON.parse(ev.data) as SetupJobEvent;
        if (ev.id && (!logsOnly || dedupes(d))) {
          if (seen.has(ev.id)) return;
          seen.add(ev.id);
        }
        if (d.type !== 'log') j = applyJobEvent(j, d);
        if (isFinal(j)) s.close();
      },
      onState: (st) => {
        if (st === 'reconnecting') s.close();
        else if (st === 'closed') resolve(j);
      },
    });
    s.start();
  });
}

describe('VIS-211: the install page shows the job\'s true end', () => {
  // the catch-up of a server before 0.11.3: the snapshot carried the id of the log line before it
  const old = [frame(24, { type: 'log', line: '[smoke] ready', ts: 1 }), frame(24, { type: 'job', job: DONE }), frame(31, { type: 'done', state: 'done', message: 'exl3-6.0 is live on port 8000.' })];

  it('a fresh page on a job whose last event is a log line holds the job (and the old rule did not)', async () => {
    const j = await followDeduped(old, true);
    expect(j?.state).toBe('done');
    expect(j?.steps.map((x) => x.state)).toEqual(['done', 'done', 'done', 'done', 'done', 'done', 'done', 'skipped']);
    // the negative control: deduping every id dropped the snapshot, and with no job the done event too
    expect(await followDeduped(old, false)).toBeNull();
  });

  it('a state event with no job in hand asks for the job by id', () => {
    expect(needsJob(null, { type: 'step', step: step('switch', 'done') })).toBe(true);
    expect(needsJob(null, { type: 'done', state: 'done', message: null })).toBe(true);
    expect(needsJob(null, { type: 'log', line: 'x', ts: 1 })).toBe(false);
    expect(needsJob(null, { type: 'job', job: DONE })).toBe(false);
    expect(needsJob(DONE, { type: 'done', state: 'done', message: null })).toBe(false);
  });

  it('the finished screen says what is ready, what it set up and how long it took', () => {
    const j: SetupJob = {
      ...DONE,
      created_at: 1760678000,
      ended_at: 1760678199,
      model: 'qwen3.8-27b',
      summary: { model: 'Qwen3.8-27B', profile: 'exl3-6.0', profile_label: 'EXL3 6.0 bits', tools: ['Laya decision models', 'Shadow reader'], slots: 4, window: 262144, side_window: 65536 },
    };
    const f = finishedLines(j, (n) => `${n / 1024}k`);
    expect(f.title).toBe('Qwen3.8-27B is ready');
    expect(f.lines).toEqual(['Weights: EXL3 6.0 bits', 'Tools: Laya decision models, Shadow reader', '4 parallel chats: 256k tokens for the first, 64k for each of the other 3', 'Took 3 min 19 s']);
    // a job before 0.11.3: its profile alone
    expect(finishedLines({ ...DONE, ended_at: null }, String)).toEqual({ title: 'The new setup is ready', lines: ['Weights: exl3-6.0'] });
    expect(finishedLines({ ...DONE, kind: 'reset' }, String).title).toBe('The shipped configuration is ready');
    expect(durationText(42)).toBe('42 s');
    expect(durationText(240)).toBe('4 min');
  });
});

describe('VIS-213: Copy diagnostics', () => {
  it('the Clipboard API where the page may use it', async () => {
    let got = '';
    expect(await copyText('report', { clipboard: { writeText: async (t: string) => void (got = t) } as unknown as Clipboard }, null)).toBe(true);
    expect(got).toBe('report');
  });
  it('plain http on a phone: no Clipboard API and no copy command: false, the page shows the text', async () => {
    const doc = {
      createElement: () => ({ value: '', setAttribute() {}, style: {}, select() {}, remove() {} }),
      body: { appendChild() {} },
      execCommand: () => false,
    } as unknown as Document;
    expect(await copyText('report', { clipboard: undefined as unknown as Clipboard }, doc)).toBe(false);
    const ok = { ...doc, execCommand: () => true } as unknown as Document;
    expect(await copyText('report', { clipboard: undefined as unknown as Clipboard }, ok)).toBe(true);
  });
});
