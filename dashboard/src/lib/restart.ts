// VIS-209: a known restart of the engine (a Setup Guide install or reset) and what the page shows
// while it runs. The engine serves the dashboard and is the process being restarted, so the page asks
// the status responder (ops/engine_status.py, OPS-175): through the reverse proxy on the same origin,
// else on its own port (`status_port` of /health, remembered here). The pure parts are unit-tested
// (tests/restart.test.ts); <qse-restart-card> (setup/restart-card.ts) renders them.

import type { SetupJob, SetupJobStep } from '../api/types';

const JOB_KEY = 'tandem.setupJob';
const PORT_KEY = 'tandem.statusPort';
/** a job older than this in localStorage is not a known restart any more (a page left open) */
const KNOWN_MAX_MS = 2 * 3600_000;

export interface SetupStatusEngine {
  state: string;
  words?: string;
  since?: number | null;
  reason?: string | null;
  left_s?: number | null;
  typical_start_s?: number;
  profile?: string | null;
}
export interface SetupStatusPhase {
  id: string;
  label: string;
  done: boolean;
}
export interface SetupStatus {
  engine: SetupStatusEngine | null;
  job: {
    id: string;
    kind: string;
    state: string;
    model?: string | null;
    profile?: string | null;
    summary?: { model?: string } | null;
    step: { index: number; total: number; id: string; label: string; state: string; detail?: string | null; started_at?: number | null } | null;
  } | null;
  load: { phases: SetupStatusPhase[]; current: string | null; weights_detail?: string | null } | null;
}

function store(): Storage | null {
  try {
    return window.localStorage;
  } catch {
    return null;
  }
}

/** the setup job this browser follows while it runs (install-progress writes it, a final job clears it) */
export function rememberJob(job: SetupJob | null, now = Date.now()): void {
  const s = store();
  if (!s) return;
  try {
    if (job && job.state !== 'done' && job.state !== 'failed' && job.state !== 'rolled_back') s.setItem(JOB_KEY, JSON.stringify({ id: job.id, at: now }));
    else s.removeItem(JOB_KEY);
  } catch {
    /* a browser that does not store: no known restart across a refresh */
  }
}

/** the id of a setup job this browser follows, when it is recent enough to be the restart's */
export function knownJob(now = Date.now(), s: Storage | null = store()): string | null {
  try {
    const raw = s?.getItem(JOB_KEY);
    if (!raw) return null;
    const v = JSON.parse(raw) as { id?: string; at?: number };
    return typeof v.id === 'string' && /^[a-f0-9]{16}$/.test(v.id) && typeof v.at === 'number' && now - v.at < KNOWN_MAX_MS ? v.id : null;
  } catch {
    return null;
  }
}

/** /health's `status_port` (OPS-175), kept for the minutes the engine does not answer */
export function rememberStatusPort(health: { status_port?: number | null } | null): void {
  const p = health?.status_port;
  try {
    if (typeof p === 'number' && p > 0 && p < 65536) store()?.setItem(PORT_KEY, String(p));
  } catch {
    /* nothing to keep */
  }
}

export function statusUrls(apiBase: string, loc: { protocol: string; hostname: string } = location, s: Storage | null = store()): string[] {
  const urls = [`${apiBase}/v1/setup/status`];
  const p = Number(s?.getItem(PORT_KEY) || 0);
  if (p > 0) urls.push(`${loc.protocol}//${loc.hostname.includes(':') ? `[${loc.hostname}]` : loc.hostname}:${p}/v1/setup/status`);
  return urls;
}

/** GET /v1/setup/status from the first source that answers with the status's shape (the engine
 *  itself, the responder through the proxy, the responder on its port); null when none does */
export async function fetchStatus(urls: string[], fetchImpl: typeof fetch = fetch): Promise<SetupStatus | null> {
  for (const u of urls) {
    try {
      const r = await fetchImpl(u, { cache: 'no-store', credentials: u.startsWith('http') ? 'omit' : 'same-origin' });
      if (!r.ok) continue;
      const j = (await r.json()) as SetupStatus;
      if (j && typeof j === 'object' && 'job' in j && 'engine' in j) return j;
    } catch {
      /* the next source */
    }
  }
  return null;
}

export interface CardModel {
  title: string;
  step: string | null;
  phases: (SetupStatusPhase & { current: boolean })[];
  detail: string | null;
  elapsed: string;
  left: string;
  /** 0..1 */
  progress: number;
  /** the restart takes far longer than its typical start, or nothing answers for that long */
  late: boolean;
  /** neither the engine nor the responder answered: the card counts the time alone */
  blind: boolean;
}

const RESTART_STEPS = new Set(['switch', 'rollback', 'smoke']);

function mins(s: number): string {
  return s >= 90 ? `${Math.round(s / 60)} min` : `${Math.max(1, Math.round(s))} s`;
}

/** The card's words and numbers from the job the page holds, the status (null: nothing answered)
 *  and the clock. `since`: when the restart began (the switch step's start, else the card's). */
export function cardModel(job: SetupJob | null, st: SetupStatus | null, now: number, since: number): CardModel {
  const model = job?.summary?.model ?? st?.job?.summary?.model ?? job?.profile ?? st?.job?.profile ?? 'the new setup';
  const steps = (job?.steps ?? []).filter((s) => s.id !== 'rollback' || (s.state !== 'pending' && s.state !== 'skipped'));
  const cur: SetupJobStep | undefined = steps.find((s) => s.state === 'running' || s.state === 'waiting');
  const step = cur ? `Step ${steps.indexOf(cur) + 1} of ${steps.length}: ${cur.label}` : st?.job?.step ? `Step ${st.job.step.index} of ${st.job.step.total}: ${st.job.step.label}` : null;
  const typical = st?.engine?.typical_start_s ?? 300;
  const elapsedS = Math.max(0, (now - since) / 1000);
  const phases = (st?.load?.phases ?? []).map((p) => ({ ...p, current: p.id === st?.load?.current }));
  const done = phases.filter((p) => p.done).length;
  const byPhase = phases.length ? done / phases.length : 0;
  const byTime = Math.min(0.95, elapsedS / Math.max(30, typical));
  const leftS = st?.engine?.left_s ?? Math.max(0, typical - elapsedS);
  const rolling = cur?.id === 'rollback';
  return {
    title: rolling ? `Going back to the previous setup` : `Restarting the engine with ${model}`,
    step,
    phases,
    detail: st?.load?.weights_detail ?? null,
    elapsed: mins(elapsedS),
    left: leftS > 0 ? `about ${mins(leftS)} left` : 'any moment now',
    progress: Math.max(byPhase, byTime * 0.9),
    late: elapsedS > 3 * typical + 300,
    blind: st == null,
  };
}

/** whether the job is in its restart: a step that stops or starts the engine runs */
export function restarting(job: SetupJob | null): boolean {
  return !!job && job.state === 'running' && job.steps.some((s) => RESTART_STEPS.has(s.id) && (s.state === 'running' || s.state === 'waiting'));
}
