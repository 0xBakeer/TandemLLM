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
