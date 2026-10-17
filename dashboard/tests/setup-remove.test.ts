// VIS-215: the weights on this box and the Remove question (src/setup/installed.ts).
import { describe, expect, it } from 'vitest';
import type { SetupInstalledWeights } from '../src/api/types';
import { removeQuestion, rows } from '../src/setup/installed';
import { finishedLines } from '../src/setup/job-view';

const W: SetupInstalledWeights = {
  active: 'exl3-6.0',
  free_gb: 412.3,
  assets: [],
  weights: [
    { model: 'qwen3.8-27b', model_name: 'Qwen3.8-27B', profile: 'exl3-4.5', label: 'EXL3 4.5 bpw', dir: '/h/q/4.5', removes: '/h/q/4.5', gb: 18.4, active: false, removable: true, shared_with: [] },
    { model: 'qwen3.8-27b', model_name: 'Qwen3.8-27B', profile: 'exl3-6.0', label: 'EXL3 6.0 bpw', dir: '/h/q/6.0', removes: '/h/q/6.0', gb: 23.3, active: true, removable: false, shared_with: [] },
    { model: 'qwen3.8-flash-next', model_name: 'Qwen3.8-Flash-Next', profile: 'flashnext-4.05', label: 'Flash-Next 4.05', dir: '/h/fn', removes: '/h/fn', gb: 80, active: false, removable: true, shared_with: ['community/x'] },
    { model: 'qwen3.8-flash-next', model_name: 'Qwen3.8-Flash-Next', profile: 'community/x', label: 'x', dir: '/h/fn', removes: '/h/fn', gb: 80, active: false, removable: true, shared_with: ['flashnext-4.05'] },
  ],
};

describe('VIS-215: remove a model in the Setup Guide', () => {
  it('the list: the running one first, then by size, a shared folder once', () => {
    expect(rows(W).map((r) => r.profile)).toEqual(['exl3-6.0', 'flashnext-4.05', 'exl3-4.5']);
    expect(rows(null)).toEqual([]);
  });
  it('the question names the GB, what stays and the line that downloads it again', () => {
    const q = removeQuestion(W.weights[0]);
    expect(q.title).toBe('Remove EXL3 4.5 bpw? It frees 18.4 GB.');
    expect(q.body).toContain('drafters and other files the models share stay');
    expect(q.again).toBe('bash ops/use-profile.sh exl3-4.5 --fetch-only');
    expect(removeQuestion(W.weights[2]).body).toBe('The folder is also community/x: they go together.');
  });
  it('the finished screen says the old weights went', () => {
    const job = { id: 'a', kind: 'apply', state: 'done', created_at: 0, ended_at: null, profile: 'exl3-4.5', steps: [], message: null, changes: [], rolled_back: false, failed_step: null, url: '', removed: { removed: [{ profile: 'exl3-6.0', dir: '/h', gb: 23.3 }], freed_gb: 23.3 } } as const;
    expect(finishedLines(job as never, String).lines).toContain('The old weights were removed: 23.3 GB freed');
  });
});
