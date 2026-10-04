// P1.5: the System view's hosts -- the words for a GPU's and a process's memory, the per-host
// history, and the mock's two hosts against the contract.

import { describe, expect, it } from 'vitest';
import { gpuMemoryWords, HOST_KEEP, procGpuWords, procLabel, pushHostSamples, rate, sourceWords } from '../src/views/hosts';
import { mockHosts } from '../mock/hosts';
import type { HostInfo, HostProcess } from '../src/api/types';

const rnd = () => 0.5;

describe('hosts', () => {
  it('says unified on a GB10 and bytes on a discrete GPU', () => {
    const [local, remote] = mockHosts(Date.now(), true, true, rnd);
    expect(gpuMemoryWords(local.gpus[0])).toBe('unified with the CPU');
    expect(gpuMemoryWords(remote.gpus[0])).toBe('88.1 GB of 102.6 GB');
    expect(gpuMemoryWords({ ...remote.gpus[0], memory_used_bytes: null })).toBe('not reported');
  });

  it('reads a process tree’s GPU memory by how it was got', () => {
    const p = mockHosts(Date.now(), true, true, rnd)[1].processes![0];
    expect(procGpuWords(p)).toBe('88.0 GB');
    const as = (gpu_memory: HostProcess['gpu_memory']) => procGpuWords({ ...p, gpu_memory });
    expect(as('unified')).toBe('unified (no figure)');
    expect(as('none')).toBe('not on the GPU');
    expect(as('not_reported')).toBe('not reported');
    expect(procLabel(p)).toBe('vLLM (Qwen3.5-4B) (vllm1)');
    expect(procLabel({ ...p, backend: 'engine', label: 'TandemLLM' })).toBe('TandemLLM (this engine)');
    expect(procLabel({ ...p, backend: 'proxy' })).toBe('this proxy');
  });

  it('keeps 30 minutes per host and drops a host that went away', () => {
    const hosts = mockHosts(0, false, true, rnd);
    let m = new Map();
    for (let i = 0; i < HOST_KEEP + 5; i++) m = pushHostSamples(m, hosts, i);
    expect(m.get('local')!.length).toBe(HOST_KEEP);
    expect(m.get('local')!.at(-1)!.t).toBe(HOST_KEEP + 4);
    expect(m.get('collector-1')!.at(-1)!.gpu).toBeCloseTo(0.61, 1);
    m = pushHostSamples(m, [hosts[0]], 999);
    expect([...m.keys()]).toEqual(['local']);
    expect(pushHostSamples(m, undefined, 1).size).toBe(0);
  });

  it('formats rates and sources', () => {
    expect(rate(null)).toBe('—');
    expect(rate(1.2e9)).toBe('1.2 GB/s');
    const h = { source: 'exporter' } as HostInfo;
    expect(sourceWords(h)).toBe('node_exporter + DCGM');
  });
});
