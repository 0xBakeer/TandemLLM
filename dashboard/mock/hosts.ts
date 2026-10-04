// P1.5: two hosts for the System view -- the Spark itself (GB10, unified memory) with the engine's
// process, and the vLLM backend's own box read through a remote collector (a discrete GPU).

import type { HostInfo } from '../src/api/types.ts';

const iso = (t: number) => new Date(t).toISOString().replace(/\.\d{3}Z$/, 'Z');

export function mockHosts(now: number, busy: boolean, withBackends: boolean, rnd: () => number): HostInfo[] {
  const jitter = (x: number, k = 0.05) => Math.round(x * (1 + (rnd() - 0.5) * k) * 1000) / 1000;
  const local: HostInfo = {
    id: 'local',
    label: 'spark',
    source: 'local',
    url: null,
    backends: ['engine'],
    state: 'ok',
    error: null,
    memory: { total_bytes: 128_452_067_328, available_bytes: Math.round(jitter(busy ? 40.1e9 : 46.3e9)), used_bytes: null, swap_total_bytes: 17_179_865_088, swap_used_bytes: 795_000_000, unified: true },
    cpu: { count: 20, utilization: jitter(busy ? 0.12 : 0.03), load: [busy ? 1.42 : 0.31, 0.88, 0.71], temperature_c: busy ? 63.5 : 47.0 },
    gpus: [
      {
        index: 0, name: 'NVIDIA GB10', uuid: 'GPU-00000000-0000', utilization: busy ? jitter(0.94) : 0, memory_used_bytes: null, memory_total_bytes: null, memory_unified: true,
        power_w: busy ? jitter(56.0) : 11.2, power_limit_w: null, temperature_c: busy ? 61 : 44, sm_clock_mhz: busy ? 2470 : 1980, mem_clock_mhz: null, source: 'nvml',
      },
    ],
    gpu_source: 'nvml',
    disks: [{ name: 'nvme0n1', nvme: true, read_bps: busy ? 1.2e9 : 0, write_bps: busy ? 3.1e8 : 12_000, busy: busy ? 0.41 : 0.0 }],
    fs: [
      { path: '/', total_bytes: 3_936_000_000_000, free_bytes: 611_000_000_000 },
    ],
    net: [{ name: 'enP7s7', rx_bps: jitter(42_000), tx_bps: jitter(180_000) }],
    processes: [
      {
        backend: 'engine', label: 'TandemLLM', port: 8000, pid: 483418, found: true, via: 'self', name: 'python', cmdline: '~/venv/bin/python -u server/app.py --host 0.0.0.0 --port 8000 --served-model qwen38-spark-engine',
        rss_bytes: 4_812_000_000, threads: 61, open_files: 212, uptime_s: 7_412, cpu: busy ? 1.04 : 0.06, gpu_memory_bytes: 76_915_195_904, gpu_memory: 'measured', children: [],
        total_rss_bytes: 4_812_000_000, total_gpu_memory_bytes: 76_915_195_904, note: 'GPU memory comes out of the same unified pool as the RSS',
      },
    ],
    sampled_at: iso(now - 1300),
    age_s: 1.3,
    cost: { samples: 412, scans: 138, cpu_ms_total: 1_731.4, last_ms: 1.9, last_scan_ms: 6.2, fast_period_s: 5, scan_period_s: 15 },
  };
  if (!withBackends) return [local];
  const remote: HostInfo = {
    id: 'collector-1',
    label: 'gpu-box',
    source: 'collector',
    url: 'http://vllm.local:9108',
    backends: ['vllm1'],
    state: 'ok',
    error: null,
    memory: { total_bytes: 270_000_000_000, available_bytes: 201_000_000_000, used_bytes: 69_000_000_000, swap_total_bytes: 0, swap_used_bytes: 0, unified: false },
    cpu: { count: 32, utilization: jitter(0.07), load: [2.1, 1.9, 1.7], temperature_c: 52.0 },
    gpus: [
      {
        index: 0, name: 'NVIDIA RTX PRO 6000 Blackwell', uuid: 'GPU-0a1b', utilization: jitter(0.61), memory_used_bytes: 88_100_000_000, memory_total_bytes: 102_600_000_000, memory_unified: false,
        power_w: jitter(388), power_limit_w: 600, temperature_c: 66, sm_clock_mhz: 2610, mem_clock_mhz: 14_001, source: 'nvml',
      },
    ],
    gpu_source: 'nvml',
    disks: [{ name: 'nvme0n1', nvme: true, read_bps: 0, write_bps: 48_000, busy: 0.01 }],
    fs: [{ path: '/', total_bytes: 1_920_000_000_000, free_bytes: 744_000_000_000 }],
    net: [{ name: 'eth0', rx_bps: jitter(910_000), tx_bps: jitter(120_000) }],
    processes: [
      {
        backend: 'vllm1', label: 'vLLM (Qwen3.5-4B)', port: 8000, pid: 20511, found: true, via: 'psutil', name: 'vllm', cmdline: '~/vllm/bin/vllm serve Qwen/Qwen3.5-4B --port 8000 --api-key ***',
        rss_bytes: 1_900_000_000, threads: 48, open_files: 131, uptime_s: 93_120, cpu: 0.42, gpu_memory_bytes: 600_000_000, gpu_memory: 'measured',
        children: [{ pid: 20640, name: 'VLLM::EngineCore', rss_bytes: 6_300_000_000, threads: 96, cpu: 0.97, gpu_memory_bytes: 87_400_000_000 }],
        total_rss_bytes: 8_200_000_000, total_gpu_memory_bytes: 88_000_000_000, note: null,
      },
    ],
    sampled_at: iso(now - 2100),
    age_s: 2.1,
    cost: { samples: 9_120, scans: 3_040, cpu_ms_total: 21_904.2, last_ms: 2.3, last_scan_ms: 4.8, fast_period_s: 5, scan_period_s: 15 },
  };
  return [local, remote];
}
