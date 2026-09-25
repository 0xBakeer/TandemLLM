// A synthetic engine log: a ring buffer with seq ids, lines of every level and source, a
// backlog seeded from the ledger's newest rows, and live lines. Mirrors SRV-30's shape.

import type { LogLevel, LogLine, LogSource } from '../src/api/types.ts';
import type { LedgerRow } from './generate.ts';

const LEVEL_RANK: Record<LogLevel, number> = { debug: 0, info: 1, warning: 2, error: 3 };

export function levelAtLeast(l: LogLevel, min: LogLevel): boolean {
  return LEVEL_RANK[l] >= LEVEL_RANK[min];
}

export function reqLine(r: LedgerRow): string {
  if (r.finish_reason === 'refused') {
    return `[req] ${r.request_id} refused ${r.status} queue full (waiting=8/8) Retry-After=5`;
  }
  const parts = [
    `[req] ${r.request_id}`,
    r.stream ? 'stream' : 'sync',
    `prompt=${r.prompt_tokens}`,
    `cached=${r.cached_tokens}`,
    `completion=${r.completion_tokens}`,
    r.thinking ? `reasoning=${r.reasoning_tokens}` : null,
    `finish=${r.finish_reason}`,
    `${Math.round(r.total_ms ?? 0)} ms`,
    r.decode_tps != null ? `${r.decode_tps.toFixed(2)} tok/s` : 'replay',
    r.blocks != null ? `blocks=${r.blocks}` : null,
    `ttft=${Math.round(r.ttft_ms ?? 0)} ms`,
    `cache=${r.cache_source}`,
    r.tool_calls ? `tool_calls=${r.tool_calls}` : null,
    r.error_type ? `error=${r.error_type}` : null,
  ];
  return parts.filter(Boolean).join(' ');
}

export function levelOf(r: LedgerRow): LogLevel {
  if (r.finish_reason === 'error') return 'error';
  if (r.finish_reason === 'timeout' || r.finish_reason === 'abandoned') return 'warning';
  return 'info';
}

const FLAVOUR: { level: LogLevel; source: LogSource; msg: string }[] = [
  { level: 'debug', source: 'http', msg: '127.0.0.1 - "POST /v1/chat/completions HTTP/1.1" 200' },
  { level: 'debug', source: 'http', msg: '127.0.0.1 - "GET /metrics HTTP/1.1" 200' },
  { level: 'debug', source: 'http', msg: '127.0.0.1 - "GET /health HTTP/1.1" 200' },
  { level: 'info', source: 'cache', msg: '[cache] prefix hit chunk=512 reused=1536 forwarded=423 entries=14 bytes=5.9G' },
  { level: 'info', source: 'cache', msg: '[cache] session hit reused=3072 forwarded=88' },
  { level: 'info', source: 'cache', msg: '[cache] put snapshot ctx=2371 bytes=412M budget=8.0G held=6.1G' },
  { level: 'info', source: 'cache', msg: '[cache] evict 2 entries (oldest) freed=824M' },
  { level: 'info', source: 'drafter', msg: '[drafter] LengthRouter arm=wide nodes=24 after=32 ms/blk=97.1' },
  { level: 'info', source: 'drafter', msg: '[drafter] lookup hit depth=32 deep chain fired' },
  { level: 'debug', source: 'drafter', msg: '[drafter] width chosen=8 accepted=3 verify=91.5 ms' },
  { level: 'info', source: 'think', msg: '[think] reasoning block closed after 230 tokens' },
  { level: 'warning', source: 'think', msg: '[think] forced close at reasoning budget 4096' },
  { level: 'info', source: 'server', msg: '[server] queue depth 2/8' },
  { level: 'warning', source: 'server', msg: '[server] LOSSY ACCEPT RULE disabled for this request (temperature 0.7)' },
  { level: 'warning', source: 'server', msg: '[server] client hung up after 1240 ms, finish=abandoned' },
  { level: 'error', source: 'server', msg: '!! RuntimeError: CUDA error: an illegal memory access was encountered (capped at 200 chars)' },
  { level: 'error', source: 'traceback', msg: 'Traceback (most recent call last):\n  File "server/app.py", line 1440, in _complete\n    for piece in eng.generate(...)\nRuntimeError: decode step failed' },
  { level: 'info', source: 'server', msg: '[server] ledger flush 41 rows in 12 ms queue=0' },
  { level: 'debug', source: 'server', msg: '[server] nvidia-smi sampled temp=41C power=27.4W sm=2405MHz' },
];

export class LogRing {
  private lines: LogLine[] = [];
  private seq = 0;
  private listeners = new Set<(l: LogLine) => void>();
  readonly capacity: number;

  constructor(capacity = 10_000) {
    this.capacity = capacity;
  }

  get lastSeq(): number {
    return this.seq;
  }

  push(level: LogLevel, source: LogSource, msg: string, request_id: string | null = null, ts = Date.now()): LogLine {
    const line: LogLine = { seq: ++this.seq, ts: new Date(ts).toISOString(), level, source, msg, request_id };
    this.lines.push(line);
    if (this.lines.length > this.capacity) this.lines.splice(0, this.lines.length - this.capacity);
    for (const l of this.listeners) l(line);
    return line;
  }

  pushRequest(r: LedgerRow): LogLine {
    return this.push(levelOf(r), 'req', reqLine(r), r.request_id, r.ts_ms);
  }

  /** Lines after `since` (exclusive), at or above `level`, containing `grep`, newest last, at most `limit`. */
  backlog(opts: { since?: number | null; level?: LogLevel; grep?: string | null; limit?: number }): LogLine[] {
    const level = opts.level ?? 'info';
    const since = opts.since ?? 0;
    const grep = opts.grep ? opts.grep.toLowerCase() : null;
    const out = this.lines.filter(
      (l) => l.seq > since && levelAtLeast(l.level, level) && (!grep || l.msg.toLowerCase().includes(grep)),
    );
    const limit = Math.min(5000, opts.limit ?? 500);
    return out.slice(-limit);
  }

  subscribe(fn: (l: LogLine) => void): () => void {
    this.listeners.add(fn);
    return () => this.listeners.delete(fn);
  }

  /** Seed the buffer: the newest rows as [req] lines, interleaved with flavour lines. */
  seed(rows: LedgerRow[], r: () => number, count = 600): void {
    const tail = rows.slice(-Math.ceil(count / 3));
    for (const row of tail) {
      const n = 1 + Math.floor(r() * 3);
      for (let i = 0; i < n; i++) {
        const f = FLAVOUR[Math.floor(r() * FLAVOUR.length)];
        this.push(f.level, f.source, f.msg, null, row.ts_ms - Math.floor(r() * 4000));
      }
      this.pushRequest(row);
    }
  }

  /** One random flavour line, for the live emitter. */
  flavour(r: () => number): LogLine {
    const f = FLAVOUR[Math.floor(r() * FLAVOUR.length)];
    return this.push(f.level, f.source, f.msg);
  }
}
