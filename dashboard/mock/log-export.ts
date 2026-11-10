// The mock engine's log export (SRV-803), in the form of server/logbuf.py `export_text`: the
// header's fields in the contract's order (docs/contract/dashboard-v1/logs-export.example.txt),
// a blank line, then every line of the ring as `<UTC time> <level> <msg>`.

import type { LogLine } from '../src/api/types.ts';

export const EXPORT_TITLE = 'TandemLLM engine log';

export interface ExportBuild {
  version?: string | null;
  gitSha?: string | null;
  uptimeS?: number | null;
  now?: Date;
  /** how many older lines the ring has already dropped */
  dropped?: number;
  capacity?: number;
  logContent?: boolean | null;
}

export function uptimeText(seconds: number | null | undefined): string {
  if (seconds == null || !Number.isFinite(seconds) || seconds < 0) return 'unknown';
  let s = Math.floor(seconds);
  const d = Math.floor(s / 86400);
  s -= d * 86400;
  const h = Math.floor(s / 3600);
  s -= h * 3600;
  const m = Math.floor(s / 60);
  s -= m * 60;
  const parts: [number, string][] = [[d, 'd'], [h, 'h'], [m, 'min']];
  while (parts.length && !parts[0][0]) parts.shift();
  return [...parts.map(([n, unit]) => `${n} ${unit}`), `${s} s`].join(' ');
}

export function exportText(lines: LogLine[], b: ExportBuild = {}): string {
  // one line each: a newline or another control character reads as a space (server/logbuf.py `known`)
  const known = (v: string | null | undefined) => (v ?? '').replace(/[\u0000-\u001f\u007f-\u009f\u2028\u2029]+/g, ' ').trim() || 'unknown';
  const dropped = b.dropped ?? 0;
  const head = [
    EXPORT_TITLE,
    `version: ${known(b.version)}`,
    `commit: ${known(b.gitSha)}`,
    `uptime: ${uptimeText(b.uptimeS)}`,
    `exported: ${(b.now ?? new Date()).toISOString().replace(/\.\d+Z$/, 'Z')} (UTC)`,
    `count: ${lines.length} lines`,
    `first line: ${lines.length ? `${lines[0].ts} (UTC)` : 'none'}`,
    `last line: ${lines.length ? `${lines[lines.length - 1].ts} (UTC)` : 'none'}`,
    dropped > 0
      ? `ring: full, the ${dropped} lines before the first one are not in this file (the ring keeps ${b.capacity ?? 10_000})`
      : 'ring: not full, no line was dropped',
    `log content: ${b.logContent == null ? 'unknown' : b.logContent ? 'on (--log-content)' : 'off'}`,
  ];
  return [...head, '', ...lines.map((l) => `${l.ts} ${l.level} ${l.msg}`)].join('\n') + '\n';
}
