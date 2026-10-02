// The Memory screen's rules (VIS-48), kept out of the component so the tests read them directly:
// a transcript block cut into its sections (the server's format, server/conductor/transcripts.py),
// the runtime lines with the hidden notes and their journal rows, a session's one-line summary,
// and the transcript settings with their defaults.

import type { MemorySessionRow, TranscriptSettings } from '../api/conductor';

export const TRANSCRIPT_DEFAULTS: TranscriptSettings = { scope: 'tools', retention_days: 30, clip_chars: 4000, loop_chars: 200 };

export type SectionKind = 'user' | 'assistant' | 'tool_result' | 'tool_call' | 'reasoning' | 'summary' | 'runtime' | 'other';
export interface Section {
  kind: SectionKind;
  /** the heading's words after `###`: "tool result (bash, 5,076 chars, clipped to 4,000)" */
  title: string;
  /** "bash" of a tool result or a call, else null */
  name: string | null;
  clipped: boolean;
  body: string;
}
export type RuntimeKind = 'hidden' | 'replayed' | 'acts' | 'checker' | 'snapshot' | 'finish' | 'redactions' | 'other';
export interface RuntimeLine {
  kind: RuntimeKind;
  text: string;
  /** the journal row a hidden note is in: a router decision or a monitor event */
  journal: string | null;
  verdict: 'PASS' | 'FAIL' | null;
}
export interface ParsedBlock {
  head: string;
  /** "compaction: 120,200 tokens in, 4,100 tokens out" of a compaction block */
  compaction: string | null;
  sections: Section[];
  runtime: RuntimeLine[];
}

// only the server's own headings start a section; a message's markdown is left in its body
const HEADING = /^### (user|assistant|summary|runtime|tool result \((.*)\)|tool call \((.*)\)|reasoning \((.*)\))$/;

function sectionOf(m: RegExpMatchArray): Omit<Section, 'body'> {
  const word = m[1];
  if (m[2] !== undefined) return { kind: 'tool_result', title: word, name: m[2].split(',')[0].trim() || null, clipped: /clipped to/.test(m[2]) };
  if (m[3] !== undefined) return { kind: 'tool_call', title: word, name: m[3].trim() || null, clipped: false };
  if (m[4] !== undefined) return { kind: 'reasoning', title: word, name: null, clipped: false };
  const kind: SectionKind = word === 'user' || word === 'assistant' || word === 'summary' || word === 'runtime' ? word : 'other';
  return { kind, title: word, name: null, clipped: false };
}

/** One block's text as its header, its sections in order and its runtime lines. */
export function parseBlock(text: string): ParsedBlock {
  const lines = text.replace(/\n+$/, '').split('\n');
  const head = lines[0] ?? '';
  let compaction: string | null = null;
  const sections: Section[] = [];
  let cur: (Omit<Section, 'body'> & { lines: string[] }) | null = null;
  const close = () => {
    if (cur) sections.push({ kind: cur.kind, title: cur.title, name: cur.name, clipped: cur.clipped, body: cur.lines.join('\n').replace(/^\n+|\n+$/g, '') });
  };
  for (const line of lines.slice(1)) {
    const m = line.match(HEADING);
    if (m) {
      close();
      cur = { ...sectionOf(m), lines: [] };
    } else if (cur) {
      cur.lines.push(line);
    } else if (line.startsWith('compaction: ')) {
      compaction = line;
    }
  }
  close();
  const rt = sections.find((x) => x.kind === 'runtime');
  return { head, compaction, sections: sections.filter((x) => x.kind !== 'runtime'), runtime: rt ? runtimeLines(rt.body) : [] };
}

/** The runtime section's lines, each with its kind; a hidden note names its journal row. */
export function runtimeLines(body: string): RuntimeLine[] {
  return body
    .split('\n')
    .filter((l) => l.trim())
    .map((text) => {
      const kind = (['hidden', 'replayed', 'acts', 'checker', 'snapshot', 'finish', 'redactions'] as RuntimeKind[]).find((k) => text.startsWith(`${k}: `)) ?? 'other';
      const j = text.match(/ · journal ([A-Za-z0-9_-]{4,40})$/);
      const v = kind === 'checker' ? text.match(/ (PASS|FAIL)(?: <-|$)/) : null;
      return { kind, text, journal: j ? j[1] : null, verdict: v ? (v[1] as 'PASS' | 'FAIL') : null };
    });
}

/** A session's line in the list: "12 turns · 3 acts · 8 snapshots · 2 compactions · calccheck minicalc PASS"
 *  (J1's turn snapshots only when the session has any). */
export function sessionLine(s: Pick<MemorySessionRow, 'counts' | 'checkers'>): string {
  if (!s.counts) return 'counts unknown (no meta.json)';
  const c = s.counts;
  const parts = [`${c.blocks} turn${c.blocks === 1 ? '' : 's'}`];
  if (c.untranscribed) parts.push(`${c.untranscribed} without tools (counted only)`);
  parts.push(`${c.acts} act${c.acts === 1 ? '' : 's'}`);
  if (c.snapshots) parts.push(`${c.snapshots} snapshot${c.snapshots === 1 ? '' : 's'}`);
  if (c.compactions) parts.push(`${c.compactions} compaction${c.compactions === 1 ? '' : 's'}`);
  if (c.redactions) parts.push(`${c.redactions} redacted`);
  if (s.checkers?.last) parts.push(s.checkers.last);
  return parts.join(' · ');
}

/** The settings document's transcript section over the defaults. */
export function transcriptSettings(doc: { memory?: { transcripts?: Partial<TranscriptSettings> } } | null | undefined): TranscriptSettings {
  return { ...TRANSCRIPT_DEFAULTS, ...(doc?.memory?.transcripts ?? {}) };
}

/** "c:ses_ab12" -> "ses_ab12" for the list; the key as the server knows it stays in the URL. */
export function shortKey(key: string | null, dir: string): string {
  if (!key) return dir;
  return key.replace(/^[ch]:/, '');
}
