// The parameter fields of the Playground (VIS-20): what each one is called in the request, its
// range, and which server flag carries its default. The panel renders from this table and the
// request builder reads it, so a field exists in exactly one place.

import type { SystemInfo } from '../api/types';
import type { ParamValues } from './types';

export type ParamKey = keyof ParamValues;

export interface ParamDef {
  key: ParamKey;
  label: string;
  /** the request field; nested ones are handled by request.ts */
  request: string;
  kind: 'number' | 'integer' | 'switch' | 'select' | 'list';
  min?: number;
  max?: number;
  step?: number;
  options?: string[];
  /** `flags.args` key that holds the server default */
  flag?: string;
  help: string;
  group: 'sampling' | 'length' | 'penalties' | 'reasoning' | 'stream';
}

export const PARAMS: ParamDef[] = [
  { key: 'temperature', label: 'temperature', request: 'temperature', kind: 'number', min: 0, max: 2, step: 0.05, flag: 'temperature', group: 'sampling', help: '0 is greedy, the exact path with the drafter; above 0 samples on the single-token path' },
  { key: 'top_p', label: 'top_p', request: 'top_p', kind: 'number', min: 0, max: 1, step: 0.01, flag: 'top_p', group: 'sampling', help: 'nucleus: keep the smallest set of tokens whose probability adds up to top_p' },
  { key: 'top_k', label: 'top_k', request: 'top_k', kind: 'integer', min: 0, max: 200, step: 1, flag: 'top_k', group: 'sampling', help: '0 is off' },
  { key: 'seed', label: 'seed', request: 'seed', kind: 'integer', min: 0, max: 2 ** 31 - 1, step: 1, group: 'sampling', help: 'a seeded sampling request reproduces exactly; blank is unseeded' },
  { key: 'draft_temperature', label: 'draft temperature', request: 'draft_temperature', kind: 'number', min: 0, max: 2, step: 0.05, group: 'sampling', help: 'the drafter’s own temperature while sampling; blank follows temperature' },
  { key: 'max_tokens', label: 'max_tokens', request: 'max_tokens', kind: 'integer', min: 1, step: 1, flag: 'default_max_tokens', group: 'length', help: 'the answer stops with finish_reason length when it is reached' },
  { key: 'stop', label: 'stop', request: 'stop', kind: 'list', group: 'length', help: 'up to 8 strings; the stop string itself is never sent back' },
  { key: 'presence_penalty', label: 'presence penalty', request: 'presence_penalty', kind: 'number', min: -2, max: 2, step: 0.05, flag: 'presence_penalty', group: 'penalties', help: 'once a token has appeared, subtract this from its logit' },
  { key: 'frequency_penalty', label: 'frequency penalty', request: 'frequency_penalty', kind: 'number', min: -2, max: 2, step: 0.05, flag: 'frequency_penalty', group: 'penalties', help: 'subtract this times the count of a token so far' },
  { key: 'repetition_penalty', label: 'repetition penalty', request: 'repetition_penalty', kind: 'number', min: 1, max: 2, step: 0.01, flag: 'rep_penalty', group: 'penalties', help: 'the HF multiplicative penalty; 1 is off' },
  { key: 'no_repeat_ngram_size', label: 'no-repeat n-gram', request: 'no_repeat_ngram_size', kind: 'integer', min: 0, max: 10, step: 1, flag: 'no_repeat_ngram', group: 'penalties', help: '0 is off; 2 or more forbids repeating an n-gram of that size' },
  { key: 'thinking', label: 'thinking', request: 'chat_template_kwargs.enable_thinking', kind: 'switch', group: 'reasoning', help: 'the template’s default is on; off sends enable_thinking false' },
  { key: 'reasoning_effort', label: 'reasoning effort', request: 'reasoning_effort', kind: 'select', options: ['low', 'medium', 'high', 'xhigh'], flag: 'reasoning_effort', group: 'reasoning', help: 'what the template asks the model for; the server default is shown' },
  { key: 'max_reasoning_tokens', label: 'max reasoning tokens', request: 'max_reasoning_tokens', kind: 'integer', min: 1, step: 1, flag: 'think_budget', group: 'reasoning', help: 'the thinking block is closed when the budget is reached; blank is unlimited' },
  { key: 'reasoning_format', label: 'reasoning format', request: 'reasoning_format', kind: 'select', options: ['tags', 'reasoning_content', 'both'], flag: 'reasoning_format', group: 'reasoning', help: 'tags: <think> in content; reasoning_content: its own delta field; both: twice' },
  { key: 'include_usage', label: 'include usage', request: 'stream_options.include_usage', kind: 'switch', group: 'stream', help: 'on: stream_options.include_usage true, the usage on its own chunk; off: no stream_options, the server\u2019s own placement (the finish chunk by default)' },
];

export const PARAM_GROUPS: { id: ParamDef['group']; label: string }[] = [
  { id: 'sampling', label: 'Sampling' },
  { id: 'length', label: 'Length' },
  { id: 'penalties', label: 'Penalties' },
  { id: 'reasoning', label: 'Reasoning' },
  { id: 'stream', label: 'Stream' },
];

export const NOT_A_PARAM = 'min_p is not a parameter of this engine; the server reads none.';

export type Defaults = Partial<Record<ParamKey, string | number | boolean | null>> & { max_len?: number };

/** The server's defaults, read from /v1/dashboard/system (flags.args + the model card). */
export function defaultsFrom(sys: SystemInfo | null | undefined): Defaults {
  const out: Defaults = {};
  if (!sys) return out;
  const args = sys.flags?.args ?? {};
  for (const p of PARAMS) {
    if (!p.flag) continue;
    const v = args[p.flag];
    if (v == null) continue;
    if (typeof v === 'number' || typeof v === 'boolean' || typeof v === 'string') out[p.key] = v;
  }
  if (out.max_reasoning_tokens === 0) out.max_reasoning_tokens = null; // 0 = unlimited on the server
  out.max_len = sys.engine?.max_len;
  out.thinking = true; // the template's default
  out.include_usage = true; // the playground's default
  return out;
}

/** "0.0", "32,768", "medium", "on" — how a default reads next to its field. */
export function defaultLabel(p: ParamDef, d: Defaults): string {
  const v = d[p.key];
  if (v == null) return p.kind === 'list' ? 'none' : p.key === 'seed' || p.key === 'draft_temperature' || p.key === 'max_reasoning_tokens' ? 'none' : 'template default';
  if (typeof v === 'boolean') return v ? 'on' : 'off';
  if (typeof v === 'number') return p.kind === 'integer' ? v.toLocaleString('en-US') : String(v);
  return String(v);
}

/** null when the value is fine, else the reason. */
export function validateParam(p: ParamDef, v: unknown, d: Defaults = {}): string | null {
  if (v === undefined) return null;
  if (p.kind === 'number' || p.kind === 'integer') {
    if (typeof v !== 'number' || !Number.isFinite(v)) return `${p.label} must be a number`;
    if (p.kind === 'integer' && !Number.isInteger(v)) return `${p.label} must be a whole number`;
    const max = p.key === 'max_tokens' ? (d.max_len ?? p.max) : p.max;
    if (p.min != null && v < p.min) return `${p.label} must be at least ${p.min}`;
    if (max != null && v > max) return `${p.label} must be at most ${max.toLocaleString('en-US')}`;
    if (p.key === 'no_repeat_ngram_size' && v === 1) return `${p.label} is 0 (off) or 2 to 10`;
    return null;
  }
  if (p.kind === 'select') return p.options?.includes(String(v)) ? null : `${p.label} must be one of ${p.options?.join(', ')}`;
  if (p.kind === 'switch') return typeof v === 'boolean' ? null : `${p.label} must be on or off`;
  if (p.kind === 'list') {
    if (!Array.isArray(v) || v.some((s) => typeof s !== 'string')) return `${p.label} must be a list of strings`;
    if (v.length > 8) return `${p.label} takes up to 8 strings`;
    return null;
  }
  return null;
}

/** Every problem in a set of values, keyed by field. */
export function validateParams(values: ParamValues, d: Defaults = {}): Partial<Record<ParamKey, string>> {
  const out: Partial<Record<ParamKey, string>> = {};
  for (const p of PARAMS) {
    const err = validateParam(p, values[p.key], d);
    if (err) out[p.key] = err;
  }
  return out;
}

/** Which fields differ from unset — what the request will carry. */
export function changedKeys(values: ParamValues): ParamKey[] {
  return PARAMS.filter((p) => values[p.key] !== undefined).map((p) => p.key);
}
