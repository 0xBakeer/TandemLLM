// The memory estimate: the TypeScript copy of server/setup/estimate.py (the single source, D5),
// line for line. The page computes it on every change so the memory bar never waits, then confirms
// it with POST /v1/setup/estimate. models/estimate-cases.json pins both: tests/setup-estimate.test.ts
// runs every case here (numbers within 0.01, the rest equal, the error cases refused with the
// same status and code). The labels and notes are the server's words, so they live here and not in
// strings.ts: the fixture holds them to the letter.
//
//   parts  = the OS + the weights + the registry's memory parts + the side slots + every tool that
//            is on (the drafters at the profile's drafters_gib) + the KV
//   KV     = pool on: (min(window, fill) + (slots - 1) x min(side, fill)) x bytes a token;
//            pool off: every window at load; worst: every slot at its full window
//   fits   = total <= device - floor (rounded to 0.01); room = budget - every part but the KV
// Every figure is GiB (2^30 bytes).

import type { SetupEstimate, SetupEstimatePart, SetupModel, SetupProfile, SetupSelection, SetupValue } from '../api/types';
import { packNoImage } from './community';
import { ENGINE_MAX_SLOTS, fieldError } from './fields';

export const GIB = 2 ** 30;
export const KV_TYPICAL_TOKENS = 32_768;
const KEYS = ['model', 'profile', 'components', 'slots', 'window', 'side_window', 'kv_pool', 'floor_gb', 'methods', 'advanced', 'device'];

/** Python's round(x, 2) for the figures here (half to even only matters on exact halves, which these sums do not hit) */
const r2 = (x: number) => Math.round(x * 100) / 100 + 0;

/** A selection the server refuses: status 400 (an unknown id) or 422, `code` as the server's. */
export class SelectionError extends Error {
  constructor(
    public status: number,
    public code: string,
    message: string,
  ) {
    super(message);
  }
}

export function deviceOf(m: Pick<SetupModel, 'devices'>, s: Pick<SetupSelection, 'device'>) {
  return m.devices.find((d) => d.id === s.device) ?? m.devices[0];
}

export function profileOf(m: Pick<SetupModel, 'profiles'>, id: string): SetupProfile {
  return m.profiles.find((p) => p.id === id) ?? m.profiles.find((p) => p.recommended) ?? m.profiles[0];
}

const fmt = (n: number) => (Number.isInteger(n) ? String(n) : String(+n.toPrecision(6)));

function intIn(name: string, v: unknown, lo: number, hi: number): number {
  if (typeof v !== 'number' || !Number.isInteger(v)) throw new SelectionError(422, 'out_of_range', `${name}: a whole number from ${fmt(lo)} to ${fmt(hi)}`);
  if (v < lo || v > hi) throw new SelectionError(422, 'out_of_range', `${name}: ${fmt(v)} is outside ${fmt(lo)} to ${fmt(hi)}`);
  return v;
}

/** A window is a whole number of the KV pool's pages (estimate.py `_on_step`). */
function onStep(name: string, v: number, step: number): number {
  if (step && v % step) throw new SelectionError(422, 'off_step', `${name}: a multiple of ${thousands(step)}, not ${thousands(v)}`);
  return v;
}

/** The selection complete and checked (estimate.py `check`); SelectionError otherwise. */
export function checkSelection(m: SetupModel, body: unknown): SetupSelection & { side_window: number; device: string } {
  if (!body || typeof body !== 'object' || Array.isArray(body)) throw new SelectionError(422, 'invalid_selection', 'the selection must be a JSON object');
  const b = body as Record<string, unknown>;
  const extra = Object.keys(b).filter((k) => !KEYS.includes(k)).sort();
  if (extra.length) throw new SelectionError(422, 'unknown_field', `unknown field '${extra[0]}'; a selection has ${KEYS.join(', ')}`);
  if (b.model !== m.id) throw new SelectionError(400, 'unknown_model', `no model '${b.model}' here`);
  const pid = b.profile ?? (m.profiles.find((p) => p.recommended) ?? m.profiles[0]).id;
  if (typeof pid !== 'string' || !m.profiles.some((p) => p.id === pid)) throw new SelectionError(400, 'unknown_profile', `no weight profile '${pid}' for ${m.id}`);
  const comps: Record<string, boolean> = {};
  for (const c of m.components) comps[c.id] = c.default;
  const given = b.components ?? {};
  if (!given || typeof given !== 'object' || Array.isArray(given)) throw new SelectionError(422, 'invalid_selection', 'components: an object of id -> true/false');
  for (const [cid, on] of Object.entries(given as Record<string, unknown>)) {
    if (!(cid in comps)) throw new SelectionError(400, 'unknown_component', `no tool '${cid}' for ${m.id}`);
    if (typeof on !== 'boolean') throw new SelectionError(422, 'invalid_selection', `components.${cid}: true or false`);
    comps[cid] = on;
  }
  // a community pack that cannot run images: the server turns the Vision tool off, so does this copy
  if ('vision' in comps && packNoImage(m, pid)) comps.vision = false;
  const bad: string[] = [];
  const label = (id: string) => m.components.find((c) => c.id === id)?.label ?? id;
  for (const c of m.components) {
    if (c.required && !comps[c.id]) bad.push(`${c.label} cannot be switched off`);
    if (!comps[c.id]) continue;
    for (const d of c.depends_on) if (!comps[d]) bad.push(`${c.label} needs ${label(d)} (${d}) on`);
    for (const d of c.conflicts) if (comps[d]) bad.push(`${c.label} and ${label(d)} (${d}) cannot both be on`);
  }
  if (bad.length) throw new SelectionError(422, 'invalid_combination', bad.join('; '));
  const p = m.parallelism;
  // the card's slots.max is how far it was measured; the engine's own limit is the cap
  const rawSlots = b.slots ?? p.slots.default;
  if (typeof rawSlots === 'number' && Number.isInteger(rawSlots) && rawSlots > ENGINE_MAX_SLOTS) throw new SelectionError(422, 'out_of_range', `slots: ${rawSlots} is above ${ENGINE_MAX_SLOTS}, the most chats the engine verifies in one pass`);
  const slots = intIn('slots', rawSlots, p.slots.min, ENGINE_MAX_SLOTS);
  const window = onStep('window', intIn('window', b.window ?? p.window.default, p.window.min, p.window.max), p.window.step);
  const mw = (m.profiles.find((x) => x.id === pid) as SetupProfile).max_window;
  if (mw && window > mw) throw new SelectionError(422, 'out_of_range', `window: ${thousands(window)} is above ${pid}'s largest window, ${thousands(mw)}`);
  // a side slot never holds more than slot 0 (server/app.py: min(side_max_len, max_len))
  const side = Math.min(window, onStep('side_window', intIn('side_window', b.side_window ?? window, p.side_window.min, p.side_window.max), p.side_window.step));
  const pool = b.kv_pool ?? p.kv_pool.default;
  if (typeof pool !== 'boolean') throw new SelectionError(422, 'invalid_selection', 'kv_pool: true or false');
  const floor = b.floor_gb ?? p.floor_gb.default;
  if (typeof floor !== 'number' || !(floor >= p.floor_gb.min && floor <= p.floor_gb.max)) throw new SelectionError(422, 'out_of_range', `floor_gb: a number from ${fmt(p.floor_gb.min)} to ${fmt(p.floor_gb.max)}`);
  const groups: Record<'methods' | 'advanced', Record<string, SetupValue>> = { methods: {}, advanced: {} };
  for (const group of ['methods', 'advanced'] as const) {
    const vals: Record<string, SetupValue> = {};
    // the fixture's model carries only what the formula reads: without the group's fields its
    // values pass unchecked (they never move the memory)
    const fields = m[group];
    for (const f of fields ?? []) vals[f.id] = f.default;
    const g = b[group] ?? {};
    if (!g || typeof g !== 'object' || Array.isArray(g)) throw new SelectionError(422, 'invalid_selection', `${group}: an object of id -> value`);
    for (const [fid, v] of Object.entries(g as Record<string, unknown>)) {
      if (!fields) {
        vals[fid] = v as SetupValue;
        continue;
      }
      const f = fields.find((x) => x.id === fid);
      if (!f) throw new SelectionError(422, 'unknown_field', `${group}.${fid}: no such field`);
      const why = fieldError(f, v);
      if (why) throw new SelectionError(422, 'invalid_value', `${group}.${fid}: ${why}`);
      vals[fid] = v as SetupValue;
    }
    groups[group] = vals;
  }
  const dev = b.device ?? m.devices[0].id;
  if (typeof dev !== 'string' || !m.devices.some((d) => d.id === dev)) throw new SelectionError(400, 'unknown_device', `no device preset '${dev}' for ${m.id}`);
  return { model: m.id, profile: pid, components: comps, slots, window, side_window: side, kv_pool: pool, floor_gb: floor, methods: groups.methods, advanced: groups.advanced, device: dev };
}

/** Which bar category a part belongs to (the memory bar's colours and legend). */
export type PartCategory = 'model' | 'helpers' | 'caches' | 'engine' | 'system' | 'chats';

export function categoryOf(m: SetupModel, id: string): PartCategory {
  if (id === 'weights' || id === 'drafters') return 'model';
  if (id === 'kv') return 'chats';
  if (id === 'os') return 'system';
  const c = m.components.find((x) => x.id === id);
  if (c) return c.group === 'roles' ? 'helpers' : c.group === 'speed' ? 'model' : 'caches';
  return 'engine';
}

const thousands = (n: number) => n.toLocaleString('en-US');

/** The KV bytes a token: a component that is on and keeps the cache in fewer bytes says how many (Flint's e4m3
 *  cache), else the model's (server/setup/estimate.py kv_bytes). */
export function kvBytes(m: SetupModel, sel: SetupSelection): number {
  for (const c of m.components) if (c.kv_bytes_per_token && sel.components?.[c.id]) return c.kv_bytes_per_token;
  return m.parallelism.kv_bytes_per_token;
}

/** The estimate of a selection (checked first, as the server does). */
export function estimate(m: SetupModel, selection: SetupSelection, availableGb: number | null = null): SetupEstimate {
  const sel = checkSelection(m, selection);
  const p = m.parallelism;
  const prof = profileOf(m, sel.profile);
  const dev = deviceOf(m, sel);
  const kvb = kvBytes(m, sel);
  const fill = p.typical_fill_tokens ?? KV_TYPICAL_TOKENS;
  const { slots, window } = sel;
  const side = sel.side_window;
  const parts: SetupEstimatePart[] = [
    { id: 'os', label: 'The OS and other programs', gb: dev.reserved_gib, kind: 'fixed' },
    { id: 'weights', label: `Weights (${prof.label})`, gb: prof.weights_gib, kind: 'fixed' },
  ];
  for (const part of m.memory.parts) parts.push({ id: part.id, label: part.label, gb: part.gib, kind: part.kind });
  // a profile's lookup table: warm, its size (the page cache); cold, its row cache (SRV-337)
  const table = prof.table;
  if (table) {
    const warm = table.policy === 'warm';
    parts.push({ id: 'table', label: warm ? 'n-gram table (page cache, kept warm)' : 'n-gram row cache (the table read past the page cache)', gb: warm ? table.gib : table.cache_gib, kind: 'fixed' });
  }
  // a part may stop growing (a prefill's workspace): max_gib
  for (const part of m.memory.per_window_parts ?? []) parts.push({ id: part.id, label: part.label, gb: Math.min((part.bytes_per_token * window) / GIB, part.max_gib ?? Infinity), kind: part.kind });
  if (slots > 1) parts.push({ id: 'side_slots', label: `${slots - 1} more slot(s), beyond their KV`, gb: (slots - 1) * p.side_slot_const_gib, kind: 'fixed' });
  for (const c of m.components) {
    if (!sel.components[c.id]) continue;
    const gb = prof.components_gib?.[c.id] ?? (c.id === 'drafters' ? (prof.drafters_gib ?? c.memory_gib) : c.memory_gib);
    if (gb <= 0 && c.memory_max_gib == null) continue;
    const part: SetupEstimatePart = { id: c.id, label: c.label, gb, kind: 'fixed' };
    if (c.memory_max_gib != null) {
      part.kind = 'reserve';
      part.worst_gb = c.memory_max_gib;
    }
    parts.push(part);
  }
  const worstRows = window + (slots - 1) * side;
  const typicalRows = Math.min(window, fill) + (slots - 1) * Math.min(side, fill);
  const kvWorst = (worstRows * kvb) / GIB;
  const kv = ((sel.kv_pool ? typicalRows : worstRows) * kvb) / GIB;
  const fixed = parts.reduce((a, x) => a + x.gb, 0);
  parts.push({ id: 'kv', label: 'Conversations (KV)' + (sel.kv_pool ? ', committed as they grow' : ', every window at load'), gb: kv, kind: 'kv', worst_gb: kvWorst });
  const total = fixed + kv;
  const budget = dev.total_gib - sel.floor_gb;
  const room = budget - fixed;
  const fits = r2(total) <= r2(budget);
  const fitsWorst = r2(room) >= r2(kvWorst);
  let ctx = Math.max(0, Math.floor((room * GIB) / kvb));
  const mw = prof.max_window === undefined ? 0 : prof.max_window;
  if (mw === null) ctx = 0;
  else if (mw) ctx = Math.min(ctx, mw * slots);
  const notes: string[] = [];
  if (mw === null) notes.push(`${prof.label} does not fit ${dev.label} at any window: it is ${r2(Math.max(0, fixed - budget)).toFixed(2)} GiB over the budget before any context.`);
  if (!fits) {
    const big = [...parts].sort((a, b) => b.gb - a.gb).slice(0, 3);
    notes.push(`Over the budget by ${r2(total - budget).toFixed(2)} GiB. The largest parts: ` + big.map((x) => `${x.label} ${x.gb.toFixed(2)} GiB`).join(', ') + '.');
    if (!sel.kv_pool) notes.push('Turn the shared KV pool on: without it every window takes its memory at load.');
    notes.push('Fewer slots, a shorter window, a smaller weight profile or fewer tools bring it down.');
  } else if (!fitsWorst) {
    notes.push(`With every slot at its full window the conversations would need ${r2(kvWorst - room).toFixed(2)} GiB more than there is. When memory runs short, a new request waits for room instead of the engine running out of memory.`);
  }
  if (room > 0) {
    const per = slots ? Math.min(window, Math.floor(ctx / slots)) : 0;
    notes.push(`About ${thousands(ctx)} tokens of conversation fit in all, ${thousands(per)} a slot when all ${slots} are full (the window is ${thousands(window)}).`);
  }
  return {
    contract_version: '1.0',
    total_gb: r2(total),
    available_gb: availableGb,
    device_gb: r2(dev.total_gib),
    floor_gb: r2(sel.floor_gb),
    budget_gb: r2(budget),
    parts: parts.map((x) => ({ ...x, gb: r2(x.gb), ...(x.worst_gb != null ? { worst_gb: r2(x.worst_gb) } : {}) })),
    fits,
    fits_worst: fitsWorst,
    context_tokens_fit: ctx,
    kv_bytes_per_token: kvb,
    typical_fill_tokens: fill,
    kv_worst_gb: r2(kvWorst),
    room_gb: r2(room),
    notes,
  };
}

/** How far over the budget (0 when it fits). */
export function overBy(e: SetupEstimate): number {
  const budget = e.budget_gb ?? e.device_gb - e.floor_gb;
  return Math.max(0, r2(e.total_gb - budget));
}

/** The page's estimate: never throws. A field that does not pass the check (a half-typed method
 *  value) does not change the memory, so it is left out; anything else wrong gives null. */
export function estimateLoose(m: SetupModel, sel: SetupSelection, availableGb: number | null = null): SetupEstimate | null {
  try {
    return estimate(m, sel, availableGb);
  } catch (e) {
    if (!(e instanceof SelectionError)) throw e;
    try {
      return estimate(m, { ...sel, methods: {}, advanced: {} }, availableGb);
    } catch {
      return null;
    }
  }
}
