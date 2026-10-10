// The components' dependency rules (VIS-78): a component whose dependency is off is disabled and
// says why; turning a dependency off turns its dependents off too and says so; a conflict disables
// the other side. Required components are always on. Wren's toggle goes through the same rules.

import type { SetupComponent, SetupModel } from '../api/types';
import { S } from './strings';

export interface Availability {
  /** the checkbox cannot be changed now */
  disabled: boolean;
  /** why, in a sentence ("Needs the conductor") */
  reason: string | null;
  required: boolean;
}

export function isOn(c: SetupComponent, sel: Record<string, boolean>): boolean {
  return c.required || (sel[c.id] ?? c.default);
}

function label(m: SetupModel, id: string): string {
  const c = m.components.find((x) => x.id === id);
  return c ? lowerLabel(c.label) : id;
}

/** "Conductor" -> "the conductor"; names that read as names stay as they are. */
export function lowerLabel(l: string): string {
  if (/^(Laya|Qwen|NVMe|EXL3|NVFP4)/.test(l)) return l;
  return `the ${l.charAt(0).toLowerCase()}${l.slice(1)}`;
}

export function availability(m: SetupModel, sel: Record<string, boolean>): Record<string, Availability> {
  const out: Record<string, Availability> = {};
  for (const c of m.components) {
    if (c.required) {
      out[c.id] = { disabled: true, reason: S.alwaysOn, required: true };
      continue;
    }
    const missing = c.depends_on.filter((d) => {
      const dc = m.components.find((x) => x.id === d);
      return !dc || !isOn(dc, sel);
    });
    if (missing.length) {
      out[c.id] = { disabled: true, reason: S.needs(missing.map((d) => label(m, d))), required: false };
      continue;
    }
    // a conflict blocks only switching on; an "on" component can always be switched off
    const clash = conflictsOf(m, c).filter((o) => isOn(o, sel));
    if (clash.length && !isOn(c, sel)) {
      out[c.id] = { disabled: true, reason: S.conflictsWith(clash.map((o) => lowerLabel(o.label))), required: false };
      continue;
    }
    out[c.id] = { disabled: false, reason: null, required: false };
  }
  return out;
}

/** Conflicts go both ways, whichever side declares them. */
function conflictsOf(m: SetupModel, c: SetupComponent): SetupComponent[] {
  return m.components.filter((o) => o.id !== c.id && (c.conflicts.includes(o.id) || o.conflicts.includes(c.id)));
}

/** Every component that needs `id`, directly or through another. */
export function dependentsOf(m: SetupModel, id: string): SetupComponent[] {
  const out: SetupComponent[] = [];
  const seen = new Set<string>([id]);
  const queue = [id];
  while (queue.length) {
    const cur = queue.shift()!;
    for (const c of m.components) {
      if (!seen.has(c.id) && c.depends_on.includes(cur)) {
        seen.add(c.id);
        out.push(c);
        queue.push(c.id);
      }
    }
  }
  return out;
}

export type ToggleResult =
  | { ok: true; components: Record<string, boolean>; /** dependents switched off with it */ cascaded: SetupComponent[]; notice: string | null }
  | { ok: false; error: string };

/** Switch one component, enforcing the rules. `ok: false` when the switch is not allowed now. */
export function toggleComponent(m: SetupModel, sel: Record<string, boolean>, id: string, on: boolean): ToggleResult {
  const c = m.components.find((x) => x.id === id);
  if (!c) return { ok: false, error: S.tool.noComponent(id) };
  if (c.required) return on ? { ok: true, components: { ...sel, [id]: true }, cascaded: [], notice: null } : { ok: false, error: S.tool.required(c.label) };
  const av = availability(m, sel)[id];
  if (on && av.disabled) return { ok: false, error: av.reason ?? S.tool.noComponent(id) };
  const next = { ...sel, [id]: on };
  if (on) return { ok: true, components: next, cascaded: [], notice: null };
  const cascaded = dependentsOf(m, id).filter((d) => isOn(d, sel) && !d.required);
  for (const d of cascaded) next[d.id] = false;
  const notice = cascaded.length ? S.cascadeOff(lowerLabel(c.label), cascaded.map((d) => d.label)) : null;
  return { ok: true, components: next, cascaded, notice };
}

/** A selection's components normalised: every registry component present, rules applied. */
export function normaliseComponents(m: SetupModel, sel: Record<string, boolean>): Record<string, boolean> {
  const out: Record<string, boolean> = {};
  for (const c of m.components) out[c.id] = c.required ? true : typeof sel[c.id] === 'boolean' ? sel[c.id] : c.default;
  // drop anything whose dependency is off (repeat: chains)
  for (let i = 0; i < m.components.length; i++) {
    let changed = false;
    for (const c of m.components) {
      if (out[c.id] && !c.required && c.depends_on.some((d) => !out[d])) {
        out[c.id] = false;
        changed = true;
      }
    }
    if (!changed) break;
  }
  return out;
}
