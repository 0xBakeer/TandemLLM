// Function calling: the tools editor's validation, the templates, the tool_choice
// shapes, and the `role: "tool"` continuation messages.

import type { Message, ToolCall, ToolChoice, ToolDef } from './types';
import { uid } from './types';

const NAME = /^[A-Za-z0-9_-]{1,64}$/;

export interface ToolsValidation {
  tools: ToolDef[];
  errors: string[];
}

/** Parse and validate the editor text. Empty text is zero tools and no error. */
export function validateTools(text: string): ToolsValidation {
  const t = text.trim();
  if (!t) return { tools: [], errors: [] };
  let parsed: unknown;
  try {
    parsed = JSON.parse(t);
  } catch (e) {
    return { tools: [], errors: [`not valid JSON: ${(e as Error).message.replace(/^JSON\.parse: /, '')}`] };
  }
  if (!Array.isArray(parsed)) return { tools: [], errors: ['tools must be a JSON array'] };
  const errors: string[] = [];
  const tools: ToolDef[] = [];
  const names = new Set<string>();
  parsed.forEach((item, i) => {
    const at = `tool ${i}`;
    if (!item || typeof item !== 'object' || Array.isArray(item)) return errors.push(`${at}: must be an object`);
    const o = item as Record<string, unknown>;
    if (o.type !== 'function') errors.push(`${at}: type must be "function"`);
    const fn = o.function as Record<string, unknown> | undefined;
    if (!fn || typeof fn !== 'object' || Array.isArray(fn)) return errors.push(`${at}: function must be an object`);
    const name = fn.name;
    if (typeof name !== 'string' || !NAME.test(name)) errors.push(`${at} function.name: letters, digits, _ or -, 1 to 64 characters`);
    else if (names.has(name)) errors.push(`${at} function.name: "${name}" is used twice`);
    else names.add(name);
    if (fn.description !== undefined && typeof fn.description !== 'string') errors.push(`${at} function.description: must be a string`);
    if (fn.parameters !== undefined) {
      const p = fn.parameters as Record<string, unknown>;
      if (!p || typeof p !== 'object' || Array.isArray(p)) errors.push(`${at} function.parameters: must be an object schema`);
      else if (p.type !== 'object') errors.push(`${at} function.parameters.type: must be "object"`);
      else if (p.properties !== undefined && (typeof p.properties !== 'object' || p.properties === null || Array.isArray(p.properties))) errors.push(`${at} function.parameters.properties: must be an object`);
    }
    tools.push(item as ToolDef);
  });
  return { tools: errors.length ? [] : tools, errors };
}

export interface ToolTemplate {
  id: string;
  label: string;
  def: ToolDef;
  /** an example result for the result box */
  example: string;
}

export const TOOL_TEMPLATES: ToolTemplate[] = [
  {
    id: 'get_weather',
    label: 'get_weather(city, unit?)',
    def: {
      type: 'function',
      function: {
        name: 'get_weather',
        description: 'Current weather for a city.',
        parameters: {
          type: 'object',
          properties: { city: { type: 'string', description: 'City name' }, unit: { type: 'string', enum: ['celsius', 'fahrenheit'] } },
          required: ['city'],
        },
      },
    },
    example: '{"temperature_c": 21, "condition": "clear", "wind_kmh": 9}',
  },
  {
    id: 'search_notes',
    label: 'search_notes(query, limit?)',
    def: {
      type: 'function',
      function: {
        name: 'search_notes',
        description: 'Full-text search over the notes; returns the best matches.',
        parameters: {
          type: 'object',
          properties: { query: { type: 'string' }, limit: { type: 'integer', minimum: 1, maximum: 20, default: 5 } },
          required: ['query'],
        },
      },
    },
    example: '{"hits": [{"title": "Caches", "snippet": "the prefix cache reuses the longest matching…"}, {"title": "Memory & budgets", "snippet": "the state store budget is 8 GiB…"}]}',
  },
  {
    id: 'run_python',
    label: 'run_python(code)',
    def: {
      type: 'function',
      function: {
        name: 'run_python',
        description: 'Run a Python snippet and return stdout.',
        parameters: { type: 'object', properties: { code: { type: 'string' } }, required: ['code'] },
      },
    },
    example: '{"stdout": "42\\n", "exit_code": 0}',
  },
];

export function templateFor(name: string): ToolTemplate | undefined {
  return TOOL_TEMPLATES.find((t) => t.def.function.name === name);
}

/** Append a template to the editor text (pretty-printed). */
export function appendTemplate(text: string, tpl: ToolTemplate): string {
  const { tools } = validateTools(text);
  const base = text.trim() ? tools : [];
  if (text.trim() && tools.length === 0) return text; // the text is invalid: do not destroy it
  return JSON.stringify([...base, tpl.def], null, 2);
}

export function removeTool(text: string, name: string): string {
  const { tools } = validateTools(text);
  const rest = tools.filter((t) => t.function.name !== name);
  return rest.length ? JSON.stringify(rest, null, 2) : '';
}

export function toolChoiceOf(kind: 'auto' | 'none' | 'required' | 'named', name?: string): ToolChoice {
  if (kind === 'named') return { type: 'function', function: { name: name ?? '' } };
  return kind;
}

export function toolChoiceKind(c: ToolChoice): 'auto' | 'none' | 'required' | 'named' {
  return typeof c === 'string' ? c : 'named';
}

/** The `role: "tool"` messages that answer the calls, in call order. A missing result is "". */
export function toolResultMessages(calls: ToolCall[], results: Record<string, string>): Message[] {
  return calls.map((c) => ({
    uid: uid(),
    role: 'tool' as const,
    tool_call_id: c.id,
    name: c.function.name,
    content: results[c.id] ?? '',
  }));
}

/** "echo the arguments" — the arguments back as the result (raw when they do not parse). */
export function echoResult(c: ToolCall): string {
  try {
    return JSON.stringify(JSON.parse(c.function.arguments));
  } catch {
    return c.function.arguments;
  }
}

/** Pretty JSON when the arguments parse, the raw text otherwise. */
export function prettyArgs(args: string): { text: string; parsed: boolean } {
  try {
    return { text: JSON.stringify(JSON.parse(args), null, 2), parsed: true };
  } catch {
    return { text: args, parsed: false };
  }
}
