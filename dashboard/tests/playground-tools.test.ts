// tools validation, templates, tool_choice, the continuation messages.
import { describe, expect, it } from 'vitest';
import { appendTemplate, echoResult, prettyArgs, removeTool, toolChoiceKind, toolChoiceOf, toolResultMessages, TOOL_TEMPLATES, validateTools } from '../src/playground/tools';

describe('validateTools', () => {
  it('empty text is zero tools and no error', () => {
    expect(validateTools('')).toEqual({ tools: [], errors: [] });
    expect(validateTools('  \n')).toEqual({ tools: [], errors: [] });
  });
  it('the three templates validate, alone and together', () => {
    for (const t of TOOL_TEMPLATES) expect(validateTools(JSON.stringify([t.def])).errors).toEqual([]);
    const all = validateTools(JSON.stringify(TOOL_TEMPLATES.map((t) => t.def)));
    expect(all.errors).toEqual([]);
    expect(all.tools.map((t) => t.function.name)).toEqual(['get_weather', 'search_notes', 'run_python']);
  });
  it('broken JSON says so', () => {
    expect(validateTools('[{')).toMatchObject({ tools: [], errors: [expect.stringMatching(/not valid JSON/)] });
  });
  it('must be an array', () => {
    expect(validateTools('{"type":"function"}').errors).toEqual(['tools must be a JSON array']);
  });
  it('names the item and the field for each rule', () => {
    const e = validateTools(
      JSON.stringify([
        { type: 'tool', function: { name: 'ok' } },
        { type: 'function', function: { name: 'get weather' } },
        { type: 'function', function: { name: 'ok' } },
        { type: 'function', function: { name: 'p', parameters: { type: 'array' } } },
        { type: 'function', function: { name: 'q', parameters: [] } },
        { type: 'function', function: { name: 'r', description: 3 } },
        { type: 'function' },
        'x',
      ]),
    ).errors;
    expect(e).toContain('tool 0: type must be "function"');
    expect(e).toContain('tool 1 function.name: letters, digits, _ or -, 1 to 64 characters');
    expect(e).toContain('tool 2 function.name: "ok" is used twice');
    expect(e).toContain('tool 3 function.parameters.type: must be "object"');
    expect(e).toContain('tool 4 function.parameters: must be an object schema');
    expect(e).toContain('tool 5 function.description: must be a string');
    expect(e).toContain('tool 6: function must be an object');
    expect(e).toContain('tool 7: must be an object');
  });
});

describe('templates in the editor', () => {
  it('append to empty and to an existing list; invalid text is left alone', () => {
    const one = appendTemplate('', TOOL_TEMPLATES[0]);
    expect(validateTools(one).tools).toHaveLength(1);
    const two = appendTemplate(one, TOOL_TEMPLATES[2]);
    expect(validateTools(two).tools.map((t) => t.function.name)).toEqual(['get_weather', 'run_python']);
    expect(appendTemplate('[{', TOOL_TEMPLATES[0])).toBe('[{');
    expect(removeTool(two, 'get_weather')).toBe(JSON.stringify([TOOL_TEMPLATES[2].def], null, 2));
    expect(removeTool(one, 'get_weather')).toBe('');
  });
});

describe('tool_choice', () => {
  it('maps the four kinds to the request shapes and back', () => {
    expect(toolChoiceOf('auto')).toBe('auto');
    expect(toolChoiceOf('none')).toBe('none');
    expect(toolChoiceOf('required')).toBe('required');
    expect(toolChoiceOf('named', 'get_weather')).toEqual({ type: 'function', function: { name: 'get_weather' } });
    expect(toolChoiceKind('required')).toBe('required');
    expect(toolChoiceKind({ type: 'function', function: { name: 'x' } })).toBe('named');
  });
});

describe('results and continuation', () => {
  const calls = [
    { id: 'call_1', type: 'function' as const, function: { name: 'get_weather', arguments: '{"city":"Berlin"}' } },
    { id: 'call_2', type: 'function' as const, function: { name: 'run_python', arguments: 'not json' } },
  ];
  it('one role: tool message per call, in call order, with tool_call_id and name; a missing result is ""', () => {
    const m = toolResultMessages(calls, { call_1: '{"temperature_c":21}' });
    expect(m.map(({ uid: _u, ...x }) => x)).toEqual([
      { role: 'tool', tool_call_id: 'call_1', name: 'get_weather', content: '{"temperature_c":21}' },
      { role: 'tool', tool_call_id: 'call_2', name: 'run_python', content: '' },
    ]);
    expect(new Set(m.map((x) => x.uid)).size).toBe(2);
  });
  it('echo returns the arguments (compact JSON, or raw when they do not parse)', () => {
    expect(echoResult(calls[0])).toBe('{"city":"Berlin"}');
    expect(echoResult(calls[1])).toBe('not json');
    expect(prettyArgs('{"a":1}')).toEqual({ text: '{\n  "a": 1\n}', parsed: true });
    expect(prettyArgs('{"a":')).toEqual({ text: '{"a":', parsed: false });
  });
});
