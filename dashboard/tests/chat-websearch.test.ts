// Web search in the Chat tab (VIS-157, QA VIS-158): the settings form's draft and checks, a chat's
// provider and its override, the tool card's result list, and the mock's routes and search run.
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { MockChatBackend } from '../src/chat/mock';
import { WEB_SEARCH } from '../src/chat/mock-data';
import { applyEvent } from '../src/chat/store';
import type { Conversation, StampedEvent, ToolCallPart, WebSearchSettings } from '../src/chat/types';
import { chatProvider, draftErrors, draftOf, draftPatch, hostOf, overrideOf, overridePatch, providerWord, resultsWord, searchResults, timeWord } from '../src/chat/websearch';

const STORED: WebSearchSettings = { ...WEB_SEARCH };

describe('the settings form', () => {
  it('a draft of the stored settings has nothing to save', () => {
    const d = draftOf(STORED);
    expect(draftPatch(STORED, d)).toEqual({});
    expect(draftErrors(d)).toEqual({});
    expect('searxng_auth_set' in d).toBe(false);
  });

  it('Save sends only the changed fields, the auth value only when typed', () => {
    const d = { ...draftOf(STORED), provider: 'searxng' as const, searxng_url: 'https://search.example.org', searxng_auth_header: 'X-API-Key', searxng_auth_value: 'k1' };
    expect(draftPatch(STORED, d)).toEqual({ provider: 'searxng', searxng_url: 'https://search.example.org', searxng_auth_header: 'X-API-Key', searxng_auth_value: 'k1' });
    const stored = { ...STORED, provider: 'searxng' as const, searxng_url: 'https://search.example.org', searxng_auth_header: 'X-API-Key', searxng_auth_set: true };
    const kept = { ...draftOf(stored), max_results: 5 };
    expect(draftPatch(stored, kept)).toEqual({ max_results: 5 });
    expect(draftPatch(stored, { ...draftOf(stored), clear_auth: true })).toEqual({ searxng_auth_value: null });
    expect(draftPatch(STORED, { ...draftOf(STORED), clear_auth: true })).toEqual({});
  });

  it('the checks name the field in the server words', () => {
    const d = draftOf(STORED);
    expect(draftErrors({ ...d, max_results: 0 })).toHaveProperty('max_results');
    expect(draftErrors({ ...d, max_results: 21 })).toHaveProperty('max_results');
    expect(draftErrors({ ...d, timeout_s: 1 })).toHaveProperty('timeout_s');
    expect(draftErrors({ ...d, language: 'german' })).toHaveProperty('language');
    expect(draftErrors({ ...d, provider: 'searxng' }).searxng_url).toMatch(/needs the instance/);
    expect(draftErrors({ ...d, searxng_url: 'ftp://x' }).searxng_url).toMatch(/http or https/);
    expect(draftErrors({ ...d, searxng_url: 'https://u:p@x' }).searxng_url).toMatch(/auth header/);
    expect(draftErrors({ ...d, searxng_url: 'https://x/search?q=1' }).searxng_url).toMatch(/without a query/);
    expect(draftErrors({ ...d, searxng_auth_header: 'Bad Header' })).toHaveProperty('searxng_auth_header');
    expect(draftErrors({ ...d, searxng_auth_value: 'k' }).searxng_auth_header).toMatch(/name the header/);
    expect(draftErrors({ ...d }, true).searxng_auth_header).toMatch(/name the header/);
    expect(draftErrors({ ...d, clear_auth: true }, true)).toEqual({});
    expect(draftErrors({ ...d, searxng_auth_header: 'X-Key', searxng_auth_value: 'a\nb' })).toHaveProperty('searxng_auth_value');
    expect(draftErrors({ ...d, provider: 'searxng', searxng_url: 'http://127.0.0.1:8888', language: 'de-DE' })).toEqual({});
  });

  it('the words', () => {
    expect(['off', 'duckduckgo', 'searxng'].map((p) => providerWord(p as never))).toEqual(['off', 'DuckDuckGo', 'SearXNG']);
    expect(providerWord(null)).toBe('off');
    expect(timeWord('')).toBe('any time');
    expect(timeWord('week')).toBe('past week');
    expect(resultsWord(0)).toBe('no results');
    expect(resultsWord(1)).toBe('1 result');
    expect(resultsWord(8)).toBe('8 results');
    expect(hostOf('https://www.example.org/a?b')).toBe('example.org');
  });
});

describe("a chat's provider", () => {
  const caps = { web_search: { provider: 'duckduckgo' as const, providers: ['off', 'duckduckgo', 'searxng'] as const } } as never;
  it('a chat follows the global provider unless it has its own', () => {
    expect(chatProvider(caps, { web_search: null })).toBe('duckduckgo');
    expect(chatProvider(caps, { web_search: { provider: 'off' } })).toBe('off');
    expect(chatProvider(null, {})).toBeNull();
    expect(overrideOf({ web_search: { provider: 'searxng', max_results: 3 } })).toBe('searxng');
    expect(overrideOf(null)).toBe('');
  });
  it('the override keeps its other fields; the default clears it', () => {
    expect(overridePatch(null, 'off')).toEqual({ provider: 'off' });
    expect(overridePatch({ provider: 'off', max_results: 3 }, 'searxng')).toEqual({ provider: 'searxng', max_results: 3 });
    expect(overridePatch({ provider: 'off', max_results: 3 }, '')).toEqual({ max_results: 3 });
    expect(overridePatch({ provider: 'off' }, '')).toBeNull();
  });
});

describe('the tool card', () => {
  const base: ToolCallPart = { type: 'tool_call', id: 'c1', name: 'web_search', input: '{"query":"rust"}', output: null, status: 'done', started_at: null, ended_at: null };
  const results = [
    { title: 'A', url: 'https://a.example/x', snippet: 'one', date: '2026-01-02' },
    { title: 'B', url: 'javascript:alert(1)', snippet: 'not a link' },
    { title: 'C', url: 'http://c.example/', snippet: '', engine: 'bing' },
  ];
  it('lists the http and https results of a done search', () => {
    const t = { ...base, output: { content: 'x', is_error: false, bytes: 1, data: { provider: 'duckduckgo' as const, query: 'rust', results } } };
    expect(searchResults(t)?.map((r) => r.title)).toEqual(['A', 'C']);
  });
  it('a failed search, another tool or a missing list has no list', () => {
    expect(searchResults({ ...base, output: { content: 'DuckDuckGo refused the search', is_error: true, bytes: 1 } })).toBeNull();
    expect(searchResults({ ...base, name: 'web_fetch', output: { content: 'x', is_error: false, bytes: 1, data: { provider: 'off', query: '', results } } })).toBeNull();
    expect(searchResults({ ...base, output: { content: 'x', is_error: false, bytes: 1 } })).toBeNull();
    expect(searchResults(base)).toBeNull();
  });
});

function follow(mock: MockChatBackend, id: string, start: Conversation) {
  const box = { conv: start };
  const sub = mock.subscribe(id, null, { onEvent: (e: StampedEvent) => (box.conv = applyEvent(box.conv, e.event)), onCurrent: () => undefined, onState: () => undefined });
  return { box, sub };
}

describe("the mock's web search", () => {
  beforeEach(() => vi.useFakeTimers());
  afterEach(() => vi.useRealTimers());

  it('stores the settings without ever answering the auth value', async () => {
    const mock = new MockChatBackend();
    const v = await mock.webSearch();
    expect(v.settings.provider).toBe('duckduckgo');
    expect((await mock.capabilities()).web_search?.provider).toBe('duckduckgo');
    const saved = await mock.saveWebSearch({ provider: 'searxng', searxng_url: 'https://search.example.org/', searxng_auth_header: 'Authorization', searxng_auth_value: 'Bearer s3cr3t' });
    expect(saved.settings.searxng_auth_set).toBe(true);
    expect(saved.settings.searxng_url).toBe('https://search.example.org');
    expect(JSON.stringify(saved)).not.toContain('s3cr3t');
    expect((await mock.capabilities()).web_search?.provider).toBe('searxng');
    await expect(mock.saveWebSearch({ max_results: 99 })).rejects.toThrow(/max_results/);
    const cleared = await mock.saveWebSearch({ searxng_auth_value: null });
    expect(cleared.settings.searxng_auth_set).toBe(false);
    mock.deleteConversation(mock.attachedId);
    mock.deleteConversation('c_rename');
  });

  it('Test answers OK with a count, or the reason', async () => {
    const mock = new MockChatBackend();
    const p = mock.testWebSearch({ provider: 'duckduckgo' });
    await vi.advanceTimersByTimeAsync(1_000);
    const r = await p;
    expect(r.ok).toBe(true);
    expect(r.count).toBe(3);
    expect(r.message).toMatch(/^DuckDuckGo answered 3 results/);
    const q = mock.testWebSearch({ provider: 'searxng', searxng_url: 'http://json-off.example:8888' });
    await vi.advanceTimersByTimeAsync(1_000);
    const bad = await q;
    expect(bad.ok).toBe(false);
    expect(bad.message).toMatch(/search\.formats/);
    mock.deleteConversation(mock.attachedId);
    mock.deleteConversation('c_rename');
  });

  it('a search question runs web_search and its card gets the list', async () => {
    const mock = new MockChatBackend();
    const c = await mock.createConversation({});
    expect(c.settings.tools).toContain('web_search');
    const { box, sub } = follow(mock, c.id, c);
    await vi.advanceTimersByTimeAsync(200);
    await mock.send(c.id, { text: 'Search the latest on speculative decoding', client_msg_id: 'cm_ws' });
    await vi.advanceTimersByTimeAsync(60_000);
    const call = box.conv.messages.at(-1)!.parts.find((x) => x.type === 'tool_call') as ToolCallPart;
    expect(call.name).toBe('web_search');
    expect(call.status).toBe('done');
    expect(searchResults(call)?.length).toBe(3);
    expect(box.conv.status).toBe('idle');
    sub.close();
    mock.deleteConversation(mock.attachedId);
    mock.deleteConversation('c_rename');
  });
});
