import { describe, expect, it, vi } from 'vitest';
import { ResumableStream, SseParser, readJsonStream } from '../src/lib/sse';

describe('SSE framing', () => {
  it('parses a complete event with event, id and data', () => {
    const p = new SseParser();
    const evs = p.feed('event: log\nid: 10234\ndata: {"seq":10234}\n\n');
    expect(evs).toEqual([{ event: 'log', id: '10234', data: '{"seq":10234}', retry: null }]);
    expect(p.lastEventId).toBe('10234');
  });

  it('reassembles events split across chunks', () => {
    const p = new SseParser();
    expect(p.feed('event: lo')).toEqual([]);
    expect(p.feed('g\ndata: {"a"')).toEqual([]);
    const evs = p.feed(':1}\n\nevent: gap\ndata: {"dropped":3}\n\n');
    expect(evs.map((e) => e.event)).toEqual(['log', 'gap']);
    expect(evs[0].data).toBe('{"a":1}');
  });

  it('ignores comments (pings) and unknown fields, joins multi-line data', () => {
    const p = new SseParser();
    const evs = p.feed(': ping\n\nfoo: bar\ndata: line1\ndata: line2\n\n');
    expect(evs).toHaveLength(1);
    expect(evs[0].event).toBe('message');
    expect(evs[0].data).toBe('line1\nline2');
  });

  it('accepts CRLF and CR line endings', () => {
    const p = new SseParser();
    const evs = p.feed('data: a\r\n\r\ndata: b\r\r');
    expect(evs.map((e) => e.data)).toEqual(['a', 'b']);
  });

  it('strips one leading space of a value and keeps the rest', () => {
    const p = new SseParser();
    expect(p.feed('data:  two\n\n')[0].data).toBe(' two');
    expect(p.feed('data:none\n\n')[0].data).toBe('none');
  });

  it('retry is parsed only when it is an integer', () => {
    const p = new SseParser();
    expect(p.feed('retry: 2500\ndata: x\n\n')[0].retry).toBe(2500);
    expect(p.feed('retry: soon\ndata: x\n\n')[0].retry).toBeNull();
  });

  it('an id-only block updates lastEventId without dispatching', () => {
    const p = new SseParser();
    expect(p.feed('id: 7\n\n')).toEqual([]);
    expect(p.lastEventId).toBe('7');
  });
});

function streamOf(chunks: string[], opts: { status?: number; endAfter?: boolean } = {}): Response {
  const enc = new TextEncoder();
  const body = new ReadableStream<Uint8Array>({
    start(c) {
      for (const ch of chunks) c.enqueue(enc.encode(ch));
      c.close();
    },
  });
  return new Response(body, { status: opts.status ?? 200, headers: { 'Content-Type': 'text/event-stream' } });
}

describe('ResumableStream', () => {
  it('reconnects with Last-Event-ID and the url gets the last id; no duplicates are emitted', async () => {
    const calls: { url: string; lastId: string | undefined }[] = [];
    const fetchImpl = vi.fn(async (url: string | URL | Request, init?: RequestInit) => {
      const h = init?.headers as Record<string, string>;
      calls.push({ url: String(url), lastId: h['Last-Event-ID'] });
      if (calls.length === 1) return streamOf(['event: log\nid: 1\ndata: {"seq":1}\n\n', 'event: log\nid: 2\ndata: {"seq":2}\n\n']);
      if (calls.length === 2) return streamOf(['event: log\nid: 3\ndata: {"seq":3}\n\n']);
      return new Promise<Response>(() => undefined); // hang: the test closes the stream
    }) as unknown as typeof fetch;
    const got: string[] = [];
    const states: string[] = [];
    const s = new ResumableStream({
      url: (last) => `/logs?since=${last ?? ''}`,
      onEvent: (e) => got.push(e.id ?? ''),
      onState: (st) => states.push(st),
      fetchImpl,
      minBackoffMs: 5,
      maxBackoffMs: 10,
    });
    s.start();
    await vi.waitFor(() => expect(got).toEqual(['1', '2', '3']), { timeout: 2000 });
    expect(calls[1].lastId).toBe('2');
    expect(calls[1].url).toBe('/logs?since=2');
    expect(states).toContain('open');
    expect(states).toContain('reconnecting');
    s.close();
    expect(s.state).toBe('closed');
  });

  it('stops on 401 and reports unauthorized', async () => {
    const fetchImpl = vi.fn(async () => new Response('{"error":{"type":"unauthorized","message":"x"}}', { status: 401 })) as unknown as typeof fetch;
    const states: string[] = [];
    const s = new ResumableStream({ url: () => '/logs', onEvent: () => undefined, onState: (st) => states.push(st), fetchImpl });
    s.start();
    await vi.waitFor(() => expect(states).toContain('unauthorized'));
    expect(fetchImpl).toHaveBeenCalledTimes(1);
  });
});

describe('readJsonStream', () => {
  it('yields parsed chunks and stops at [DONE]', async () => {
    const res = streamOf(['data: {"a":1}\n\n', 'data: {"b":2}\n\ndata: [DONE]\n\ndata: {"c":3}\n\n']);
    const out: unknown[] = [];
    for await (const c of readJsonStream(res.body!)) out.push(c);
    expect(out).toEqual([{ a: 1 }, { b: 2 }]);
  });
});
