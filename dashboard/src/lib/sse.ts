// Server-sent events over fetch: a framing parser (unit-tested) and a resumable stream client
// that reconnects with Last-Event-ID / `since`, backs off, and reports gaps. EventSource is not
// used because the console needs to re-open with a different `level`/`grep`, to know the
// reconnect state, and to run the same code in tests without a browser.

export interface SseEvent {
  event: string; // 'message' when the server sent none
  data: string;
  id: string | null;
  retry: number | null;
}

/** Incremental SSE parser. Feed chunks; get complete events. Handles CRLF, comments, multi-line data. */
export class SseParser {
  private buf = '';
  private event = '';
  private data: string[] = [];
  private id: string | null = null;
  private retry: number | null = null;
  lastEventId: string | null = null;

  feed(chunk: string): SseEvent[] {
    this.buf += chunk;
    const out: SseEvent[] = [];
    let idx: number;
    // Normalise CRLF / CR to LF.
    this.buf = this.buf.replace(/\r\n?/g, '\n');
    while ((idx = this.buf.indexOf('\n')) >= 0) {
      const line = this.buf.slice(0, idx);
      this.buf = this.buf.slice(idx + 1);
      const ev = this.line(line);
      if (ev) out.push(ev);
    }
    return out;
  }

  private line(line: string): SseEvent | null {
    if (line === '') return this.dispatch();
    if (line.startsWith(':')) return null; // comment / ping
    const colon = line.indexOf(':');
    const field = colon < 0 ? line : line.slice(0, colon);
    let val = colon < 0 ? '' : line.slice(colon + 1);
    if (val.startsWith(' ')) val = val.slice(1);
    switch (field) {
      case 'event':
        this.event = val;
        break;
      case 'data':
        this.data.push(val);
        break;
      case 'id':
        if (!val.includes('\0')) this.id = val;
        break;
      case 'retry': {
        const n = Number(val);
        if (Number.isInteger(n) && n >= 0) this.retry = n;
        break;
      }
      default:
        break; // unknown fields are ignored per spec
    }
    return null;
  }

  private dispatch(): SseEvent | null {
    if (this.data.length === 0 && this.event === '' && this.id === null && this.retry === null) return null;
    if (this.id !== null) this.lastEventId = this.id;
    const ev: SseEvent = {
      event: this.event || 'message',
      data: this.data.join('\n'),
      id: this.id,
      retry: this.retry,
    };
    this.event = '';
    this.data = [];
    this.id = null;
    this.retry = null;
    // An event with no data lines is not dispatched (spec), but an id-only block still updates lastEventId.
    if (ev.data === '' && ev.event === 'message') return null;
    return ev;
  }
}

export type StreamState = 'connecting' | 'open' | 'reconnecting' | 'closed' | 'unauthorized' | 'error';

export interface StreamOptions {
  url: (lastId: string | null) => string;
  onEvent: (ev: SseEvent) => void;
  onState?: (s: StreamState, detail?: string) => void;
  fetchImpl?: typeof fetch;
  minBackoffMs?: number;
  maxBackoffMs?: number;
}

/** A resumable stream: opens `url(lastId)`, reconnects with Last-Event-ID, backs off 1 s → 30 s. */
export class ResumableStream {
  private ctrl: AbortController | null = null;
  private closed = false;
  private backoff: number;
  lastId: string | null = null;
  state: StreamState = 'closed';
  attempts = 0;

  constructor(private opts: StreamOptions) {
    this.backoff = opts.minBackoffMs ?? 1000;
  }

  private setState(s: StreamState, detail?: string) {
    this.state = s;
    this.opts.onState?.(s, detail);
  }

  start(): void {
    this.closed = false;
    void this.loop();
  }

  close(): void {
    this.closed = true;
    this.ctrl?.abort();
    this.ctrl = null;
    this.setState('closed');
  }

  private async loop(): Promise<void> {
    const f = this.opts.fetchImpl ?? fetch;
    while (!this.closed) {
      this.ctrl = new AbortController();
      this.setState(this.attempts === 0 ? 'connecting' : 'reconnecting');
      try {
        const headers: Record<string, string> = { Accept: 'text/event-stream' };
        if (this.lastId) headers['Last-Event-ID'] = this.lastId;
        const res = await f(this.opts.url(this.lastId), {
          headers,
          credentials: 'same-origin',
          signal: this.ctrl.signal,
          cache: 'no-store',
        });
        if (res.status === 401) {
          this.setState('unauthorized');
          return;
        }
        if (!res.ok || !res.body) {
          throw new Error(`HTTP ${res.status}`);
        }
        this.setState('open');
        this.backoff = this.opts.minBackoffMs ?? 1000;
        const parser = new SseParser();
        const reader = res.body.getReader();
        const dec = new TextDecoder();
        for (;;) {
          const { value, done } = await reader.read();
          if (done) break;
          for (const ev of parser.feed(dec.decode(value, { stream: true }))) {
            if (ev.id) this.lastId = ev.id;
            if (ev.retry) this.backoff = Math.max(250, ev.retry);
            this.opts.onEvent(ev);
          }
        }
        if (this.closed) return;
        // The server ended the stream: say so at once, then reconnect after the backoff.
        this.setState('reconnecting');
      } catch (e) {
        if (this.closed || (e as Error).name === 'AbortError') return;
        this.setState('error', (e as Error).message);
      }
      this.attempts++;
      await sleep(this.backoff);
      this.backoff = Math.min(this.opts.maxBackoffMs ?? 30_000, this.backoff * 2);
    }
  }
}

function sleep(ms: number): Promise<void> {
  return new Promise((r) => setTimeout(r, ms));
}

/**
 * Read a streaming chat completion (OpenAI SSE with `data: {json}` lines, `data: [DONE]`).
 * Yields parsed chunks. Used by the Dev tab's test request box.
 */
export async function* readJsonStream(body: ReadableStream<Uint8Array>): AsyncGenerator<unknown> {
  const parser = new SseParser();
  const reader = body.getReader();
  const dec = new TextDecoder();
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    for (const ev of parser.feed(dec.decode(value, { stream: true }))) {
      if (ev.data === '[DONE]') return;
      try {
        yield JSON.parse(ev.data);
      } catch {
        // a malformed chunk is skipped, the stream continues
      }
    }
  }
}
