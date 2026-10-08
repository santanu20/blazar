/**
 * blazar-sdk — zero-dependency TypeScript client for the blazar
 * inference control plane.
 *
 * House style mirrors sdk/python: one file, no runtime dependencies,
 * environment-driven defaults. Node's global fetch does the transport.
 *
 * Missing in v1 (python SDK covers these today): realtime voice,
 * anthropic batch endpoints, speech/transcribe.
 */

/** Default gateway origin; overridden by `BLAZAR_URL` then `Client` arg. */
export const BLAZAR_URL = "http://127.0.0.1:11435";

export class BlazarError extends Error {
  readonly status: number;
  readonly blazarCode: string | null;

  constructor(status: number, message: string, blazarCode: string | null) {
    super(message);
    this.name = "BlazarError";
    this.status = status;
    this.blazarCode = blazarCode;
  }
}

export interface ChatMessage {
  role: "system" | "user" | "assistant" | "tool";
  content: string;
  [k: string]: unknown;
}

export interface ChatUsage {
  prompt_tokens?: number;
  completion_tokens?: number;
  total_tokens?: number;
  [k: string]: unknown;
}

export interface ChatResponse {
  id: string;
  model: string;
  created: number;
  choices: Array<{
    index?: number;
    message?: ChatMessage;
    delta?: Partial<ChatMessage>;
    finish_reason?: string | null;
    [k: string]: unknown;
  }>;
  usage?: ChatUsage;
  [k: string]: unknown;
}

export interface ChatOptions {
  temperature?: number;
  max_tokens?: number;
  top_p?: number;
  seed?: number;
  stop?: string | string[];
  signal?: AbortSignal;
  [k: string]: unknown;
}

export interface PsRow {
  model: string;
  [k: string]: unknown;
}

export interface ExplainCard {
  model: string;
  [k: string]: unknown;
}

export interface McpServerInfo {
  [k: string]: unknown;
}

export interface FailoverReport {
  [k: string]: unknown;
}

export interface ModelInfo {
  id: string;
  [k: string]: unknown;
}

/**
 * Incremental server-sent-events parser: feed() accepts arbitrary byte
 * chunks (may split a line anywhere, including mid-CR), nextEvent()
 * yields parsed `{ data }` frames and skips comment lines. Follows the
 * SSE framing rules the gateway emits: `\n`, `\r\n`, or `\r` line ends,
 * `:`-prefixed comments, multi-line `data:` joined with `\n`.
 */
export class SseParser {
  private buf = "";

  feed(chunk: string): void {
    this.buf += chunk;
  }

  /** Pops one complete event, or null while the buffer holds no full line. */
  nextEvent(): { data: string } | null {
    for (;;) {
      const idx = this.findLineEnd();
      if (idx === null) return null;
      const line = this.buf.slice(0, idx.pos);
      this.buf = this.buf.slice(idx.next);
      if (!line.trim()) continue; // event separator
      if (line.startsWith(":")) continue; // comment / keep-alive
      if (line.startsWith("data:")) return { data: line.slice(5).replace(/^ /, "") };
      // Other fields (event:, id:, retry:) are not used by the gateway.
    }
  }

  private findLineEnd(): { pos: number; next: number } | null {
    const lf = this.buf.indexOf("\n");
    const cr = this.buf.indexOf("\r");
    if (lf === -1 && cr === -1) return null;
    if (cr === -1 || (lf !== -1 && lf < cr)) return { pos: lf, next: lf + 1 };
    // CR found first: it ends the line unless it is the final byte in the
    // buffer, where a following LF may still arrive in the next chunk.
    if (cr === this.buf.length - 1) return null;
    const two = this.buf[cr + 1] === "\n" ? 2 : 1;
    return { pos: cr, next: cr + two };
  }
}

export class Client {
  readonly base: string;
  private readonly apiKey: string | null;
  readonly timeoutMs: number;

  constructor(baseUrl?: string, apiKey?: string, timeoutMs = 300_000) {
    this.base = (
      baseUrl ??
      process.env.BLAZAR_URL ??
      BLAZAR_URL
    ).replace(/\/+$/, "");
    this.apiKey = apiKey ?? process.env.BLAZAR_API_KEY ?? null;
    this.timeoutMs = timeoutMs;
  }

  /** One-shot chat completion (non-streaming). */
  async chat(model: string, messages: ChatMessage[], opts: ChatOptions = {}): Promise<ChatResponse> {
    const { signal, ...body } = opts;
    return this.json<ChatResponse>(
      "/v1/chat/completions",
      { model, messages, stream: false, ...body },
      signal,
      "POST",
    );
  }

  /** Streaming chat: yields incremental delta objects, then a final usage chunk if sent. */
  async *chatStream(
    model: string,
    messages: ChatMessage[],
    opts: ChatOptions = {},
  ): AsyncGenerator<Record<string, unknown>> {
    const { signal, ...body } = opts;
    const timeout = AbortSignal.timeout(this.timeoutMs);
    const merged = signal ? AbortSignal.any([signal, timeout]) : timeout;
    const res = await fetch(`${this.base}/v1/chat/completions`, {
      method: "POST",
      headers: this.headers(),
      body: JSON.stringify({ model, messages, stream: true, ...body }),
      signal: merged,
    });
    if (!res.ok) throw await blazarError(res);
    if (!res.body) throw new BlazarError(0, "gateway returned no body", null);
    const decoder = new TextDecoder();
    const parser = new SseParser();
    const reader = res.body.getReader();
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      parser.feed(decoder.decode(value, { stream: true }));
      let ev: { data: string } | null;
      while ((ev = parser.nextEvent()) !== null) {
        if (ev.data === "[DONE]") return;
        const parsed = JSON.parse(ev.data) as Record<string, unknown>;
        yield parsed;
      }
    }
  }

  /** OpenAI-style model listing. */
  async models(signal?: AbortSignal): Promise<{ data: ModelInfo[] }> {
    return this.json("/v1/models", undefined, signal, "GET");
  }

  /** Resident children snapshot (`blazar ps` surface). */
  async ps(signal?: AbortSignal): Promise<PsRow[]> {
    const r = await this.json<{ models?: PsRow[] }>("/api/ps", undefined, signal, "GET");
    return r.models ?? [];
  }

  /** Routing decision card for a model (`blazar explain` surface). */
  async explain(model: string, signal?: AbortSignal): Promise<ExplainCard> {
    return this.json(`/api/explain/${encodeURIComponent(model)}`, undefined, signal, "GET");
  }

  /** Federation failover report. */
  async failover(signal?: AbortSignal): Promise<FailoverReport> {
    return this.json("/api/failover", undefined, signal, "GET");
  }

  /** Configured MCP servers. */
  async mcpServers(signal?: AbortSignal): Promise<McpServerInfo[]> {
    const r = await this.json<{ servers?: McpServerInfo[] }>("/api/mcp", undefined, signal, "GET");
    return r.servers ?? [];
  }

  private headers(): Record<string, string> {
    const h: Record<string, string> = { "content-type": "application/json" };
    if (this.apiKey) h.authorization = `Bearer ${this.apiKey}`;
    return h;
  }

  private async json<T>(
    path: string,
    body: unknown,
    signal: AbortSignal | undefined,
    method: "GET" | "POST",
  ): Promise<T> {
    const timeout = AbortSignal.timeout(this.timeoutMs);
    const merged = signal ? AbortSignal.any([signal, timeout]) : timeout;
    const res = await fetch(`${this.base}${path}`, {
      method,
      headers: this.headers(),
      body: body === undefined ? undefined : JSON.stringify(body),
      signal: merged,
    });
    if (!res.ok) throw await blazarError(res);
    return (await res.json()) as T;
  }
}

async function blazarError(res: Response): Promise<BlazarError> {
  let message = `${res.status} ${res.statusText}`;
  let blazarCode: string | null = null;
  try {
    const v = (await res.json()) as {
      error?: { message?: string; blazar_code?: string };
    };
    if (v.error?.message) message = v.error.message;
    blazarCode = v.error?.blazar_code ?? null;
  } catch {
    // non-JSON error body: keep the status line
  }
  return new BlazarError(res.status, message, blazarCode);
}
