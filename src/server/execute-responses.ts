/**
 * Responses-API wire format for Azure AI Foundry.
 *
 * Pure wire-shape code: builds input[] of typed ResponseItems, streams
 * /v1/responses, accumulates output items via response.output_item.done.
 * The agent loop scaffolding lives in common.ts.
 *
 * Wire shapes are lifted from openai/codex; see docs/RESPONSES-API-NOTES.md
 * for source citations.
 */
import {
  DEFAULT_DEPLOYMENT,
  OPENAI_V1_PATH,
} from "../shared/constants.js";
import {
  emit,
  emitLog,
  postWithRateLimitRetry,
  resolveApiKey,
  type ExecuteContext,
  type ExecuteResult,
  type NormalizedToolCall,
  type TurnOutcome,
  type TurnRunner,
  runAgentLoop,
} from "./common.js";
import type { ToolDefinition } from "./tools.js";

// ---------------------------------------------------------------------------
// Responses API wire types (subset we use)
// ---------------------------------------------------------------------------

type ResponsesContent =
  | { type: "input_text"; text: string }
  | { type: "output_text"; text: string };

export type ResponsesItem =
  | { type: "message"; role: "user" | "assistant" | "system" | "developer"; content: ResponsesContent[]; id?: string }
  | { type: "function_call"; call_id: string; name: string; arguments: string; id?: string }
  | { type: "function_call_output"; call_id: string; output: string }
  | { type: "reasoning"; id?: string; summary: unknown[]; content?: unknown[]; encrypted_content?: string };

type ResponsesTool = {
  type: "function";
  name: string;
  description?: string;
  strict?: boolean;
  parameters: object;
};

interface SseEvent {
  type: string;
  data: Record<string, unknown>;
}

// ---------------------------------------------------------------------------
// SSE parsing — Responses uses event:/data: framing with response.completed
// terminator (NOT [DONE] like chat completions).
// ---------------------------------------------------------------------------

export async function* parseResponsesSse(
  body: ReadableStream<Uint8Array>,
  signal?: AbortSignal,
): AsyncGenerator<SseEvent> {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  for (;;) {
    if (signal?.aborted) throw new Error("aborted");
    const { value, done } = await reader.read();
    if (done) return;
    buf += decoder.decode(value, { stream: true });
    let nl;
    while ((nl = buf.indexOf("\n\n")) >= 0) {
      const block = buf.slice(0, nl);
      buf = buf.slice(nl + 2);
      let evType = "";
      let dataStr = "";
      for (const line of block.split("\n")) {
        if (line.startsWith("event:")) evType = line.slice(6).trim();
        else if (line.startsWith("data:")) dataStr += (dataStr ? "\n" : "") + line.slice(5).trim();
      }
      if (!dataStr) continue;
      try {
        yield { type: evType, data: JSON.parse(dataStr) };
      } catch {
        // drop malformed payloads (codex does the same)
      }
    }
  }
}

/**
 * Convert chat-completions tool definition shape (nested under "function")
 * into Responses-API flat shape.
 */
function flattenTools(tools: ToolDefinition[]): ResponsesTool[] {
  return tools.map((t) => ({
    type: "function" as const,
    name: t.function.name,
    description: t.function.description,
    parameters: t.function.parameters,
  }));
}

/**
 * Responses-API state carries both the input array AND the resolved system
 * prompt — Responses API takes "instructions" as a top-level body field
 * (NOT as a state item like chat-completions does), so each request needs
 * the instructions string handy.
 */
interface ResponsesState {
  items: ResponsesItem[];
  instructions: string | null;
}

const responsesTurnRunner: TurnRunner<ResponsesState> = {
  surface: "responses",

  buildInitialState(prompt, instructions) {
    return {
      items: [
        {
          type: "message",
          role: "user",
          content: [{ type: "input_text", text: prompt }],
        },
      ],
      instructions,
    };
  },

  async runTurn({ ctx, state, tools, abort }) {
    const cfg = ctx.config;
    const endpoint = String(cfg.endpoint ?? process.env.AZURE_FOUNDRY_ENDPOINT ?? "")
      .trim()
      .replace(/\/+$/, "");
    const apiKey = resolveApiKey(cfg);
    // Paperclip's native Model dropdown writes the selection to cfg.model;
    // cfg.deployment is accepted for back-compat. Either is the Azure deployment.
    const deployment =
      String(cfg.deployment ?? cfg.model ?? DEFAULT_DEPLOYMENT).trim();

    const url = `${endpoint}${OPENAI_V1_PATH}/responses`;

    const body: Record<string, unknown> = {
      model: deployment,
      input: state.items,
      stream: true,
      // store=false → server doesn't persist items across requests, so we
      // re-send the full input array each turn. We also strip server-assigned
      // ids in state advancement to avoid id-mismatch errors. Codex does
      // the same when not using its WebSocket transport.
      store: false,
      parallel_tool_calls: true,
    };
    if (state.instructions) body.instructions = state.instructions;
    if (cfg.reasoningEffort) {
      body.reasoning = { effort: cfg.reasoningEffort, summary: "auto" };
      body.include = ["reasoning.encrypted_content"];
    }
    if (cfg.maxOutputTokens !== undefined) body.max_output_tokens = cfg.maxOutputTokens;
    if (tools) body.tools = flattenTools(tools);

    const res = await postWithRateLimitRetry(url, apiKey, body, abort, (m) => emitLog(ctx, m));
    if (!res.ok || !res.body) {
      const text = await res.text().catch(() => "");
      throw new Error(`Foundry HTTP ${res.status}: ${text.slice(0, 500)}`);
    }

    const outputItems: ResponsesItem[] = [];
    let contentDelta = "";
    let finishReason = "unknown";
    let usage = { input: 0, output: 0, reasoning: 0 };

    for await (const ev of parseResponsesSse(res.body, abort.signal)) {
      const k = (ev.data?.type as string) ?? ev.type;
      const handler = mapResponsesEvent(k);
      if (!handler) continue;
      const update = await handler(ev.data, async (text) => {
        contentDelta += text;
        await emit(ctx, { kind: "token", text });
      }, async (text) => {
        await emit(ctx, { kind: "thinking", text });
      });
      if (update?.outputItem) outputItems.push(update.outputItem);
      if (update?.finishReason) finishReason = update.finishReason;
      if (update?.usage) usage = { ...usage, ...update.usage };
      if (update?.fallbackOutputItems && outputItems.length === 0) {
        outputItems.push(...update.fallbackOutputItems);
      }
      if (update?.terminate) break;
      if (update?.error) throw new Error(update.error);
    }

    // Strip ids before threading items back; with store=false, ids aren't
    // persisted server-side and re-sending them trips HTTP 400 "Item with
    // id ... not found".
    const itemsAfter: ResponsesItem[] = [...state.items];
    for (const item of outputItems) {
      // eslint-disable-next-line @typescript-eslint/no-unused-vars
      const { id: _id, ...rest } = item as ResponsesItem & { id?: string };
      // Reasoning items without encrypted_content can't be replayed.
      if (rest.type === "reasoning") {
        const r = rest as Extract<ResponsesItem, { type: "reasoning" }>;
        if (!r.encrypted_content) continue;
      }
      itemsAfter.push(rest as ResponsesItem);
    }

    const toolCalls: NormalizedToolCall[] = outputItems
      .filter((it): it is Extract<ResponsesItem, { type: "function_call" }> => it.type === "function_call")
      .map((tc) => ({ callId: tc.call_id, name: tc.name, arguments: tc.arguments }));

    return {
      state: { items: itemsAfter, instructions: state.instructions },
      contentDelta,
      toolCalls,
      finishReason,
      usage,
    } satisfies TurnOutcome<ResponsesState>;
  },

  appendToolResult(state, callId, resultJson) {
    return {
      items: [...state.items, { type: "function_call_output", call_id: callId, output: resultJson }],
      instructions: state.instructions,
    };
  },
};

// ---------------------------------------------------------------------------
// Per-event handlers, isolated for testability.
// ---------------------------------------------------------------------------

interface EventUpdate {
  outputItem?: ResponsesItem;
  finishReason?: string;
  usage?: { input?: number; output?: number; reasoning?: number };
  fallbackOutputItems?: ResponsesItem[];
  terminate?: boolean;
  error?: string;
}

type EventHandler = (
  data: Record<string, unknown>,
  emitToken: (text: string) => Promise<void>,
  emitThinking: (text: string) => Promise<void>,
) => Promise<EventUpdate | null> | EventUpdate | null;

export function mapResponsesEvent(kind: string): EventHandler | null {
  switch (kind) {
    case "response.created":
      return () => null;
    case "response.output_text.delta":
      return async (data, emitToken) => {
        const delta = (data.delta as string) ?? "";
        if (delta) await emitToken(delta);
        return null;
      };
    case "response.output_item.done":
      return (data) => {
        const item = data.item as ResponsesItem | undefined;
        return item ? { outputItem: item } : null;
      };
    case "response.reasoning_summary_text.delta":
    case "response.reasoning_text.delta":
      return async (data, _emitToken, emitThinking) => {
        const delta = (data.delta as string) ?? "";
        if (delta) await emitThinking(delta);
        return null;
      };
    case "response.completed":
      return (data) => {
        const resp = data.response as
          | { id?: string; usage?: Record<string, number>; output?: ResponsesItem[] }
          | undefined;
        const u = resp?.usage as
          | {
              input_tokens?: number;
              output_tokens?: number;
              total_tokens?: number;
              input_tokens_details?: { cached_tokens?: number };
              output_tokens_details?: { reasoning_tokens?: number };
            }
          | undefined;
        return {
          finishReason: "stop",
          usage: u
            ? {
                input: u.input_tokens ?? 0,
                output: u.output_tokens ?? 0,
                reasoning: u.output_tokens_details?.reasoning_tokens ?? 0,
              }
            : undefined,
          fallbackOutputItems: resp?.output && Array.isArray(resp.output) ? resp.output : undefined,
          terminate: true,
        };
      };
    case "response.failed":
      return (data) => {
        const e = (data.response as { error?: { code?: string; message?: string } } | undefined)?.error ?? {};
        return { error: `response.failed: ${e.code ?? "unknown"} — ${e.message ?? ""}` };
      };
    case "response.incomplete":
      return (data) => {
        const reason = (data.response as { incomplete_details?: { reason?: string } } | undefined)
          ?.incomplete_details?.reason;
        return { finishReason: `incomplete:${reason ?? "unknown"}` };
      };
    default:
      return null;
  }
}

/** Public entry point — kept exported so unit tests can call directly. */
export async function executeResponses(ctx: ExecuteContext): Promise<ExecuteResult> {
  return runAgentLoop(ctx, responsesTurnRunner);
}
