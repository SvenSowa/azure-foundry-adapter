/**
 * Chat-completions wire format for Azure AI Foundry.
 *
 * Pure wire-shape code: builds messages[], streams /v1/chat/completions,
 * accumulates token deltas + tool_call deltas via index. The agent loop
 * scaffolding lives in common.ts.
 */
import {
  DEFAULT_API_VERSION,
  DEFAULT_DEPLOYMENT,
  OPENAI_V1_PATH,
} from "../shared/constants.js";
import type { ChatMessage } from "../shared/types.js";
import type { ToolDefinition } from "./tools.js";
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

// Soft-cap older tool_result bodies once the conversation grows.
const TOOL_RESULT_HISTORY_BUDGET = 1500;
const TOOL_RESULT_RECENT_KEEP = 2;

interface PartialToolCall {
  id: string;
  name: string;
  argumentsRaw: string;
}

function buildUrl(endpoint: string, apiVersion: string): string {
  // The /openai/v1/ path is Azure's modern "v1 API" surface; no api-version
  // query param. Legacy /openai/deployments/{name}/... requires one.
  const useLegacy = apiVersion !== DEFAULT_API_VERSION && apiVersion !== "";
  if (useLegacy) {
    // Caller passes deployment via the body's "model" field; the legacy URL
    // also names it in the path. We don't currently support legacy here —
    // sticking with v1 is correct for Foundry resources today.
  }
  return `${endpoint}${OPENAI_V1_PATH}/chat/completions`;
}

const chatTurnRunner: TurnRunner<ChatMessage[]> = {
  surface: "chat",

  buildInitialState(prompt, instructions) {
    const msgs: ChatMessage[] = [];
    if (instructions) msgs.push({ role: "system", content: instructions });
    msgs.push({ role: "user", content: prompt });
    return msgs;
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
    const apiVersion = String(cfg.apiVersion ?? DEFAULT_API_VERSION).trim();

    truncateOlderToolResults(state);

    const body: Record<string, unknown> = {
      model: deployment,
      messages: state,
      stream: true,
      stream_options: { include_usage: true },
    };
    if (cfg.temperature !== undefined) body.temperature = cfg.temperature;
    if (cfg.maxOutputTokens !== undefined) body.max_completion_tokens = cfg.maxOutputTokens;
    if (cfg.reasoningEffort) body.reasoning_effort = cfg.reasoningEffort;
    if (tools) body.tools = tools;

    const url = buildUrl(endpoint, apiVersion);
    const res = await postWithRateLimitRetry(url, apiKey, body, abort, (m) => emitLog(ctx, m));
    if (!res.ok || !res.body) {
      const text = await res.text().catch(() => "");
      throw new Error(`Foundry HTTP ${res.status}: ${text.slice(0, 500)}`);
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";

    let contentDelta = "";
    let finishReason = "unknown";
    let inputTokens = 0;
    let outputTokens = 0;
    const toolCallsByIndex = new Map<number, PartialToolCall>();

    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      let nl;
      while ((nl = buf.indexOf("\n")) >= 0) {
        const rawLine = buf.slice(0, nl).trim();
        buf = buf.slice(nl + 1);
        if (!rawLine || !rawLine.startsWith("data:")) continue;
        const data = rawLine.slice("data:".length).trim();
        if (data === "[DONE]") {
          if (finishReason === "unknown") finishReason = "stop";
          continue;
        }
        let chunk: ChatStreamChunk;
        try {
          chunk = JSON.parse(data);
        } catch {
          continue;
        }
        const delta = chunk.choices?.[0]?.delta;
        if (delta?.content) {
          contentDelta += delta.content;
          await emit(ctx, { kind: "token", text: delta.content });
        }
        if (delta?.tool_calls) {
          for (const tc of delta.tool_calls) {
            const slot = toolCallsByIndex.get(tc.index) ?? { id: "", name: "", argumentsRaw: "" };
            if (tc.id) slot.id = tc.id;
            if (tc.function?.name) slot.name = tc.function.name;
            if (tc.function?.arguments) slot.argumentsRaw += tc.function.arguments;
            toolCallsByIndex.set(tc.index, slot);
          }
        }
        const fr = chunk.choices?.[0]?.finish_reason;
        if (fr) finishReason = fr;
        if (chunk.usage) {
          inputTokens = chunk.usage.prompt_tokens ?? inputTokens;
          outputTokens = chunk.usage.completion_tokens ?? outputTokens;
        }
      }
    }

    const toolCalls: NormalizedToolCall[] = Array.from(toolCallsByIndex.entries())
      .sort((a, b) => a[0] - b[0])
      .map(([, v]) => ({ callId: v.id, name: v.name, arguments: v.argumentsRaw }))
      .filter((t) => t.name.length > 0);

    // Append the assistant message to state (mirrors what the model just said,
    // including any tool_calls so subsequent turns reference them by call_id).
    const newState: ChatMessage[] = [
      ...state,
      {
        role: "assistant",
        content: contentDelta || null,
        ...(toolCalls.length > 0
          ? {
              tool_calls: toolCalls.map((tc) => ({
                id: tc.callId,
                type: "function" as const,
                function: { name: tc.name, arguments: tc.arguments },
              })),
            }
          : {}),
      },
    ];

    return {
      state: newState,
      contentDelta,
      toolCalls,
      finishReason,
      usage: { input: inputTokens, output: outputTokens },
    } satisfies TurnOutcome<ChatMessage[]>;
  },

  appendToolResult(state, callId, resultJson) {
    return [...state, { role: "tool", tool_call_id: callId, content: resultJson }];
  },
};

function truncateOlderToolResults(messages: ChatMessage[]): void {
  const toolResultIdx: number[] = [];
  for (let i = 0; i < messages.length; i++) {
    if (messages[i].role === "tool") toolResultIdx.push(i);
  }
  const truncateBefore = toolResultIdx.length - TOOL_RESULT_RECENT_KEEP;
  for (let k = 0; k < truncateBefore; k++) {
    const i = toolResultIdx[k];
    const m = messages[i];
    const content = typeof m.content === "string" ? m.content : "";
    if (content.length > TOOL_RESULT_HISTORY_BUDGET) {
      messages[i] = {
        ...m,
        content:
          content.slice(0, TOOL_RESULT_HISTORY_BUDGET) +
          `\n…[truncated ${content.length - TOOL_RESULT_HISTORY_BUDGET} chars from older tool result]`,
      };
    }
  }
}

interface ChatStreamChunk {
  choices?: Array<{
    delta?: {
      content?: string;
      tool_calls?: Array<{
        index: number;
        id?: string;
        type?: string;
        function?: { name?: string; arguments?: string };
      }>;
    };
    finish_reason?: string;
  }>;
  usage?: { prompt_tokens?: number; completion_tokens?: number };
}

/** Public entry point — kept exported so unit tests can call directly. */
export async function executeChat(ctx: ExecuteContext): Promise<ExecuteResult> {
  return runAgentLoop(ctx, chatTurnRunner);
}
