/**
 * Shared infrastructure used by both execute-chat and execute-responses.
 *
 * The two executors differ only in their wire format — request body shape,
 * SSE event taxonomy, and tool-call accumulation. Everything else (timeout
 * handling, prompt building, instructions loading, sandbox setup, hop
 * budget, exit-code mapping, AF:: stdout protocol, rate-limit retry, etc.)
 * is identical and lives here.
 */
import { promises as fs } from "node:fs";
import {
  DEFAULT_GRACE_SEC,
  DEFAULT_TIMEOUT_SEC,
} from "../shared/constants.js";
import type {
  AzureFoundryConfig,
  AzureFoundrySessionParams,
} from "../shared/types.js";
import { DEFAULT_TOOLS, dispatchToolCall, type ToolDefinition } from "./tools.js";
import { makeSandboxOptions, type SandboxOptions } from "./sandbox.js";

// ---------------------------------------------------------------------------
// Shared types
// ---------------------------------------------------------------------------

export interface ExecuteContext {
  prompt?: string;
  config: AzureFoundryConfig & Record<string, unknown>;
  context?: Record<string, unknown>;
  agent?: { id?: string; name?: string; companyId?: string };
  runtime?: { sessionParams?: Record<string, unknown> | null; taskKey?: string | null };
  session?: AzureFoundrySessionParams | null;
  templateVars?: Record<string, string>;
  workspaceRoot?: string;
  tools?: ToolDefinition[];
  onLog: (stream: "stdout" | "stderr", chunk: string) => Promise<void> | void;
  onSession?: (params: AzureFoundrySessionParams) => Promise<void> | void;
  signal?: AbortSignal;
}

export interface ExecuteResult {
  exitCode: number;
  finishReason: string;
  outputText: string;
  usage: { input: number; output: number; reasoning?: number };
  toolHops: number;
}

/** Tool call shape that bridges chat and responses wire formats. */
export interface NormalizedToolCall {
  /** Stable id from the wire (call_id on responses, id on chat). */
  callId: string;
  /** Function name. */
  name: string;
  /** Arguments as a JSON string (empty string if none/streamed-incomplete). */
  arguments: string;
}

/** What runTurn returns — common shape regardless of wire underneath. */
export interface TurnOutcome<TState> {
  /** Updated state with this turn's assistant output appended. */
  state: TState;
  /** Visible text content emitted in this turn. */
  contentDelta: string;
  /** Tool calls the model wants executed before the next turn. */
  toolCalls: NormalizedToolCall[];
  /** Wire-side finish reason (stop, tool_calls, length, etc.). */
  finishReason: string;
  /** Token usage for this single turn. */
  usage: { input: number; output: number; reasoning?: number };
}

/** Per-path glue that runAgentLoop calls into. */
export interface TurnRunner<TState> {
  /** Build the initial conversation state from the user prompt + system instructions. */
  buildInitialState(prompt: string, instructions: string | null): TState;
  /** Stream a single turn against the wire and return the outcome. */
  runTurn(args: {
    ctx: ExecuteContext;
    state: TState;
    tools: ToolDefinition[] | null;
    abort: AbortController;
  }): Promise<TurnOutcome<TState>>;
  /** Append a tool result to the state in the wire-specific shape. */
  appendToolResult(state: TState, callId: string, resultJson: string): TState;
  /** Path label for logs / events (e.g. "chat" or "responses"). */
  surface: string;
}

// ---------------------------------------------------------------------------
// AF:: stdout protocol
// ---------------------------------------------------------------------------

export const PREFIX = "AF::";

export function emit(ctx: ExecuteContext, event: Record<string, unknown>) {
  return ctx.onLog("stdout", `${PREFIX}${JSON.stringify(event)}\n`);
}
export function emitLog(ctx: ExecuteContext, msg: string) {
  return ctx.onLog("stdout", `[azure_foundry] ${msg}\n`);
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const DEFAULT_HEARTBEAT_PROMPT = `You are an autonomous agent woken up on a scheduled heartbeat.

You have shell, file, and search tools to inspect the workspace. If there's a
task assigned to you (look for taskId / taskTitle / taskBody in your context),
work on it. Otherwise, look for unassigned issues to claim, or report idle.

When done, summarize what you accomplished in plain text.`;

export function asStr(v: unknown): string | undefined {
  if (typeof v !== "string") return undefined;
  const t = v.trim();
  return t.length > 0 ? t : undefined;
}

export function expandTemplate(text: string | undefined | null, vars: Record<string, string>): string {
  if (typeof text !== "string" || text.length === 0) return "";
  return text.replace(/\{\{\s*([a-zA-Z0-9_]+)\s*\}\}/g, (_, key) => {
    const v = vars[key];
    return typeof v === "string" ? v : "";
  });
}

export function buildPrompt(ctx: ExecuteContext): string {
  const ctxPrompt = asStr(ctx.context?.prompt);
  if (ctxPrompt) return ctxPrompt;
  if (asStr(ctx.prompt)) return ctx.prompt!;
  const taskId = asStr(ctx.context?.taskId);
  const taskTitle = asStr(ctx.context?.taskTitle);
  const taskBody = asStr(ctx.context?.taskBody);
  if (taskId || taskTitle || taskBody) {
    return [
      `You have a task assigned to you.`,
      taskId ? `Task ID: ${taskId}` : "",
      taskTitle ? `Title: ${taskTitle}` : "",
      taskBody ? `\n${taskBody}` : "",
      `\nWork on the task using your tools, then summarize what you did.`,
    ].filter(Boolean).join("\n");
  }
  return DEFAULT_HEARTBEAT_PROMPT;
}

export function buildTemplateVars(ctx: ExecuteContext): Record<string, string> {
  const out: Record<string, string> = { ...(ctx.templateVars ?? {}) };
  const c = ctx.context ?? {};
  for (const k of [
    "agentId", "agentName", "companyId", "companyName", "runId",
    "taskId", "taskTitle", "taskBody", "projectName",
  ]) {
    const v = asStr(c[k]);
    if (v) out[k] = v;
  }
  if (ctx.agent?.name && !out.agentName) out.agentName = ctx.agent.name;
  if (ctx.agent?.id && !out.agentId) out.agentId = ctx.agent.id;
  if (ctx.agent?.companyId && !out.companyId) out.companyId = ctx.agent.companyId;
  return out;
}

export async function loadInstructions(
  cfg: AzureFoundryConfig,
  vars: Record<string, string>,
): Promise<string | null> {
  if (!cfg.instructionsFilePath) return null;
  try {
    const raw = await fs.readFile(cfg.instructionsFilePath, "utf8");
    return expandTemplate(raw, vars);
  } catch (err) {
    return `[failed to read instructions: ${(err as Error).message}]`;
  }
}

export function resolveApiKey(cfg: AzureFoundryConfig): string {
  const k = cfg.apiKey;
  if (typeof k === "string") return k;
  if (k && typeof k === "object" && "value" in k) return (k as { value: string }).value;
  return process.env.AZURE_FOUNDRY_API_KEY?.trim() ?? "";
}

// ---------------------------------------------------------------------------
// Rate-limit retry — same policy for both wire shapes.
// ---------------------------------------------------------------------------

const RATE_LIMIT_RETRIES = 4;
const RATE_LIMIT_BASE_MS = 1500;
// Don't honor an absurd Retry-After that would blow the run's time budget; cap
// the wait so we fail fast with a clear message instead of stalling.
const RATE_LIMIT_MAX_WAIT_MS = 30_000;

/** Sleep that resolves early (rejecting) if the abort signal fires. */
function abortableDelay(ms: number, signal: AbortSignal): Promise<void> {
  return new Promise<void>((resolve, reject) => {
    if (signal.aborted) {
      reject(new DOMException("Aborted during rate-limit backoff", "AbortError"));
      return;
    }
    const onAbort = () => {
      clearTimeout(timer);
      reject(new DOMException("Aborted during rate-limit backoff", "AbortError"));
    };
    const timer = setTimeout(() => {
      signal.removeEventListener("abort", onAbort);
      resolve();
    }, ms);
    signal.addEventListener("abort", onAbort, { once: true });
  });
}

export async function postWithRateLimitRetry(
  url: string,
  apiKey: string,
  body: Record<string, unknown>,
  ctrl: AbortController,
  log: (msg: string) => void | Promise<void>,
): Promise<Response> {
  let attempt = 0;
  for (;;) {
    const res = await fetch(url, {
      method: "POST",
      headers: {
        "api-key": apiKey,
        "content-type": "application/json",
        accept: "text/event-stream",
      },
      body: JSON.stringify(body),
      signal: ctrl.signal,
    });
    if (res.status !== 429 || attempt >= RATE_LIMIT_RETRIES) return res;
    const retryAfter = Number(res.headers.get("retry-after"));
    const rawDelayMs =
      Number.isFinite(retryAfter) && retryAfter > 0
        ? retryAfter * 1000
        : RATE_LIMIT_BASE_MS * Math.pow(2, attempt);
    const delayMs = Math.min(rawDelayMs, RATE_LIMIT_MAX_WAIT_MS);
    await log(
      `Foundry 429 (rate limited) — backing off ${delayMs}ms (attempt ${attempt + 1}/${RATE_LIMIT_RETRIES})`,
    );
    await res.text().catch(() => "");
    // Abort-aware wait: if the run times out or is cancelled mid-backoff, stop
    // immediately rather than sleeping out the full Retry-After.
    await abortableDelay(delayMs, ctrl.signal);
    attempt++;
  }
}

// ---------------------------------------------------------------------------
// Generic agent loop — both wire paths plug into this.
// ---------------------------------------------------------------------------

const DEFAULT_MAX_TOOL_HOPS = 20;

export async function runAgentLoop<TState>(
  ctx: ExecuteContext,
  runner: TurnRunner<TState>,
): Promise<ExecuteResult> {
  const cfg = ctx.config;

  // Validate required config up front.
  const endpoint = (cfg.endpoint ?? process.env.AZURE_FOUNDRY_ENDPOINT ?? "")
    .toString()
    .trim()
    .replace(/\/+$/, "");
  if (!endpoint) {
    await emit(ctx, { kind: "error", message: "AZURE_FOUNDRY_ENDPOINT not configured" });
    return zeroResult(2, "config_error");
  }
  const apiKey = resolveApiKey(cfg);
  if (!apiKey) {
    await emit(ctx, { kind: "error", message: "AZURE_FOUNDRY_API_KEY not configured" });
    return zeroResult(2, "config_error");
  }

  const enableToolLoop = cfg.enableToolLoop !== false;
  const maxToolHops =
    typeof cfg.maxToolHops === "number" && cfg.maxToolHops > 0
      ? cfg.maxToolHops
      : DEFAULT_MAX_TOOL_HOPS;
  const timeoutMs = (cfg.timeoutSec ?? DEFAULT_TIMEOUT_SEC) * 1000;
  const _graceMs = (cfg.graceSec ?? DEFAULT_GRACE_SEC) * 1000;

  await emitLog(
    ctx,
    `Foundry run: surface=${runner.surface} deployment=${cfg.deployment ?? cfg.model ?? "(default)"} toolLoop=${enableToolLoop} maxHops=${maxToolHops}`,
  );

  // Build prompt + instructions and the runner-specific initial state.
  const vars = buildTemplateVars(ctx);
  const userContent = expandTemplate(buildPrompt(ctx), vars);
  const systemContent = await loadInstructions(cfg, vars);
  let state = runner.buildInitialState(userContent, systemContent);

  // Sandbox setup is shared.
  const sandboxOpts: SandboxOptions = makeSandboxOptions({
    rootDir: ctx.workspaceRoot ?? process.cwd(),
  });
  const tools = ctx.tools ?? DEFAULT_TOOLS;

  // Timeout / abort plumbing.
  const ctrl = new AbortController();
  let timedOut = false;
  const timer = setTimeout(() => {
    timedOut = true;
    ctrl.abort();
  }, timeoutMs);
  if (ctx.signal) {
    ctx.signal.addEventListener("abort", () => ctrl.abort(), { once: true });
  }

  let finalText = "";
  let totalInput = 0;
  let totalOutput = 0;
  let totalReasoning = 0;
  let finishReason = "unknown";
  let toolHops = 0;

  try {
    for (let hop = 0; hop <= maxToolHops; hop++) {
      const turn = await runner.runTurn({
        ctx,
        state,
        tools: enableToolLoop ? tools : null,
        abort: ctrl,
      });
      state = turn.state;
      totalInput += turn.usage.input;
      totalOutput += turn.usage.output;
      totalReasoning += turn.usage.reasoning ?? 0;
      finalText += turn.contentDelta;
      finishReason = turn.finishReason;

      if (turn.toolCalls.length === 0) break;

      if (!enableToolLoop) {
        for (const tc of turn.toolCalls) {
          await emit(ctx, {
            kind: "tool_call",
            id: tc.callId,
            name: tc.name,
            arguments: tc.arguments,
          });
        }
        break;
      }

      if (hop === maxToolHops) {
        await emit(ctx, { kind: "error", message: `tool hop budget exhausted (${maxToolHops})` });
        finishReason = "hop_budget_exhausted";
        break;
      }

      // Execute tool calls and append results in the wire-specific shape.
      for (const tc of turn.toolCalls) {
        await emit(ctx, {
          kind: "tool_call",
          id: tc.callId,
          name: tc.name,
          arguments: tc.arguments,
        });
        let parsedArgs: Record<string, unknown> = {};
        try {
          parsedArgs = tc.arguments ? JSON.parse(tc.arguments) : {};
        } catch (err) {
          await emit(ctx, {
            kind: "log",
            level: "warn",
            message: `bad JSON in tool args for ${tc.name}: ${(err as Error).message}`,
          });
        }
        const dispatched = await dispatchToolCall(tc.name, parsedArgs, sandboxOpts);
        const resultPayload = dispatched.ok
          ? dispatched.result
          : { error: dispatched.errorMessage ?? "unknown error" };
        const resultJson = JSON.stringify(resultPayload);
        await emit(ctx, {
          kind: "tool_result",
          id: tc.callId,
          content:
            resultJson.length > 2000
              ? resultJson.slice(0, 2000) + "…[truncated]"
              : resultJson,
        });
        state = runner.appendToolResult(state, tc.callId, resultJson);
      }
      toolHops++;
    }

    await emit(ctx, {
      kind: "usage",
      input: totalInput,
      output: totalOutput,
      ...(totalReasoning > 0 ? { reasoning: totalReasoning } : {}),
    });
    await emit(ctx, { kind: "finish", reason: finishReason });

    if (ctx.onSession) {
      const conversationId =
        ctx.session?.conversationId ??
        `af-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
      await ctx.onSession({
        conversationId,
        totalInputTokens: (ctx.session?.totalInputTokens ?? 0) + totalInput,
        totalOutputTokens: (ctx.session?.totalOutputTokens ?? 0) + totalOutput,
      });
    }

    const softTerminations = new Set([
      "stop", "tool_calls", "hop_budget_exhausted", "length",
    ]);
    return {
      exitCode: softTerminations.has(finishReason) ? 0 : 1,
      finishReason,
      outputText: finalText,
      usage: { input: totalInput, output: totalOutput, reasoning: totalReasoning },
      toolHops,
    };
  } catch (err) {
    const e = err as Error;
    const aborted = e?.name === "AbortError" || /\baborted\b/i.test(e?.message ?? "");
    let message = e?.message ?? String(err);
    if (aborted) {
      message = timedOut
        ? `Run timed out after ${Math.round(timeoutMs / 1000)}s. The Azure deployment was rate-limiting (HTTP 429) and the model did not respond in time. Increase the deployment's Tokens-Per-Minute (TPM) rate limit in Azure AI Foundry, or reduce request volume/concurrency.`
        : "Run was cancelled before completion.";
    }
    await emit(ctx, { kind: "error", message });
    return {
      exitCode: 1,
      finishReason: timedOut ? "timeout" : aborted ? "aborted" : "exception",
      outputText: finalText,
      usage: { input: totalInput, output: totalOutput, reasoning: totalReasoning },
      toolHops,
    };
  } finally {
    clearTimeout(timer);
  }
}

function zeroResult(exitCode: number, finishReason: string): ExecuteResult {
  return { exitCode, finishReason, outputText: "", usage: { input: 0, output: 0 }, toolHops: 0 };
}
