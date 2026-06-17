/**
 * Server-side adapter module exports.
 */
import { ADAPTER_TYPE, ADAPTER_LABEL } from "../shared/constants.js";
import { execute } from "./execute.js";
import { testEnvironment } from "./test.js";
import { detectModel } from "./detect-model.js";
import { listSkills, syncSkills } from "./skills.js";
import { listDeployments, type DeploymentModel } from "./deployments.js";
import { models, agentConfigurationDoc } from "../index.js";

export { execute, testEnvironment, detectModel, listSkills, syncSkills, listDeployments };

/**
 * Merge live Azure deployments with the static suggestion list.
 *
 * Live deployments (fetched with the server-level Foundry credentials) take
 * precedence and appear first; any static suggestion not already covered is
 * appended so the picker is never empty even before a resource is wired up.
 */
async function resolveModels(force = false): Promise<DeploymentModel[]> {
  const live = await listDeployments({ force });
  if (live.length === 0) return models;
  const seen = new Set(live.map((m) => m.id));
  const extras = models.filter((m) => !seen.has(m.id));
  return [...live, ...extras];
}

function readNonEmptyString(value: unknown): string | null {
  return typeof value === "string" && value.trim().length > 0
    ? value.trim()
    : null;
}

/**
 * Declarative config schema — the Paperclip UI fetches this from
 * GET /api/adapters/azure_foundry/config-schema and renders the right
 * form fields automatically (labels, hints, types, dropdowns).
 *
 * Async so the `deployment` combobox can be populated with the resource's
 * live deployments (fetched via server-level Foundry credentials). Falls back
 * to the static suggestion list when no credentials are configured yet.
 */
async function getConfigSchema() {
  const deploymentOptions = (await resolveModels()).map((m) => ({
    value: m.id,
    label: m.label,
  }));
  return {
    fields: [
      {
        key: "endpoint",
        label: "Endpoint",
        type: "text" as const,
        required: true,
        hint: "Foundry resource base URL, e.g. https://foundry-coxshire-eastus2.cognitiveservices.azure.com/",
        group: "Connection",
      },
      {
        key: "apiKey",
        label: "API key",
        type: "text" as const,
        required: true,
        hint: "Resource API key. Stored encrypted as a Paperclip secret.",
        group: "Connection",
      },
      {
        key: "deployment",
        label: "Deployment",
        type: "combobox" as const,
        required: true,
        default: deploymentOptions[0]?.value ?? "gpt-5-5",
        options: deploymentOptions,
        hint: "Azure-side deployment name. Live deployments are listed automatically; you can also type a custom name and press Enter.",
        group: "Connection",
      },
      {
        key: "apiVersion",
        label: "API version",
        type: "text" as const,
        hint: "Optional. Defaults to the modern /openai/v1 surface (no api-version param).",
        group: "Connection",
      },
      {
        key: "reasoningEffort",
        label: "Reasoning effort",
        type: "select" as const,
        options: [
          { value: "", label: "(model default)" },
          { value: "minimal", label: "minimal" },
          { value: "low", label: "low" },
          { value: "medium", label: "medium" },
          { value: "high", label: "high" },
          { value: "xhigh", label: "xhigh" },
        ],
        hint: "Passed via reasoning_effort to deployments that support it.",
        group: "Generation",
      },
      {
        key: "temperature",
        label: "Temperature",
        type: "number" as const,
        hint: "Sampling temperature. Leave blank for model default.",
        group: "Generation",
      },
      {
        key: "maxOutputTokens",
        label: "Max output tokens",
        type: "number" as const,
        hint: "Hard cap on output tokens.",
        group: "Generation",
      },
      {
        key: "enableToolLoop",
        label: "Enable tool loop",
        type: "toggle" as const,
        default: true,
        hint: "When on, the agent executes sandbox tools (read_file, run_shell, etc.) and loops until the model returns a tool-free response.",
        group: "Behavior",
      },
      {
        key: "maxToolHops",
        label: "Max tool hops",
        type: "number" as const,
        default: 20,
        hint: "Maximum tool-call rounds before the loop bails out.",
        group: "Behavior",
      },
      {
        key: "timeoutSec",
        label: "Timeout (seconds)",
        type: "number" as const,
        default: 300,
        hint: "Run timeout in seconds.",
        group: "Behavior",
      },
      {
        key: "instructionsFilePath",
        label: "Instructions file",
        type: "text" as const,
        hint: "Absolute path to a markdown file injected as the system prompt.",
        group: "Instructions",
      },
    ],
  };
}

/**
 * Plugin-loader entry point.
 *
 * Paperclip's external-adapter plugin loader calls `createServerAdapter()`
 * from the package root after install. We return a `ServerAdapterModule`
 * shape (duck-typed against @paperclipai/adapter-utils so the package
 * still compiles standalone).
 */
export function createServerAdapter() {
  return {
    type: ADAPTER_TYPE,
    label: ADAPTER_LABEL,
    execute,
    testEnvironment,
    sessionCodec,
    listSkills,
    syncSkills,
    models,
    supportsLocalAgentJwt: false,
    supportsInstructionsBundle: true,
    instructionsPathKey: "instructionsFilePath",
    agentConfigurationDoc,
    getConfigSchema,
    listModels: async () => resolveModels(),
    refreshModels: async () => resolveModels(true),
    detectModel: async () => {
      const d = detectModel();
      return { model: d.model, provider: "azure_foundry", source: d.source };
    },
  };
}

/**
 * Session codec — Paperclip persists this between heartbeats so the same
 * conversation can continue across runs.
 *
 * Foundry chat completions are stateless; we only need a stable conversationId
 * for log grouping plus running token totals. The full message history is
 * reconstructed from Paperclip's run/event store on each heartbeat.
 */
export const sessionCodec = {
  deserialize(raw: unknown) {
    if (typeof raw !== "object" || raw === null || Array.isArray(raw)) return null;
    const record = raw as Record<string, unknown>;
    const conversationId =
      readNonEmptyString(record.conversationId) ??
      readNonEmptyString(record.conversation_id);
    if (!conversationId) return null;
    return {
      conversationId,
      totalInputTokens:
        typeof record.totalInputTokens === "number" ? record.totalInputTokens : 0,
      totalOutputTokens:
        typeof record.totalOutputTokens === "number" ? record.totalOutputTokens : 0,
    };
  },
  serialize(params: unknown) {
    if (!params || typeof params !== "object") return null;
    const p = params as Record<string, unknown>;
    const conversationId =
      readNonEmptyString(p.conversationId) ??
      readNonEmptyString(p.conversation_id);
    if (!conversationId) return null;
    return {
      conversationId,
      totalInputTokens: typeof p.totalInputTokens === "number" ? p.totalInputTokens : 0,
      totalOutputTokens: typeof p.totalOutputTokens === "number" ? p.totalOutputTokens : 0,
    };
  },
  getDisplayId(params: unknown) {
    if (!params || typeof params !== "object") return null;
    const p = params as Record<string, unknown>;
    return (
      readNonEmptyString(p.conversationId) ??
      readNonEmptyString(p.conversation_id)
    );
  },
};
