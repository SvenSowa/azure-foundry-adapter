import { OPENAI_V1_PATH } from "../shared/constants.js";
import { getApiSurface, KNOWN_API_SURFACE } from "../shared/capability.js";

/**
 * Paperclip's adapter-utils contract:
 *
 *   testEnvironment(ctx: AdapterEnvironmentTestContext): Promise<AdapterEnvironmentTestResult>
 *
 * The UI fetches POST /api/companies/:companyId/adapters/:type/test-environment
 * and renders the returned `checks` array. Returning the wrong shape blanks the
 * configuration page (the UI assumes `result.checks.map(...)` is iterable).
 */

interface AdapterEnvironmentTestContext {
  companyId: string;
  adapterType: string;
  config: Record<string, unknown>;
  deployment?: unknown;
}

type CheckLevel = "info" | "warn" | "error";

interface AdapterEnvironmentCheck {
  code: string;
  level: CheckLevel;
  message: string;
  detail?: string | null;
  hint?: string | null;
}

interface AdapterEnvironmentTestResult {
  adapterType: string;
  status: "pass" | "warn" | "fail";
  checks: AdapterEnvironmentCheck[];
  testedAt: string;
}

/** Unwrap Paperclip's secret envelope `{type:"plain"|"ref",value:...}`. */
function asString(v: unknown): string {
  if (typeof v === "string") return v;
  if (v && typeof v === "object" && "value" in v && typeof (v as { value: unknown }).value === "string") {
    return (v as { value: string }).value;
  }
  return "";
}

export async function testEnvironment(
  ctx: AdapterEnvironmentTestContext,
): Promise<AdapterEnvironmentTestResult> {
  const checks: AdapterEnvironmentCheck[] = [];
  const config = ctx.config ?? {};

  const endpoint = (asString(config.endpoint) || process.env.AZURE_FOUNDRY_ENDPOINT || "")
    .trim()
    .replace(/\/+$/, "");
  const apiKey = (asString(config.apiKey) || process.env.AZURE_FOUNDRY_API_KEY || "").trim();
  // Paperclip's native Model dropdown writes the selection to config.model;
  // config.deployment is accepted for back-compat. Either is the Azure deployment.
  const deployment = (asString(config.deployment) || asString(config.model)).trim();

  if (!endpoint) {
    checks.push({
      code: "endpoint_missing",
      level: "error",
      message: "Endpoint is not configured.",
      hint: "Set the Endpoint field to your Foundry resource base URL, e.g. https://foundry-coxshire-eastus2.cognitiveservices.azure.com/",
    });
  } else if (!/^https?:\/\//i.test(endpoint)) {
    checks.push({
      code: "endpoint_malformed",
      level: "error",
      message: `Endpoint does not start with http:// or https://`,
      detail: endpoint,
    });
  } else {
    checks.push({
      code: "endpoint_ok",
      level: "info",
      message: `Endpoint configured`,
      detail: endpoint,
    });
  }

  if (!apiKey) {
    checks.push({
      code: "api_key_missing",
      level: "error",
      message: "API key is not configured.",
      hint: "Paste the resource API key into the API key field, or set AZURE_FOUNDRY_API_KEY in the server env.",
    });
  } else {
    checks.push({
      code: "api_key_present",
      level: "info",
      message: "API key configured",
      detail: `key length ${apiKey.length}`,
    });
  }

  // Live reachability — hit /v1/models which validates endpoint + API key.
  // (This endpoint returns Azure's model catalog, NOT the user's deployments.
  // Deployment names live behind /openai/deployments/... and don't appear here.)
  if (endpoint && apiKey && /^https?:\/\//i.test(endpoint)) {
    const url = `${endpoint}${OPENAI_V1_PATH}/models`;
    try {
      const ctrl = new AbortController();
      const timer = setTimeout(() => ctrl.abort(), 8000);
      const res = await fetch(url, {
        method: "GET",
        headers: { "api-key": apiKey, accept: "application/json" },
        signal: ctrl.signal,
      });
      clearTimeout(timer);

      if (res.ok) {
        const body = (await res.json()) as { data?: Array<{ id: string }> };
        const catalogSize = (body.data ?? []).length;
        checks.push({
          code: "endpoint_reachable",
          level: "info",
          message: `Foundry reachable — ${catalogSize} model${catalogSize === 1 ? "" : "s"} in catalog.`,
        });
      } else if (res.status === 401 || res.status === 403) {
        checks.push({
          code: "auth_failed",
          level: "error",
          message: `Authentication failed — HTTP ${res.status} from /openai/v1/models`,
          hint: "Verify the API key matches the resource (Azure portal → Keys and Endpoint).",
        });
      } else if (res.status === 404) {
        checks.push({
          code: "endpoint_not_found",
          level: "error",
          message: `Endpoint returned 404 for /openai/v1/models`,
          hint: "Confirm the endpoint URL points at a Foundry/AI-services resource.",
        });
      } else {
        const txt = await res.text().catch(() => "");
        checks.push({
          code: "http_error",
          level: "error",
          message: `Foundry returned HTTP ${res.status}`,
          detail: txt.slice(0, 200) || null,
        });
      }
    } catch (err) {
      const message = (err as Error).message;
      checks.push({
        code: "network_error",
        level: "error",
        message: `Network error reaching Foundry`,
        detail: message,
        hint: "Check that the endpoint hostname resolves and the server can reach Azure (firewall, VPN, DNS).",
      });
    }
  }

  // Real deployment validation — fire a small ping against the actual API
  // surface this deployment uses (chat completions or Responses API).
  // Skipped when prerequisites are missing or already failed above.
  const hadFatal = checks.some((c) => c.level === "error");
  const apiSurface = deployment
    ? getApiSurface(deployment, asString(config.apiSurface))
    : "chat";

  // Surface a hint when we can't find the deployment in the known table —
  // user might have typed a new one and forgot to set apiSurface.
  if (deployment && !KNOWN_API_SURFACE[deployment] && !asString(config.apiSurface)) {
    checks.push({
      code: "deployment_unknown_to_table",
      level: "info",
      message: `Deployment "${deployment}" isn't in the known capability table; defaulting to "chat" path.`,
      hint: "If this deployment only supports the Responses API, set adapterConfig.apiSurface to \"responses\".",
    });
  }

  if (deployment && endpoint && apiKey && !hadFatal) {
    const pingUrl =
      apiSurface === "responses"
        ? `${endpoint}${OPENAI_V1_PATH}/responses`
        : `${endpoint}${OPENAI_V1_PATH}/chat/completions`;
    const pingBody =
      apiSurface === "responses"
        ? {
            model: deployment,
            input: [{ type: "message", role: "user", content: [{ type: "input_text", text: "ping" }] }],
            max_output_tokens: 64,
            store: false,
            stream: false,
          }
        : {
            model: deployment,
            messages: [{ role: "user", content: "ping" }],
            max_completion_tokens: 64,
            stream: false,
          };
    try {
      const ctrl = new AbortController();
      const timer = setTimeout(() => ctrl.abort(), 15_000);
      const res = await fetch(pingUrl, {
        method: "POST",
        headers: {
          "api-key": apiKey,
          "content-type": "application/json",
          accept: "application/json",
        },
        body: JSON.stringify(pingBody),
        signal: ctrl.signal,
      });
      clearTimeout(timer);

      if (res.ok) {
        checks.push({
          code: "deployment_ok",
          level: "info",
          message: `Deployment "${deployment}" responds (${apiSurface} path).`,
        });
      } else {
        const txt = await res.text().catch(() => "");
        // A 400 with a max_tokens-related message means the deployment IS
        // reachable and responded — it just couldn't finish in 64 tokens.
        // That's a passing signal for our purposes.
        const isMaxTokensHint =
          res.status === 400 &&
          (txt.includes("max_tokens") ||
            txt.includes("max_output_tokens") ||
            txt.includes("model output limit"));
        // Common "wrong API surface" signature: chat completions returning
        // "operation is unsupported" for a responses-only model. Surface it
        // distinctly so users know to flip apiSurface or pick a different
        // deployment.
        const isWrongSurface =
          res.status === 400 &&
          txt.includes("operation is unsupported");
        const isUnknownModel =
          !isMaxTokensHint &&
          !isWrongSurface &&
          (txt.includes("DeploymentNotFound") ||
            txt.includes("does not exist") ||
            txt.includes("model_not_found") ||
            res.status === 404);
        if (isMaxTokensHint) {
          checks.push({
            code: "deployment_ok",
            level: "info",
            message: `Deployment "${deployment}" responds (${apiSurface} path — reasoning model token cap hit on ping, but reachable).`,
          });
        } else if (isWrongSurface) {
          const otherSurface = apiSurface === "chat" ? "responses" : "chat";
          checks.push({
            code: "deployment_wrong_surface",
            level: "error",
            message: `Deployment "${deployment}" rejected the ${apiSurface} API.`,
            detail: txt.slice(0, 200) || null,
            hint: `This deployment likely belongs on the ${otherSurface} path. Either pick a deployment that supports ${apiSurface}, or set adapterConfig.apiSurface to "${otherSurface}".`,
          });
        } else if (isUnknownModel) {
          checks.push({
            code: "deployment_not_found",
            level: "error",
            message: `Deployment "${deployment}" not found on this Foundry resource.`,
            detail: txt.slice(0, 200) || null,
            hint: "Check the spelling against the Azure portal → Deployments page.",
          });
        } else if (res.status === 429) {
          checks.push({
            code: "deployment_rate_limited",
            level: "warn",
            message: `Deployment "${deployment}" is rate-limited (HTTP 429), but it exists.`,
          });
        } else {
          checks.push({
            code: "deployment_ping_failed",
            level: "warn",
            message: `Deployment "${deployment}" returned HTTP ${res.status} on a ${apiSurface} ping.`,
            detail: txt.slice(0, 200) || null,
          });
        }
      }
    } catch (err) {
      checks.push({
        code: "deployment_ping_failed",
        level: "warn",
        message: `Deployment ping threw an exception.`,
        detail: (err as Error).message,
      });
    }
  } else if (!deployment) {
    checks.push({
      code: "deployment_missing",
      level: "warn",
      message: "Deployment not set — runs will fail without one.",
      hint: "Pick a deployment from the dropdown.",
    });
  }

  // Roll up overall status.
  let status: "pass" | "warn" | "fail" = "pass";
  for (const c of checks) {
    if (c.level === "error") {
      status = "fail";
      break;
    }
    if (c.level === "warn") status = "warn";
  }

  return {
    adapterType: ctx.adapterType,
    status,
    checks,
    testedAt: new Date().toISOString(),
  };
}
