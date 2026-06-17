/**
 * Live deployment discovery.
 *
 * Paperclip's model-discovery hooks (`listModels`, `refreshModels`,
 * `getConfigSchema`) are invoked WITHOUT a per-agent config object, so the
 * only credentials available here are the server-level environment variables
 * the Home Assistant add-on exports:
 *
 *   AZURE_FOUNDRY_ENDPOINT
 *   AZURE_FOUNDRY_API_KEY
 *
 * Azure's OpenAI-compatible `/openai/v1/models` route returns the full model
 * *catalog* (hundreds of entries), NOT the deployments the user actually
 * created. The real deployment names are served by the data-plane management
 * route `GET {endpoint}/openai/deployments?api-version=...`, which is what we
 * query here. Results are cached briefly and every failure falls back to the
 * caller's static list, so discovery never blocks agent creation.
 */
import { DEFAULT_API_VERSION } from "../shared/constants.js";

export interface DeploymentModel {
  id: string;
  label: string;
}

/**
 * Candidate api-versions for the data-plane deployments list, tried in order
 * until one returns data. Newer Azure OpenAI resources answer the GA
 * versions; `services.ai.azure.com` (Foundry) resources currently only serve
 * the deployments list under the older `2023-03-15-preview` version, so it is
 * kept as the final fallback.
 */
const DEPLOYMENT_API_VERSIONS = [
  DEFAULT_API_VERSION,
  "2023-05-15",
  "2023-03-15-preview",
];

const CACHE_TTL_MS = 5 * 60_000;
const REQUEST_TIMEOUT_MS = 8000;

interface CacheEntry {
  at: number;
  key: string;
  models: DeploymentModel[];
}

let cache: CacheEntry | null = null;

function resolveCreds(): { endpoint: string; apiKey: string } {
  const endpoint = (process.env.AZURE_FOUNDRY_ENDPOINT || "")
    .trim()
    .replace(/\/+$/, "");
  const apiKey = (process.env.AZURE_FOUNDRY_API_KEY || "").trim();
  return { endpoint, apiKey };
}

interface RawDeployment {
  id?: string;
  model?: string;
  status?: string;
  properties?: { model?: { name?: string }; provisioningState?: string };
}

function normalize(items: RawDeployment[]): DeploymentModel[] {
  const out: DeploymentModel[] = [];
  const seen = new Set<string>();
  for (const d of items) {
    const id = (d.id ?? "").trim();
    if (!id || seen.has(id)) continue;
    seen.add(id);
    const model = (d.model ?? d.properties?.model?.name ?? "").trim();
    out.push({ id, label: model && model !== id ? `${id} (${model})` : id });
  }
  out.sort((a, b) => a.id.localeCompare(b.id));
  return out;
}

async function fetchOnce(
  endpoint: string,
  apiKey: string,
  apiVersion: string,
): Promise<DeploymentModel[]> {
  const url = `${endpoint}/openai/deployments?api-version=${encodeURIComponent(apiVersion)}`;
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), REQUEST_TIMEOUT_MS);
  try {
    const res = await fetch(url, {
      method: "GET",
      headers: { "api-key": apiKey, accept: "application/json" },
      signal: ctrl.signal,
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const body = (await res.json()) as { data?: RawDeployment[]; value?: RawDeployment[] };
    return normalize(body.data ?? body.value ?? []);
  } finally {
    clearTimeout(timer);
  }
}

/**
 * Return the live deployment list for the configured Foundry resource.
 *
 * - Returns `[]` when credentials are absent (caller should fall back to its
 *   static suggestion list).
 * - Caches successful lookups for {@link CACHE_TTL_MS}; pass `{ force: true }`
 *   (the Refresh button in the UI) to bypass the cache.
 * - Never throws: any network/HTTP failure yields the last cached value or
 *   an empty list.
 */
export async function listDeployments(
  opts: { force?: boolean } = {},
): Promise<DeploymentModel[]> {
  const { endpoint, apiKey } = resolveCreds();
  if (!endpoint || !apiKey) return [];

  const cacheKey = `${endpoint}::${apiKey.length}`;
  if (
    !opts.force &&
    cache &&
    cache.key === cacheKey &&
    Date.now() - cache.at < CACHE_TTL_MS
  ) {
    return cache.models;
  }

  for (const apiVersion of DEPLOYMENT_API_VERSIONS) {
    try {
      const models = await fetchOnce(endpoint, apiKey, apiVersion);
      if (models.length > 0) {
        cache = { at: Date.now(), key: cacheKey, models };
        return models;
      }
    } catch {
      // Try the next api-version.
    }
  }

  // Nothing worked: serve the previous cache if we have one, else empty.
  if (cache && cache.key === cacheKey) return cache.models;
  return [];
}
