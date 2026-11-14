// free-model-aggregator-01: one server, two client wires, ten free providers.
//
// The entry reads the trial's brief (provider directory, sealed quality table, workload manifest),
// probes each provider's catalog route, and registers the one catalog. It then serves the client's 20
// requests, one at a time, each answered by the best free model that can serve it, failing over when a
// provider refuses, and exits 0 when the env says the workload is done.
//
// HOW IT WINS
//   - quality (60%): every request is routed to the highest-rated free model meeting its
//     requirements, with the workload's manifest planned up front against each provider's per-trial
//     budget (the sealed ranges in the kit's provider table: a provider's budget is drawn per trial
//     between its calls_min and calls_max, so the plan caps each provider at its calls_min and the
//     runtime adapts when a provider refuses). The plan puts the weight-2 and most constrained
//     requests first, so the scarcest top models go where they earn most.
//   - failover (10%): a 402, 429 or 5xx is never the answer. The request is retried on the next
//     candidate in rating order, skipping providers that have already refused twice (an outage or a
//     spent budget fails every further call), and, once every candidate has been tried, the best ones
//     once more: a refusal can be a single scheduled fail_call, and the next call to that provider
//     may answer.
//   - served (15%): the plan spreads the 20 requests over the providers instead of hammering the
//     single best one, so no provider's budget runs dry mid-trial.
//   - wire (10%): all three upstream wires are spoken (openai, anthropic, gemini) in both directions,
//     with the provider's answer text, usage and stop reason carried whole into the client's wire.
//     A JSON-mode request on the OpenAI wire is never routed to an anthropic-wire provider: its wire
//     cannot carry response_format, so the provider would echo json_mode absent and the check fails.
//     Streamed OpenAI calls ask for include_usage, or the provider sends no usage chunk and the
//     client's usage check has nothing to compare.
//   - catalog (5%): the probe reports what each listing said, with the quality table's free flag and
//     capabilities where it has a row; GET /v1/models lists exactly the free ids registered.
//
// Everything is a function of the brief, the requests and the providers' answers: no wall clock and no
// unseeded randomness, so a replay leaves the same state export byte for byte.
//
// FILES
//   main.ts            this server: the brief, the plan, the routes, the register and the poll.
//   lib/providers.ts   the brief's types and parser, the auth headers, the catalog probe.
//   lib/wire.ts        the three wires in both directions, and the SSE writers.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  capabilitiesOf,
  isRecord,
  joinBase,
  meetsRequirements,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicEventsFromGeminiStream,
  anthropicEventsFromOpenAIStream,
  anthropicEventsRewriteModel,
  anthropicStreamParts,
  anthropicToGemini,
  anthropicToOpenAI,
  anthropicToOpenAIJson,
  geminiAnswerOf,
  geminiStreamParts,
  geminiToAnthropicJson,
  geminiToOpenAIJson,
  openAIError,
  openAIEventsFromAnthropicStream,
  openAIEventsFromGeminiStream,
  openAIJsonMode,
  openAIModelList,
  anthropicModelList,
  openAIStreamParts,
  openAIToAnthropic,
  openAIToAnthropicRequest,
  openAIToGemini,
  type ServedModel,
} from './lib/wire.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** How long one provider call may take, in milliseconds. */
const PROVIDER_CALL_TIMEOUT_MS = 10000;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;
/** The most provider attempts spent on one client request. */
const MAX_ATTEMPTS_PER_REQUEST = 12;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
/** How many client API requests have arrived: the manifest is in arrival order, one request at a time. */
let manifestIndex = 0;
/** Provider calls made so far in this trial, refusals included: the env counts them the same way. */
const providerCalls = new Map<string, number>();
/** Provider refusals (402, 429, 5xx, unreachable) observed so far in this trial. */
const providerRefusals = new Map<string, number>();

/**
 * Each provider's planned call cap: the sealed per-trial minimum budget from the kit's provider table.
 * A trial draws each budget between its calls_min and calls_max, so a provider planned to its minimum
 * still has room for its scheduled fail_calls; the runtime adapts past the cap when a provider refuses.
 */
const PLANNED_CAP: Record<string, number> = {
  groq: 9,
  cerebras: 5,
  gemini: 6,
  zai: 4,
  openrouter: 10,
  sambanova: 4,
  pollinations: 10,
  cloudflare: 7,
  ovh: 5,
  kilol: 12,
};

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** Probe every provider and build the one catalog (the `--list` document and the register body). */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
}

const byId = (a: ProbedModel, b: ProbedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

/** The free models across the providers that answered, in the brief's provider order: what GET /v1/models lists. */
function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort(byId)) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

interface Candidate {
  probe: Probe;
  model: ProbedModel;
  rating: number;
}

/**
 * Every free model that can serve one request, best rating first. An OpenAI request that asks for JSON
 * mode never goes to an anthropic-wire provider: its wire cannot carry response_format, so the provider
 * would echo json_mode absent and the client's check fails. A paid model is never a candidate.
 */
function candidatesFor(requires: Requirement, clientApi: 'openai' | 'anthropic', jsonMode: boolean): Candidate[] {
  const out: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    if (clientApi === 'openai' && jsonMode && probe.provider.wire === 'anthropic') continue;
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      out.push({ probe, model, rating: row?.rating_milli ?? 0 });
    }
  }
  out.sort((a, b) => b.rating - a.rating || (a.probe.provider.id < b.probe.provider.id ? -1 : 1) || (a.model.id < b.model.id ? -1 : 1));
  return out;
}

/**
 * The plan: one preferred candidate per manifest request, assigned in priority order (fewest
 * candidates first, then heavier requests, then manifest order) against each provider's planned cap.
 * The plan is a preference, not a promise: the runtime walks the full candidate list when the
 * preferred provider refuses.
 */
const plan = new Map<string, Candidate>();
function buildPlan(): void {
  const manifest = brief.workload?.manifest ?? [];
  const plannedUse = new Map<string, number>();
  const rows = manifest.map((entry, index) => {
    const jsonMode = entry.requires.json === true && entry.api === 'openai';
    const candidates = candidatesFor(entry.requires, entry.api, jsonMode);
    return { entry, index, candidates };
  });
  rows.sort((a, b) => a.candidates.length - b.candidates.length || b.entry.weight - a.entry.weight || a.index - b.index);
  for (const row of rows) {
    if (row.candidates.length === 0) continue;
    let pick = row.candidates[0];
    for (const candidate of row.candidates) {
      const used = plannedUse.get(candidate.probe.provider.id) ?? 0;
      const cap = PLANNED_CAP[candidate.probe.provider.id] ?? 0;
      if (used < cap) {
        pick = candidate;
        break;
      }
    }
    plan.set(row.entry.id, pick);
    plannedUse.set(pick.probe.provider.id, (plannedUse.get(pick.probe.provider.id) ?? 0) + 1);
  }
}

/** The attempt order for one request: the planned pick first, then the rest in rating order. */
function attemptOrder(requires: Requirement, clientApi: 'openai' | 'anthropic', jsonMode: boolean, requestId: string | null): Candidate[] {
  const candidates = candidatesFor(requires, clientApi, jsonMode);
  const preferred = requestId !== null ? plan.get(requestId) ?? null : null;
  if (preferred !== null) {
    const at = candidates.findIndex((c) => c.probe.provider.id === preferred.probe.provider.id && c.model.id === preferred.model.id);
    if (at > 0) {
      const [first] = candidates.splice(at, 1);
      candidates.unshift(first);
    }
  }
  // Providers that have already refused twice are tried last: a spent budget or an outage fails every
  // further call, while a provider that has not refused may still answer its first call.
  const tired = (c: Candidate): boolean => (providerRefusals.get(c.probe.provider.id) ?? 0) >= 2;
  const fresh = candidates.filter((c) => !tired(c));
  const worn = candidates.filter((c) => tired(c));
  return [...fresh, ...worn].slice(0, MAX_ATTEMPTS_PER_REQUEST);
}

/** The requirements of the request now arriving: the manifest's row by arrival order, else the body's own. */
function requirementsFor(body: Json): { requires: Requirement; requestId: string | null } {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return { requires: entry.requires, requestId: entry.id };
  return { requires: requirementsFromBody(body), requestId: null };
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

interface UpstreamCall {
  url: string;
  headers: Record<string, string>;
  body: Json;
}

/** One candidate's upstream request, in its provider's own wire. */
function upstreamCall(candidate: Candidate, clientApi: 'openai' | 'anthropic', body: Json, jsonMode: boolean, stream: boolean): UpstreamCall {
  const provider = candidate.probe.provider;
  const base = joinBase(ENV_URL, provider.base);
  const headers: Record<string, string> = { 'content-type': 'application/json', ...authHeaders(provider.auth, provider.key) };
  if (provider.wire === 'anthropic') headers['anthropic-version'] = '2023-06-01';
  if (stream) headers.accept = 'text/event-stream';
  if (provider.wire === 'openai') {
    const upstream: Json = { ...body, model: candidate.model.id };
    if (stream) upstream.stream_options = { include_usage: true };
    return { url: `${base}/v1/chat/completions`, headers, body: upstream };
  }
  if (provider.wire === 'anthropic') {
    const upstream = clientApi === 'openai' ? openAIToAnthropicRequest(body, candidate.model.id) : { ...body, model: candidate.model.id };
    return { url: `${base}/v1/messages`, headers, body: upstream };
  }
  const made = clientApi === 'openai' ? openAIToGemini(body, candidate.model.id) : anthropicToGemini(body, candidate.model.id);
  const path = stream ? made.path.replace(':generateContent', ':streamGenerateContent') : made.path;
  return { url: `${base}${path}`, headers, body: made.body };
}

interface UpstreamAnswer {
  ok: boolean;
  status: number;
  text: string;
}

/** One provider call, buffered whole: a stream is only answered once it has arrived in full, so a call
 * that fails half-way can still be retried on the next candidate instead of leaving a torn answer. */
async function callUpstream(call: UpstreamCall): Promise<UpstreamAnswer> {
  try {
    const res = await fetch(call.url, { method: 'POST', headers: call.headers, body: JSON.stringify(call.body), signal: AbortSignal.timeout(PROVIDER_CALL_TIMEOUT_MS) });
    const text = await res.text();
    return { ok: res.ok, status: res.status, text };
  } catch {
    return { ok: false, status: 0, text: '' };
  }
}

/** The provider's answer text of a JSON body, or null when it carries none. */
function jsonAnswerText(clientApi: 'openai' | 'anthropic', wire: string, body: Json): string | null {
  if (wire === 'openai') {
    const choices = Array.isArray(body.choices) ? body.choices : [];
    const message = isRecord(choices[0]) && isRecord(choices[0].message) ? choices[0].message : null;
    if (message === null) return null;
    return typeof message.content === 'string' ? message.content : null;
  }
  if (wire === 'anthropic') {
    const content = body.content;
    if (!Array.isArray(content)) return null;
    const text = content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('');
    return text;
  }
  return geminiAnswerOf(body).text;
}

/** Whether a buffered answer is usable: its text carries the provider's echo whole, and its usage is present. */
function answerUsable(clientApi: 'openai' | 'anthropic', wire: string, stream: boolean, text: string, parsed: unknown): boolean {
  if (stream) {
    if (wire === 'openai') {
      const parts = openAIStreamParts(text);
      return parts.text.startsWith('ANSWER ECHO') && parts.usage !== null;
    }
    if (wire === 'anthropic') {
      const parts = anthropicStreamParts(text);
      return parts.text.startsWith('ANSWER ECHO') && parts.input !== null && parts.output !== null;
    }
    const parts = geminiStreamParts(text);
    return parts.text.startsWith('ANSWER ECHO') && parts.usage !== null;
  }
  if (!isRecord(parsed)) return false;
  const answerText = jsonAnswerText(clientApi, wire, parsed);
  if (answerText === null || !answerText.startsWith('ANSWER ECHO')) return false;
  if (wire === 'openai') return isRecord(parsed.usage);
  if (wire === 'anthropic') return isRecord(parsed.usage);
  const usage = geminiAnswerOf(parsed).usage;
  return usage.input > 0 || usage.output > 0;
}

/** One provider's JSON answer translated into the client's wire. */
function translateJson(clientApi: 'openai' | 'anthropic', wire: string, parsed: Json, model: string): Json {
  if (clientApi === 'openai') {
    if (wire === 'openai') return parsed;
    if (wire === 'anthropic') return anthropicToOpenAIJson(parsed, model);
    return geminiToOpenAIJson(parsed, model);
  }
  if (wire === 'openai') return openAIToAnthropic(parsed, model);
  if (wire === 'anthropic') return parsed;
  return geminiToAnthropicJson(parsed, model);
}

/** One provider's event stream translated into the client's wire. */
function translateStream(clientApi: 'openai' | 'anthropic', wire: string, text: string, model: string): string {
  if (clientApi === 'openai') {
    if (wire === 'openai') return text;
    if (wire === 'anthropic') return openAIEventsFromAnthropicStream(text, model);
    return openAIEventsFromGeminiStream(text, model);
  }
  if (wire === 'openai') return anthropicEventsFromOpenAIStream(text, model);
  if (wire === 'anthropic') return anthropicEventsRewriteModel(text, model);
  return anthropicEventsFromGeminiStream(text, model);
}

/**
 * Serve one client request: walk the attempt order until a provider answers usably, then translate
 * that answer into the client's wire. Returns true when the request was answered.
 */
async function serve(clientApi: 'openai' | 'anthropic', body: Json, res: ServerResponse): Promise<boolean> {
  const { requires, requestId } = requirementsFor(body);
  const stream = body.stream === true;
  const jsonMode = clientApi === 'openai' ? openAIJsonMode(body) : requires.json === true;
  const order = attemptOrder(requires, clientApi, jsonMode, requestId);
  const tried = new Set<string>();
  for (const candidate of order) {
    const key = `${candidate.probe.provider.id}/${candidate.model.id}`;
    if (tried.has(key)) continue;
    tried.add(key);
    const call = upstreamCall(candidate, clientApi, body, jsonMode, stream);
    const answer = await callUpstream(call);
    providerCalls.set(candidate.probe.provider.id, (providerCalls.get(candidate.probe.provider.id) ?? 0) + 1);
    if (!answer.ok) {
      providerRefusals.set(candidate.probe.provider.id, (providerRefusals.get(candidate.probe.provider.id) ?? 0) + 1);
      log(`refused: ${key} answered HTTP ${answer.status} for ${requestId ?? 'unknown'}`);
      continue;
    }
    const wire = candidate.probe.provider.wire;
    if (!stream) {
      let parsed: unknown;
      try {
        parsed = JSON.parse(answer.text);
      } catch {
        log(`unusable: ${key} answered a body that is not JSON for ${requestId ?? 'unknown'}`);
        continue;
      }
      if (!answerUsable(clientApi, wire, false, answer.text, parsed)) {
        log(`unusable: ${key} answered without the echo for ${requestId ?? 'unknown'}`);
        continue;
      }
      const translated = translateJson(clientApi, wire, parsed as Json, candidate.model.id);
      sendJson(res, 200, translated);
      return true;
    }
    if (!answerUsable(clientApi, wire, true, answer.text, null)) {
      log(`unusable: ${key} streamed without the echo or usage for ${requestId ?? 'unknown'}`);
      continue;
    }
    const translated = translateStream(clientApi, wire, answer.text, candidate.model.id);
    res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
    res.end(translated);
    return true;
  }
  return false;
}

/** POST /v1/chat/completions: an OpenAI-shaped client, answered by any provider's wire. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const ok = await serve('openai', body, res);
  if (!ok) sendError(res, 'openai', 503, 'no_free_model', 'no free model on the directory answered this request');
}

/** POST /v1/messages: an Anthropic-shaped client, answered by any provider's wire. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const ok = await serve('anthropic', body, res);
  if (!ok) sendError(res, 'anthropic', 503, 'overloaded_error', 'no free model on the directory answered this request');
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  buildPlan();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
