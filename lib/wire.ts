// The client wires and the three upstream wires.
//
// The trial's client calls the entry as an OpenAI service (POST /v1/chat/completions) or an Anthropic
// one (POST /v1/messages), with a JSON body or with server-sent events. Each provider speaks one of the
// same three wires: openai, anthropic or gemini. This file translates every request into the provider's
// wire and every answer back into the client's wire, preserving exactly what the provider echoes:
//
//   - the system text, the user text and any image (by the sha8 of their bytes), the tool names in
//     order, max_tokens, the stop sequences in order and JSON mode. A field rewritten or dropped loses
//     the request, so every translation carries them all across.
//   - the answer's text whole, the provider's usage counts as that wire reports them, and a stop reason
//     in the same family (stop / end_turn / STOP are one family; the providers only ever answer that
//     way, so every answer maps there).
//
// Two wires cannot carry JSON mode: Anthropic's Messages API has no response_format, so an OpenAI
// request that asks for JSON mode must never be routed to an anthropic-wire provider (main.ts enforces
// that); Gemini carries it as generationConfig.responseMimeType "application/json".
//
// Everything here is a pure function of its arguments: no clock and no randomness, so the same body and
// the same provider answer give the same bytes on a replay. The two model listings use fixed timestamps
// for the same reason.

import { isRecord, type Json } from './providers.ts';

/** One model the entry serves, as GET /v1/models lists it. */
export interface ServedModel {
  id: string;
  provider: string;
  display_name: string;
}

/** Token counts in one wire's spelling, normalised to input/output. */
export interface Usage {
  input: number;
  output: number;
}

/** A fixed epoch second for every listed model's `created`: nothing in the reply reads the clock. */
const FIXED_OPENAI_CREATED = 1767225600;
/** A fixed instant for every listed model's `created_at`: nothing in the reply reads the clock. */
const FIXED_ANTHROPIC_CREATED_AT = '2026-01-01T00:00:00Z';
/** How wide the text pieces of a synthesised stream are, in characters. */
const STREAM_PIECE_CHARS = 13;

/** GET /v1/models on an OpenAI-shaped request: the models this entry serves, in OpenAI's list shape. */
export function openAIModelList(models: readonly ServedModel[]): Json {
  return {
    object: 'list',
    data: models.map((m) => ({ id: m.id, object: 'model', created: FIXED_OPENAI_CREATED, owned_by: m.provider })),
  };
}

/** GET /v1/models on an Anthropic-shaped request (an x-api-key header): Anthropic's model list shape. */
export function anthropicModelList(models: readonly ServedModel[]): Json {
  return {
    data: models.map((m) => ({ type: 'model', id: m.id, display_name: m.display_name, created_at: FIXED_ANTHROPIC_CREATED_AT })),
    has_more: false,
  };
}

/** An error in Anthropic's shape (the Messages API's error envelope). */
export function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

/** An error in OpenAI's shape (Chat Completions). */
export function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, code } };
}

/** The text of a string or of a block array's text blocks; '' when there is none. */
function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}

/** The text of a message's content as the provider's echo reads it: text blocks joined with ''. */
function answerText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('');
}

/** One Anthropic content block as an OpenAI content part: text or an inline image; null for the rest. */
function partOf(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') return { type: 'text', text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
      return { type: 'image_url', image_url: { url: `data:${media};base64,${source.data}` } };
    }
    if (source.type === 'url' && typeof source.url === 'string') return { type: 'image_url', image_url: { url: source.url } };
  }
  return null;
}

/** A tool_result block's content as text: its text blocks, or its JSON when it holds something else. */
function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = textOf(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/** A tool call's arguments as an object: the provider's JSON string, or {} when it is not one. */
function argumentsOf(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}

function numberOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

/** OpenAI's finish_reason as Anthropic's stop_reason. */
export function stopReasonOf(finish: unknown): string {
  if (finish === 'length') return 'max_tokens';
  if (finish === 'tool_calls' || finish === 'function_call') return 'tool_use';
  if (finish === 'content_filter') return 'refusal';
  return 'end_turn';
}

/** Anthropic's stop_reason as OpenAI's finish_reason. */
export function finishReasonOf(stop: unknown): string {
  if (stop === 'max_tokens') return 'length';
  if (stop === 'tool_use') return 'tool_calls';
  if (stop === 'refusal') return 'content_filter';
  return 'stop';
}

/** Whether an OpenAI body asks for JSON mode. */
export function openAIJsonMode(body: Json): boolean {
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return format === 'json_object' || format === 'json_schema';
}

/** Split text into pieces of at most STREAM_PIECE_CHARS characters (never empty input: one empty piece). */
function splitText(text: string): string[] {
  const pieces: string[] = [];
  for (let at = 0; at < text.length; at += STREAM_PIECE_CHARS) pieces.push(text.slice(at, at + STREAM_PIECE_CHARS));
  return pieces.length === 0 ? [''] : pieces;
}

/**
 * An Anthropic Messages request as an OpenAI Chat Completions request that names `model`: the system
 * prompt becomes a system message, image blocks become image_url data URLs, tool_use blocks become
 * tool_calls, tool_result blocks become tool messages, tools and tool_choice are rewritten, and the
 * output cap, temperature, top_p, stop sequences and streaming pass through. When `jsonMode` is true the
 * OpenAI body carries response_format json_object (the caller only sets it when the request asked for
 * JSON in a wire that can say so).
 */
export function anthropicToOpenAI(body: Json, model: string, jsonMode = false): Json {
  const messages: Json[] = [];
  const system = textOf(body.system);
  if (system !== '') messages.push({ role: 'system', content: system });
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: assistant ? 'assistant' : 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (assistant) {
      const toolCalls: Json[] = [];
      for (const block of content) {
        if (!isRecord(block) || block.type !== 'tool_use') continue;
        toolCalls.push({
          id: typeof block.id === 'string' ? block.id : '',
          type: 'function',
          function: {
            name: typeof block.name === 'string' ? block.name : '',
            arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
          },
        });
      }
      const text = textOf(content);
      messages.push({
        role: 'assistant',
        content: text !== '' ? text : null,
        ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
      });
      continue;
    }
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_result') {
        messages.push({
          role: 'tool',
          tool_call_id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
          content: toolResultText(block.content),
        });
        continue;
      }
      const part = partOf(block);
      if (part !== null) parts.push(part);
    }
    if (parts.length > 0) messages.push({ role: 'user', content: parts });
  }

  const out: Json = { model, messages };
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) out.stop = body.stop_sequences;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== 'string') return [];
      return [
        {
          type: 'function',
          function: {
            name: raw.name,
            description: typeof raw.description === 'string' ? raw.description : '',
            parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
          },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') out.tool_choice = 'auto';
    else if (choice.type === 'any') out.tool_choice = 'required';
    else if (choice.type === 'tool' && typeof choice.name === 'string') out.tool_choice = { type: 'function', function: { name: choice.name } };
  }
  if (jsonMode) out.response_format = { type: 'json_object' };
  if (body.stream === true) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

/** A data URL's media type and base64 data, or null when it is not one. */
function dataUrlOf(url: unknown): { media: string; data: string } | null {
  if (typeof url !== 'string') return null;
  const match = /^data:([^;,]*);base64,(.*)$/s.exec(url);
  if (match === null) return null;
  return { media: match[1] === '' ? 'image/png' : match[1], data: match[2] };
}

/**
 * An OpenAI Chat Completions request as an Anthropic Messages request that names `model`: the system
 * message becomes the system prompt, image_url parts become image blocks, tool_calls become tool_use
 * blocks, tool messages become tool_result blocks on the next user message, tools and tool_choice are
 * rewritten, and the output cap, temperature, top_p, stop sequences and streaming pass through. JSON
 * mode is dropped: Anthropic's wire cannot carry it, so the caller never routes a JSON-mode request
 * here.
 */
export function openAIToAnthropicRequest(body: Json, model: string): Json {
  const messages: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  let system = '';
  /** Tool results waiting for the next user message to carry them. */
  let pendingToolResults: Json[] = [];
  const flushToolResults = (into: Json[]): void => {
    if (pendingToolResults.length === 0) return;
    into.push(...pendingToolResults);
    pendingToolResults = [];
  };
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      const text = textOf(raw.content);
      system = system === '' ? text : `${system}\n${text}`;
      continue;
    }
    if (raw.role === 'tool') {
      pendingToolResults.push({
        type: 'tool_result',
        tool_use_id: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '',
        content: typeof raw.content === 'string' ? raw.content : answerText(raw.content),
      });
      continue;
    }
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (assistant) {
      const blocks: Json[] = [];
      const text = textOf(content);
      if (text !== '') blocks.push({ type: 'text', text });
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        blocks.push({
          type: 'tool_use',
          id: typeof call.id === 'string' ? call.id : '',
          name: typeof call.function.name === 'string' ? call.function.name : '',
          input: argumentsOf(call.function.arguments),
        });
      }
      if (blocks.length > 0) messages.push({ role: 'assistant', content: blocks });
      continue;
    }
    // A user message: text and image parts, with any pending tool results ahead of them.
    const blocks: Json[] = [];
    flushToolResults(blocks);
    if (typeof content === 'string') {
      if (content !== '') blocks.push({ type: 'text', text: content });
    } else if (Array.isArray(content)) {
      for (const part of content) {
        if (!isRecord(part)) continue;
        if (part.type === 'text' && typeof part.text === 'string') blocks.push({ type: 'text', text: part.text });
        else if (part.type === 'image_url' && isRecord(part.image_url)) {
          const data = dataUrlOf(part.image_url.url);
          if (data !== null) blocks.push({ type: 'image', source: { type: 'base64', media_type: data.media, data: data.data } });
          else if (typeof part.image_url.url === 'string') blocks.push({ type: 'image', source: { type: 'url', url: part.image_url.url } });
        }
      }
    }
    if (blocks.length > 0) messages.push({ role: 'user', content: blocks });
  }
  if (pendingToolResults.length > 0) messages.push({ role: 'user', content: pendingToolResults });

  const out: Json = { model, messages };
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  else if (typeof body.max_completion_tokens === 'number') out.max_tokens = body.max_completion_tokens;
  if (system !== '') out.system = system;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop) && body.stop.length > 0) out.stop_sequences = body.stop.filter((s) => typeof s === 'string');
  else if (typeof body.stop === 'string' && body.stop !== '') out.stop_sequences = [body.stop];
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
      return [{ name: raw.function.name, description: typeof raw.function.description === 'string' ? raw.function.description : '', input_schema: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} } }];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (choice === 'auto') out.tool_choice = { type: 'auto' };
  else if (choice === 'required') out.tool_choice = { type: 'any' };
  else if (isRecord(choice) && choice.type === 'function' && isRecord(choice.function) && typeof choice.function.name === 'string') {
    out.tool_choice = { type: 'tool', name: choice.function.name };
  }
  if (body.stream === true) out.stream = true;
  return out;
}

/** One OpenAI content part as a Gemini part: text, or an inline image; null for the rest. */
function geminiPartOf(part: Json): Json | null {
  if (part.type === 'text' && typeof part.text === 'string') return { text: part.text };
  if (part.type === 'image_url' && isRecord(part.image_url)) {
    const data = dataUrlOf(part.image_url.url);
    if (data !== null) return { inlineData: { mimeType: data.media, data: data.data } };
    if (typeof part.image_url.url === 'string') return { fileData: { mimeType: 'image/png', fileUri: part.image_url.url } };
  }
  return null;
}

/** The generation config of a Gemini request from an OpenAI or Anthropic body. */
function geminiConfig(body: Json, jsonMode: boolean): Json {
  const config: Json = {};
  const maxTokens = typeof body.max_tokens === 'number' ? body.max_tokens : typeof body.max_completion_tokens === 'number' ? body.max_completion_tokens : null;
  if (maxTokens !== null) config.maxOutputTokens = maxTokens;
  const stops = Array.isArray(body.stop) ? body.stop.filter((s) => typeof s === 'string') : Array.isArray(body.stop_sequences) ? body.stop_sequences.filter((s) => typeof s === 'string') : [];
  if (stops.length > 0) config.stopSequences = stops;
  if (typeof body.temperature === 'number') config.temperature = body.temperature;
  if (typeof body.top_p === 'number') config.topP = body.top_p;
  if (jsonMode) config.responseMimeType = 'application/json';
  return config;
}

/** One OpenAI message as Gemini contents, with tool messages as function responses. */
function geminiContentsOfOpenAI(body: Json): Json[] {
  const contents: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw) || raw.role === 'system') continue;
    if (raw.role === 'tool') {
      contents.push({ role: 'user', parts: [{ functionResponse: { name: '', response: { content: typeof raw.content === 'string' ? raw.content : answerText(raw.content) } } }] });
      continue;
    }
    const assistant = raw.role === 'assistant';
    const parts: Json[] = [];
    const content = raw.content;
    if (typeof content === 'string') {
      if (content !== '') parts.push({ text: content });
    } else if (Array.isArray(content)) {
      for (const part of content) {
        if (!isRecord(part)) continue;
        if (part.type === 'text' && typeof part.text === 'string') parts.push({ text: part.text });
        else {
          const geminiPart = geminiPartOf(part);
          if (geminiPart !== null) parts.push(geminiPart);
        }
      }
    }
    const calls = assistant && Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
    for (const call of calls) {
      if (!isRecord(call) || !isRecord(call.function)) continue;
      parts.push({ functionCall: { name: typeof call.function.name === 'string' ? call.function.name : '', args: argumentsOf(call.function.arguments) } });
    }
    if (parts.length > 0) contents.push({ role: assistant ? 'model' : 'user', parts });
  }
  return contents;
}

/**
 * An OpenAI Chat Completions request as a Gemini generateContent request: the system message becomes
 * the system instruction, user and assistant messages become contents, image parts become inline data,
 * tools become function declarations, and the output cap, stop sequences and JSON mode become the
 * generation config.
 */
export function openAIToGemini(body: Json, model: string): { path: string; body: Json } {
  const systemParts: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw) || raw.role !== 'system') continue;
    const text = textOf(raw.content);
    if (text !== '') systemParts.push({ text });
  }
  const out: Json = { contents: geminiContentsOfOpenAI(body) };
  if (systemParts.length > 0) out.systemInstruction = { parts: systemParts };
  const tools = Array.isArray(body.tools)
    ? body.tools.flatMap((raw) => {
        if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
        return [{ name: raw.function.name, description: typeof raw.function.description === 'string' ? raw.function.description : '', parameters: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} } }];
      })
    : [];
  if (tools.length > 0) out.tools = [{ functionDeclarations: tools }];
  const config = geminiConfig(body, openAIJsonMode(body));
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return { path: `/v1beta/models/${encodeURIComponent(model)}:generateContent`, body: out };
}

/**
 * An Anthropic Messages request as a Gemini generateContent request: the system prompt becomes the
 * system instruction, messages become contents (tool_use as function calls, tool_result as function
 * responses), tools become function declarations, and the output cap and stop sequences become the
 * generation config. Anthropic carries no JSON mode, so none is set.
 */
export function anthropicToGemini(body: Json, model: string): { path: string; body: Json } {
  const system = textOf(body.system);
  const contents: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      if (content !== '') contents.push({ role: assistant ? 'model' : 'user', parts: [{ text: content }] });
      continue;
    }
    if (!Array.isArray(content)) continue;
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'text' && typeof block.text === 'string') parts.push({ text: block.text });
      else if (block.type === 'image' && isRecord(block.source)) {
        if (block.source.type === 'base64' && typeof block.source.data === 'string') {
          parts.push({ inlineData: { mimeType: typeof block.source.media_type === 'string' ? block.source.media_type : 'image/png', data: block.source.data } });
        } else if (block.source.type === 'url' && typeof block.source.url === 'string') {
          parts.push({ fileData: { mimeType: 'image/png', fileUri: block.source.url } });
        }
      } else if (block.type === 'tool_use') {
        parts.push({ functionCall: { name: typeof block.name === 'string' ? block.name : '', args: isRecord(block.input) ? block.input : {} } });
      } else if (block.type === 'tool_result') {
        parts.push({ functionResponse: { name: typeof block.tool_use_id === 'string' ? block.tool_use_id : '', response: { content: toolResultText(block.content) } } });
      }
    }
    if (parts.length > 0) contents.push({ role: assistant ? 'model' : 'user', parts });
  }
  const out: Json = { contents };
  if (system !== '') out.systemInstruction = { parts: [{ text: system }] };
  const tools = Array.isArray(body.tools)
    ? body.tools.flatMap((raw) => {
        if (!isRecord(raw) || typeof raw.name !== 'string') return [];
        return [{ name: raw.name, description: typeof raw.description === 'string' ? raw.description : '', parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} } }];
      })
    : [];
  if (tools.length > 0) out.tools = [{ functionDeclarations: tools }];
  const config = geminiConfig(body, false);
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return { path: `/v1beta/models/${encodeURIComponent(model)}:generateContent`, body: out };
}

/** A Chat Completions answer as an Anthropic Messages object. */
export function openAIToAnthropic(body: Json, model: string): Json {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const content: Json[] = [];
  const text = typeof message.content === 'string' ? message.content : answerText(message.content);
  if (text !== '') content.push({ type: 'text', text });
  const calls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
  for (const call of calls) {
    if (!isRecord(call) || !isRecord(call.function)) continue;
    content.push({
      type: 'tool_use',
      id: typeof call.id === 'string' ? call.id : 'toolu_unknown',
      name: typeof call.function.name === 'string' ? call.function.name : '',
      input: argumentsOf(call.function.arguments),
    });
  }
  if (content.length === 0) content.push({ type: 'text', text: '' });
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    id: `msg_${typeof body.id === 'string' ? body.id : 'aggregate'}`,
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: stopReasonOf(choice.finish_reason),
    stop_sequence: null,
    usage: { input_tokens: numberOrZero(usage.prompt_tokens), output_tokens: numberOrZero(usage.completion_tokens) },
  };
}

/** An Anthropic Messages answer as a Chat Completions object. */
export function anthropicToOpenAIJson(body: Json, model: string): Json {
  const content = body.content;
  const text = typeof content === 'string' ? content : answerText(content);
  const usage = isRecord(body.usage) ? body.usage : {};
  const input = numberOrZero(usage.input_tokens);
  const output = numberOrZero(usage.output_tokens);
  return {
    id: `chatcmpl_${typeof body.id === 'string' ? body.id : 'aggregate'}`,
    object: 'chat.completion',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [{ index: 0, message: { role: 'assistant', content: text }, finish_reason: finishReasonOf(body.stop_reason), logprobs: null }],
    usage: { prompt_tokens: input, completion_tokens: output, total_tokens: input + output },
  };
}

/** The text of a Gemini answer's candidates, and its usage. */
export function geminiAnswerOf(body: Json): { text: string; usage: Usage; stop: string } {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const candidate = isRecord(candidates[0]) ? candidates[0] : {};
  const content = isRecord(candidate.content) ? candidate.content : {};
  const parts = Array.isArray(content.parts) ? content.parts : [];
  const text = parts.flatMap((p) => (isRecord(p) && typeof p.text === 'string' ? [p.text] : [])).join('');
  const meta = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  return {
    text,
    usage: { input: numberOrZero(meta.promptTokenCount), output: numberOrZero(meta.candidatesTokenCount) },
    stop: typeof candidate.finishReason === 'string' ? candidate.finishReason : 'STOP',
  };
}

/** A Gemini answer as a Chat Completions object. */
export function geminiToOpenAIJson(body: Json, model: string): Json {
  const answer = geminiAnswerOf(body);
  return {
    id: 'chatcmpl_gemini',
    object: 'chat.completion',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [{ index: 0, message: { role: 'assistant', content: answer.text }, finish_reason: finishReasonOf(answer.stop), logprobs: null }],
    usage: { prompt_tokens: answer.usage.input, completion_tokens: answer.usage.output, total_tokens: answer.usage.input + answer.usage.output },
  };
}

/** A Gemini answer as an Anthropic Messages object. */
export function geminiToAnthropicJson(body: Json, model: string): Json {
  const answer = geminiAnswerOf(body);
  return {
    id: 'msg_gemini',
    type: 'message',
    role: 'assistant',
    model,
    content: [{ type: 'text', text: answer.text }],
    stop_reason: stopReasonOf(answer.stop),
    stop_sequence: null,
    usage: { input_tokens: answer.usage.input, output_tokens: answer.usage.output },
  };
}

/** One server-sent event: `event: <name>` when named, then the JSON data line, then the blank line. */
export function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/** One parsed SSE frame: its event name (null when unnamed) and its data lines joined. */
export interface SseFrame {
  event: string | null;
  data: string;
}

/** The SSE frames of a buffered event stream, in order. */
export function parseSseFrames(text: string): SseFrame[] {
  const frames: SseFrame[] = [];
  const normalized = text.replace(/\r\n/g, '\n');
  for (const frame of normalized.split('\n\n')) {
    if (frame.trim() === '') continue;
    let event: string | null = null;
    const data: string[] = [];
    for (const line of frame.split('\n')) {
      if (line.startsWith('event:')) event = line.slice('event:'.length).trim();
      else if (line.startsWith('data:')) data.push(line.slice('data:'.length).replace(/^ /, ''));
    }
    if (data.length > 0) frames.push({ event, data: data.join('\n') });
  }
  return frames;
}

/** The text, usage and stop reason of a buffered OpenAI event stream. */
export function openAIStreamParts(text: string): { text: string; usage: Usage | null; finish: string | null } {
  let out = '';
  let usage: Usage | null = null;
  let finish: string | null = null;
  for (const frame of parseSseFrames(text)) {
    const payload = frame.data.trim();
    if (payload === '[DONE]') continue;
    let chunk: unknown;
    try {
      chunk = JSON.parse(payload);
    } catch {
      continue;
    }
    if (!isRecord(chunk)) continue;
    if (isRecord(chunk.usage)) usage = { input: numberOrZero(chunk.usage.prompt_tokens), output: numberOrZero(chunk.usage.completion_tokens) };
    const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
    if (choices.length === 0) continue;
    const choice = isRecord(choices[0]) ? choices[0] : null;
    if (choice === null) continue;
    const delta = isRecord(choice.delta) ? choice.delta : {};
    if (typeof delta.content === 'string') out += delta.content;
    if (typeof choice.finish_reason === 'string') finish = choice.finish_reason;
  }
  return { text: out, usage, finish };
}

/** The text, usage and stop reason of a buffered Anthropic event stream. */
export function anthropicStreamParts(text: string): { text: string; input: number | null; output: number | null; stop: string | null } {
  let out = '';
  let input: number | null = null;
  let output: number | null = null;
  let stop: string | null = null;
  for (const frame of parseSseFrames(text)) {
    let data: unknown;
    try {
      data = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isRecord(data)) continue;
    if (frame.event === 'message_start' && isRecord(data.message) && isRecord(data.message.usage)) {
      input = numberOrZero(data.message.usage.input_tokens);
    } else if (frame.event === 'content_block_delta' && isRecord(data.delta) && data.delta.type === 'text_delta' && typeof data.delta.text === 'string') {
      out += data.delta.text;
    } else if (frame.event === 'message_delta') {
      if (isRecord(data.delta) && typeof data.delta.stop_reason === 'string') stop = data.delta.stop_reason;
      if (isRecord(data.usage)) output = numberOrZero(data.usage.output_tokens);
    }
  }
  return { text: out, input, output, stop };
}

/** The text and usage of a buffered Gemini event stream. */
export function geminiStreamParts(text: string): { text: string; usage: Usage | null; stop: string | null } {
  let out = '';
  let usage: Usage | null = null;
  let stop: string | null = null;
  for (const frame of parseSseFrames(text)) {
    let data: unknown;
    try {
      data = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isRecord(data)) continue;
    const candidates = Array.isArray(data.candidates) ? data.candidates : [];
    const candidate = isRecord(candidates[0]) ? candidates[0] : null;
    if (candidate !== null) {
      const content = isRecord(candidate.content) ? candidate.content : {};
      const parts = Array.isArray(content.parts) ? content.parts : [];
      for (const part of parts) if (isRecord(part) && typeof part.text === 'string') out += part.text;
      if (typeof candidate.finishReason === 'string') stop = candidate.finishReason;
    }
    if (isRecord(data.usageMetadata)) {
      usage = { input: numberOrZero(data.usageMetadata.promptTokenCount), output: numberOrZero(data.usageMetadata.candidatesTokenCount) };
    }
  }
  return { text: out, usage, stop };
}

/** A plain text and its usage as an OpenAI event stream (role chunk, content pieces, finish, usage, [DONE]). */
export function openAIEventsFromText(text: string, usage: Usage, model: string): string {
  const frames: string[] = [
    sseEvent(null, { id: 'chatcmpl-aggregate', object: 'chat.completion.chunk', created: FIXED_OPENAI_CREATED, model, choices: [{ index: 0, delta: { role: 'assistant', content: '' }, finish_reason: null }] }),
  ];
  for (const piece of splitText(text)) {
    frames.push(sseEvent(null, { id: 'chatcmpl-aggregate', object: 'chat.completion.chunk', created: FIXED_OPENAI_CREATED, model, choices: [{ index: 0, delta: { content: piece }, finish_reason: null }] }));
  }
  frames.push(sseEvent(null, { id: 'chatcmpl-aggregate', object: 'chat.completion.chunk', created: FIXED_OPENAI_CREATED, model, choices: [{ index: 0, delta: {}, finish_reason: 'stop' }] }));
  frames.push(
    sseEvent(null, {
      id: 'chatcmpl-aggregate',
      object: 'chat.completion.chunk',
      created: FIXED_OPENAI_CREATED,
      model,
      choices: [],
      usage: { prompt_tokens: usage.input, completion_tokens: usage.output, total_tokens: usage.input + usage.output },
    }),
  );
  frames.push('data: [DONE]\n\n');
  return frames.join('');
}

/** A plain text and its usage as an Anthropic event stream (start, one text block, delta, stop). */
export function anthropicEventsFromText(text: string, usage: Usage, model: string): string {
  const frames: string[] = [
    sseEvent('message_start', {
      type: 'message_start',
      message: { id: 'msg_stream', type: 'message', role: 'assistant', model, content: [], stop_reason: null, stop_sequence: null, usage: { input_tokens: usage.input, output_tokens: 0 } },
    }),
    sseEvent('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } }),
  ];
  for (const piece of splitText(text)) {
    frames.push(sseEvent('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: piece } }));
  }
  frames.push(
    sseEvent('content_block_stop', { type: 'content_block_stop', index: 0 }),
    sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: 'end_turn', stop_sequence: null }, usage: { output_tokens: usage.output } }),
    sseEvent('message_stop', { type: 'message_stop' }),
  );
  return frames.join('');
}

/**
 * A Chat Completions SSE stream as the Anthropic event sequence: message_start on the first chunk,
 * content_block_start/delta/stop per text block or tool call, then message_delta with the stop reason and
 * usage, then message_stop. `model` is the model the client asked for, named back to it. The usage
 * chunk of the OpenAI stream (sent because the upstream call asked for include_usage) supplies the
 * token counts; without one the counts stay 0 and the client's usage check fails, so callers must
 * request it upstream.
 */
export function anthropicEventsFromOpenAIStream(text: string, model: string): string {
  const parts = openAIStreamParts(text);
  return anthropicEventsFromText(parts.text, parts.usage ?? { input: 0, output: 0 }, model);
}

/**
 * An Anthropic event stream as a Chat Completions SSE stream: the text deltas become content deltas,
 * the message_delta's stop reason becomes the finish chunk, and the message_start / message_delta
 * usage becomes the trailing usage chunk.
 */
export function openAIEventsFromAnthropicStream(text: string, model: string): string {
  const parts = anthropicStreamParts(text);
  const usage: Usage = { input: parts.input ?? 0, output: parts.output ?? 0 };
  return openAIEventsFromText(parts.text, usage, model);
}

/** An Anthropic event stream with its message_start model rewritten to `model`; the rest passes through. */
export function anthropicEventsRewriteModel(text: string, model: string): string {
  const out: string[] = [];
  for (const frame of parseSseFrames(text)) {
    if (frame.event === 'message_start') {
      try {
        const data: unknown = JSON.parse(frame.data);
        if (isRecord(data) && isRecord(data.message)) {
          out.push(sseEvent('message_start', { ...data, message: { ...data.message, model } }));
          continue;
        }
      } catch {
        // Fall through and keep the frame as it is.
      }
    }
    out.push(`${frame.event === null ? '' : `event: ${frame.event}\n`}data: ${frame.data}\n\n`);
  }
  return out.join('');
}

/** A Gemini event stream as a Chat Completions SSE stream, with the final frame's usage. */
export function openAIEventsFromGeminiStream(text: string, model: string): string {
  const parts = geminiStreamParts(text);
  return openAIEventsFromText(parts.text, parts.usage ?? { input: 0, output: 0 }, model);
}

/** A Gemini event stream as an Anthropic event stream, with the final frame's usage. */
export function anthropicEventsFromGeminiStream(text: string, model: string): string {
  const parts = geminiStreamParts(text);
  return anthropicEventsFromText(parts.text, parts.usage ?? { input: 0, output: 0 }, model);
}
