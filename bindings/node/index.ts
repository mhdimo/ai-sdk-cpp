/**
 * AI SDK C++ - Native Node.js bindings
 *
 * Usage:
 *   import { createAnthropic, generateText, Agent, tool } from 'ai-sdk-cpp';
 *
 *   const anthropic = createAnthropic();
 *   const model = anthropic('claude-sonnet-4-20250514');
 *
 *   const result = await generateText({
 *     model,
 *     prompt: 'Hello!',
 *   });
 */

// The compiled addon lands in build/Release/ (node-gyp output). This file ships
// from dist/, so the addon is one level up.
const native = require('../build/Release/ai_sdk_native.node') as NativeBinding;

interface NativeBinding {
  Context: new () => NativeContext;
  Provider: new (ctx: NativeContext, name: string, apiKey: string | null, baseUrl: string | null) => NativeProvider;
  Model: new (provider: NativeProvider, modelId: string) => NativeModel;
  ToolSet: new () => NativeToolSet;
  Agent: new (model: NativeModel, tools: NativeToolSet, instructions: string, maxSteps: number) => NativeAgent;
  generateText(model: NativeModel, opts: NativeGenerateOpts): NativeResult;
  streamText(model: NativeModel, opts: NativeGenerateOpts, callback: StreamCallback): void;
  Session: new (agent: NativeAgent, memoryDir?: string, maxContextTokens?: number) => NativeSession;
  MemoryStore: new (dir: string) => NativeMemoryStore;
  Batch: new (provider: NativeProvider, modelId: string) => NativeBatch;
  standardToolkit(): NativeToolSet;
  withPermissions(tools: NativeToolSet, policy: PermissionPolicy): NativeToolSet;
  version(): string;
  mergeToolSets(dest: NativeToolSet, src: NativeToolSet): void;
  mcpToolsetFromServer(ctx: NativeContext, configJson: string): NativeToolSet;
}

interface NativeSession {
  send(prompt: string): NativeResult;
  sendStream(prompt: string, callback: StreamCallback): void;
}

interface NativeMemoryStore {
  save(scope: string, key: string, content: string): void;
}
interface NativeBatchRequest { customId?: string; prompt?: string; system?: string; maxOutputTokens?: number; temperature?: number; }
interface NativeBatchResult { batchId: string; status: string; items: Array<{ customId: string; result: string | null; error: string | null }>; }
interface NativeBatch {
  run(requests: NativeBatchRequest[], pollIntervalMs?: number): NativeBatchResult;
}

type NativeContext = object;
type NativeProvider = object;
type NativeModel = object;
interface NativeToolSet {
  add(name: string, description: string, schemaJson: string, callback: ToolCallback): void;
}
interface NativeAgent {
  call(prompt: string): NativeResult;
}
type ToolCallback = (toolName: string, inputJson: string) => Promise<{ output: string; isError: boolean }> | { output: string; isError: boolean };
type StreamCallback = (type: string, text: string | null, toolName: string | null, toolCallId: string | null, usage: { inputTokens: number; outputTokens: number } | null) => void;

interface NativeGenerateOpts {
  prompt?: string;
  system?: string;
  messagesJson?: string;
  maxSteps?: number;
  maxOutputTokens?: number;
  temperature?: number;
  toolSet?: NativeToolSet;
}

interface NativeResult {
  text: string;
  finishReason: string;
  inputTokens: number;
  outputTokens: number;
  steps: number;
}

// --- Public API ---

let _ctx: NativeContext | null = null;
function getCtx(): NativeContext {
  if (!_ctx) _ctx = new native.Context();
  return _ctx;
}

export interface ProviderOptions {
  apiKey?: string;
  baseUrl?: string;
}

export interface Model {
  _native: NativeModel;
  provider: string;
  modelId: string;
}

export interface ProviderInstance {
  (modelId: string): Model;
  model(modelId: string): Model;
  _native: NativeProvider;
}

function createProvider(name: string, opts: ProviderOptions = {}): ProviderInstance {
  const ctx = getCtx();
  const nativeProvider = new native.Provider(ctx, name, opts.apiKey ?? null, opts.baseUrl ?? null);

  const provider = (modelId: string): Model => {
    const nativeModel = new native.Model(nativeProvider, modelId);
    return { _native: nativeModel, provider: name, modelId };
  };

  provider.model = provider;
  (provider as any)._native = nativeProvider;
  return provider as unknown as ProviderInstance;
}

export const createAnthropic = (opts?: ProviderOptions) => createProvider('anthropic', opts);
export const createOpenAI = (opts?: ProviderOptions) => createProvider('openai', opts);
export const createGoogle = (opts?: ProviderOptions) => createProvider('google', opts);
export const createDeepSeek = (opts?: ProviderOptions) => createProvider('deepseek', opts);
export const createZai = (opts?: ProviderOptions) => createProvider('zai', opts);
export const createDeepSeekAnthropic = (opts?: ProviderOptions) => createProvider('deepseek-anthropic', opts);
export const createZaiOpenAI = (opts?: ProviderOptions) => createProvider('zai-openai', opts);

export interface ToolDefinition {
  name: string;
  description: string;
  schema: Record<string, unknown>;
  execute: (input: Record<string, unknown>) => Promise<unknown> | unknown;
}

export function tool(
  name: string,
  schema: Record<string, unknown>,
  description: string,
  execute: (input: Record<string, unknown>) => Promise<unknown> | unknown
): ToolDefinition {
  return { name, description, schema, execute };
}

export interface GenerateTextOptions {
  model: Model;
  prompt?: string;
  system?: string;
  messages?: Array<{ role: string; content: string }>;
  tools?: ToolDefinition[];
  maxSteps?: number;
  maxOutputTokens?: number;
  temperature?: number;
}

export interface GenerateResult {
  text: string;
  finishReason: string;
  usage: {
    inputTokens: number;
    outputTokens: number;
  };
  steps: number;
}

export async function generateText(opts: GenerateTextOptions): Promise<GenerateResult> {
  let nativeToolSet: NativeToolSet | undefined;

  if (opts.tools && opts.tools.length > 0) {
    nativeToolSet = new native.ToolSet();
    for (const t of opts.tools) {
      nativeToolSet.add(t.name, t.description, JSON.stringify(t.schema), async (toolName, inputJson) => {
        const input = JSON.parse(inputJson);
        try {
          const result = await t.execute(input);
          const output = typeof result === 'string' ? result : JSON.stringify(result);
          return { output, isError: false };
        } catch (e: any) {
          return { output: e.message ?? String(e), isError: true };
        }
      });
    }
  }

  const nativeOpts: NativeGenerateOpts = {
    prompt: opts.prompt,
    system: opts.system,
    maxSteps: opts.maxSteps,
    maxOutputTokens: opts.maxOutputTokens,
    temperature: opts.temperature,
    toolSet: nativeToolSet,
  };

  if (opts.messages) {
    nativeOpts.messagesJson = JSON.stringify(opts.messages);
  }

  const result = native.generateText(opts.model._native, nativeOpts);

  return {
    text: result.text,
    finishReason: result.finishReason,
    usage: {
      inputTokens: result.inputTokens,
      outputTokens: result.outputTokens,
    },
    steps: result.steps,
  };
}

export interface StreamEvent {
  type: 'text_delta' | 'tool_call_start' | 'tool_call_delta' | 'tool_call_end' | 'reasoning_start' | 'reasoning_delta' | 'reasoning_end' | 'tool_result' | 'step_finish' | 'finish' | 'error';
  /** Token usage, populated on the finish event. */
  usage?: { inputTokens: number; outputTokens: number };
  text?: string;
  toolName?: string;
  toolCallId?: string;
}

export async function* streamText(opts: GenerateTextOptions): AsyncGenerator<StreamEvent> {
  const queue: StreamEvent[] = [];
  let resolveWait: (() => void) | null = null;
  let finished = false;

  let nativeToolSet: NativeToolSet | undefined;
  if (opts.tools && opts.tools.length > 0) {
    nativeToolSet = new native.ToolSet();
    for (const t of opts.tools) {
      nativeToolSet.add(t.name, t.description, JSON.stringify(t.schema), async (toolName, inputJson) => {
        const input = JSON.parse(inputJson);
        try {
          const result = await t.execute(input);
          const output = typeof result === 'string' ? result : JSON.stringify(result);
          return { output, isError: false };
        } catch (e: any) {
          return { output: e.message ?? String(e), isError: true };
        }
      });
    }
  }

  const nativeOpts: NativeGenerateOpts = {
    prompt: opts.prompt,
    system: opts.system,
    messagesJson: opts.messages ? JSON.stringify(opts.messages) : undefined,
    maxSteps: opts.maxSteps,
    maxOutputTokens: opts.maxOutputTokens,
    temperature: opts.temperature,
    toolSet: nativeToolSet,
  };

  // native.streamText returns immediately; events arrive asynchronously.
  native.streamText(opts.model._native, nativeOpts, (type, text, toolName, toolCallId, usage) => {
    const event: StreamEvent = { type: type as StreamEvent['type'] };
    if (text) event.text = text;
    if (toolName) event.toolName = toolName;
    if (toolCallId) event.toolCallId = toolCallId;
    if (usage) event.usage = usage;
    queue.push(event);
    if (type === 'finish' || type === 'error') finished = true;
    if (resolveWait) { const r = resolveWait; resolveWait = null; r(); }
  });

  while (!finished || queue.length > 0) {
    if (queue.length === 0) {
      await new Promise<void>((r) => { resolveWait = r; });
    }
    while (queue.length > 0) yield queue.shift()!;
  }
}

export class Agent {
  private _native: NativeAgent;

  constructor(opts: {
    model: Model;
    tools: ToolDefinition[];
    instructions?: string;
    maxSteps?: number;
    extraToolSets?: StandardToolSet[];
  }) {
    const toolSet = new native.ToolSet();
    for (const t of opts.tools) {
      toolSet.add(t.name, t.description, JSON.stringify(t.schema), async (toolName, inputJson) => {
        const input = JSON.parse(inputJson);
        try {
          const result = await t.execute(input);
          const output = typeof result === 'string' ? result : JSON.stringify(result);
          return { output, isError: false };
        } catch (e: any) {
          return { output: e.message ?? String(e), isError: true };
        }
      });
    }
    // Merge in extra tool sets (e.g. MCP tools from mcpToolsetFromServer).
    if (opts.extraToolSets) {
      for (const extra of opts.extraToolSets) {
        native.mergeToolSets(toolSet, extra);
      }
    }

    this._native = new native.Agent(
      opts.model._native,
      toolSet,
      opts.instructions ?? '',
      opts.maxSteps ?? 50,
    );
  }

  async call(prompt: string): Promise<GenerateResult> {
    const result = this._native.call(prompt);
    return {
      text: result.text,
      finishReason: result.finishReason,
      usage: {
        inputTokens: result.inputTokens,
        outputTokens: result.outputTokens,
      },
      steps: result.steps,
    };
  }
}

export function version(): string {
  return native.version();
}

// --- MCP ---

/** Merge `src` tool set into `dest` (e.g. combine custom tools with MCP tools). */
export function mergeToolSets(dest: StandardToolSet, src: StandardToolSet): void {
  native.mergeToolSets(dest, src);
}

/** Connect to an MCP server and return its tools as a ToolSet. */
export function mcpToolsetFromServer(configJson: string): StandardToolSet {
  return native.mcpToolsetFromServer(getCtx(), configJson);
}

// --- Standard toolkit + permissions + session ---

export type PermissionPolicy = (tool: string, inputJson: string) => number;
export type StandardToolSet = NativeToolSet;

export function standardToolkit(): StandardToolSet {
  return native.standardToolkit();
}

export function withPermissions(tools: StandardToolSet, policy: PermissionPolicy): StandardToolSet {
  return native.withPermissions(tools, policy);
}

export interface SessionOptions {
  /** If set, the C++ session uses a MemoryContextStrategy: relevant persisted
   *  memory is auto-injected before each turn, and history auto-compacts
   *  (sliding window) near maxContextTokens. */
  memoryDir?: string;
  maxContextTokens?: number;
}

export class Session {
  private _native: NativeSession;

  constructor(agent: Agent, opts?: SessionOptions) {
    if (opts?.memoryDir) {
      this._native = new native.Session(agent['_native'], opts.memoryDir, opts.maxContextTokens ?? 0);
    } else {
      this._native = new native.Session(agent['_native']);
    }
  }

  send(prompt: string): GenerateResult {
    const result = this._native.send(prompt);
    return {
      text: result.text,
      finishReason: result.finishReason,
      usage: { inputTokens: result.inputTokens, outputTokens: result.outputTokens },
      steps: result.steps,
    };
  }

  /** Stream a session turn as an async iterable of events. Non-blocking: the
   *  native stream runs on a background thread; events are delivered on the JS
   *  event loop, so the UI (e.g. Ink) stays responsive. */
  async *sendStream(prompt: string): AsyncGenerator<StreamEvent> {
    const queue: StreamEvent[] = [];
    let resolveWait: (() => void) | null = null;
    let finished = false;

    this._native.sendStream(prompt, (type, text, toolName, toolCallId, usage) => {
      const ev: StreamEvent = { type: type as StreamEvent['type'] };
      if (text) ev.text = text;
      if (toolName) ev.toolName = toolName;
      if (toolCallId) ev.toolCallId = toolCallId;
      if (usage) ev.usage = usage;
      queue.push(ev);
      if (type === "finish" || type === "error") finished = true;
      if (resolveWait) { const r = resolveWait; resolveWait = null; r(); }
    });

    while (!finished || queue.length > 0) {
      if (queue.length === 0) {
        await new Promise<void>((r) => { resolveWait = r; });
      }
      while (queue.length > 0) yield queue.shift()!;
    }
  }
}

// --- Memory + Batch ---

export class MemoryStore {
  private _native: NativeMemoryStore;
  constructor(dir: string) {
    this._native = new native.MemoryStore(dir);
  }
  save(scope: string, key: string, content: string): void {
    this._native.save(scope, key, content);
  }
}

export interface BatchRequest {
  customId?: string;
  prompt?: string;
  system?: string;
  maxOutputTokens?: number;
  temperature?: number;
}
export interface BatchResult {
  batchId: string;
  status: string;
  items: Array<{ customId: string; result: string | null; error: string | null }>;
}

export class Batch {
  private _native: NativeBatch;
  constructor(provider: ProviderInstance, modelId: string) {
    this._native = new native.Batch(provider._native, modelId);
  }
  run(requests: BatchRequest[], pollIntervalMs = 5000): BatchResult {
    return this._native.run(requests, pollIntervalMs);
  }
}
