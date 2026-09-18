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

// Resolve the compiled addon. node-gyp-build looks in build/Release first —
// the layout node-gyp produces in a source checkout — and falls back to
// prebuilds/<platform>-<arch>/node.napi.node, which is what a published tarball
// ships. The same require therefore works in-repo and after npm install.
//
// The argument is the package root, not __dirname: this file is emitted to
// dist/, and both locations sit beside that directory rather than inside it.
// The addon's own rpath is @loader_path, so whatever directory it is found in
// must also hold libai_sdk.
const native = require('node-gyp-build')(
  require('path').join(__dirname, '..')
) as NativeBinding;

interface NativeBinding {
  Context: new () => NativeContext;
  Provider: new (ctx: NativeContext, name: string, apiKey: string | null, baseUrl: string | null) => NativeProvider;
  Model: new (provider: NativeProvider, modelId: string) => NativeModel;
  ToolSet: new () => NativeToolSet;
  Agent: new (model: NativeModel, tools: NativeToolSet, instructions: string, maxSteps: number, opts?: NativeAgentOptions) => NativeAgent;
  generateText(model: NativeModel, opts: NativeGenerateOpts): NativeResult;
  streamText(model: NativeModel, opts: NativeGenerateOpts, callback: StreamCallback): void;
  Session: new (agent: NativeAgent, opts?: NativeSessionOptions) => NativeSession;
  MemoryStore: new (dir: string) => NativeMemoryStore;
  Batch: new (provider: NativeProvider, modelId: string) => NativeBatch;
  standardToolkit(): NativeToolSet;
  withPermissions(tools: NativeToolSet, policy: PermissionPolicy, approver?: Approver): NativeToolSet;
  /** Present only on addons that accept the third argument above. */
  supportsApprover?: boolean;
  version(): string;
  mergeToolSets(dest: NativeToolSet, src: NativeToolSet): void;
  mcpToolsetFromServer(ctx: NativeContext, configJson: string): NativeToolSet;
}

interface NativeSessionOptions {
  memoryDir?: string;
  maxContextTokens?: number;
  enableCheckpoint?: boolean;
}

interface NativeAgentOptions {
  providerOptions?: Record<string, Record<string, unknown>>;
}

interface NativeSession {
  send(prompt: string): Promise<NativeResult>;
  sendStream(prompt: string, callback: StreamCallback): void;
  addUser(text: string): void;
  addAssistant(text: string): void;
  setSystem(text: string): void;
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
  call(prompt: string): Promise<NativeResult>;
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
  providerOptions?: Record<string, Record<string, unknown>>;
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
  providerOptions?: Record<string, Record<string, unknown>>;
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
    providerOptions: opts.providerOptions,
    toolSet: nativeToolSet,
  };

  if (opts.messages) {
    nativeOpts.messagesJson = JSON.stringify(opts.messages);
  }

  const result = await native.generateText(opts.model._native, nativeOpts);

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

/** Events whose payload carries `text`. Deltas are often legitimately empty
 *  (providers emit them between real tokens), so `text` is always set for these
 *  — otherwise `out += ev.text` silently appends the string "undefined". */
const TEXT_EVENTS: ReadonlySet<StreamEvent['type']> = new Set([
  'text_delta',
  'tool_call_delta',
  'reasoning_delta',
  'tool_result',
  'error',
]);

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
    providerOptions: opts.providerOptions,
    toolSet: nativeToolSet,
  };

  // native.streamText returns immediately; events arrive asynchronously.
  native.streamText(opts.model._native, nativeOpts, (type, text, toolName, toolCallId, usage) => {
    const event: StreamEvent = { type: type as StreamEvent['type'] };
    if (text || TEXT_EVENTS.has(event.type)) event.text = text ?? '';
    if (toolName) event.toolName = toolName;
    if (toolCallId) event.toolCallId = toolCallId;
    if (usage) event.usage = usage;
    queue.push(event);
    if ((type === 'finish' && (usage as any)?.finishReason !== 'tool_calls') || type === 'error') {
      finished = true;
      // The C binding emits tool-call-result events AFTER the stream finishes.
      // Brief wait for them to arrive before the generator exits.
      setTimeout(() => { if (resolveWait) { const r = resolveWait; resolveWait = null; r(); } }, 200);
    }
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
  private _toolSet: any;  // prevent GC — the C++ agent references the tool callbacks

  constructor(opts: {
    model: Model;
    tools: ToolDefinition[];
    instructions?: string;
    maxSteps?: number;
    extraToolSets?: StandardToolSet[];
    providerOptions?: Record<string, Record<string, unknown>>;
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

    this._toolSet = toolSet;  // prevent GC of the ToolSet wrapper
    this._native = new native.Agent(
      opts.model._native,
      toolSet,
      opts.instructions ?? '',
      opts.maxSteps ?? 50,
      opts.providerOptions ? { providerOptions: opts.providerOptions } : undefined,
    );
  }

  async call(prompt: string): Promise<GenerateResult> {
    const result = await this._native.call(prompt);
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

/** Mirrors the C enum: what a policy or approver answers with. */
export const PermissionDecision = {
  Allow: 0,
  Deny: 1,
  /** Policy only: escalate to the approver (without one this fails closed). */
  Ask: 2,
  /** Approver only: allow now and stop asking for this tool this session. */
  AllowAlways: 3,
} as const;
export type PermissionDecision = (typeof PermissionDecision)[keyof typeof PermissionDecision];

/** A decision plus the sentence to show the model when refusing a call.
 *  Returning a bare number means "no explanation". */
export interface PermissionVerdict {
  decision: number;
  /** Shown to the model as part of the tool result. Say what would have to
   *  change — a model told only that a call was refused retries variants of
   *  it, and every retry is another prompt the user already answered. */
  reason?: string;
}

/** Synchronous rule: answer Allow/Deny/Ask with no I/O. */
export type PermissionPolicy = (
  tool: string,
  inputJson: string,
) => number | PermissionVerdict;

/** Interactive approver, called only when the policy answers `Ask`. It is
 *  expected to be async — the usual implementation awaits a UI prompt — and
 *  its verdict is what allows or refuses the call. `rationale` is a
 *  human-readable line from the engine, suitable for showing to the user.
 *
 *  Anything that is not one of the `PermissionDecision` values (or a rejected
 *  promise) refuses the call: an approver is a permission gate, so a missing
 *  `return` must not read as approval.
 *
 *  Caveat: a prompt can only be awaited from the streaming/async entry points
 *  (`Session.sendStream`, `streamText`, and the promise-returning `call`s),
 *  which run the loop off the JS thread. A blocking main-thread call — the C
 *  API's `ai_agent_call`, reached from `generateText`/`Agent.call` on the
 *  synchronous path — cannot park for a prompt, so a promise-returning
 *  approver there refuses the call and its late verdict is discarded. */
export type Approver = (
  tool: string,
  inputJson: string,
  rationale: string,
) => number | PermissionVerdict | Promise<number | PermissionVerdict>;

export type StandardToolSet = NativeToolSet;

export function standardToolkit(): StandardToolSet {
  return native.standardToolkit();
}

/** True when this addon understands the third `withPermissions` argument.
 *  An older addon would ignore it and silently fail closed on every `Ask`,
 *  so callers that need a prompt should check rather than assume. */
export const supportsApprover: boolean = native.supportsApprover === true;

export function withPermissions(
  tools: StandardToolSet,
  policy: PermissionPolicy,
  approver?: Approver,
): StandardToolSet {
  if (approver && !supportsApprover) {
    throw new Error(
      'ai-sdk-cpp: this native addon predates interactive approvals (rebuild it), ' +
        'so an `Ask` could not reach your approver',
    );
  }
  return native.withPermissions(tools, policy, approver);
}

export interface SessionOptions {
  /** If set, the C++ session uses a MemoryContextStrategy: relevant persisted
   *  memory is auto-injected before each turn, and history auto-compacts
   *  (sliding window) near maxContextTokens. */
  memoryDir?: string;
  maxContextTokens?: number;
  /** Auto-checkpoint the conversation into memory every 5 turns.
   *  This calls the model to summarize history — costs an extra API call.
   *  Default true for backward compatibility. Set to false if you don't
   *  need automatic memory improvement or want to avoid extra API calls. */
  enableCheckpoint?: boolean;
}

export class Session {
  private _native: NativeSession;
  private _activePromise: Promise<void> | null = null;

  constructor(agent: Agent, opts?: SessionOptions) {
    if (opts?.memoryDir) {
      this._native = new native.Session(agent['_native'], {
        memoryDir: opts.memoryDir,
        maxContextTokens: opts.maxContextTokens ?? 0,
        enableCheckpoint: opts.enableCheckpoint ?? true,
      });
    } else {
      this._native = new native.Session(agent['_native']);
    }
  }

  send(prompt: string): Promise<GenerateResult> {
    return this._native.send(prompt).then((result) => ({
      text: result.text,
      finishReason: result.finishReason,
      usage: { inputTokens: result.inputTokens, outputTokens: result.outputTokens },
      steps: result.steps,
    }));
  }

  addUser(text: string): void {
    this._native.addUser(text);
  }

  addAssistant(text: string): void {
    this._native.addAssistant(text);
  }

  setSystem(text: string): void {
    this._native.setSystem(text);
  }

  /** Stream a session turn as an async iterable of events. Non-blocking: the
   *  native stream runs on a background thread; events are delivered on the JS
   *  event loop, so the UI (e.g. Ink) stays responsive. */
  async *sendStream(prompt: string): AsyncGenerator<StreamEvent> {
    if (this._activePromise) {
      await this._activePromise;
    }
    let resolveActive: any = null;
    this._activePromise = new Promise<void>((r) => { resolveActive = r; });

    const queue: StreamEvent[] = [];
    let resolveWait: (() => void) | null = null;
    let finished = false;

    this._native.sendStream(prompt, (type, text, toolName, toolCallId, usage) => {
      const ev: StreamEvent = { type: type as StreamEvent['type'] };
      if (text || TEXT_EVENTS.has(ev.type)) ev.text = text ?? '';
      if (toolName) ev.toolName = toolName;
      if (toolCallId) ev.toolCallId = toolCallId;
      if (usage) ev.usage = usage;
      queue.push(ev);
      if ((type === "finish" && (usage as any)?.finishReason !== "tool_calls") || type === "error") {
        finished = true;
        setTimeout(() => {
          if (resolveWait) { const r = resolveWait; resolveWait = null; r(); }
          if (resolveActive) { resolveActive(); }
        }, 200);
      }
      if (resolveWait) { const r = resolveWait; resolveWait = null; r(); }
    });

    try {
      while (!finished || queue.length > 0) {
        if (queue.length === 0) {
          await new Promise<void>((r) => { resolveWait = r; });
        }
        while (queue.length > 0) yield queue.shift()!;
      }
    } finally {
      if (finished && resolveActive) {
        resolveActive();
      }
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
