# Node binding parity with the C++ core

What the Node binding (`ai-sdk-cpp`) exposes relative to the C++ core
(`libai_sdk`). Three layers decide what reaches JavaScript:

```
C++ core        include/ai, src/, providers/
   ↓             selected entry points only
C ABI           bindings/c/ai_sdk.h
   ↓             N-API wrappers
Node binding    bindings/node (index.ts + src/addon.cpp)
```

A feature marked **core-only** exists in C++ but has no C ABI entry point, so
the Node binding cannot reach it without an ABI change. **C ABI-only** means
the ABI exports it but the addon does not wrap it.

Legend: ✅ exposed · ⚠ partial or lossy · ◐ C ABI-only · ○ core-only · — absent

## Text generation

| Capability | Core | C ABI | Node | Notes |
|---|---|---|---|---|
| `generate_text` | ✅ | ✅ | ✅ | `generateText` |
| `stream_text` | ✅ | ✅ | ✅ | `streamText`, async iterable |
| `prompt` / `system` / `messages` | ✅ | ✅ | ✅ | `messages` is text-only across the C ABI; tool and file/image parts are dropped by `deserialize_prompt` |
| Tools + multi-step tool loop | ✅ | ✅ | ✅ | Async JS tools supported |
| `maxSteps`, `maxOutputTokens`, `temperature` | ✅ | ✅ | ✅ | |
| `providerOptions` (provider-namespaced JSON) | ✅ | ✅ | ✅ | Passed through verbatim |
| `toolChoice` | ✅ | — | — | `ToolChoiceAuto/None/Required/Specific` |
| `top_p`, `top_k`, `stop_sequences` | ✅ | — | — | Present in `GenerateTextOptions`, absent from the C ABI |
| `seed`, penalties, `user_id`, headers | ✅ (`CallOptions`) | — | — | Reachable only through batch/custom models |
| `repair_tool_call` | ✅ | — | — | |
| `generate_object` / `stream_object` | ✅ | — | — | Structured output not reachable from Node |
| `embed` / `embed_many` | ✅ | — | — | |
| `rerank` | ✅ | — | — | Interface only; no concrete model ships |
| `generate_image` / `generate_speech` / `transcribe` | ✅ | — | — | Interfaces only; no concrete models ship |
| `middleware` (caching / logging / telemetry) | ✅ | — | — | |
| Retry (`max_retries`) | ⚠ | — | — | Declared but inert: no core call site consumes it |
| Cancellation token | ✅ | — | — | No abort/`AbortSignal` path |
| Token counting / `trimToFit` | ✅ | — | — | |
| Warnings, request/response metadata, per-step results | ✅ | — | — | Node gets text, finish reason, token totals, and a step count |

## Agents, sessions, tools

| Capability | Core | C ABI | Node | Notes |
|---|---|---|---|---|
| `Agent.call` | ✅ | ✅ | ✅ | `Agent.call` |
| `Agent.stream` | ✅ | ✅ | ✅ | `Agent.callStream` (bound in this release) |
| Agent `instructions`, `maxSteps`, `providerOptions` | ✅ | ✅ | ✅ | |
| Agent `temperature`, `maxOutputTokens`, `toolChoice`, `on_step_finish`, middleware | ✅ | — | — | The C ABI's agent options do not carry them |
| `Session.send` / `send_stream` | ✅ | ✅ | ✅ | |
| `Session.addUser` / `addAssistant` / `setSystem` | ✅ | ✅ | ✅ | |
| Session memory + auto-checkpoint | ✅ | ✅ | ✅ | `memoryDir`, `enableCheckpoint` |
| `Session.history` / `metadata` / id | ✅ | — | — | |
| Session snapshot / restore + `JsonFileSessionStore` | ✅ | — | — | Manual persistence is core-only |
| Custom context strategy / summarizer | ✅ | — | — | |
| Standard toolkit (6 tools) | ✅ | ✅ | ✅ | `standardToolkit()` |
| Custom tool sets, merge, describe | ✅ | ✅ | ✅ | |
| Permission policy + async approver | ✅ | ✅ | ✅ | `withPermissions`; fail-closed |
| Memory store: add/update/get/list/remove | ✅ | — | — | |
| Memory store: save(scope,key,content) | ✅ | ✅ | ✅ | `MemoryStore.save` |
| Memory retrievers + `memory_tools` | ✅ | — | — | Recall/save/checkpoint tools are core-only |
| MCP tools | ✅ | ✅ | ✅ | `mcpToolsetFromServer` |
| MCP resources / prompts / sampling / progress | ✅ | — | — | |
| Batch run (submit → poll → results) | ✅ | ✅ | ✅ | |
| Batch status / cancel / list | ✅ | — | — | |
| `CodingAgent` facade | ✅ | — | — | Compose `Agent` + `Session` instead |
| Provider registry (`provider:model` lookup) | ✅ | — | — | No auto-registration in core either |

## Providers

Ten provider factories exist in C++; the C ABI maps seven:

| Native factory | C ABI / Node |
|---|---|
| `create_anthropic` | ✅ `createAnthropic` |
| `create_openai` | ✅ `createOpenAI` |
| `create_google` | ✅ `createGoogle` (compiled in only when the provider is enabled) |
| `create_deepseek` | ✅ `createDeepSeek` |
| `create_deepseek_anthropic` | ✅ `createDeepSeekAnthropic` |
| `create_zai` | ✅ `createZai` |
| `create_zai_openai` | ✅ `createZaiOpenAI` |
| `create_moonshotai` | — |
| `create_bedrock` | — |
| `create_openai_compatible` | — |

Provider file storage (`file_storage()`) is core-only.

## Stream events

| Core `StreamPart` | C ABI event | Node |
|---|---|---|
| `TextDelta` | `text_delta` | ✅ |
| `ToolInputStart` / `Delta` / `End` | `tool_call_start` / `_delta` / `_end` | ✅ |
| `FinishPart` | `finish` | ✅ |
| `ErrorPart` | `error` | ✅ (text only) |
| `ReasoningStart` / `Delta` / `End` | `reasoning_start` / `_delta` / `_end` | ✅ (signature/redacted data dropped) |
| `ToolResult` (synthesized) | `tool_result` | ⚠ session streams only |
| `StreamStart`, `TextStart` / `End`, `ResponseMetadataPart`, `RawPart` | — | — |
| Step boundary | `step_finish` | ⚠ only wired through the C-only `on_event` option, which Node never sets |

## Errors

| Layer | What is available |
|---|---|
| Core | `AiError` hierarchy with `error_type()`, HTTP status, `is_retryable()`, `retry_after()`, response body |
| C ABI | Nine coarse `ai_status_t` values and `ai_status_message()` |
| Node | Rejected promises carry `err.code` with the status name (`AI_ERROR_AUTHENTICATION`, …) and the engine message. Stream failures carry text only. |

## Requirements filed for the next engine release

These need work below `bindings/node`, so they are documented rather than
implemented here. Ordered by impact on Node consumers:

1. **Stream error status.** Add a status/code field to `ai_stream_event_t` so
   stream failures stop being message-only. Already on the engine queue.
2. **Cancellation.** Expose a cancel handle and `ai_*_cancel` entry points;
   then map a JS `AbortSignal` onto it.
3. **Structured errors.** Add `ai_last_error_*` accessors for error type, HTTP
   status, retryability, `retry-after`, and response body.
4. **Tool results on every stream path.** `stream_text` and
   `ai_agent_call_stream` do not emit `AI_STREAM_TOOL_RESULT`; only sessions
   do. Tool output is otherwise invisible to streaming callers.
5. **Multipart prompts.** Replace the lossy `deserialize_prompt` path for
   `messages_json` so tool messages and file/image parts survive.
6. **Structured output and embeddings** in the C ABI (`generate_object`,
   `stream_object`, `embed`, `embed_many`).
7. **Token counting** (`count_tokens`, `trim_to_fit`) for client-side budget
   checks.
8. **Provider name map** for `moonshotai`, `bedrock`, `openai-compatible`.
9. **Memory and sessions**: full memory CRUD, retrievers/`memory_tools`, and
   session snapshot/restore + `JsonFileSessionStore`.
10. **MCP resources and prompts.**
11. **Retry.** Wire `max_retries` into the core call path before exposing it —
    today the option is accepted and ignored.
12. **Middleware / telemetry hooks** in the ABI if native caching, logging, or
    tracing should be configurable from Node.

## Node-layer changes in this release

- `Agent.callStream()` — binds the existing `ai_agent_call_stream` ABI.
- Rejected calls carry `err.code` from `ai_status_t`.
- One shared stream-drain helper now backs `streamText`, `Session.sendStream`,
  and `Agent.callStream`; one shared tool-set builder backs `generateText`,
  `streamText`, and `Agent`.
- Tests: agent streaming, tool-loop streaming, stream failure, error codes.
