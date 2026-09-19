# Changelog

All notable changes to ai-sdk-cpp. Format loosely based on
[Keep a Changelog](https://keepachangelog.com/).

## [1.0.0] - 2026-09-18

The bindings stop being a preview. The Node binding is now covered by a
hermetic test suite (mock provider server, no API keys) that runs in CI, and
the defects that suite found are fixed — including six that blocked release.

### Changed — BREAKING
- **Node: `Session.send()` and `Agent.call()` are now `async`.** They return a
  promise and no longer block the JS event loop for the whole turn. A turn that
  calls a tool has to hand control back to the event loop the tool's own JS
  runs on; blocking there deadlocked the process. `await` both. `generateText`
  and `streamText` were already async and are unchanged.

### Added — Node binding
- **Test suite** (`bindings/node/test/`): 63 cases over the public surface —
  every provider factory, tool-calling through each entry point, streaming
  events, sessions, memory, batch, MCP, standard toolkit, permissions, and
  tool-set merging. Hermetic: a mock provider server stands in for the vendor
  APIs, so it needs no keys and no network. Cases that drive a JS tool callback
  run out-of-process under a wall-clock budget, because a deadlock in one of
  those blocks the event loop so completely that an in-process timeout could
  never fire to fail the test.
- **CI job** (`node-binding`) building the C library and running the suite.
- `streamText` accepts `messages`, `tools`, and `maxSteps`.
- `Session` with persistent memory (auto-inject recall + auto-compaction),
  the checkpoint writer, and token usage on the stream finish event.
- `mergeToolSets()`, `mcpToolsetFromServer()`, `standardToolkit()`,
  `withPermissions()`, `MemoryStore`, `Batch`, and `Agent` extra tool sets.
- **Tool-set introspection**: `describeToolSet()` in Node, backed by
  `ai_tool_set_describe_json()` in the C API, reports the tools in a set as
  `{name, description, inputSchema}` ordered by name. Ordered because the
  underlying container is an unordered map and a description that comes back in
  a different order every run cannot be compared against anything — and it is
  the sets you cannot otherwise see into (an MCP server's tools, a
  permission-wrapped set) that this is for.
- Provider options can be passed through to agents from C and Node.

### Fixed
- **A failed `Session.sendStream()` turn reported success.** When the request
  never got a response — refused connection, DNS failure, timeout — the
  session entry point returned its error status without emitting a terminal
  event, and the binding, whose fallback exists to stop a consumer waiting
  forever, filled the gap with a *finish*. The caller got a turn that had
  succeeded and produced nothing: zero tokens, empty `finishReason`, no error.
  The status-code failures (401/429/500) were never affected — those arrive as
  a response and were already surfaced from inside the stream — and neither was
  an agent carrying tools, which routes the same failure through the stream.
  It took a tool-less agent plus a server that never answered to reach it.
  `ai_stream_text` has always emitted `AI_STREAM_ERROR` from its catch; the
  session entry point now does too, and the binding's fallback fails closed
  rather than reporting a finish it never received.
- **`streamText` silently dropped tool calls.** It called the model's
  `do_stream()` directly, so a tool call was streamed to the caller and never
  executed. It now goes through `ai::stream_text`, which runs the tool loop —
  fixed in the C binding, so the Python, Rust, and Go bindings inherit it.
- **A tool call hung the blocking entry points at 100% CPU forever.** The C
  binding busy-waits on the event loop while the coroutine awaits a JS tool
  callback, but the callback could only run on the event loop the caller was
  blocking. Tool callbacks now run on a worker thread and resolve a promise.
- **`Batch.run()` hung at 100% CPU**, having polled exactly once. `run_batch()`
  waits between polls on an event-loop timer; an in-flight HTTP call left the
  loop with no queued work, which *stops* it, so the timer could never fire.
  The driver now restarts the loop before each `run_one()`.
- **Batch results carried the wrong `customId`** — every item but the last got
  a dangling pointer, because request strings were stored via `c_str()` while
  the container they lived in was still reallocating.
- **`withPermissions()` aborted the process** (`Fatal error in
  v8::HandleScope::CreateHandle`): the policy callback called into JS from the
  tool's worker thread, which has no handle scope. It now goes through a
  `ThreadSafeFunction`.
- **Empty `text_delta` events rendered as the literal string `"undefined"`.**
  Providers legitimately emit empty deltas between real tokens; events whose
  payload is text now always carry a string.
- **`createGoogle` ignored `baseUrl`** and had no `GOOGLE_BASE_URL` env
  fallback, so the provider could not be pointed anywhere but the default host.
- **`Task` start-then-await**: `co_await` on a `Task` that was already
  `start()`ed re-entered the coroutine mid-await and read an empty result
  (silent nulls / UB). `await_suspend` now registers the continuation and
  suspends for in-flight tasks; only fresh tasks are launched. Move
  construction carries the started flag.
- **Checkpoint writer never fired for streamed turns**: `send_stream` did not
  increment the session turn counter.
- **Use-after-free on GC** in the Node tool callback path (`SIGTRAP`).

### Changed
- **Parallel tool execution**: `execute_tools` (generate + stream paths)
  launches all tool calls in a step before awaiting results, so independent
  calls interleave on the event loop instead of serializing behind the slowest
  tool. Results stay in tool-call order; per-call error isolation unchanged.
- **Provider-scoped reasoning effort**, and an opt-in checkpoint summarizer
  with a configurable proactive-compaction threshold.
- **The version string has one source of truth.** `ai_sdk_version()` and the
  MCP `clientInfo` handshake are compiled from the CMake project version
  (`include/ai/version.hpp`) instead of a retyped literal, so they cannot drift
  from the release number the package manifests carry. A test asserts the
  binding's `version()` and `package.json` agree.

### Tests
- 162 offline unit tests (`ctest`), plus the 61-case Node suite. Both run in CI
  on every push; neither needs an API key or network access.
- `scripts/coverage.sh` reports what those tests actually exercise (Clang
  source-based coverage). It covers the SDK but not the socket layer or the C
  binding — the unit tests substitute their own `IHttpClient`, so those are
  covered by the Node suite instead. The script says so in its output.

### Notes & known limitations
- **Building requires GCC 13 or newer, or Clang.** GCC 11 and 12 miscompile the
  library's coroutines: 30 of the 158 tests the suite held when this was measured
  crashed on them, at `-O0` and `-O3` alike, while GCC 13 passed all 158 on the
  same OS, standard library and Boost. A build that succeeded on those compilers
  would therefore still be a library that segfaults, so the build now refuses
  them by name instead of letting that through —
  `-DAI_SDK_ALLOW_UNSUPPORTED_COMPILER=ON` overrides it. This lands hardest on
  Ubuntu 22.04, RHEL 9 and Debian 12, whose default repositories ship GCC 11 or
  12. On Ubuntu 22.04 a newer GCC is one package away
  (`ppa:ubuntu-toolchain-r/test` carries `g++-13`), and that is the better route
  there than Clang: this project builds Clang with `-stdlib=libc++`, and libc++
  is not installed by default on any of those distributions.
  `sudo apt-get install libc++-dev libc++abi-dev` is the other way, with the
  caveat that anything you link against it then needs libc++ at runtime too.
- **An ambient `ANTHROPIC_AUTH_TOKEN` outranks an explicitly passed `apiKey`.**
  The Anthropic provider advertises two credentials and prefers the Bearer
  token (`docs/providers.md`), resolving each independently, so if the
  environment carries `ANTHROPIC_AUTH_TOKEN` — which it does for anyone running
  Claude Code against an Anthropic-compatible gateway — `createAnthropic({
  apiKey })` authenticates with the ambient token instead. Requests fail, or
  are billed elsewhere, with nothing in the error to say why. Unset the
  variable, or set `authToken` deliberately, if you meant the key you passed.
  Changing the precedence is a candidate for a future release, not this one.
- **Streaming reports failures as a terminal `error` event, not a throw.**
  `streamText`/`sendStream` yield `{ type: 'error' }` and end; they do not
  reject. Callers must check for it.
- **Node: passing the wrong kind of object to an entry point can abort the
  process instead of throwing.** Eighteen call sites across five wrapper types
  (context, provider, model, tool set, agent) unwrap their arguments without
  checking that the unwrap succeeded, so `withPermissions({}, policy)` or the
  like reaches a null handle and dies rather than raising a catchable error.
  `describeToolSet()` is guarded; the rest are unchanged as of this release,
  and guarding them is a candidate for 1.0.1 — mechanical, but wide enough that
  it did not belong in a release that is otherwise done.
- **Google** compiles and is wired through `createGoogle` (including
  `baseUrl`), but has not been exercised against the live API — treat it as
  experimental.

## [0.1.0] - 2026-06-22

### Added — native agent capabilities
- **Session & context management** (`include/ai/session/`): `Session` holds
  conversation history across turns and applies a pluggable `ContextStrategy` —
  `SlidingWindowStrategy` (default, tool-call/tool-result pair-safe) and opt-in
  `SummarizationStrategy` (compaction). Owns the stateless `ToolLoopAgent`.
- **Permission / approval hooks** (`include/ai/permission/`): `with_permissions`
  wraps a `ToolSet` with a sync `PermissionPolicy` + async interactive
  `Approver`, with a per-session allow-always cache and fail-closed default.
- **Standard toolkit** (`include/ai/tools/standard/`): `standard_toolkit()`
  with `read_file`, `write_file`, `edit_file` (exact `str_replace`, unique
  unless `replace_all`), `glob`, `grep`, `bash`.
- **Session persistence & resume** (`include/ai/session/store.hpp`):
  `SessionStore` interface + `JsonFileSessionStore` with **full-fidelity**
  message serialization (system, user text/file-url, assistant text/tool-call/
  reasoning, tool results) so tool conversations resume correctly.
- **Persistent memory** (`include/ai/memory/`): `MarkdownMemoryStore` +
  `KeywordRetriever` / `EmbeddingRetriever` (uses the SDK's own `embed()` —
  semantic recall) / `HybridRetriever`, `MemoryContextStrategy` (auto-injects
  relevant memory within a token budget), memory tools
  (`recall_memory`/`save_memory`/`add_note`/`update_checkpoint`/
  `log_task_progress`), and `make_checkpoint_writer` (post-turn auto-checkpoint
  via a `Session::set_on_turn_finish` hook). No new dependencies.
- **`CodingAgent` facade** (`include/ai/coding_agent.hpp`): bundles Session +
  permissioned standard toolkit + optional persistent memory — a Codex-style
  CLI in ~20 lines.
- **Batch** (`src/core/batch.cpp`): `run_batch()` orchestrator +
  `Provider::batch_processor()` polymorphic factory; **C binding** (`ai_batch_*`).
- **`IHttpClient`** interface (`include/ai/http/client.hpp`) with an injectable
  seam on Anthropic, OpenAI, and Google providers for offline provider testing
  via `tests/unit/fake_http_client.hpp`.
- **Anthropic structured output**: tool-use pattern so `generate_object` /
  `stream_object` work on Anthropic (request + response sides).
- Engineering baseline: `.clang-format`, `.clang-tidy`, `vcpkg.json`,
  `CONTRIBUTING.md`, `SECURITY.md`. CI workflow (`.github/workflows/ci.yml`).
  Gated live smoke test (`tests/unit/test_live_smoke.cpp`).

### Added — provider parity & packaging
- **z.ai (GLM)** provider (Anthropic-compatible; strips the `[1m]` Claude-Code
  alias that the API rejects) + **DeepSeek Anthropic** and **z.ai OpenAI**
  endpoints. See `docs/providers.md` for the provider × protocol matrix.
- **`find_package(ai-sdk-cpp 0.1.0)`**: CMake package config exports the core
  library, C API, and every provider that was built as `ai-sdk-cpp::*` targets.
- **OpenAI**: streaming token usage (`stream_options.include_usage`),
  `reasoning_content` surfacing (DeepSeek/R1-style), `json_schema`→`json_object`
  auto-downgrade on non-OpenAI hosts, `refusal`, reasoning-effort guard.
- **Anthropic**: extended-thinking `signature` + `redacted_thinking` round-trip
  for multi-turn tool loops; `pause_turn`/`refusal`/`model_context_window_exceeded`
  stop reasons; `tool_choice:none`; `budget_tokens` validation; `metadata.user_id`;
  `anthropic-beta` header; adaptive-thinking override.
- **Session streaming** in the C API + Python binding (`ai_session_send_stream`),
  with `reasoning_*` and `tool_result` stream events.

### Fixed
- **`AsyncGenerator` handshake**: `yield_value` returned `suspend_always` and
  never resumed the consumer — every `co_await gen.next()` loop deadlocked
  (broke `stream_text` multi-step, `stream_object`, `streaming_chat`, the C
  `ai_stream_text`). Now uses symmetric transfer on yield + final_suspend.
- **Swallowed exceptions**: `next()` checked `done()` before the stored
  exception, silently dropping errors raised after the last yield.
- **`Task<void>`** lacked `done()`/`get()` — added.
- **Dangling `SseParser`** in `do_stream` across Anthropic, OpenAI, Google: the
  parse coroutine held `&sse_parser` to a frame-local destroyed when the Task
  completed. Parser is now owned inside the generator frame.
- **`stream_object`** never populated final object/usage and skipped schema
  validation; now returns a shared `StreamObjectFinalState` and validates.
- **`stream_text.full_result`** was always null; now resolves after draining.
- **`ai_stream_text` / `ai_agent_call_stream`** delivered nothing / was a stub;
  both now consume the stream and dispatch to the callback.
- **Docs/reality**: README MIT (was Apache-2.0), dropped fictional Rust/Go
  binding Quick Start, corrected provider count, reclassified Cohere, fixed
  `CLAUDE.md` provider-structure diagram.

### Removed
- `generate_video` placeholder header (shipped in public API, did nothing).

### Tests
- 105 offline unit tests across streaming, batch, session, context strategies,
  permissions, toolkit, persistence, memory, the facade, MCP, embeddings/rerank,
  and OpenAI/Anthropic reasoning + structured-output parsing. Live-verified
  paths: OpenAI, Anthropic, DeepSeek, z.ai, Moonshot, Bedrock.

### Notes & known limitations
- **Providers**: OpenAI, Anthropic, DeepSeek, z.ai (GLM), MoonshotAI, Amazon
  Bedrock, Google. **Google** compiles but is not yet live-verified — treat as
  experimental. The 7 minor OpenAI-compatible wrappers (Groq, xAI, Mistral,
  Fireworks, TogetherAI, Perplexity, Cohere) were removed; use the generic
  OpenAI-compatible provider or the OpenAI provider with a custom `base_url`.
- **Bindings** (Python/Node/Go/Rust) wrap the C API and ship as a **preview** —
  build from source. Registry publication (PyPI/npm/crates) is deferred.
- Extended-thinking signature round-trip is unit-tested but not yet exercised
  against a live Claude extended-thinking tool loop.
