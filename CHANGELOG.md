# Changelog

All notable changes to ai-sdk-cpp. Format loosely based on
[Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

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
- 33 → 82+ offline unit tests across streaming, batch, session, context
  strategies, permissions, toolkit, persistence, memory, the facade, and
  Anthropic/OpenAI/Google provider parsing.

### Known gaps (not in this release)
- MCP: only `tools/list` + `tools/call`, no transport seam for offline tests,
  missing resources/prompts/sampling/progress/streamable-HTTP.
- Rust/Go bindings: still DIY-FFI (no in-repo packages).
- Python/Node bindings do not yet cover Session/permissions/toolkit/memory.
- Not yet verified against live provider APIs (offline suite only).
