# Native Agent Capabilities — Design Spec

**Date:** 2026-06-19
**Goal:** Make `ai-sdk-cpp` the best C++ library for building native CLI agents /
Codex-like apps by closing the four caps identified in audit, plus a persistent
memory subsystem.

**Status:** Design approved (2026-06-19). Forks resolved to recommended options (see
"Resolved decisions").

---

## Context & Goals

The SDK already has a working agent loop (`ToolLoopAgent`), streaming, structured
output, multi-provider support, and an HTTP-mock seam. After this session's fixes the
public surface is correct and tested (48 tests). What's missing for a *competitive*
native-agent library:

1. No conversation/context management — the agent is stateless; no session memory,
   context-window compaction, or sliding window at the orchestration layer.
2. No permission/approval hooks — dangerous tools (shell, file-write) run with no gate.
3. Thin test coverage across providers/MCP/agent-loop edge cases.
4. No diff/edit tool or session persistence.
5. No persistent, cross-session project memory.

This spec adds all five as **composable primitives behind interfaces**, plus a
`CodingAgent` convenience facade, so a Codex-style CLI is ~20 lines while power users
can assemble the pieces.

## Non-goals

- No new mandatory dependencies. SQLite (FTS5) is explicitly *optional* / future.
- No GUI; targets CLI/TUI agents and app backends.
- Not re-implementing the model/providers — these layer on top of the existing
  `LanguageModel` / `Agent` / `ToolSet`.

## Architecture overview

Interface-first, each capability independently testable and replaceable:

```
CodingAgent (facade)
 └─ Session  ──────► ContextStrategy ──► MemoryContextStrategy ──► MemoryStore
     │                  (sliding window,       (inject relevant       (markdown files)
     │                   summarization)         memory per turn)
     │
     ├─ uses ─► ToolLoopAgent (existing, stateless)
     ├─ tools ─► SecureToolSet(standard_toolkit(), PermissionPolicy, Approver)
     └─ persist► SessionStore (JSON files)
```

New code lives in new namespaces/headers; existing code is reused, not rewritten.

## Existing primitives reused (do not duplicate)

- `TokenCounter` / `ApproximateTokenCounter` and `ContextWindow` (`max_tokens`,
  `reserved_output_tokens`, `fits()`, **`trim_to_fit()`**) — `include/ai/util/token_count.hpp`.
- `ToolLoopAgent` is **stateless** (`call()` forwards fresh to `generate_text`) — the
  clean seam `Session` wraps.
- `embed()` / `embed_many()` — `include/ai/core/embed.hpp` — powers semantic memory.
- `workflow::serialize_prompt` / `deserialize_prompt` — `include/ai/workflow/serialize.hpp`
  — message serialization for persistence.
- `FakeHttpClient` (`IHttpClient`) — provider test mocking seam (added this session).

---

## Sub-project 1 — Session & Context Management

**Purpose:** hold conversation history across turns; manage it to fit the context
window; never break tool-call/tool-result pairing.

**Components** (`include/ai/session/`):

- `class Session` — owns `Prompt history_`, an `Agent&` (or `LanguageModelPtr`+opts),
  a `ContextWindow`, a `TokenCounter`, and **one** `ContextStrategy`. (Memory, when
  enabled, is a `MemoryContextStrategy` that *decorates* an inner window/compaction
  strategy — see #5.) API:
  - `Task<GenerateTextResult> send(std::string user_msg, CancellationToken = {})`
  - `Task<StreamTextResult> send_stream(std::string user_msg, CancellationToken = {})`
  - `void add_user(...)` / `add_assistant(...)` for manual composition
  - `const Prompt& history() const`, `SessionMetadata metadata()`
  - Each `send`: append user → apply strategy → run agent → append assistant + tool
    messages → return result.
- `struct ContextStrategy` (interface) —
  `Prompt manage(Prompt history, const ContextWindow&, const TokenCounter&, Summarizer&)`.
  - **`SlidingWindowStrategy`** (default) — uses `ContextWindow::trim_to_fit`, but:
    protects the system message and the current turn; trims **tool-call/tool-result
    matched pairs** only (providers reject an orphaned `tool_use`).
  - **`SummarizationStrategy`** (opt-in) — on overflow, evict the oldest pair, summarize
    the evicted tail via an LLM `Summarizer` into one compact `## Earlier in this
    session` message kept after the system prompt; keep the last N turns verbatim.
- `using Summarizer = std::function<Task<std::string>(const Prompt& to_summarize)>;`
  (caller wires it to a cheap model; default = no-op when summarization off).

**Resolved decision (Fork 1):** sliding window always on; summarization **opt-in**
(adds LLM cost/latency per compaction).

---

## Sub-project 2 — Permission / Approval Hooks

**Purpose:** authorize tool execution *before* `execute` runs; support auto-rules and
interactive CLI approval.

**Components** (`include/ai/permission/`):

- `enum class PermissionDecision { Allow, Deny, Ask };`
- `using PermissionPolicy = std::function<PermissionDecision(const std::string& tool, const boost::json::value& input)>;`
  — synchronous rules (e.g. allow `read_file`, deny `rm -rf`).
- `using Approver = std::function<Task<PermissionDecision>(const std::string& tool, const boost::json::value& input, const std::string& rationale)>;`
  — async interactive path (CLI prompt) for the `Ask` outcome.
- `ToolSet with_permissions(ToolSet tools, PermissionPolicy policy, Approver approver = {})`
  — returns a `ToolSet` whose each tool's `execute` runs the gate: `Allow`→run,
  `Deny`→return a structured error result (`{ "error": "denied", ... }`), `Ask`→
  `co_await approver`. Maintains per-session **"allow always for this tool"** cache.
- Integrates by composing: `Session` is handed `with_permissions(standard_toolkit(), …)`.

**Resolved decision (Fork 2):** policy + async approver, chained (rules first; `Ask`
escalates to user).

---

## Sub-project 3 — Standard Agent Toolkit

**Purpose:** a first-class, tested toolkit factory promoting the example into the library.

**Components** (`include/ai/tools/standard/`):

- `ToolSet standard_toolkit(ToolkitOptions opts)` returning:
  - `read_file` — path, optional line range, truncation cap.
  - `write_file` — create/overwrite.
  - `edit_file` — **exact string-replace**: `{path, old_string, new_string, replace_all?}`;
    errors if `old_string` absent or **not unique** (unless `replace_all`). Deterministic.
  - `glob` — find files by pattern.
  - `grep` — content search (ripgrep-style).
  - `bash` / `run_command` — timeout, working_dir, output cap.
- Each tool is a real implementation with unit tests (filesystem ops in temp dirs).

**Resolved decision (Fork 3):** exact-match `str_replace` semantics (matches Claude Code;
deterministic, testable). Fuzzy/unified-diff variants are future.

---

## Sub-project 4 — Session Persistence & Resume

**Purpose:** serialize/resume the *conversation transcript*.

**Components** (`include/ai/session/store.hpp`):

- `class SessionStore` (interface) — `Task<void> save(const Session&)`,
  `Task<std::optional<Session>> load(const std::string& id)`, `Task<std::vector<SessionMeta>> list()`,
  `Task<void> remove(const std::string& id)`.
- `class JsonFileSessionStore` — writes `<dir>/<id>.json` via
  `workflow::serialize_prompt`; metadata = `{id, created_at, updated_at, model, provider,
  token totals, compaction_summary}`. Default dir `.agent/sessions/`.
- `Session::id()`, `Session::metadata()`. Resume = `store.load(id)` then `send(...)`.

**Resolved decision (Fork 4):** JSON files (inspectable, matches Codex/Claude Code);
interface leaves SQLite pluggable later.

---

## Sub-project 5 — Persistent Memory

**Purpose:** cross-session *project knowledge* — retrieved by relevance, agent-writable,
auto-injected on resume; "the agent remembers without relearning." Distinct from
transcript persistence (#4).

**File layout** (`.agent/memory/`): `MEMORY.md` (project knowledge/rules/decisions),
`checkpoint.md` (auto session-state snapshot), `notes.md` (scratch),
`tasks/<id>/progress.md` (per-task logs).

**Components** (`include/ai/memory/`):

- `struct MemoryRecord { std::string id, scope, key, content; std::vector<std::string> tags; boost::json::object metadata; timestamps; };`
- `class MemoryStore` (interface) — `add/update/remove/get/list`. **`MarkdownMemoryStore`**
  v1 (files, zero deps, git-friendly). `SqliteFtsMemoryStore` is a future option.
- `class MemoryRetriever` (interface) — `std::vector<ScoredRecord> query(const std::string& q, int k)`:
  - **`KeywordRetriever`** — inverted index / grep over the markdown files (zero-dep default).
  - **`EmbeddingRetriever`** — uses the SDK's **`embed()`**: embed memory chunks
    (`embed_many`) + the query, cosine-rank. Vectors cached in a sidecar file. **Semantic**
    recall that FTS cannot do — a differentiator leveraging existing SDK capability.
  - **`HybridRetriever`** — keyword ∪ embedding, merged scores.
- `class MemoryContextStrategy : public ContextStrategy` — **decorates an inner
  `ContextStrategy`** (e.g. sliding-window + summarization). On **resume**: inject
  `MEMORY.md` + latest `checkpoint.md` as a system block; **per turn**: retrieve top-K
  relevant records for the user message and inject a compact `## Relevant project memory`
  block within a token budget coordinated with the `ContextWindow`, **then delegate** the
  remaining window/compaction to the inner strategy (memory injection counts toward the
  budget).
- **Memory tools** (permission-gated) — provided by a `memory_tools(MemoryStore&,
  MemoryRetriever&)` factory, composed into the `ToolSet` alongside `standard_toolkit()`
  when memory is enabled: `recall_memory(query)`, `save_memory(scope, content)`,
  `update_checkpoint(state)`, `add_note(text)`, `log_task_progress(task_id, entry)` — the
  agent actively maintains memory.
- `class CheckpointWriter` — a `Session` **post-turn hook** that every *N* turns (or on
  close) runs a small LLM call to refresh `checkpoint.md` from the transcript. The
  "continuously improves itself" loop.

**Resolved decisions (Forks 5–7):** hybrid retrieval (keyword always + embeddings when an
embedding model is configured); markdown files; auto-checkpoint on + throttled. **No new
dependencies.** Cost: embeddings = embed API calls; checkpoint = small LLM calls — both
opt-in/throttled.

---

## Sub-project 6 — Test Coverage

Woven through 1–5 **plus** backfill:

- **Session:** history growth; sliding window trims to fit; summarization summarizes the
  evicted tail (mock `Summarizer`); **tool-call/tool-result pair integrity preserved**
  after trim.
- **Permissions:** allow/deny rules; `Ask`→async approver; "allow always" caching; deny
  returns structured error.
- **Toolkit:** `edit_file` match / unique / error / `replace_all`; read/write/glob/grep in
  temp dirs; bash timeout.
- **Persistence:** JSON round-trip save/load; resume continues history; metadata preserved.
- **Memory:** add/update/query; keyword retriever recall; embedding retriever recall (mock
  embedding model); `MemoryContextStrategy` injects within budget on resume and per-turn;
  `CheckpointWriter` refreshes checkpoint.
- **Backfill:** **MCP stdio fake** (in-memory JSON-RPC pipe → `tools/list`, `tools/call`,
  progress); **OpenAI/Google provider parsing** via `FakeHttpClient` (extend the existing
  pattern); agent-loop edge cases (max_steps, tool error propagation).

---

## Facade — `CodingAgent`

`include/ai/coding_agent.hpp`: bundles `Session` + `SecureToolSet(standard_toolkit())` +
`MemoryContextStrategy` (optional) + provider, exposing `run()` (REPL) and
`send_once(...)`. A Codex-style CLI is ~20 lines. Power users skip it and compose
primitives.

**Resolved decision:** include the facade.

---

## Resolved decisions (all forks)

| # | Decision | Choice |
|---|----------|--------|
| 1 | Compaction default | Sliding window always; summarization opt-in |
| 2 | Permission model | Sync `PermissionPolicy` + async `Approver`, chained |
| 3 | Edit tool | Exact-match `str_replace` (unique unless `replace_all`) |
| 4 | Persistence store | JSON files; `SessionStore` interface (SQLite later) |
| 5 | Memory retrieval | Hybrid (keyword + embeddings) |
| 6 | Memory store | Markdown files; `MemoryStore` interface (SQLite FTS5 later) |
| 7 | Auto-checkpoint | On, throttled (every N turns + on close) |
| 8 | Facade | Include `CodingAgent` |
| — | Dependencies | No new mandatory deps (SQLite optional/future) |

## File layout (new)

```
include/ai/session/         session.hpp, context_strategy.hpp, store.hpp
include/ai/permission/      permission.hpp
include/ai/tools/standard/  standard_toolkit.hpp
include/ai/memory/          memory_store.hpp, retriever.hpp, memory_strategy.hpp, checkpoint.hpp
include/ai/coding_agent.hpp
src/session/ src/permission/ src/tools/standard/ src/memory/
tests/unit/test_session.cpp test_permission.cpp test_standard_toolkit.cpp
            test_session_store.cpp test_memory.cpp (+ MCP/provider backfill)
```

## Build / integration

- Add new sources to `AI_SDK_CORE_SOURCES` in root `CMakeLists.txt`.
- New headers re-exported via `include/ai/ai.hpp`.
- Test files added to `tests/unit/CMakeLists.txt`; provider backfill conditionally links
  provider libs (pattern already established for `test_anthropic_structured.cpp`).
- The `examples/claude_code_clone/` is refactored to use `CodingAgent` (proves the facade).

## Out of scope / future

- SQLite FTS5 / vector-DB backends (interfaces leave room).
- Semantic deduplication of memory; memory decay/forgetting policies.
- Multi-agent / subagent orchestration beyond the `CheckpointWriter` hook.
- Fuzzy/unified-diff edit variants.
- Publishing Rust/Go binding packages (separate effort).
