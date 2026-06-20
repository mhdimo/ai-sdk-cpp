# Native Agents — API reference

ai-sdk-cpp's high-level building blocks for CLI / app agents (Codex / Claude
Code class). Everything is composable; `CodingAgent` bundles the common case.

All types live in `ai::` (memory in `ai::memory::`, mcp in `ai::mcp::`) and are
re-exported from `<ai/ai.hpp>`.

## CodingAgent (facade)

The one-liner entry point: a context-managed session driving a tool-loop agent
with the standard toolkit, optional permission gating, and optional persistent
memory.

```cpp
#include <ai/ai.hpp>
#include <ai/providers/anthropic/anthropic.hpp>

boost::asio::io_context ioc;
auto model = ai::providers::anthropic::create_anthropic({.io_context = ioc})
                 ->language_model("claude-sonnet-4-5");

ai::CodingAgentOptions opts;
opts.model = model;
opts.instructions = "You are a careful coding assistant.";
opts.enable_memory = true;            // opt-in cross-session memory (.agent/memory)
// opts.permission_policy = my_policy; // optional dangerous-tool gating
ai::CodingAgent agent(opts);

auto task = agent.send("Add a unit test for foo()");
task.start();
while (!task.done()) ioc.run_one();
std::cout << task.get().text << "\n";
```

## Session & context management — `ai::Session`

Owns conversation history across turns and runs a stateless `Agent` on a
context-managed slice. Default strategy is `SlidingWindowStrategy`
(tool-call/tool-result pair-safe); `SummarizationStrategy` adds compaction.

```cpp
ai::Session session(agent);            // sensible defaults
session.set_system("You are helpful.");
auto r = co_await session.send("What is 2+2?");  // appends user + assistant turns
session.metadata();                    // turns, token totals, compaction_summary
session.snapshot(); / session.restore(snap);      // for persistence
session.set_on_turn_finish(checkpoint_hook);      // e.g. memory checkpoint writer
```

## Permission / approval hooks — `ai::with_permissions`

Wraps a `ToolSet` so each tool's `execute` is gated by a sync `PermissionPolicy`
(+ optional async `Approver`). `Ask` with no approver fails closed.

```cpp
ai::PermissionPolicy policy = [](auto& tool, auto& input) {
    if (tool == "read_file") return ai::PermissionDecision::Allow;
    if (tool == "bash")      return ai::PermissionDecision::Ask;
    return ai::PermissionDecision::Deny;
};
ai::ToolSet gated = ai::with_permissions(ai::standard_toolkit(), policy, approver);
```

## Standard toolkit — `ai::standard_toolkit`

`read_file`, `write_file`, **`edit_file`** (exact `str_replace`; unique unless
`replace_all`), `glob`, `grep`, `bash`. Returns a `ToolSet`.

## Session persistence — `ai::JsonFileSessionStore`

`SessionStore` interface; the JSON store serializes the full conversation
(text + tool calls + tool results) so tool conversations resume correctly.

```cpp
ai::JsonFileSessionStore store(".agent/sessions");
co_await store.save(session.snapshot());
auto snap = co_await store.load(session.id());   // std::optional
```

## Persistent memory — `ai::memory::`

Cross-session project knowledge, retrieved by relevance and auto-injected.

- `MarkdownMemoryStore` — human-editable per-record markdown (scopes:
  `project` / `checkpoint` / `scratch` / `task`).
- Retrievers: `KeywordRetriever`, `EmbeddingRetriever` (uses the SDK's own
  `embed()` for semantic recall), `HybridRetriever`.
- `MemoryContextStrategy` — a `ContextStrategy` that injects project memory +
  checkpoint + top-K relevant records within a token budget, then delegates to
  an inner window/compaction strategy.
- `memory_tools()` — `recall_memory` / `save_memory` / `add_note` /
  `update_checkpoint` / `log_task_progress`.
- `make_checkpoint_writer()` — a `Session` post-turn hook that periodically
  summarizes the transcript into a checkpoint record.

```cpp
auto store = std::make_shared<ai::memory::MarkdownMemoryStore>(".agent/memory");
auto retr = std::make_shared<ai::memory::HybridRetriever>(
    std::make_shared<ai::memory::KeywordRetriever>(*store),
    std::make_shared<ai::memory::EmbeddingRetriever>(*store, embedding_model));
auto strat = std::make_shared<ai::memory::MemoryContextStrategy>(
    store, retr, std::make_shared<ai::SlidingWindowStrategy>());
```

## MCP client — `ai::mcp::McpClient`

JSON-RPC 2.0 over a pluggable `Transport`: `StdioTransport` (subprocess,
NDJSON), `StreamableHttpTransport` (MCP 2025, session-aware), `InMemoryTransport`
(tests). Supports tools, resources, prompts, server-initiated sampling, and
progress notifications.

```cpp
ai::mcp::McpServerConfig cfg{.transport="stdio", .command="npx",
                             .args={"-y","@modelcontextprotocol/server-filesystem","."}};
ai::mcp::McpClient client(cfg);
co_await client.connect();
auto tools = co_await client.list_tools();
ai::ToolSet ts = ai::mcp::mcp_tools_to_toolset(client_shared, tools);  // drop into an agent
```
