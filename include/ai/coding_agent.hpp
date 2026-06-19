#pragma once

#include <ai/agent/tool_loop_agent.hpp>
#include <ai/session/session.hpp>
#include <ai/permission/permission.hpp>
#include <ai/tools/standard/standard_toolkit.hpp>
#include <ai/memory/memory.hpp>
#include <ai/model/language_model.hpp>
#include <ai/model/embedding_model.hpp>

#include <filesystem>
#include <string>

namespace ai {

/// Options for the high-level CodingAgent facade.
struct CodingAgentOptions {
    LanguageModelPtr model;
    std::string instructions;
    ContextWindow window{.max_tokens = 128 * 1024, .reserved_output_tokens = 4096};
    int max_steps = 50;

    ToolkitOptions toolkit_options;
    PermissionPolicy permission_policy;  ///< optional; empty = allow all
    Approver approver;                   ///< optional interactive approvals

    /// Persistent cross-session memory (opt-in).
    bool enable_memory = false;
    std::filesystem::path memory_dir = ".agent/memory";
    EmbeddingModelPtr embedding_model;  ///< optional; enables semantic retrieval
    Summarizer checkpoint_summarizer;   ///< optional; defaults to one from `model`
    int checkpoint_every_n_turns = 5;
};

/// Convenience facade that bundles everything needed for a Codex/Claude-Code
/// style CLI agent: a context-managed Session driving a tool-loop agent with
/// the standard toolkit, optional permission gating, and optional persistent
/// memory (auto-injected context + auto-checkpointing). Power users can skip
/// this and compose the primitives directly.
class CodingAgent {
public:
    explicit CodingAgent(CodingAgentOptions opts);

    /// Send a user message through the managed session and return the result.
    Task<GenerateTextResult> send(std::string user_msg, CancellationToken cancel = {});

    const Session& session() const { return session_; }
    Session& session() { return session_; }

    const std::shared_ptr<memory::MemoryStore>& memory_store() const { return store_; }

private:
    LanguageModelPtr model_;
    std::shared_ptr<memory::MemoryStore> store_;
    std::shared_ptr<memory::MemoryRetriever> retriever_;
    ToolLoopAgent agent_;
    Session session_;
};

} // namespace ai
