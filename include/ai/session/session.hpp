#pragma once

#include <ai/agent/agent.hpp>
#include <ai/core/generate_text.hpp>
#include <ai/core/stream_text.hpp>
#include <ai/prompt/message.hpp>
#include <ai/session/context_strategy.hpp>
#include <ai/stream/async_generator.hpp>
#include <ai/util/cancellation.hpp>
#include <ai/util/token_count.hpp>

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace ai {

/// Lightweight bookkeeping exposed by Session for UIs / persistence.
struct SessionMetadata {
    int turns = 0;                ///< number of completed send()/send_stream() turns
    int total_input_tokens = 0;   ///< cumulative input tokens reported by the model
    int total_output_tokens = 0;  ///< cumulative output tokens reported by the model
    std::optional<std::string> compaction_summary;  ///< last evicted-tail summary, if any
};

/// Options for constructing a Session.
struct SessionOptions {
    /// Defaults applied when not overridden by an explicit Session constructor.
    ContextWindow window{.max_tokens = 128 * 1024, .reserved_output_tokens = 4096};
    std::shared_ptr<TokenCounter> counter;
    std::shared_ptr<ContextStrategy> strategy;
    /// Optional summarizer; only consulted by strategies that summarize.
    Summarizer summarizer;
};

/// Serializable snapshot of a Session: enough to persist and resume a
/// conversation (id, full message history, bookkeeping, model/provider refs).
struct SessionSnapshot {
    std::string id;
    Prompt history;
    SessionMetadata metadata;
    std::string model_id;      ///< optional, for display
    std::string provider_id;   ///< optional, for display
};

/// Owns conversation `history_` across turns and runs a stateless `Agent` on
/// a context-managed slice of it. Each `send` appends the user message,
/// applies the configured ContextStrategy (sliding-window by default),
/// invokes the agent, appends the turn's assistant + tool messages, and
/// returns the result.
class Session {
public:
    /// Full-control constructor: bring your own window, counter, and strategy.
    Session(
        Agent& agent,
        ContextWindow window,
        std::shared_ptr<TokenCounter> counter,
        std::shared_ptr<ContextStrategy> strategy,
        Summarizer summarizer = {}
    );

    /// Convenience constructor using sensible defaults: an
    /// ApproximateTokenCounter and a SlidingWindowStrategy, plus the given
    /// context window.
    Session(Agent& agent, ContextWindow window = {});

    ~Session();
    Session(const Session&) = delete;
    Session& operator=(const Session&) = delete;
    Session(Session&&) noexcept;
    Session& operator=(Session&&) noexcept;

    /// Send a user message, run the agent on the managed history, append the
    /// resulting assistant + tool messages, and return the result.
    Task<GenerateTextResult> send(
        std::string user_msg,
        CancellationToken cancel = {}
    );

    /// Streaming variant. Appends the user message, applies the context
    /// strategy, and returns a StreamTextResult whose `stream` and
    /// `full_result` the caller drives as usual. NOTE: because the caller
    /// drains the stream asynchronously, the streamed assistant turn is NOT
    /// auto-appended to history; after draining, call add_assistant(...) if you
    /// need the reply recorded for the next turn.
    Task<StreamTextResult> send_stream(
        std::string user_msg,
        CancellationToken cancel = {}
    );

    /// Manually append a user message without running the agent.
    void add_user(std::string text);
    /// Manually append an assistant message without running the agent.
    void add_assistant(std::string text);

    /// Seed or replace the leading system message. If a SystemMessage is
    /// already at the front it is replaced in place; otherwise one is prepended.
    void set_system(std::string text);

    /// The full managed conversation history (post-compaction).
    const Prompt& history() const& { return history_; }
    Prompt history() const&& = delete;

    /// Bookkeeping metadata.
    SessionMetadata metadata() const { return metadata_; }

    /// Stable session id (auto-generated at construction; overridable).
    const std::string& id() const { return id_; }
    void set_id(std::string id) { id_ = std::move(id); }

    /// Capture/restore the full conversation state for persistence + resume.
    /// `model_id`/`provider_id` are informational (the Agent is NOT serialized).
    SessionSnapshot snapshot(
        std::string model_id = {},
        std::string provider_id = {}
    ) const;
    void restore(SessionSnapshot snapshot);

    const ContextWindow& window() const { return window_; }

    /// Optional post-turn hook (e.g. a checkpoint writer). Invoked at the end
    /// of `send()` with the session state; errors are swallowed (best-effort).
    void set_on_turn_finish(std::function<Task<void>(const Session&)> hook) {
        on_turn_finish_ = std::move(hook);
    }

    /// Manually fire the on_turn_finish hook (e.g. after appending a streamed
    /// turn via send_stream + add_assistant). Best-effort: errors swallowed.
    Task<void> fire_turn_finish();

private:
    Agent* agent_;
    ContextWindow window_;
    std::shared_ptr<TokenCounter> counter_;
    std::shared_ptr<ContextStrategy> strategy_;
    Summarizer summarizer_;
    Prompt history_;
    SessionMetadata metadata_{};
    std::string id_;
    std::function<Task<void>(const Session&)> on_turn_finish_;
};

} // namespace ai
