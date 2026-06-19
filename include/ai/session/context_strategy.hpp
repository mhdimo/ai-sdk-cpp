#pragma once

#include <ai/prompt/message.hpp>
#include <ai/util/token_count.hpp>
#include <ai/stream/async_generator.hpp>
#include <functional>
#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace ai {

/// A callback that compacts a slice of conversation history into a single
/// summary string. Callers wire this to a cheap model (or a mock in tests).
/// Implementations are coroutines so they may call into the SDK's own
/// `generate_text` / provider layer. When summarization is off, the strategy
/// simply does not invoke it.
using Summarizer = std::function<Task<std::string>(const Prompt& to_summarize)>;

/// Strategy interface. A Session owns exactly one ContextStrategy and applies
/// it (via `manage`) before each model call to keep the history within the
/// model's context window. Implementations must never orphan a tool_use from
/// its tool_result (providers reject an unpaired tool message) and must
/// respect `ContextWindow::available_input_tokens()`.
class ContextStrategy {
public:
    virtual ~ContextStrategy() = default;

    /// Return a (possibly trimmed/restructured) copy of `history` that fits
    /// `window` according to `counter`. `summarizer` is used only by
    /// strategies that summarize (it may be null for sliding-window use).
    ///
    /// This is a coroutine so that summarizing strategies can `co_await` the
    /// `Summarizer` (which may itself call into the SDK's provider layer) on
    /// the caller's io_context.
    virtual Task<Prompt> manage(
        Prompt history,
        const ContextWindow& window,
        const TokenCounter& counter,
        Summarizer& summarizer
    ) = 0;
};

// ---------------------------------------------------------------------------
// Internal helpers (exposed in the header so unit tests can exercise them
// directly without going through the full Session).
// ---------------------------------------------------------------------------

/// Collect every tool_call_id referenced by AssistantMessage tool-call parts
/// in the range [first, last). Used to decide which ToolMessage parts must
/// stay paired with their originating assistant turn.
template <typename Iter>
std::unordered_set<std::string> collect_tool_call_ids(Iter first, Iter last) {
    std::unordered_set<std::string> ids;
    for (auto it = first; it != last; ++it) {
        if (auto* asst = std::get_if<AssistantMessage>(&*it)) {
            for (auto& part : asst->content) {
                if (auto* tc = std::get_if<ToolCallPart>(&part)) {
                    ids.insert(tc->tool_call_id);
                }
            }
        }
    }
    return ids;
}

/// True if `msg` references any tool_call_id in `kept_ids` (i.e. it is the
/// matching result for an assistant tool call we are keeping).
inline bool tool_message_has_kept_ids(
    const ToolMessage& msg,
    const std::unordered_set<std::string>& kept_ids
) {
    for (auto& part : msg.content) {
        if (kept_ids.count(part.tool_call_id)) {
            return true;
        }
    }
    return false;
}

/// Removes from the FRONT of `messages` (after any system messages) whole
/// "turn units", where a turn unit is:
///   - one UserMessage, or
///   - one AssistantMessage immediately followed by all ToolMessages whose
///     tool_call_ids it introduced (a matched tool-call/tool-result group).
///
/// Turn units are dropped oldest-first until the remaining messages fit in
/// `available_tokens` or only the protected tail remains. The system messages
/// at the front are always preserved, and `protected_tail` messages at the end
/// are never evicted (the current/final turn). Because an assistant tool-call
/// group and its tool results are dropped *together*, this never orphans a
/// tool_use from its tool_result.
Prompt evict_turns_from_front(
    Prompt messages,
    const TokenCounter& counter,
    int available_tokens,
    std::size_t protected_tail
);

// ---------------------------------------------------------------------------
// SlidingWindowStrategy (default)
// ---------------------------------------------------------------------------

/// The default compaction strategy: protect the leading system messages and
/// the final/current turn, then drop the oldest complete turn units until the
/// history fits. Tool-call/tool-result groups are kept or evicted as matched
/// pairs.
class SlidingWindowStrategy : public ContextStrategy {
public:
    Task<Prompt> manage(
        Prompt history,
        const ContextWindow& window,
        const TokenCounter& counter,
        Summarizer& /*summarizer*/
    ) override {
        const int available = window.available_input_tokens();
        if ((int)counter.count_messages_tokens(history) <= available) {
            co_return history;
        }
        // Protect the system block and the most recent turn so the agent still
        // sees "what the user just asked".
        co_return evict_turns_from_front(
            std::move(history), counter, available, /*protected_tail=*/1u);
    }
};

// ---------------------------------------------------------------------------
// SummarizationStrategy (opt-in)
// ---------------------------------------------------------------------------

/// On overflow this strategy evicts the oldest turn units (as matched pairs),
/// summarizes the evicted tail via the `Summarizer` callback into one compact
/// `## Earlier in this session` message kept immediately after the system
/// prompt, and keeps the last `kept_turns` turns verbatim. If `summarizer`
/// is null it degrades to plain sliding-window behavior.
class SummarizationStrategy : public ContextStrategy {
public:
    explicit SummarizationStrategy(std::size_t kept_turns = 4)
        : kept_turns_(kept_turns) {}

    Task<Prompt> manage(
        Prompt history,
        const ContextWindow& window,
        const TokenCounter& counter,
        Summarizer& summarizer
    ) override;

private:
    std::size_t kept_turns_;
};

} // namespace ai
