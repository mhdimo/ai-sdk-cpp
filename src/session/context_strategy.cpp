#include <ai/session/context_strategy.hpp>

#include <algorithm>
#include <utility>

namespace ai {

namespace {

/// Count leading system messages in the prompt.
std::size_t count_leading_system(const Prompt& messages) {
    std::size_t n = 0;
    for (auto& m : messages) {
        if (std::holds_alternative<SystemMessage>(m)) {
            ++n;
        } else {
            break;
        }
    }
    return n;
}

/// Given an index `i` pointing at the start of a turn unit (a UserMessage or
/// an AssistantMessage that introduces tool calls), return the index
/// one-past the end of that turn unit. A turn unit starting at an
/// AssistantMessage consumes itself plus any immediately following
/// ToolMessages whose tool_call_ids it introduced (matched pairs).
/// For any other message type the turn unit is just that single message.
std::size_t turn_unit_end(const Prompt& messages, std::size_t i) {
    if (i >= messages.size()) {
        return messages.size();
    }

    if (auto* asst = std::get_if<AssistantMessage>(&messages[i])) {
        // Gather the tool_call_ids introduced by this assistant message.
        std::unordered_set<std::string> ids;
        for (auto& part : asst->content) {
            if (auto* tc = std::get_if<ToolCallPart>(&part)) {
                ids.insert(tc->tool_call_id);
            }
        }
        std::size_t end = i + 1;
        if (!ids.empty()) {
            // Consume trailing ToolMessages that carry results for these ids.
            while (end < messages.size()) {
                auto* tool = std::get_if<ToolMessage>(&messages[end]);
                if (!tool) break;
                bool belongs = false;
                for (auto& part : tool->content) {
                    if (ids.count(part.tool_call_id)) {
                        belongs = true;
                        break;
                    }
                }
                if (!belongs) break;
                ++end;
            }
        }
        return end;
    }

    // UserMessage, standalone ToolMessage, etc.: single-message unit.
    return i + 1;
}

} // namespace

Prompt evict_turns_from_front(
    Prompt messages,
    const TokenCounter& counter,
    int available_tokens,
    std::size_t protected_tail
) {
    if (available_tokens <= 0) {
        return messages;
    }

    // Preserve leading system messages and the final `protected_tail` messages
    // (the current turn). Evict whole turn units between those two regions,
    // oldest first, until we fit.
    const std::size_t system_count = count_leading_system(messages);
    std::size_t protected_start =
        messages.size() > protected_tail ? messages.size() - protected_tail
                                         : system_count;
    if (protected_start < system_count) {
        protected_start = system_count;
    }

    while (static_cast<int>(counter.count_messages_tokens(messages)) > available_tokens) {
        // Index of the first evictable message: just after the system block,
        // but never inside the protected tail.
        std::size_t start = system_count;
        if (start >= protected_start) {
            break; // nothing left to evict without touching the protected tail
        }

        std::size_t end = turn_unit_end(messages, start);
        if (end <= start) {
            end = start + 1; // safety: always make progress
        }
        // Never split a turn unit: if evicting this unit would cross into the
        // protected tail, stop evicting (keeping tool-call/tool-result pairs
        // intact) rather than orphan one. Over-budget is preferable to a
        // malformed history that providers reject.
        if (end > protected_start) {
            break;
        }
        messages.erase(messages.begin() + start,
                       messages.begin() + end);

        // Recompute the protected region start after shrinking.
        protected_start =
            messages.size() > protected_tail ? messages.size() - protected_tail
                                             : system_count;
        if (protected_start < system_count) {
            protected_start = system_count;
        }
    }

    return messages;
}

Task<Prompt> SummarizationStrategy::manage(
    Prompt history,
    const ContextWindow& window,
    const TokenCounter& counter,
    Summarizer& summarizer
) {
    const int available = window.available_input_tokens();
    if (static_cast<int>(counter.count_messages_tokens(history)) <= available) {
        co_return history;
    }

    const std::size_t system_count = count_leading_system(history);

    // The verbatim tail we keep is `kept_turns_` turn units. Walk backwards
    // from the end to find where the kept region starts.
    std::size_t kept_start = history.size();
    std::size_t turns_kept = 0;
    // Walk turn units from the back, but never cross into the system block.
    while (kept_start > system_count && turns_kept < kept_turns_) {
        // Find the start of the turn unit ending at kept_start.
        std::size_t unit_start = kept_start - 1;
        // If kept_start is at a ToolMessage, roll back to the assistant that
        // introduced its tool calls (so the pair stays together).
        if (std::get_if<ToolMessage>(&history[unit_start])) {
            while (unit_start > system_count &&
                   std::get_if<ToolMessage>(&history[unit_start - 1])) {
                --unit_start;
            }
            // Now move back to the AssistantMessage that produced them.
            if (unit_start > system_count) {
                --unit_start; // the assistant message
            }
        }
        // Move back further past any preceding non-tool messages that belong
        // to this same turn (e.g. the user message is its own unit).
        ++turns_kept;
        kept_start = unit_start;
    }
    if (kept_start < system_count) {
        kept_start = system_count;
    }

    // The evicted middle region is what we summarize.
    Prompt evicted(history.begin() + system_count,
                   history.begin() + kept_start);

    Prompt kept_tail(history.begin() + kept_start, history.end());

    Prompt result;
    // 1. System messages.
    for (std::size_t i = 0; i < system_count; ++i) {
        result.push_back(history[i]);
    }
    // 2. Summary of evicted tail (only if we evicted something and have a
    //    summarizer; otherwise fall back to plain sliding window).
    if (!evicted.empty() && summarizer) {
        // manage() is a coroutine running on the caller's io_context, so we
        // can await the Summarizer directly — it may itself call into the
        // SDK's provider layer (generate_text / a model).
        std::string summary = co_await summarizer(evicted);
        if (!summary.empty()) {
            result.push_back(SystemMessage{
                .content = std::string("## Earlier in this session\n") + summary,
            });
        }
    } else if (!evicted.empty()) {
        // No summarizer available: degrade to sliding-window eviction of the
        // middle region so the window still fits.
        Prompt system_block(history.begin(), history.begin() + system_count);
        Prompt middle(std::move(system_block));
        for (auto& m : kept_tail) middle.push_back(m);
        co_return evict_turns_from_front(
            std::move(middle), counter, available, /*protected_tail=*/1u);
    }

    // 3. Kept verbatim tail.
    for (auto& m : kept_tail) {
        result.push_back(std::move(m));
    }

    // If it still doesn't fit (e.g. summary + tail too large), apply plain
    // sliding-window eviction as a safety net.
    if (static_cast<int>(counter.count_messages_tokens(result)) > available) {
        result = evict_turns_from_front(
            std::move(result), counter, available, /*protected_tail=*/1u);
    }

    co_return result;
}

} // namespace ai
