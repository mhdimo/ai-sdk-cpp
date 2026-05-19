#include <ai/util/token_count.hpp>
#include <ai/prompt/content_part.hpp>
#include <boost/json.hpp>
#include <algorithm>

namespace ai {

namespace {

int text_to_tokens(std::string_view text) {
    return std::max(1, (int)(text.size() + 3) / 4);
}

int content_tokens(const UserContent& content) {
    int total = 0;
    for (auto& part : content) {
        if (auto* t = std::get_if<TextPart>(&part)) {
            total += text_to_tokens(t->text);
        }
    }
    return total;
}

int assistant_content_tokens(const AssistantContent& content) {
    int total = 0;
    for (auto& part : content) {
        if (auto* t = std::get_if<TextPart>(&part)) {
            total += text_to_tokens(t->text);
        } else if (auto* tc = std::get_if<ToolCallPart>(&part)) {
            total += text_to_tokens(tc->tool_name);
            total += text_to_tokens(boost::json::serialize(tc->input));
        }
    }
    return total;
}

} // namespace

int ApproximateTokenCounter::count_tokens(std::string_view text) const {
    return text_to_tokens(text);
}

int ApproximateTokenCounter::count_message_tokens(const Message& msg) const {
    int overhead = 4; // per-message overhead
    return overhead + std::visit([](auto& m) -> int {
        using T = std::decay_t<decltype(m)>;
        if constexpr (std::is_same_v<T, SystemMessage>) {
            return text_to_tokens(m.content);
        } else if constexpr (std::is_same_v<T, UserMessage>) {
            return content_tokens(m.content);
        } else if constexpr (std::is_same_v<T, AssistantMessage>) {
            return assistant_content_tokens(m.content);
        } else if constexpr (std::is_same_v<T, ToolMessage>) {
            int total = 0;
            for (auto& r : m.content) {
                total += text_to_tokens(r.tool_name) + 10;
                std::visit([&](auto& output) {
                    using O = std::decay_t<decltype(output)>;
                    if constexpr (std::is_same_v<O, TextOutput>) {
                        total += text_to_tokens(output.value);
                    } else if constexpr (std::is_same_v<O, JsonOutput>) {
                        total += text_to_tokens(boost::json::serialize(output.value));
                    } else if constexpr (std::is_same_v<O, ErrorTextOutput>) {
                        total += text_to_tokens(output.value);
                    } else if constexpr (std::is_same_v<O, ErrorJsonOutput>) {
                        total += text_to_tokens(boost::json::serialize(output.value));
                    } else {
                        total += 20;
                    }
                }, r.output);
            }
            return total;
        }
        return 0;
    }, msg);
}

int ApproximateTokenCounter::count_messages_tokens(const Prompt& messages) const {
    int total = 3; // conversation overhead
    for (auto& msg : messages) {
        total += count_message_tokens(msg);
    }
    return total;
}

Prompt ContextWindow::trim_to_fit(Prompt messages, const TokenCounter& counter) const {
    int available = available_input_tokens();

    while (counter.count_messages_tokens(messages) > available && messages.size() > 1) {
        // Find first non-system message and remove it
        auto it = messages.begin();
        for (; it != messages.end(); ++it) {
            if (!std::holds_alternative<SystemMessage>(*it)) break;
        }
        if (it == messages.end()) break;
        messages.erase(it);
    }

    return messages;
}

} // namespace ai
