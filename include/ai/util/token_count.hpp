#pragma once

#include <ai/prompt/message.hpp>
#include <string_view>
#include <vector>

namespace ai {

class TokenCounter {
public:
    virtual ~TokenCounter() = default;
    virtual int count_tokens(std::string_view text) const = 0;
    virtual int count_message_tokens(const Message& msg) const = 0;
    virtual int count_messages_tokens(const Prompt& messages) const = 0;
};

class ApproximateTokenCounter : public TokenCounter {
public:
    int count_tokens(std::string_view text) const override;
    int count_message_tokens(const Message& msg) const override;
    int count_messages_tokens(const Prompt& messages) const override;
};

struct ContextWindow {
    int max_tokens;
    int reserved_output_tokens;

    int available_input_tokens() const { return max_tokens - reserved_output_tokens; }

    bool fits(const Prompt& messages, const TokenCounter& counter) const {
        return counter.count_messages_tokens(messages) <= available_input_tokens();
    }

    Prompt trim_to_fit(Prompt messages, const TokenCounter& counter) const;
};

} // namespace ai
