#pragma once

#include <ai/prompt/content_part.hpp>
#include <string>
#include <vector>
#include <variant>

namespace ai {

using UserContent = std::vector<std::variant<TextPart, FilePart>>;
using AssistantContent = std::vector<std::variant<TextPart, FilePart, ReasoningPart, ToolCallPart, ToolResultPart>>;
using ToolContent = std::vector<ToolResultPart>;

struct SystemMessage {
    std::string content;
    ProviderOptions provider_options;
};

struct UserMessage {
    UserContent content;
    ProviderOptions provider_options;
};

struct AssistantMessage {
    AssistantContent content;
    ProviderOptions provider_options;
};

struct ToolMessage {
    ToolContent content;
    ProviderOptions provider_options;
};

using Message = std::variant<SystemMessage, UserMessage, AssistantMessage, ToolMessage>;
using Prompt = std::vector<Message>;

} // namespace ai
