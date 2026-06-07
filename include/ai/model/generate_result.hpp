#pragma once

#include <ai/stream/stream_part.hpp>
#include <ai/prompt/content_part.hpp>
#include <string>
#include <vector>
#include <optional>
#include <variant>
#include <boost/json.hpp>

namespace ai {

struct TextContent {
    std::string text;
};

struct ReasoningContent {
    std::string text;
};

struct ToolCallContent {
    std::string tool_call_id;
    std::string tool_name;
    std::string input; // JSON string
};

struct FileContent {
    FileData data;
    std::string media_type;
    std::optional<std::string> filename;
};

using Content = std::variant<TextContent, ReasoningContent, ToolCallContent, FileContent>;

struct RequestMetadata {
    std::optional<boost::json::value> body;
};

struct ResponseMetadata {
    std::optional<std::unordered_map<std::string, std::string>> headers;
    std::optional<boost::json::value> body;
    std::optional<std::string> id;
    std::optional<std::string> model_id;
};

struct GenerateResult {
    std::vector<Content> content;
    FinishReason finish_reason;
    Usage usage;
    std::vector<Warning> warnings;
    std::optional<boost::json::object> provider_metadata;
    std::optional<RequestMetadata> request;
    std::optional<ResponseMetadata> response;

    std::string text() const {
        std::string result;
        for (auto& c : content) {
            if (auto* t = std::get_if<TextContent>(&c)) {
                result += t->text;
            }
        }
        return result;
    }

    std::vector<ToolCallContent> tool_calls() const {
        std::vector<ToolCallContent> calls;
        for (auto& c : content) {
            if (auto* tc = std::get_if<ToolCallContent>(&c)) {
                calls.push_back(*tc);
            }
        }
        return calls;
    }
};

} // namespace ai
