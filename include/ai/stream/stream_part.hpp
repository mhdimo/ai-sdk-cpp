#pragma once

#include <string>
#include <variant>
#include <optional>
#include <vector>
#include <chrono>
#include <boost/json.hpp>

namespace ai {

enum class FinishReason {
    Stop,
    Length,
    ContentFilter,
    ToolCalls,
    Error,
    Other
};

struct Usage {
    struct InputTokens {
        std::optional<int> total;
        std::optional<int> no_cache;
        std::optional<int> cache_read;
        std::optional<int> cache_write;
    };

    struct OutputTokens {
        std::optional<int> total;
        std::optional<int> text;
        std::optional<int> reasoning;
    };

    InputTokens input_tokens;
    OutputTokens output_tokens;
    std::optional<boost::json::object> raw;

    Usage& operator+=(const Usage& other) {
        if (other.input_tokens.total)
            input_tokens.total = input_tokens.total.value_or(0) + *other.input_tokens.total;
        if (other.output_tokens.total)
            output_tokens.total = output_tokens.total.value_or(0) + *other.output_tokens.total;
        return *this;
    }
};

struct Warning {
    std::string type;
    std::string message;
    std::optional<std::string> details;
};

struct StreamStart {
    std::vector<Warning> warnings;
};

struct TextStart {
    std::string id;
};

struct TextDelta {
    std::string id;
    std::string delta;
};

struct TextEnd {
    std::string id;
};

struct ReasoningStart {
    std::string id;
};

struct ReasoningDelta {
    std::string id;
    std::string delta;
};

struct ReasoningEnd {
    std::string id;
};

struct ToolInputStart {
    std::string id;
    std::string tool_name;
    bool provider_executed = false;
};

struct ToolInputDelta {
    std::string id;
    std::string delta;
};

struct ToolInputEnd {
    std::string id;
};

struct ResponseMetadataPart {
    std::optional<std::string> id;
    std::optional<std::string> model_id;
    std::optional<std::chrono::system_clock::time_point> timestamp;
};

struct FinishPart {
    FinishReason reason;
    Usage usage;
    std::optional<boost::json::object> provider_metadata;
};

struct ErrorPart {
    std::string message;
    std::optional<int> code;
};

struct RawPart {
    boost::json::value raw_value;
};

using StreamPart = std::variant<
    StreamStart, TextStart, TextDelta, TextEnd,
    ReasoningStart, ReasoningDelta, ReasoningEnd,
    ToolInputStart, ToolInputDelta, ToolInputEnd,
    ResponseMetadataPart, FinishPart, ErrorPart, RawPart>;

} // namespace ai
