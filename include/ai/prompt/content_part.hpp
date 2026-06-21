#pragma once

#include <string>
#include <vector>
#include <variant>
#include <optional>
#include <cstdint>
#include <boost/json.hpp>

namespace ai {

struct DataFileData {
    std::vector<uint8_t> data;
};

struct UrlFileData {
    std::string url;
};

struct ReferenceFileData {
    boost::json::object reference;
};

struct TextFileData {
    std::string text;
};

using FileData = std::variant<DataFileData, UrlFileData, ReferenceFileData, TextFileData>;

using ProviderOptions = boost::json::object;

struct TextPart {
    std::string text;
    ProviderOptions provider_options;
};

struct FilePart {
    FileData data;
    std::string media_type;
    std::optional<std::string> filename;
    ProviderOptions provider_options;
};

struct ReasoningPart {
    std::string text;
    std::optional<std::string> signature;
    std::optional<std::string> redacted_data;
    ProviderOptions provider_options;
};

struct ToolCallPart {
    std::string tool_call_id;
    std::string tool_name;
    boost::json::value input;
    bool provider_executed = false;
    ProviderOptions provider_options;
};

struct TextOutput {
    std::string value;
};

struct JsonOutput {
    boost::json::value value;
};

struct ErrorTextOutput {
    std::string value;
};

struct ErrorJsonOutput {
    boost::json::value value;
};

struct ExecutionDenied {
    std::optional<std::string> reason;
};

struct ContentOutputPart {
    std::variant<TextPart, FilePart> part;
};

struct ContentOutput {
    std::vector<ContentOutputPart> value;
};

using ToolResultOutput = std::variant<
    TextOutput, JsonOutput, ErrorTextOutput,
    ErrorJsonOutput, ExecutionDenied, ContentOutput>;

struct ToolResultPart {
    std::string tool_call_id;
    std::string tool_name;
    ToolResultOutput output;
    ProviderOptions provider_options;
};

} // namespace ai
