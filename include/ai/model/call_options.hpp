#pragma once

#include <ai/prompt/message.hpp>
#include <ai/tool/tool_choice.hpp>
#include <ai/schema/json_schema.hpp>
#include <ai/util/cancellation.hpp>
#include <string>
#include <vector>
#include <optional>
#include <unordered_map>
#include <boost/json.hpp>

namespace ai {

struct ResponseFormat {
    std::string type; // "text" or "json"
    std::optional<schema::JsonSchema> schema;
    std::optional<std::string> name;
    std::optional<std::string> description;
};

struct FunctionTool {
    std::string name;
    std::optional<std::string> description;
    schema::JsonSchema input_schema;
    bool strict = false;
    ProviderOptions provider_options;
};

struct CallOptions {
    Prompt prompt;

    std::optional<int> max_output_tokens;
    std::optional<double> temperature;
    std::optional<double> top_p;
    std::optional<int> top_k;
    std::optional<double> presence_penalty;
    std::optional<double> frequency_penalty;
    std::optional<std::vector<std::string>> stop_sequences;
    std::optional<int> seed;

    std::optional<ResponseFormat> response_format;

    std::vector<FunctionTool> tools;
    std::optional<ToolChoice> tool_choice;

    std::optional<std::string> reasoning;

    std::unordered_map<std::string, std::string> headers;
    CancellationToken cancel;
    bool include_raw_chunks = false;

    boost::json::object provider_options;
};

} // namespace ai
