#pragma once

#include <ai/schema/json_schema.hpp>
#include <ai/stream/async_generator.hpp>
#include <ai/prompt/message.hpp>
#include <ai/util/cancellation.hpp>
#include <boost/json.hpp>
#include <string>
#include <optional>
#include <functional>

namespace ai {

struct ToolExecutionContext {
    std::string tool_call_id;
    std::string tool_name;
    std::vector<Message> messages;
    CancellationToken cancel;
};

struct ToolDefinition {
    std::string name;
    std::optional<std::string> description;
    schema::JsonSchema input_schema;
    bool strict = false;

    using ExecuteFn = std::function<Task<boost::json::value>(
        boost::json::value input,
        ToolExecutionContext ctx
    )>;
    std::optional<ExecuteFn> execute;
};

inline ToolDefinition tool(
    std::string name,
    schema::JsonSchema input_schema,
    std::string description,
    ToolDefinition::ExecuteFn execute
) {
    return ToolDefinition{
        .name = std::move(name),
        .description = std::move(description),
        .input_schema = std::move(input_schema),
        .strict = false,
        .execute = std::move(execute),
    };
}

inline ToolDefinition tool(
    std::string name,
    schema::JsonSchema input_schema,
    std::string description
) {
    return ToolDefinition{
        .name = std::move(name),
        .description = std::move(description),
        .input_schema = std::move(input_schema),
        .strict = false,
        .execute = std::nullopt,
    };
}

} // namespace ai
