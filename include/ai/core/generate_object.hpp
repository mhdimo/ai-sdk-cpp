#pragma once

#include <ai/model/language_model.hpp>
#include <ai/model/middleware.hpp>
#include <ai/schema/json_schema.hpp>
#include <ai/schema/validator.hpp>
#include <ai/stream/async_generator.hpp>
#include <ai/stream/stream_part.hpp>
#include <ai/util/cancellation.hpp>
#include <boost/json.hpp>
#include <string>
#include <optional>
#include <vector>
#include <memory>

namespace ai {

struct GenerateObjectOptions {
    LanguageModelPtr model;
    schema::JsonSchema schema;
    std::optional<std::string> schema_name;
    std::optional<std::string> schema_description;

    std::optional<std::string> prompt;
    std::optional<std::vector<Message>> messages;
    std::optional<std::string> system;

    std::optional<int> max_output_tokens;
    std::optional<double> temperature;
    std::optional<double> top_p;

    std::vector<MiddlewarePtr> middleware;
    CancellationToken cancel;
    int max_retries = 2;

    boost::json::object provider_options;
};

struct GenerateObjectResult {
    boost::json::value object;
    FinishReason finish_reason;
    Usage usage;
    std::vector<Warning> warnings;
};

struct StreamObjectOptions {
    LanguageModelPtr model;
    schema::JsonSchema schema;
    std::optional<std::string> schema_name;
    std::optional<std::string> schema_description;

    std::optional<std::string> prompt;
    std::optional<std::vector<Message>> messages;
    std::optional<std::string> system;

    std::optional<int> max_output_tokens;
    std::optional<double> temperature;
    std::optional<double> top_p;

    std::vector<MiddlewarePtr> middleware;
    CancellationToken cancel;
    int max_retries = 2;

    boost::json::object provider_options;
};

/// Final state of a stream_object() call, populated incrementally as the
/// partial-object stream is drained. Read these fields *after* the consumer has
/// finished iterating partial_object_stream.
struct StreamObjectFinalState {
    boost::json::value object;
    Usage usage;
    FinishReason finish_reason = FinishReason::Other;
    bool had_output = false;
    bool validated = false;
};

struct StreamObjectResult {
    AsyncGenerator<boost::json::value> partial_object_stream;
    std::shared_ptr<StreamObjectFinalState> final_state;
};

Task<GenerateObjectResult> generate_object(GenerateObjectOptions options);
Task<StreamObjectResult> stream_object(StreamObjectOptions options);

} // namespace ai
