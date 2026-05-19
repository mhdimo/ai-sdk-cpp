#include <ai/core/generate_object.hpp>
#include <ai/util/json.hpp>
#include <ai/error/ai_error.hpp>

namespace ai {

Task<GenerateObjectResult> generate_object(GenerateObjectOptions options) {
    Prompt prompt;

    if (options.system) {
        prompt.push_back(SystemMessage{.content = *options.system});
    }

    if (options.messages) {
        for (auto& msg : *options.messages) {
            prompt.push_back(msg);
        }
    } else if (options.prompt) {
        UserContent content;
        content.push_back(TextPart{.text = *options.prompt});
        prompt.push_back(UserMessage{.content = std::move(content)});
    }

    CallOptions call_opts;
    call_opts.prompt = prompt;
    call_opts.max_output_tokens = options.max_output_tokens;
    call_opts.temperature = options.temperature;
    call_opts.top_p = options.top_p;
    call_opts.cancel = options.cancel;
    call_opts.provider_options = options.provider_options;

    call_opts.response_format = ResponseFormat{
        .type = "json",
        .schema = options.schema,
        .name = options.schema_name,
        .description = options.schema_description,
    };

    auto result = co_await options.model->do_generate(std::move(call_opts));

    auto text = result.text();
    if (text.empty()) {
        throw error::NoOutputGeneratedError("Model did not generate any output");
    }

    auto parsed = ai::json::parse(text);

    schema::Validator validator(options.schema);
    auto validation = validator.validate(parsed);
    if (!validation.success) {
        std::string error_msg = "Generated object does not match schema: ";
        for (auto& e : validation.errors) {
            error_msg += e + "; ";
        }
        throw error::TypeValidationError(std::move(error_msg), parsed);
    }

    co_return GenerateObjectResult{
        .object = std::move(parsed),
        .finish_reason = result.finish_reason,
        .usage = result.usage,
        .warnings = std::move(result.warnings),
    };
}

namespace {

std::optional<boost::json::value> try_parse_partial(const std::string& buffer) {
    // Try direct parse first
    boost::system::error_code ec;
    auto val = boost::json::parse(buffer, ec);
    if (!ec) return val;

    // Try appending closing braces to recover partial objects
    for (auto suffix : {"}", "}}", "]}}", "\"}", "\"]}"}) {
        val = boost::json::parse(buffer + suffix, ec);
        if (!ec) return val;
    }
    return std::nullopt;
}

} // namespace

Task<StreamObjectResult> stream_object(StreamObjectOptions options) {
    Prompt prompt;
    if (options.system) {
        prompt.push_back(SystemMessage{.content = *options.system});
    }
    if (options.messages) {
        for (auto& msg : *options.messages) prompt.push_back(msg);
    } else if (options.prompt) {
        UserContent content;
        content.push_back(TextPart{.text = *options.prompt});
        prompt.push_back(UserMessage{.content = std::move(content)});
    }

    CallOptions call_opts;
    call_opts.prompt = prompt;
    call_opts.max_output_tokens = options.max_output_tokens;
    call_opts.temperature = options.temperature;
    call_opts.cancel = options.cancel;
    call_opts.provider_options = options.provider_options;
    call_opts.response_format = ResponseFormat{
        .type = "json",
        .schema = options.schema,
        .name = options.schema_name,
        .description = options.schema_description,
    };

    auto stream_result = co_await options.model->do_stream(std::move(call_opts));

    auto gen = [](AsyncGenerator<StreamPart> stream, schema::JsonSchema schema)
        -> AsyncGenerator<boost::json::value> {
        std::string buffer;
        boost::json::value last_valid;

        while (auto part = co_await stream.next()) {
            if (auto* delta = std::get_if<TextDelta>(&*part)) {
                buffer += delta->delta;
                auto parsed = try_parse_partial(buffer);
                if (parsed && *parsed != last_valid) {
                    last_valid = *parsed;
                    co_yield last_valid;
                }
            }
        }

        // Final yield with complete parsed object
        boost::system::error_code ec;
        auto final_val = boost::json::parse(buffer, ec);
        if (!ec && final_val != last_valid) {
            co_yield final_val;
        }
    }(std::move(stream_result.stream), options.schema);

    co_return StreamObjectResult{
        .partial_object_stream = std::move(gen),
        .object = boost::json::value(),
        .usage = {},
    };
}

} // namespace ai
