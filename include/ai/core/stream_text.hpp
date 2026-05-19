#pragma once

#include <ai/core/generate_text.hpp>
#include <ai/model/language_model.hpp>
#include <ai/model/middleware.hpp>
#include <ai/tool/tool_set.hpp>
#include <ai/tool/tool_choice.hpp>
#include <ai/tool/tool_call.hpp>
#include <ai/stream/async_generator.hpp>
#include <ai/stream/stream_part.hpp>
#include <ai/util/cancellation.hpp>
#include <string>
#include <vector>
#include <optional>
#include <functional>

namespace ai {

struct StreamTextOptions {
    LanguageModelPtr model;
    ToolSet tools;
    ToolChoice tool_choice = ToolChoiceAuto{};

    std::optional<std::string> prompt;
    std::optional<std::vector<Message>> messages;
    std::optional<std::string> system;

    std::optional<int> max_output_tokens;
    std::optional<double> temperature;
    std::optional<double> top_p;
    std::optional<int> top_k;
    std::optional<std::vector<std::string>> stop_sequences;

    int max_steps = 1;

    std::function<void(const StreamPart&)> on_chunk;
    std::function<void(const StepResult&)> on_step_finish;

    std::vector<MiddlewarePtr> middleware;
    CancellationToken cancel;
    int max_retries = 2;

    boost::json::object provider_options;
};

struct TextStreamPart {
    std::variant<
        TextDelta,
        ToolInputStart,
        ToolInputDelta,
        ToolInputEnd,
        ReasoningDelta,
        FinishPart,
        ErrorPart
    > part;
};

struct StreamTextResult {
    AsyncGenerator<StreamPart> stream;
    Task<GenerateTextResult> full_result;
};

Task<StreamTextResult> stream_text(StreamTextOptions options);

} // namespace ai
