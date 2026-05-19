#pragma once

#include <ai/model/language_model.hpp>
#include <ai/model/generate_result.hpp>
#include <ai/model/middleware.hpp>
#include <ai/tool/tool_set.hpp>
#include <ai/tool/tool_call.hpp>
#include <ai/tool/tool_choice.hpp>
#include <ai/stream/async_generator.hpp>
#include <ai/util/cancellation.hpp>
#include <string>
#include <vector>
#include <optional>
#include <functional>

namespace ai {

struct StepResult {
    GenerateResult result;
    std::vector<ToolCallResult> tool_results;
    int step_number;
};

struct GenerateTextOptions {
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

    std::function<void(const StepResult&)> on_step_finish;

    // Tool input repair: if tool input JSON is invalid, call this function
    // to attempt repair. Takes the bad input string and error message,
    // returns a corrected JSON string (or throws to abort).
    std::optional<std::function<Task<std::string>(std::string bad_input, std::string error)>> repair_tool_call;

    std::vector<MiddlewarePtr> middleware;
    CancellationToken cancel;
    int max_retries = 2;

    boost::json::object provider_options;
};

struct GenerateTextResult {
    std::string text;
    std::vector<ToolCallResult> tool_calls;
    FinishReason finish_reason;
    Usage usage;
    std::vector<StepResult> steps;
    std::vector<Message> response_messages;
    std::vector<Warning> warnings;

    const StepResult& final_step() const { return steps.back(); }
};

Task<GenerateTextResult> generate_text(GenerateTextOptions options);

} // namespace ai
