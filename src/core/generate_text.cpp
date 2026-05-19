#include <ai/core/generate_text.hpp>
#include <ai/model/wrap_language_model.hpp>
#include <ai/util/json.hpp>
#include <ai/error/ai_error.hpp>

namespace ai {

namespace {

Prompt build_prompt(const GenerateTextOptions& opts) {
    Prompt prompt;

    if (opts.system) {
        prompt.push_back(SystemMessage{.content = *opts.system});
    }

    if (opts.messages) {
        for (auto& msg : *opts.messages) {
            prompt.push_back(msg);
        }
    } else if (opts.prompt) {
        UserContent content;
        content.push_back(TextPart{.text = *opts.prompt});
        prompt.push_back(UserMessage{.content = std::move(content)});
    }

    return prompt;
}

CallOptions build_call_options(const GenerateTextOptions& opts, const Prompt& prompt) {
    CallOptions call_opts;
    call_opts.prompt = prompt;
    call_opts.max_output_tokens = opts.max_output_tokens;
    call_opts.temperature = opts.temperature;
    call_opts.top_p = opts.top_p;
    call_opts.top_k = opts.top_k;
    call_opts.stop_sequences = opts.stop_sequences;
    call_opts.cancel = opts.cancel;
    call_opts.provider_options = opts.provider_options;

    if (!opts.tools.empty()) {
        for (auto* tool : opts.tools.all()) {
            call_opts.tools.push_back(FunctionTool{
                .name = tool->name,
                .description = tool->description,
                .input_schema = tool->input_schema,
                .strict = tool->strict,
            });
        }
        call_opts.tool_choice = opts.tool_choice;
    }

    return call_opts;
}

void append_assistant_message(Prompt& prompt, const GenerateResult& result) {
    AssistantContent content;
    for (auto& c : result.content) {
        if (auto* text = std::get_if<TextContent>(&c)) {
            content.push_back(TextPart{.text = text->text});
        } else if (auto* reasoning = std::get_if<ReasoningContent>(&c)) {
            content.push_back(ReasoningPart{.text = reasoning->text});
        } else if (auto* tc = std::get_if<ToolCallContent>(&c)) {
            auto input = ai::json::safe_parse(tc->input);
            content.push_back(ToolCallPart{
                .tool_call_id = tc->tool_call_id,
                .tool_name = tc->tool_name,
                .input = input.value_or(boost::json::value(tc->input)),
            });
        }
    }
    prompt.push_back(AssistantMessage{.content = std::move(content)});
}

void append_tool_results(Prompt& prompt, const std::vector<ToolCallResult>& results) {
    ToolContent content;
    for (auto& r : results) {
        ToolResultOutput output;
        if (r.is_error) {
            output = ErrorJsonOutput{.value = r.output};
        } else {
            output = JsonOutput{.value = r.output};
        }
        content.push_back(ToolResultPart{
            .tool_call_id = r.tool_call_id,
            .tool_name = r.tool_name,
            .output = std::move(output),
        });
    }
    prompt.push_back(ToolMessage{.content = std::move(content)});
}

Task<std::vector<ToolCallResult>> execute_tools(
    const std::vector<ToolCallContent>& tool_calls,
    const ToolSet& tools,
    const Prompt& messages,
    CancellationToken cancel,
    const std::optional<std::function<Task<std::string>(std::string, std::string)>>& repair_tool_call
) {
    std::vector<ToolCallResult> results;
    results.reserve(tool_calls.size());

    for (auto& tc : tool_calls) {
        auto* tool_def = tools.find(tc.tool_name);
        if (!tool_def || !tool_def->execute) {
            results.push_back(ToolCallResult{
                .tool_call_id = tc.tool_call_id,
                .tool_name = tc.tool_name,
                .input = ai::json::safe_parse(tc.input).value_or(boost::json::value(tc.input)),
                .output = boost::json::value("Tool not found or not executable: " + tc.tool_name),
                .is_error = true,
            });
            continue;
        }

        // Attempt to parse tool input JSON, with optional repair
        auto parsed_input = ai::json::safe_parse(tc.input);
        if (!parsed_input && repair_tool_call) {
            // Tool input JSON is invalid; attempt repair
            try {
                std::string error_msg = "Invalid JSON in tool call input for tool '" +
                    tc.tool_name + "': failed to parse input";
                auto repaired = co_await (*repair_tool_call)(tc.input, error_msg);
                parsed_input = ai::json::safe_parse(repaired);
                if (!parsed_input) {
                    // Repair returned non-JSON; use the repaired string as-is
                    parsed_input = boost::json::value(repaired);
                }
            } catch (const std::exception&) {
                // Repair failed; fall through with original input as string value
                parsed_input = boost::json::value(tc.input);
            }
        }

        auto input = parsed_input.value_or(boost::json::value(tc.input));
        ToolExecutionContext ctx{
            .tool_call_id = tc.tool_call_id,
            .tool_name = tc.tool_name,
            .messages = std::vector<Message>(messages.begin(), messages.end()),
            .cancel = cancel,
        };

        try {
            auto output = co_await (*tool_def->execute)(input, std::move(ctx));
            results.push_back(ToolCallResult{
                .tool_call_id = tc.tool_call_id,
                .tool_name = tc.tool_name,
                .input = input,
                .output = std::move(output),
                .is_error = false,
            });
        } catch (const std::exception& e) {
            results.push_back(ToolCallResult{
                .tool_call_id = tc.tool_call_id,
                .tool_name = tc.tool_name,
                .input = input,
                .output = boost::json::value(std::string("Error: ") + e.what()),
                .is_error = true,
            });
        }
    }

    co_return results;
}

} // namespace

Task<GenerateTextResult> generate_text(GenerateTextOptions options) {
    // Apply middleware to the model if any are specified
    auto model = options.middleware.empty()
        ? options.model
        : wrap_language_model(options.model, options.middleware);

    auto prompt = build_prompt(options);
    std::vector<StepResult> steps;
    Usage total_usage{};
    std::vector<ToolCallResult> all_tool_calls;

    for (int step = 0; step < options.max_steps; ++step) {
        options.cancel.throw_if_cancelled();

        auto call_opts = build_call_options(options, prompt);
        auto result = co_await model->do_generate(std::move(call_opts));

        total_usage += result.usage;

        auto tool_calls_content = result.tool_calls();
        std::vector<ToolCallResult> step_tool_results;

        if (result.finish_reason == FinishReason::ToolCalls && !tool_calls_content.empty()) {
            append_assistant_message(prompt, result);
            step_tool_results = co_await execute_tools(
                tool_calls_content, options.tools, prompt, options.cancel,
                options.repair_tool_call
            );
            append_tool_results(prompt, step_tool_results);
            all_tool_calls.insert(all_tool_calls.end(),
                step_tool_results.begin(), step_tool_results.end());
        }

        StepResult step_result{
            .result = std::move(result),
            .tool_results = std::move(step_tool_results),
            .step_number = step,
        };

        if (options.on_step_finish) {
            options.on_step_finish(step_result);
        }

        bool should_continue = step_result.result.finish_reason == FinishReason::ToolCalls
                               && !options.tools.empty();
        steps.push_back(std::move(step_result));

        if (!should_continue) break;
    }

    std::string final_text;
    FinishReason final_reason = FinishReason::Stop;
    std::vector<Warning> warnings;

    if (!steps.empty()) {
        final_text = steps.back().result.text();
        final_reason = steps.back().result.finish_reason;
        warnings = steps.back().result.warnings;
    }

    co_return GenerateTextResult{
        .text = std::move(final_text),
        .tool_calls = std::move(all_tool_calls),
        .finish_reason = final_reason,
        .usage = total_usage,
        .steps = std::move(steps),
        .response_messages = std::move(prompt),
        .warnings = std::move(warnings),
    };
}

} // namespace ai
