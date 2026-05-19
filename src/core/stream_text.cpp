#include <ai/core/stream_text.hpp>
#include <ai/model/wrap_language_model.hpp>
#include <ai/util/json.hpp>
#include <ai/error/ai_error.hpp>

namespace ai {

namespace {

Prompt build_prompt(const StreamTextOptions& opts) {
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

CallOptions build_call_options(const StreamTextOptions& opts, const Prompt& prompt) {
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

void append_assistant_message(Prompt& prompt, const std::string& text,
                              const std::vector<ToolCallContent>& tool_calls) {
    AssistantContent content;
    if (!text.empty()) {
        content.push_back(TextPart{.text = text});
    }
    for (auto& tc : tool_calls) {
        auto input = ai::json::safe_parse(tc.input);
        content.push_back(ToolCallPart{
            .tool_call_id = tc.tool_call_id,
            .tool_name = tc.tool_name,
            .input = input.value_or(boost::json::value(tc.input)),
        });
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
    CancellationToken cancel
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

        auto input = ai::json::safe_parse(tc.input).value_or(boost::json::value(tc.input));
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

/// Internal coroutine that produces a multi-step stream with tool execution.
/// Yields StreamPart values from ALL steps seamlessly.
AsyncGenerator<StreamPart> stream_text_multi_step(
    LanguageModelPtr model,
    StreamTextOptions options,
    Prompt prompt
) {
    std::vector<StepResult> steps;
    Usage total_usage{};
    std::vector<ToolCallResult> all_tool_calls;

    for (int step = 0; step < options.max_steps; ++step) {
        options.cancel.throw_if_cancelled();

        auto call_opts = build_call_options(options, prompt);
        auto stream_result = co_await model->do_stream(std::move(call_opts));

        // Consume the stream, accumulating text and tool calls
        std::string accumulated_text;
        std::vector<ToolCallContent> accumulated_tool_calls;
        FinishReason finish_reason = FinishReason::Stop;
        Usage step_usage{};
        std::vector<Warning> step_warnings;

        // Track in-progress tool calls by id
        std::unordered_map<std::string, ToolCallContent> pending_tool_calls;

        while (!stream_result.stream.done()) {
            auto maybe_part = co_await stream_result.stream.next();
            if (!maybe_part) break;

            auto& part = *maybe_part;

            // Process the part to accumulate state
            if (auto* text_delta = std::get_if<TextDelta>(&part)) {
                accumulated_text += text_delta->delta;
            } else if (auto* tool_start = std::get_if<ToolInputStart>(&part)) {
                pending_tool_calls[tool_start->id] = ToolCallContent{
                    .tool_call_id = tool_start->id,
                    .tool_name = tool_start->tool_name,
                    .input = "",
                };
            } else if (auto* tool_delta = std::get_if<ToolInputDelta>(&part)) {
                auto it = pending_tool_calls.find(tool_delta->id);
                if (it != pending_tool_calls.end()) {
                    it->second.input += tool_delta->delta;
                }
            } else if (auto* tool_end = std::get_if<ToolInputEnd>(&part)) {
                auto it = pending_tool_calls.find(tool_end->id);
                if (it != pending_tool_calls.end()) {
                    accumulated_tool_calls.push_back(std::move(it->second));
                    pending_tool_calls.erase(it);
                }
            } else if (auto* finish = std::get_if<FinishPart>(&part)) {
                finish_reason = finish->reason;
                step_usage = finish->usage;
            } else if (auto* stream_start = std::get_if<StreamStart>(&part)) {
                step_warnings = stream_start->warnings;
            }

            // Yield the part to the consumer
            co_yield part;
        }

        total_usage += step_usage;

        // Build result for this step
        std::vector<Content> content;
        if (!accumulated_text.empty()) {
            content.push_back(TextContent{.text = accumulated_text});
        }
        for (auto& tc : accumulated_tool_calls) {
            content.push_back(tc);
        }

        GenerateResult step_gen_result{
            .content = std::move(content),
            .finish_reason = finish_reason,
            .usage = step_usage,
            .warnings = step_warnings,
        };

        std::vector<ToolCallResult> step_tool_results;

        // If finished with tool calls, execute them and continue
        if (finish_reason == FinishReason::ToolCalls && !accumulated_tool_calls.empty()
            && !options.tools.empty()) {
            append_assistant_message(prompt, accumulated_text, accumulated_tool_calls);

            step_tool_results = co_await execute_tools(
                accumulated_tool_calls, options.tools, prompt, options.cancel
            );
            append_tool_results(prompt, step_tool_results);

            all_tool_calls.insert(all_tool_calls.end(),
                step_tool_results.begin(), step_tool_results.end());
        }

        StepResult step_result{
            .result = std::move(step_gen_result),
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
}

} // namespace

Task<StreamTextResult> stream_text(StreamTextOptions options) {
    // Apply middleware to the model if any are specified
    auto model = options.middleware.empty()
        ? options.model
        : wrap_language_model(options.model, options.middleware);

    auto prompt = build_prompt(options);

    // If single step and no tools, use simplified path
    if (options.max_steps <= 1 || options.tools.empty()) {
        auto call_opts = build_call_options(options, prompt);
        auto stream_result = co_await model->do_stream(std::move(call_opts));

        co_return StreamTextResult{
            .stream = std::move(stream_result.stream),
            .full_result = Task<GenerateTextResult>{nullptr},
        };
    }

    // Multi-step streaming with tool execution
    auto stream = stream_text_multi_step(std::move(model), std::move(options), std::move(prompt));

    co_return StreamTextResult{
        .stream = std::move(stream),
        .full_result = Task<GenerateTextResult>{nullptr},
    };
}

} // namespace ai
