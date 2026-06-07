#include <ai/agent/tool_loop_agent.hpp>

namespace ai {

ToolLoopAgent::ToolLoopAgent(ToolLoopAgentOptions options)
    : options_(std::move(options)) {}

Task<GenerateTextResult> ToolLoopAgent::call(
    std::string prompt,
    CancellationToken cancel
) {
    co_return co_await generate_text(GenerateTextOptions{
        .model = options_.model,
        .tools = options_.tools,
        .tool_choice = options_.tool_choice,
        .prompt = std::move(prompt),
        .system = options_.instructions,
        .max_output_tokens = options_.max_output_tokens,
        .temperature = options_.temperature,
        .max_steps = options_.max_steps,
        .on_step_finish = options_.on_step_finish,
        .middleware = options_.middleware,
        .cancel = std::move(cancel),
        .max_retries = options_.max_retries,
        .provider_options = options_.provider_options,
    });
}

Task<GenerateTextResult> ToolLoopAgent::call(
    std::vector<Message> messages,
    CancellationToken cancel
) {
    co_return co_await generate_text(GenerateTextOptions{
        .model = options_.model,
        .tools = options_.tools,
        .tool_choice = options_.tool_choice,
        .messages = std::move(messages),
        .system = options_.instructions,
        .max_output_tokens = options_.max_output_tokens,
        .temperature = options_.temperature,
        .max_steps = options_.max_steps,
        .on_step_finish = options_.on_step_finish,
        .middleware = options_.middleware,
        .cancel = std::move(cancel),
        .max_retries = options_.max_retries,
        .provider_options = options_.provider_options,
    });
}

Task<StreamTextResult> ToolLoopAgent::stream(
    std::string prompt,
    CancellationToken cancel
) {
    co_return co_await stream_text(StreamTextOptions{
        .model = options_.model,
        .tools = options_.tools,
        .tool_choice = options_.tool_choice,
        .prompt = std::move(prompt),
        .system = options_.instructions,
        .max_output_tokens = options_.max_output_tokens,
        .temperature = options_.temperature,
        .max_steps = options_.max_steps,
        .on_step_finish = options_.on_step_finish,
        .middleware = options_.middleware,
        .cancel = std::move(cancel),
        .max_retries = options_.max_retries,
        .provider_options = options_.provider_options,
    });
}

Task<StreamTextResult> ToolLoopAgent::stream(
    std::vector<Message> messages,
    CancellationToken cancel
) {
    co_return co_await stream_text(StreamTextOptions{
        .model = options_.model,
        .tools = options_.tools,
        .tool_choice = options_.tool_choice,
        .messages = std::move(messages),
        .system = options_.instructions,
        .max_output_tokens = options_.max_output_tokens,
        .temperature = options_.temperature,
        .max_steps = options_.max_steps,
        .on_step_finish = options_.on_step_finish,
        .middleware = options_.middleware,
        .cancel = std::move(cancel),
        .max_retries = options_.max_retries,
        .provider_options = options_.provider_options,
    });
}

} // namespace ai
