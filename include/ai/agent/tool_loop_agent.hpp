#pragma once

#include <ai/agent/agent.hpp>
#include <ai/model/language_model.hpp>
#include <ai/model/middleware.hpp>
#include <ai/tool/tool_set.hpp>
#include <ai/tool/tool_choice.hpp>
#include <string>
#include <vector>
#include <optional>
#include <functional>

namespace ai {

struct ToolLoopAgentOptions {
    LanguageModelPtr model;
    ToolSet tools;
    std::string instructions;
    int max_steps = 20;
    ToolChoice tool_choice = ToolChoiceAuto{};
    std::vector<MiddlewarePtr> middleware;

    std::optional<double> temperature;
    std::optional<int> max_output_tokens;

    std::function<void(const StepResult&)> on_step_finish;
    int max_retries = 2;

    boost::json::object provider_options;
};

class ToolLoopAgent : public Agent {
public:
    explicit ToolLoopAgent(ToolLoopAgentOptions options);

    Task<GenerateTextResult> call(
        std::string prompt,
        CancellationToken cancel = {}
    ) override;

    Task<GenerateTextResult> call(
        std::vector<Message> messages,
        CancellationToken cancel = {}
    ) override;

    Task<StreamTextResult> stream(
        std::string prompt,
        CancellationToken cancel = {}
    );

    Task<StreamTextResult> stream(
        std::vector<Message> messages,
        CancellationToken cancel = {}
    );

private:
    ToolLoopAgentOptions options_;
};

} // namespace ai
