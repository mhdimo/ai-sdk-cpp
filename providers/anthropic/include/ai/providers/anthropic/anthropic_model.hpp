#pragma once

#include <ai/model/language_model.hpp>
#include <ai/http/client.hpp>
#include <string>
#include <memory>

namespace ai::providers::anthropic {

class AnthropicProvider;

class AnthropicLanguageModel : public LanguageModel {
public:
    AnthropicLanguageModel(std::string model_id, AnthropicProvider& provider);

    std::string_view provider() const override { return "anthropic"; }
    std::string_view model_id() const override { return model_id_; }

    Task<GenerateResult> do_generate(CallOptions options) override;
    Task<StreamResult> do_stream(CallOptions options) override;

    // Pure transformation of CallOptions into the Anthropic Messages API body.
    // Public to allow unit-testing structured-output / request construction.
    boost::json::value build_request_body(const CallOptions& options, bool stream);

private:
    std::string model_id_;
    AnthropicProvider& provider_;
    boost::json::array convert_messages(const Prompt& prompt);
    boost::json::array convert_tools(const std::vector<FunctionTool>& tools);
    boost::json::value convert_tool_choice(const std::optional<ToolChoice>& choice);

    GenerateResult parse_response(const boost::json::value& response);
    std::vector<StreamPart> parse_stream_chunk(const boost::json::value& chunk);

    int default_max_tokens() const;
};

} // namespace ai::providers::anthropic
