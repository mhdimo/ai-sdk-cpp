#pragma once

#include <ai/model/language_model.hpp>
#include <ai/http/client.hpp>
#include <string>
#include <memory>

namespace ai::providers::openai {

class OpenAIProvider;

class OpenAIChatLanguageModel : public LanguageModel {
public:
    OpenAIChatLanguageModel(std::string model_id, std::shared_ptr<OpenAIProvider> provider);

    std::string_view provider() const override { return "openai"; }
    std::string_view model_id() const override { return model_id_; }

    Task<GenerateResult> do_generate(CallOptions options) override;
    Task<StreamResult> do_stream(CallOptions options) override;

private:
    std::string model_id_;
    std::shared_ptr<OpenAIProvider> provider_;

    boost::json::value build_request_body(const CallOptions& options, bool stream);
    GenerateResult parse_response(const boost::json::value& response);
};

} // namespace ai::providers::openai
