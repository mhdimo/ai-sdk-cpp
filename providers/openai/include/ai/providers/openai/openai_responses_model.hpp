#pragma once

#include <ai/model/language_model.hpp>
#include <ai/model/call_options.hpp>
#include <ai/model/generate_result.hpp>
#include <ai/model/stream_result.hpp>
#include <string>

namespace ai::providers::openai {

class OpenAIProvider;

class OpenAIResponsesModel : public LanguageModel {
public:
    OpenAIResponsesModel(std::string model_id, OpenAIProvider& provider);

    std::string_view provider() const override { return "openai"; }
    std::string_view model_id() const override { return model_id_; }
    Task<GenerateResult> do_generate(CallOptions options) override;
    Task<StreamResult> do_stream(CallOptions options) override;

private:
    std::string model_id_;
    OpenAIProvider& provider_;

    boost::json::object build_request_body(const CallOptions& options, bool stream = false);
    GenerateResult parse_response(const boost::json::value& response);
};

} // namespace ai::providers::openai
