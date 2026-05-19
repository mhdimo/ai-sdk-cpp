#pragma once

#include <ai/model/language_model.hpp>
#include <ai/http/client.hpp>
#include <string>

namespace ai::providers::google {

class GoogleProvider;

class GoogleLanguageModel : public LanguageModel {
public:
    GoogleLanguageModel(std::string model_id, GoogleProvider& provider);

    std::string_view provider() const override { return "google"; }
    std::string_view model_id() const override { return model_id_; }

    Task<GenerateResult> do_generate(CallOptions options) override;
    Task<StreamResult> do_stream(CallOptions options) override;

private:
    std::string model_id_;
    GoogleProvider& provider_;

    boost::json::value build_request_body(const CallOptions& options);
    boost::json::array convert_contents(const Prompt& prompt, std::string& system_text);
    boost::json::array convert_tools(const std::vector<FunctionTool>& tools);
    GenerateResult parse_response(const boost::json::value& response);
};

} // namespace ai::providers::google
