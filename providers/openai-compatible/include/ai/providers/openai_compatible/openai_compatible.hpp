#pragma once

#include <ai/model/provider.hpp>
#include <ai/http/client.hpp>
#include <boost/asio.hpp>
#include <string>
#include <optional>

namespace ai::providers::openai_compatible {

struct OpenAICompatibleOptions {
    std::string name;
    std::string base_url;
    std::optional<std::string> api_key;
    std::string api_key_env_var = "OPENAI_COMPATIBLE_API_KEY";
    boost::asio::io_context& io_context;
};

class OpenAICompatibleProvider : public Provider {
public:
    explicit OpenAICompatibleProvider(OpenAICompatibleOptions options);

    LanguageModelPtr language_model(std::string_view model_id) override;
    std::string_view provider_id() const override { return provider_name_; }

private:
    OpenAICompatibleOptions options_;
    std::string provider_name_;
};

ProviderPtr create_openai_compatible(OpenAICompatibleOptions options);

} // namespace ai::providers::openai_compatible
