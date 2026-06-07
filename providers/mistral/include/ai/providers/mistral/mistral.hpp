#pragma once

#include <ai/model/provider.hpp>
#include <boost/asio.hpp>
#include <string>
#include <optional>

namespace ai::providers::mistral {

struct MistralOptions {
    std::optional<std::string> api_key;
    std::string base_url = "https://api.mistral.ai/v1";
    boost::asio::io_context& io_context;
};

class MistralProvider : public Provider {
public:
    explicit MistralProvider(MistralOptions options);

    LanguageModelPtr language_model(std::string_view model_id) override;
    std::string_view provider_id() const override { return "mistral"; }

private:
    MistralOptions options_;
};

ProviderPtr create_mistral(MistralOptions options);

} // namespace ai::providers::mistral
