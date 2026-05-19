#pragma once

#include <ai/model/provider.hpp>
#include <ai/http/client.hpp>
#include <boost/asio.hpp>
#include <string>
#include <optional>

namespace ai::providers::moonshotai {

struct MoonshotAIOptions {
    std::optional<std::string> api_key;
    std::string base_url = "https://api.moonshot.ai/v1";
    boost::asio::io_context& io_context;
};

class MoonshotAIProvider : public Provider {
public:
    explicit MoonshotAIProvider(MoonshotAIOptions options);

    LanguageModelPtr language_model(std::string_view model_id) override;
    std::string_view provider_id() const override { return "moonshotai"; }

private:
    MoonshotAIOptions options_;
};

ProviderPtr create_moonshotai(MoonshotAIOptions options);

} // namespace ai::providers::moonshotai
