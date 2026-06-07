#pragma once

#include <ai/model/provider.hpp>
#include <boost/asio.hpp>
#include <string>
#include <optional>

namespace ai::providers::togetherai {

struct TogetherAIOptions {
    std::optional<std::string> api_key;
    std::string base_url = "https://api.together.xyz/v1";
    boost::asio::io_context& io_context;
};

class TogetherAIProvider : public Provider {
public:
    explicit TogetherAIProvider(TogetherAIOptions options);

    LanguageModelPtr language_model(std::string_view model_id) override;
    std::string_view provider_id() const override { return "togetherai"; }

private:
    TogetherAIOptions options_;
};

ProviderPtr create_togetherai(TogetherAIOptions options);

} // namespace ai::providers::togetherai
