#pragma once

#include <ai/model/provider.hpp>
#include <ai/http/client.hpp>
#include <boost/asio.hpp>
#include <string>
#include <optional>

namespace ai::providers::deepseek {

struct DeepSeekOptions {
    std::optional<std::string> api_key;
    std::string base_url = "https://api.deepseek.com/v1";
    boost::asio::io_context& io_context;
};

class DeepSeekProvider : public Provider {
public:
    explicit DeepSeekProvider(DeepSeekOptions options);

    LanguageModelPtr language_model(std::string_view model_id) override;
    std::string_view provider_id() const override { return "deepseek"; }

private:
    DeepSeekOptions options_;
};

ProviderPtr create_deepseek(DeepSeekOptions options);

} // namespace ai::providers::deepseek
