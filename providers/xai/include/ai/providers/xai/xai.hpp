#pragma once

#include <ai/model/provider.hpp>
#include <boost/asio.hpp>
#include <string>
#include <optional>

namespace ai::providers::xai {

struct XAIOptions {
    std::optional<std::string> api_key;
    std::string base_url = "https://api.x.ai/v1";
    boost::asio::io_context& io_context;
};

class XAIProvider : public Provider {
public:
    explicit XAIProvider(XAIOptions options);

    LanguageModelPtr language_model(std::string_view model_id) override;
    std::string_view provider_id() const override { return "xai"; }

private:
    XAIOptions options_;
};

ProviderPtr create_xai(XAIOptions options);

} // namespace ai::providers::xai
