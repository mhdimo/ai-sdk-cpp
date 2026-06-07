#pragma once

#include <ai/model/provider.hpp>
#include <boost/asio.hpp>
#include <string>
#include <optional>

namespace ai::providers::perplexity {

struct PerplexityOptions {
    std::optional<std::string> api_key;
    std::string base_url = "https://api.perplexity.ai";
    boost::asio::io_context& io_context;
};

class PerplexityProvider : public Provider {
public:
    explicit PerplexityProvider(PerplexityOptions options);

    LanguageModelPtr language_model(std::string_view model_id) override;
    std::string_view provider_id() const override { return "perplexity"; }

private:
    PerplexityOptions options_;
};

ProviderPtr create_perplexity(PerplexityOptions options);

} // namespace ai::providers::perplexity
