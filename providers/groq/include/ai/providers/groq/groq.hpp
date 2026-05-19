#pragma once

#include <ai/model/provider.hpp>
#include <boost/asio.hpp>
#include <string>
#include <optional>

namespace ai::providers::groq {

struct GroqOptions {
    std::optional<std::string> api_key;
    std::string base_url = "https://api.groq.com/openai/v1";
    boost::asio::io_context& io_context;
};

class GroqProvider : public Provider {
public:
    explicit GroqProvider(GroqOptions options);

    LanguageModelPtr language_model(std::string_view model_id) override;
    std::string_view provider_id() const override { return "groq"; }

private:
    GroqOptions options_;
};

ProviderPtr create_groq(GroqOptions options);

} // namespace ai::providers::groq
