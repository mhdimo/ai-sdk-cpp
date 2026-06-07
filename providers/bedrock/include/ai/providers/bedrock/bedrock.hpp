#pragma once

#include <ai/model/provider.hpp>
#include <ai/http/client.hpp>
#include <boost/asio.hpp>
#include <string>
#include <optional>

namespace ai::providers::bedrock {

struct BedrockOptions {
    std::optional<std::string> region;
    std::optional<std::string> access_key_id;
    std::optional<std::string> secret_access_key;
    std::optional<std::string> session_token;
    std::optional<std::string> bearer_token;
    std::optional<std::string> base_url;
    boost::asio::io_context& io_context;
};

class BedrockProvider : public Provider {
public:
    explicit BedrockProvider(BedrockOptions options);

    LanguageModelPtr language_model(std::string_view model_id) override;
    std::string_view provider_id() const override { return "amazon-bedrock"; }

    const BedrockOptions& options() const { return options_; }
    std::string runtime_base_url() const;
    http::Headers auth_headers(const std::string& url, const std::string& body) const;
    http::HttpClient& http_client() { return http_client_; }

private:
    BedrockOptions options_;
    http::HttpClient http_client_;
    std::string resolved_region_;
    std::string resolved_access_key_id_;
    std::string resolved_secret_access_key_;
    std::string resolved_session_token_;
    std::string resolved_bearer_token_;
    bool use_bearer_token_ = false;
};

ProviderPtr create_bedrock(BedrockOptions options);

} // namespace ai::providers::bedrock
