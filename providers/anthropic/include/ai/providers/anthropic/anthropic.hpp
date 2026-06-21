#pragma once

#include <ai/model/provider.hpp>
#include <ai/model/file_storage.hpp>
#include <ai/http/client.hpp>
#include <boost/asio.hpp>
#include <boost/json.hpp>
#include <string>
#include <optional>
#include <functional>
#include <vector>
#include <memory>

namespace ai::providers::anthropic {

struct AnthropicBuiltinTool {
    std::string type;  // "web_search_20260209", "code_execution_20250522", etc.
    std::string name;
    boost::json::object config;  // tool-specific config
};

struct AnthropicOptions {
    std::optional<std::string> api_key;
    std::optional<std::string> auth_token;
    std::string base_url = "https://api.anthropic.com";
    std::optional<std::string> api_version;
    // Optional anthropic-beta features (e.g. "interleaved-thinking-2025-05-14"),
    // joined into the anthropic-beta header.
    std::optional<std::vector<std::string>> anthropic_beta;
    boost::asio::io_context& io_context;
    std::optional<std::function<http::Headers()>> extra_headers;
    // Optional injected client (used by tests to avoid real network). When null,
    // the provider constructs its own HttpClient bound to io_context.
    std::shared_ptr<http::IHttpClient> http_client;
};

class AnthropicProvider : public Provider, public std::enable_shared_from_this<AnthropicProvider> {
public:
    explicit AnthropicProvider(AnthropicOptions options);

    LanguageModelPtr language_model(std::string_view model_id) override;
    std::shared_ptr<ai::batch::BatchProcessor> batch_processor(std::string_view model_id) override;
    FileStoragePtr file_storage();
    std::string_view provider_id() const override { return "anthropic"; }

    const AnthropicOptions& options() const { return options_; }
    http::Headers auth_headers() const;
    std::string messages_url() const;
    http::IHttpClient& http_client() { return *http_client_; }

private:
    AnthropicOptions options_;
    std::shared_ptr<http::IHttpClient> http_client_;
    std::string resolved_api_key_;
    std::string resolved_auth_token_;
};

ProviderPtr create_anthropic(AnthropicOptions options);

} // namespace ai::providers::anthropic
