#pragma once

#include <ai/model/provider.hpp>
#include <ai/http/client.hpp>
#include <boost/asio.hpp>
#include <string>
#include <optional>

namespace ai::providers::google {

struct GoogleOptions {
    std::optional<std::string> api_key;
    std::string base_url = "https://generativelanguage.googleapis.com/v1beta";
    boost::asio::io_context& io_context;
    // Optional injected client (tests); null = construct a real HttpClient.
    std::shared_ptr<http::IHttpClient> http_client;
};

class GoogleProvider : public Provider {
public:
    explicit GoogleProvider(GoogleOptions options);

    LanguageModelPtr language_model(std::string_view model_id) override;
    std::string_view provider_id() const override { return "google"; }

    const GoogleOptions& options() const { return options_; }
    http::IHttpClient& http_client() { return *http_client_; }

    // Build generate URL: /models/{model}:generateContent?key={apiKey}
    std::string generate_url(std::string_view model_id) const;
    // Build streaming URL: /models/{model}:streamGenerateContent?alt=sse&key={apiKey}
    std::string stream_url(std::string_view model_id) const;
    // Auth headers (content-type only, key is in query param)
    http::Headers content_headers() const;

private:
    GoogleOptions options_;
    std::shared_ptr<http::IHttpClient> http_client_;
    std::string resolved_api_key_;
};

ProviderPtr create_google(GoogleOptions options);

} // namespace ai::providers::google
