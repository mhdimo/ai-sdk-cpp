#pragma once

#include <ai/model/provider.hpp>
#include <ai/model/embedding_model.hpp>
#include <ai/model/file_storage.hpp>
#include <ai/http/client.hpp>
#include <boost/asio.hpp>
#include <string>
#include <optional>

namespace ai::providers::openai {

struct OpenAIOptions {
    std::optional<std::string> api_key;
    std::string base_url = "https://api.openai.com/v1";
    std::optional<std::string> organization;
    std::optional<std::string> project;
    boost::asio::io_context& io_context;
    // Optional injected client (tests); null = construct a real HttpClient.
    std::shared_ptr<http::IHttpClient> http_client;
};

class OpenAIProvider : public Provider {
public:
    explicit OpenAIProvider(OpenAIOptions options);

    LanguageModelPtr language_model(std::string_view model_id) override;
    std::shared_ptr<ai::batch::BatchProcessor> batch_processor(std::string_view model_id) override;
    LanguageModelPtr responses_model(std::string_view model_id);
    EmbeddingModelPtr embedding_model(std::string_view model_id);
    FileStoragePtr file_storage();
    std::string_view provider_id() const override { return "openai"; }

    const OpenAIOptions& options() const { return options_; }
    http::Headers auth_headers() const;
    std::string chat_completions_url() const;
    http::IHttpClient& http_client() { return *http_client_; }

private:
    OpenAIOptions options_;
    std::shared_ptr<http::IHttpClient> http_client_;
    std::string resolved_api_key_;
};

ProviderPtr create_openai(OpenAIOptions options);

} // namespace ai::providers::openai
