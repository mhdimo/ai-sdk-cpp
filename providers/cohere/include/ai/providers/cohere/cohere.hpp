#pragma once

#include <ai/model/provider.hpp>
#include <ai/http/client.hpp>
#include <boost/asio.hpp>
#include <string>
#include <optional>
#include <vector>

namespace ai::providers::cohere {

struct CohereOptions {
    std::optional<std::string> api_key;
    std::string base_url = "https://api.cohere.com/v2";
    boost::asio::io_context& io_context;
};

struct RerankDocument {
    std::string text;
};

struct RerankResult {
    int index;
    double relevance_score;
};

struct RerankOptions {
    std::string model = "rerank-v3.5";
    std::string query;
    std::vector<RerankDocument> documents;
    std::optional<int> top_n;
    CancellationToken cancel;
};

struct RerankResponse {
    std::vector<RerankResult> results;
};

class CohereProvider : public Provider {
public:
    explicit CohereProvider(CohereOptions options);

    LanguageModelPtr language_model(std::string_view model_id) override;
    std::string_view provider_id() const override { return "cohere"; }

    Task<RerankResponse> rerank(RerankOptions options);

    http::HttpClient& http_client() { return http_client_; }
    http::Headers auth_headers() const;

private:
    CohereOptions options_;
    std::string resolved_api_key_;
    http::HttpClient http_client_;
};

ProviderPtr create_cohere(CohereOptions options);

} // namespace ai::providers::cohere
