#include <ai/providers/cohere/cohere.hpp>
#include <ai/providers/openai/openai.hpp>
#include <ai/util/json.hpp>
#include <ai/error/api_call_error.hpp>
#include <cstdlib>
#include <stdexcept>

namespace ai::providers::cohere {

namespace json = boost::json;

CohereProvider::CohereProvider(CohereOptions options)
    : options_(std::move(options))
    , http_client_(options_.io_context) {
    if (options_.api_key) {
        resolved_api_key_ = *options_.api_key;
    } else {
        const char* env = std::getenv("COHERE_API_KEY");
        if (env) resolved_api_key_ = env;
    }
}

http::Headers CohereProvider::auth_headers() const {
    http::Headers headers;
    if (!resolved_api_key_.empty()) {
        headers["Authorization"] = "Bearer " + resolved_api_key_;
    }
    return headers;
}

LanguageModelPtr CohereProvider::language_model(std::string_view model_id) {
    openai::OpenAIOptions openai_opts{
        .api_key = resolved_api_key_,
        .base_url = options_.base_url,
        .io_context = options_.io_context,
    };
    auto openai_provider = openai::create_openai(std::move(openai_opts));
    return openai_provider->language_model(model_id);
}

Task<RerankResponse> CohereProvider::rerank(RerankOptions options) {
    json::object body;
    body["model"] = options.model;
    body["query"] = options.query;

    json::array docs;
    for (auto& doc : options.documents) {
        docs.push_back(json::object{{"text", doc.text}});
    }
    body["documents"] = std::move(docs);

    if (options.top_n) {
        body["top_n"] = *options.top_n;
    }

    std::string url = options_.base_url + "/rerank";
    auto headers = auth_headers();

    auto response = co_await http_client_.post_json(
        url, json::value(std::move(body)), headers, options.cancel);

    auto parsed = ai::json::parse(response.body);
    RerankResponse result;

    if (parsed.is_object()) {
        auto results_it = parsed.as_object().find("results");
        if (results_it != parsed.as_object().end() && results_it->value().is_array()) {
            for (auto& item : results_it->value().as_array()) {
                if (!item.is_object()) continue;
                auto& obj = item.as_object();
                RerankResult rr;
                if (auto idx = obj.find("index"); idx != obj.end() && idx->value().is_int64())
                    rr.index = static_cast<int>(idx->value().as_int64());
                if (auto score = obj.find("relevance_score"); score != obj.end() && score->value().is_double())
                    rr.relevance_score = score->value().as_double();
                else if (auto score = obj.find("relevance_score"); score != obj.end() && score->value().is_int64())
                    rr.relevance_score = static_cast<double>(score->value().as_int64());
                result.results.push_back(rr);
            }
        }
    }

    co_return result;
}

ProviderPtr create_cohere(CohereOptions options) {
    return std::make_shared<CohereProvider>(std::move(options));
}

} // namespace ai::providers::cohere
