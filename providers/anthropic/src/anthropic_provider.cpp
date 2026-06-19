#include <ai/providers/anthropic/anthropic.hpp>
#include <ai/providers/anthropic/anthropic_model.hpp>
#include <ai/providers/anthropic/anthropic_file_storage.hpp>
#include <ai/providers/anthropic/anthropic_batch.hpp>
#include <cstdlib>
#include <stdexcept>

namespace ai::providers::anthropic {

AnthropicProvider::AnthropicProvider(AnthropicOptions options)
    : options_(std::move(options))
    , http_client_(options_.http_client
        ? options_.http_client
        : std::make_shared<http::HttpClient>(options_.io_context)) {
    if (options_.api_key) {
        resolved_api_key_ = *options_.api_key;
    } else if (options_.auth_token) {
        // auth_token mode - no api_key needed
    } else {
        const char* env_key = std::getenv("ANTHROPIC_API_KEY");
        if (env_key) {
            resolved_api_key_ = env_key;
        }
    }
}

LanguageModelPtr AnthropicProvider::language_model(std::string_view model_id) {
    return std::make_shared<AnthropicLanguageModel>(std::string(model_id), shared_from_this());
}

std::shared_ptr<ai::batch::BatchProcessor> AnthropicProvider::batch_processor(std::string_view model_id) {
    return std::make_shared<AnthropicBatchProcessor>(*this, std::string(model_id));
}

FileStoragePtr AnthropicProvider::file_storage() {
    return std::make_shared<AnthropicFileStorage>(*this);
}

http::Headers AnthropicProvider::auth_headers() const {
    http::Headers headers;
    headers["anthropic-version"] = options_.api_version.value_or("2023-06-01");
    headers["content-type"] = "application/json";

    if (options_.auth_token) {
        headers["Authorization"] = "Bearer " + *options_.auth_token;
    } else if (!resolved_api_key_.empty()) {
        headers["x-api-key"] = resolved_api_key_;
    }

    if (options_.extra_headers) {
        auto extra = (*options_.extra_headers)();
        for (auto& [k, v] : extra) {
            headers[k] = v;
        }
    }

    return headers;
}

std::string AnthropicProvider::messages_url() const {
    return options_.base_url + "/v1/messages";
}

ProviderPtr create_anthropic(AnthropicOptions options) {
    return std::make_shared<AnthropicProvider>(std::move(options));
}

} // namespace ai::providers::anthropic
