#include <ai/providers/google/google.hpp>
#include <ai/providers/google/google_model.hpp>
#include <cstdlib>

namespace ai::providers::google {

GoogleProvider::GoogleProvider(GoogleOptions options)
    : options_(std::move(options))
    , http_client_(options_.io_context) {
    if (options_.api_key) {
        resolved_api_key_ = *options_.api_key;
    } else {
        const char* env_key = std::getenv("GOOGLE_GENERATIVE_AI_API_KEY");
        if (env_key) {
            resolved_api_key_ = env_key;
        }
    }
}

LanguageModelPtr GoogleProvider::language_model(std::string_view model_id) {
    return std::make_shared<GoogleLanguageModel>(std::string(model_id), *this);
}

std::string GoogleProvider::generate_url(std::string_view model_id) const {
    return options_.base_url + "/models/" + std::string(model_id) +
           ":generateContent?key=" + resolved_api_key_;
}

std::string GoogleProvider::stream_url(std::string_view model_id) const {
    return options_.base_url + "/models/" + std::string(model_id) +
           ":streamGenerateContent?alt=sse&key=" + resolved_api_key_;
}

http::Headers GoogleProvider::content_headers() const {
    http::Headers headers;
    headers["Content-Type"] = "application/json";
    return headers;
}

ProviderPtr create_google(GoogleOptions options) {
    return std::make_shared<GoogleProvider>(std::move(options));
}

} // namespace ai::providers::google
