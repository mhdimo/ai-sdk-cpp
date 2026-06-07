#include <ai/providers/mistral/mistral.hpp>
#include <ai/providers/openai/openai.hpp>
#include <cstdlib>
#include <stdexcept>

namespace ai::providers::mistral {

namespace {

std::string resolve_api_key(const std::optional<std::string>& key) {
    if (key && !key->empty()) return *key;
    if (auto* env = std::getenv("MISTRAL_API_KEY")) return env;
    throw std::runtime_error("Mistral API key not found. Set MISTRAL_API_KEY or pass api_key.");
}

} // namespace

MistralProvider::MistralProvider(MistralOptions options)
    : options_(std::move(options)) {}

LanguageModelPtr MistralProvider::language_model(std::string_view model_id) {
    openai::OpenAIOptions openai_opts{
        .api_key = resolve_api_key(options_.api_key),
        .base_url = options_.base_url,
        .io_context = options_.io_context,
    };
    auto openai_provider = openai::create_openai(std::move(openai_opts));
    return openai_provider->language_model(model_id);
}

ProviderPtr create_mistral(MistralOptions options) {
    return std::make_shared<MistralProvider>(std::move(options));
}

} // namespace ai::providers::mistral
