#include <ai/providers/xai/xai.hpp>
#include <ai/providers/openai/openai.hpp>
#include <cstdlib>
#include <stdexcept>

namespace ai::providers::xai {

namespace {

std::string resolve_api_key(const std::optional<std::string>& key) {
    if (key && !key->empty()) return *key;
    if (auto* env = std::getenv("XAI_API_KEY")) return env;
    throw std::runtime_error("xAI API key not found. Set XAI_API_KEY or pass api_key.");
}

} // namespace

XAIProvider::XAIProvider(XAIOptions options)
    : options_(std::move(options)) {}

LanguageModelPtr XAIProvider::language_model(std::string_view model_id) {
    openai::OpenAIOptions openai_opts{
        .api_key = resolve_api_key(options_.api_key),
        .base_url = options_.base_url,
        .io_context = options_.io_context,
    };
    auto openai_provider = openai::create_openai(std::move(openai_opts));
    return openai_provider->language_model(model_id);
}

ProviderPtr create_xai(XAIOptions options) {
    return std::make_shared<XAIProvider>(std::move(options));
}

} // namespace ai::providers::xai
