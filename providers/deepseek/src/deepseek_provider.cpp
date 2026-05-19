#include <ai/providers/deepseek/deepseek.hpp>
#include <ai/providers/openai/openai.hpp>
#include <cstdlib>
#include <stdexcept>

namespace ai::providers::deepseek {

namespace {

std::string resolve_api_key(const std::optional<std::string>& key) {
    if (key && !key->empty()) return *key;
    if (auto* env = std::getenv("DEEPSEEK_API_KEY")) return env;
    throw std::runtime_error("DeepSeek API key not found. Set DEEPSEEK_API_KEY or pass api_key.");
}

} // namespace

DeepSeekProvider::DeepSeekProvider(DeepSeekOptions options)
    : options_(std::move(options)) {}

LanguageModelPtr DeepSeekProvider::language_model(std::string_view model_id) {
    openai::OpenAIOptions openai_opts{
        .api_key = resolve_api_key(options_.api_key),
        .base_url = options_.base_url,
        .io_context = options_.io_context,
    };
    auto openai_provider = openai::create_openai(std::move(openai_opts));
    return openai_provider->language_model(model_id);
}

ProviderPtr create_deepseek(DeepSeekOptions options) {
    return std::make_shared<DeepSeekProvider>(std::move(options));
}

} // namespace ai::providers::deepseek
