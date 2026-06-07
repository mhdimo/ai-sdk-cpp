#include <ai/providers/moonshotai/moonshotai.hpp>
#include <ai/providers/openai/openai.hpp>
#include <cstdlib>
#include <stdexcept>

namespace ai::providers::moonshotai {

namespace {

std::string resolve_api_key(const std::optional<std::string>& key) {
    if (key && !key->empty()) return *key;
    if (auto* env = std::getenv("MOONSHOT_API_KEY")) return env;
    throw std::runtime_error("Moonshot API key not found. Set MOONSHOT_API_KEY or pass api_key.");
}

} // namespace

MoonshotAIProvider::MoonshotAIProvider(MoonshotAIOptions options)
    : options_(std::move(options)) {}

LanguageModelPtr MoonshotAIProvider::language_model(std::string_view model_id) {
    openai::OpenAIOptions openai_opts{
        .api_key = resolve_api_key(options_.api_key),
        .base_url = options_.base_url,
        .io_context = options_.io_context,
    };
    auto openai_provider = openai::create_openai(std::move(openai_opts));
    return openai_provider->language_model(model_id);
}

ProviderPtr create_moonshotai(MoonshotAIOptions options) {
    return std::make_shared<MoonshotAIProvider>(std::move(options));
}

} // namespace ai::providers::moonshotai
