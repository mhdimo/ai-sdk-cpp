#include <ai/providers/openai_compatible/openai_compatible.hpp>
#include <ai/providers/openai/openai.hpp>
#include <cstdlib>
#include <stdexcept>

namespace ai::providers::openai_compatible {

namespace {

std::string resolve_api_key(const std::optional<std::string>& key, const std::string& env_var) {
    if (key && !key->empty()) return *key;
    if (auto* env = std::getenv(env_var.c_str())) return env;
    throw std::runtime_error("API key not found. Set " + env_var + " or pass api_key.");
}

} // namespace

OpenAICompatibleProvider::OpenAICompatibleProvider(OpenAICompatibleOptions options)
    : options_(std::move(options))
    , provider_name_(options_.name) {}

LanguageModelPtr OpenAICompatibleProvider::language_model(std::string_view model_id) {
    openai::OpenAIOptions openai_opts{
        .api_key = resolve_api_key(options_.api_key, options_.api_key_env_var),
        .base_url = options_.base_url,
        .io_context = options_.io_context,
    };
    auto openai_provider = openai::create_openai(std::move(openai_opts));
    return openai_provider->language_model(model_id);
}

ProviderPtr create_openai_compatible(OpenAICompatibleOptions options) {
    return std::make_shared<OpenAICompatibleProvider>(std::move(options));
}

} // namespace ai::providers::openai_compatible
