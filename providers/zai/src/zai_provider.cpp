#include <ai/providers/zai/zai.hpp>
#include <cstdlib>
#include <stdexcept>

namespace ai::providers::zai {

namespace {

std::string resolve_api_key(const std::optional<std::string>& key) {
    if (key && !key->empty()) return *key;
    if (auto* env = std::getenv("ZAI_API_KEY")) return env;
    throw std::runtime_error("z.ai API key not found. Set ZAI_API_KEY or pass api_key.");
}

} // namespace

std::string sanitize_model_id(std::string_view model_id) {
    std::string id(model_id);
    // Strip a single trailing `[...]` group — the Claude-Code context-window
    // alias (e.g. `glm-5.2[1m]`) that the z.ai API rejects with
    // `[1211] Unknown Model`. Only a trailing alias is removed so genuine
    // bracketed substrings inside a model name are preserved.
    auto end = id.size();
    if (end >= 2 && id.back() == ']') {
        auto open = id.rfind('[');
        if (open != std::string::npos && open < end - 1) {
            id.erase(open);
        }
    }
    return id;
}

ZaiProvider::ZaiProvider(ZaiOptions options)
    : options_(std::move(options)) {
    anthropic::AnthropicOptions ao{
        .auth_token = resolve_api_key(options_.api_key),
        .base_url = options_.base_url,
        .io_context = options_.io_context,
    };
    anthropic_ = std::static_pointer_cast<anthropic::AnthropicProvider>(
        anthropic::create_anthropic(std::move(ao)));
}

LanguageModelPtr ZaiProvider::language_model(std::string_view model_id) {
    // glm-5.2[1m] -> glm-5.2 before the API ever sees it.
    return anthropic_->language_model(sanitize_model_id(model_id));
}

std::shared_ptr<ai::batch::BatchProcessor> ZaiProvider::batch_processor(std::string_view model_id) {
    return anthropic_->batch_processor(sanitize_model_id(model_id));
}

ProviderPtr create_zai(ZaiOptions options) {
    return std::make_shared<ZaiProvider>(std::move(options));
}

ProviderPtr create_zai_openai(ZaiOpenAiOptions options) {
    openai::OpenAIOptions o{
        .api_key = resolve_api_key(options.api_key),
        .base_url = options.base_url,
        .io_context = options.io_context,
    };
    return openai::create_openai(std::move(o));
}

} // namespace ai::providers::zai
