#pragma once

#include <ai/model/provider.hpp>
#include <ai/providers/anthropic/anthropic.hpp>
#include <ai/providers/openai/openai.hpp>
#include <boost/asio.hpp>
#include <string>
#include <optional>

namespace ai::providers::zai {

// z.ai GLM, Anthropic-compatible endpoint (default). Authenticated with a
// Bearer token (ZAI_API_KEY) sent as `Authorization: Bearer ...` by the
// underlying Anthropic provider.
struct ZaiOptions {
    std::optional<std::string> api_key;   // Bearer token; falls back to ZAI_API_KEY
    std::string base_url = "https://api.z.ai/api/anthropic";
    boost::asio::io_context& io_context;
};

// z.ai GLM, OpenAI-compatible endpoint. Bearer token from ZAI_API_KEY.
struct ZaiOpenAiOptions {
    std::optional<std::string> api_key;   // falls back to ZAI_API_KEY
    std::string base_url = "https://api.z.ai/api/paas/v4";
    boost::asio::io_context& io_context;
};

// Owns an inner Anthropic provider and sanitizes the model id (strips the
// trailing `[...]` Claude-Code context alias, e.g. `glm-5.2[1m]` -> `glm-5.2`)
// before delegating to it. Everything else (request building, parsing,
// streaming, batching) is forwarded to the Anthropic provider.
class ZaiProvider : public Provider {
public:
    explicit ZaiProvider(ZaiOptions options);

    LanguageModelPtr language_model(std::string_view model_id) override;
    std::shared_ptr<ai::batch::BatchProcessor> batch_processor(std::string_view model_id) override;
    std::string_view provider_id() const override { return "zai"; }

private:
    ZaiOptions options_;
    std::shared_ptr<anthropic::AnthropicProvider> anthropic_;
};

// Sanitize a z.ai model id by stripping a trailing `[...]` alias (e.g. the
// `[1m]` Claude-Code context window alias that the z.ai API rejects with
// `[1211] Unknown Model`). Exposed for unit testing.
std::string sanitize_model_id(std::string_view model_id);

ProviderPtr create_zai(ZaiOptions options);
ProviderPtr create_zai_openai(ZaiOpenAiOptions options);

} // namespace ai::providers::zai
