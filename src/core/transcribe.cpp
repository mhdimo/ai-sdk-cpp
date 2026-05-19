#include <ai/core/transcribe.hpp>

namespace ai {

Task<TranscribeResult> transcribe(TranscribeOptions options) {
    TranscriptionModelOptions model_opts{
        .audio = std::move(options.audio),
        .media_type = std::move(options.media_type),
        .language = std::move(options.language),
        .prompt = std::move(options.prompt),
        .provider_options = std::move(options.provider_options),
    };

    auto result = co_await options.model->do_transcribe(std::move(model_opts));

    co_return TranscribeResult{
        .text = std::move(result.text),
        .segments = std::move(result.segments),
        .language = std::move(result.language),
        .usage = result.usage,
    };
}

} // namespace ai
