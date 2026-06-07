#include <ai/core/generate_speech.hpp>

namespace ai {

Task<GenerateSpeechResult> generate_speech(GenerateSpeechOptions options) {
    SpeechModelOptions model_opts{
        .text = std::move(options.text),
        .voice = std::move(options.voice),
        .speed = options.speed,
        .format = std::move(options.format),
        .provider_options = std::move(options.provider_options),
    };

    auto result = co_await options.model->do_generate(std::move(model_opts));

    co_return GenerateSpeechResult{
        .audio = std::move(result.audio),
        .media_type = std::move(result.media_type),
        .usage = result.usage,
    };
}

} // namespace ai
