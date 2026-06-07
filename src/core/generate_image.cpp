#include <ai/core/generate_image.hpp>

namespace ai {

Task<GenerateImageResult> generate_image(GenerateImageOptions options) {
    ImageModelOptions model_opts{
        .prompt = std::move(options.prompt),
        .size = std::move(options.size),
        .style = std::move(options.style),
        .quality = std::move(options.quality),
        .n = options.n,
        .provider_options = std::move(options.provider_options),
    };

    auto result = co_await options.model->do_generate(std::move(model_opts));

    co_return GenerateImageResult{
        .images = std::move(result.images),
        .usage = result.usage,
        .warnings = std::move(result.warnings),
    };
}

} // namespace ai
