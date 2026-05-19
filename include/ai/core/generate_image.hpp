#pragma once

#include <ai/model/image_model.hpp>
#include <ai/util/cancellation.hpp>
#include <string>
#include <optional>

namespace ai {

struct GenerateImageOptions {
    ImageModelPtr model;
    std::string prompt;
    std::optional<std::string> size;
    std::optional<std::string> style;
    std::optional<std::string> quality;
    std::optional<int> n;
    CancellationToken cancel;
    boost::json::object provider_options;
};

struct GenerateImageResult {
    std::vector<GeneratedImage> images;
    Usage usage;
    std::vector<Warning> warnings;
};

Task<GenerateImageResult> generate_image(GenerateImageOptions options);

} // namespace ai
