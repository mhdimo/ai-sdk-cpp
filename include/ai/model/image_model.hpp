#pragma once

#include <ai/stream/async_generator.hpp>
#include <ai/stream/stream_part.hpp>
#include <boost/json.hpp>
#include <string>
#include <vector>
#include <optional>
#include <memory>

namespace ai {

struct GeneratedImage {
    std::string base64;
    std::optional<std::string> url;
    std::string media_type;
};

struct ImageModelOptions {
    std::string prompt;
    std::optional<std::string> size;
    std::optional<std::string> style;
    std::optional<std::string> quality;
    std::optional<int> n;
    boost::json::object provider_options;
};

struct ImageModelResult {
    std::vector<GeneratedImage> images;
    Usage usage;
    std::vector<Warning> warnings;
};

class ImageModel {
public:
    virtual ~ImageModel() = default;
    virtual std::string_view provider() const = 0;
    virtual std::string_view model_id() const = 0;
    virtual Task<ImageModelResult> do_generate(ImageModelOptions options) = 0;
};

using ImageModelPtr = std::shared_ptr<ImageModel>;

} // namespace ai
