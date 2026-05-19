#pragma once

#include <ai/stream/async_generator.hpp>
#include <ai/stream/stream_part.hpp>
#include <boost/json.hpp>
#include <string>
#include <vector>
#include <optional>

namespace ai {

struct GenerateVideoOptions {
    std::string prompt;
    std::optional<std::string> duration;
    std::optional<std::string> resolution;
    boost::json::object provider_options;
};

struct GenerateVideoResult {
    std::vector<uint8_t> video;
    std::string media_type;
    Usage usage;
};

// Placeholder — no providers implement video generation yet
// Task<GenerateVideoResult> generate_video(GenerateVideoOptions options);

} // namespace ai
