#pragma once

#include <ai/stream/async_generator.hpp>
#include <ai/stream/stream_part.hpp>
#include <ai/model/generate_result.hpp>
#include <optional>

namespace ai {

struct StreamResult {
    AsyncGenerator<StreamPart> stream;
    std::optional<RequestMetadata> request;
    std::optional<std::unordered_map<std::string, std::string>> response_headers;
};

} // namespace ai
