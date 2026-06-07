#pragma once

#include <ai/core/generate_text.hpp>
#include <ai/core/stream_text.hpp>
#include <ai/stream/async_generator.hpp>
#include <string>
#include <vector>

namespace ai {

class Agent {
public:
    virtual ~Agent() = default;

    virtual Task<GenerateTextResult> call(
        std::string prompt,
        CancellationToken cancel = {}
    ) = 0;

    virtual Task<GenerateTextResult> call(
        std::vector<Message> messages,
        CancellationToken cancel = {}
    ) = 0;
};

} // namespace ai
