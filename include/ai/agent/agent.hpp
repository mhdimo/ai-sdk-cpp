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

    /// Streaming entry point. Default implementation throws (agents that do
    /// not support streaming are still valid Agent subclasses). Subclasses
    /// such as ToolLoopAgent override this.
    virtual Task<StreamTextResult> stream(
        std::string prompt,
        CancellationToken cancel = {}
    );

    virtual Task<StreamTextResult> stream(
        std::vector<Message> messages,
        CancellationToken cancel = {}
    );
};

} // namespace ai
