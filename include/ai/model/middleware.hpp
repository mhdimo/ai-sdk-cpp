#pragma once

#include <ai/model/call_options.hpp>
#include <ai/model/generate_result.hpp>
#include <ai/model/stream_result.hpp>
#include <ai/stream/async_generator.hpp>
#include <functional>
#include <memory>

namespace ai {

class Middleware {
public:
    virtual ~Middleware() = default;

    using GenerateFn = std::function<Task<GenerateResult>()>;
    using StreamFn = std::function<Task<StreamResult>()>;

    virtual Task<GenerateResult> wrap_generate(
        GenerateFn do_generate,
        StreamFn do_stream,
        CallOptions& options
    ) {
        return do_generate();
    }

    virtual Task<StreamResult> wrap_stream(
        GenerateFn do_generate,
        StreamFn do_stream,
        CallOptions& options
    ) {
        return do_stream();
    }
};

using MiddlewarePtr = std::shared_ptr<Middleware>;

} // namespace ai
