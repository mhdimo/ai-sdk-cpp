#pragma once

#include <ai/model/call_options.hpp>
#include <ai/model/generate_result.hpp>
#include <ai/model/stream_result.hpp>
#include <ai/stream/async_generator.hpp>
#include <memory>
#include <string>
#include <string_view>

namespace ai {

class LanguageModel {
public:
    virtual ~LanguageModel() = default;

    virtual std::string_view provider() const = 0;
    virtual std::string_view model_id() const = 0;

    virtual Task<GenerateResult> do_generate(CallOptions options) = 0;
    virtual Task<StreamResult> do_stream(CallOptions options) = 0;
};

using LanguageModelPtr = std::shared_ptr<LanguageModel>;

} // namespace ai
