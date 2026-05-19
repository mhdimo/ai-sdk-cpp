#pragma once

#include <ai/stream/stream_part.hpp>
#include <ai/stream/async_generator.hpp>
#include <ai/util/cancellation.hpp>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace ai {

struct EmbedOptions {
    std::vector<std::string> values;
    CancellationToken cancel;
};

struct EmbedResult {
    std::vector<std::vector<float>> embeddings;
    Usage usage;
};

class EmbeddingModel {
public:
    virtual ~EmbeddingModel() = default;

    virtual std::string_view provider() const = 0;
    virtual std::string_view model_id() const = 0;

    virtual Task<EmbedResult> do_embed(EmbedOptions options) = 0;
};

using EmbeddingModelPtr = std::shared_ptr<EmbeddingModel>;

} // namespace ai
