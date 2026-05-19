#pragma once

#include <ai/model/embedding_model.hpp>
#include <ai/stream/async_generator.hpp>
#include <ai/util/cancellation.hpp>
#include <string>
#include <vector>
#include <optional>

namespace ai {

struct SingleEmbedOptions {
    EmbeddingModelPtr model;
    std::string value;
    CancellationToken cancel;
    int max_retries = 2;
};

struct SingleEmbedResult {
    std::vector<float> embedding;
    Usage usage;
};

struct EmbedManyOptions {
    EmbeddingModelPtr model;
    std::vector<std::string> values;
    CancellationToken cancel;
    int max_retries = 2;
};

struct EmbedManyResult {
    std::vector<std::vector<float>> embeddings;
    Usage usage;
};

Task<SingleEmbedResult> embed(SingleEmbedOptions options);
Task<EmbedManyResult> embed_many(EmbedManyOptions options);

} // namespace ai
