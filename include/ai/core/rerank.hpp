#pragma once

#include <ai/stream/async_generator.hpp>
#include <ai/util/cancellation.hpp>
#include <string>
#include <vector>
#include <optional>
#include <functional>

namespace ai {

struct RerankDocument {
    std::string text;
};

struct RankedResult {
    int index;
    double relevance_score;
    std::string document_text;
};

struct RerankModelResult {
    std::vector<RankedResult> results;
};

class RerankModel {
public:
    virtual ~RerankModel() = default;
    virtual std::string_view provider() const = 0;
    virtual std::string_view model_id() const = 0;

    struct RerankCallOptions {
        std::string query;
        std::vector<RerankDocument> documents;
        std::optional<int> top_n;
        CancellationToken cancel;
    };

    virtual Task<RerankModelResult> do_rerank(RerankCallOptions options) = 0;
};

using RerankModelPtr = std::shared_ptr<RerankModel>;

struct RerankOptions {
    RerankModelPtr model;
    std::string query;
    std::vector<std::string> documents;
    std::optional<int> top_n;
    CancellationToken cancel;
    int max_retries = 2;
};

struct RerankResult {
    std::vector<RankedResult> results;
};

Task<RerankResult> rerank(RerankOptions options);

} // namespace ai
