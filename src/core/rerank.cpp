#include <ai/core/rerank.hpp>
#include <stdexcept>

namespace ai {

Task<RerankResult> rerank(RerankOptions options) {
    if (!options.model) {
        throw std::invalid_argument("rerank: model is required");
    }
    if (options.query.empty()) {
        throw std::invalid_argument("rerank: query must not be empty");
    }
    if (options.documents.empty()) {
        throw std::invalid_argument("rerank: documents must not be empty");
    }

    std::vector<std::string> original_docs = options.documents;

    RerankModel::RerankCallOptions call_opts;
    call_opts.query = std::move(options.query);
    call_opts.top_n = options.top_n;
    call_opts.cancel = std::move(options.cancel);

    for (auto& doc : options.documents) {
        call_opts.documents.push_back(RerankDocument{.text = std::move(doc)});
    }

    auto result = co_await options.model->do_rerank(std::move(call_opts));

    RerankResult output;
    for (auto& r : result.results) {
        if (r.index >= 0 && r.index < static_cast<int>(original_docs.size())) {
            r.document_text = original_docs[r.index];
        }
        output.results.push_back(std::move(r));
    }

    co_return output;
}

} // namespace ai
