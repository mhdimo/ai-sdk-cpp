#include <catch2/catch_test_macros.hpp>

#include <ai/core/rerank.hpp>

#include <boost/asio.hpp>

#include <memory>
#include <set>
#include <string>
#include <vector>

namespace {

template <typename T>
T run(ai::Task<T> task, boost::asio::io_context& ioc) {
    task.start();
    int guard = 0;
    while (!task.done() && guard++ < 100'000) {
        ioc.run_one();
    }
    REQUIRE(task.done());
    return task.get();
}

// Ranks documents by how many query characters appear in the text (descending).
class FakeRerankModel : public ai::RerankModel {
public:
    std::string_view provider() const override { return "fake"; }
    std::string_view model_id() const override { return "fake-rerank"; }

    ai::Task<ai::RerankModelResult> do_rerank(ai::RerankModel::RerankCallOptions opts) override {
        ai::RerankModelResult out;
        for (std::size_t i = 0; i < opts.documents.size(); ++i) {
            int overlap = 0;
            for (char c : opts.query) {
                if (opts.documents[i].text.find(c) != std::string::npos) ++overlap;
            }
            out.results.push_back(ai::RankedResult{
                .index = static_cast<int>(i),
                .relevance_score = static_cast<double>(overlap),
                .document_text = opts.documents[i].text,
            });
        }
        std::sort(out.results.begin(), out.results.end(),
                  [](const auto& a, const auto& b) { return a.relevance_score > b.relevance_score; });
        if (opts.top_n && *opts.top_n < static_cast<int>(out.results.size())) {
            out.results.resize(*opts.top_n);
        }
        co_return out;
    }
};

} // namespace

TEST_CASE("rerank orders documents and applies top_n", "[rerank]") {
    boost::asio::io_context ioc;
    auto model = std::make_shared<FakeRerankModel>();

    auto result = run(ai::rerank({
        .model = model,
        .query = "cat",
        .documents = {"dog", "category", "caterpillar", "fish"},
        .top_n = 2,
    }), ioc);

    REQUIRE(result.results.size() == 2);
    // "caterpillar" and "category" both share 3 chars with "cat" (the top 2),
    // in either order — don't assume tie-break order.
    REQUIRE(result.results[0].relevance_score == 3.0);
    REQUIRE(result.results[1].relevance_score == 3.0);
    std::set<std::string> top{result.results[0].document_text,
                              result.results[1].document_text};
    REQUIRE(top.count("category"));
    REQUIRE(top.count("caterpillar"));
}

TEST_CASE("rerank without top_n returns all documents", "[rerank]") {
    boost::asio::io_context ioc;
    auto model = std::make_shared<FakeRerankModel>();

    auto result = run(ai::rerank({
        .model = model,
        .query = "x",
        .documents = {"a", "b", "c"},
    }), ioc);

    REQUIRE(result.results.size() == 3);
}
