#include <catch2/catch_test_macros.hpp>

#include <ai/core/embed.hpp>
#include <ai/model/embedding_model.hpp>

#include <boost/asio.hpp>

#include <memory>
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

// Embeds each value as a 1-d vector equal to its length — enough to assert the
// core plumbing (model invocation + result mapping) offline.
class FakeEmbeddingModel : public ai::EmbeddingModel {
public:
    std::string_view provider() const override { return "fake"; }
    std::string_view model_id() const override { return "fake-embed"; }

    ai::Task<ai::EmbedResult> do_embed(ai::EmbedOptions opts) override {
        ai::EmbedResult r;
        r.embeddings.reserve(opts.values.size());
        for (auto& v : opts.values) {
            r.embeddings.push_back({static_cast<float>(v.size())});
        }
        co_return r;
    }
};

} // namespace

TEST_CASE("embed returns the model's vector for one value", "[embeddings]") {
    boost::asio::io_context ioc;
    auto model = std::make_shared<FakeEmbeddingModel>();

    auto result = run(ai::embed({.model = model, .value = "hello"}), ioc);

    REQUIRE(result.embedding.size() == 1);
    REQUIRE(result.embedding[0] == 5.0f);
}

TEST_CASE("embed_many returns a vector per input", "[embeddings]") {
    boost::asio::io_context ioc;
    auto model = std::make_shared<FakeEmbeddingModel>();

    auto result = run(ai::embed_many({.model = model, .values = {"a", "bb", "ccc"}}), ioc);

    REQUIRE(result.embeddings.size() == 3);
    REQUIRE(result.embeddings[0].size() == 1);
    REQUIRE(result.embeddings[0][0] == 1.0f);
    REQUIRE(result.embeddings[1][0] == 2.0f);
    REQUIRE(result.embeddings[2][0] == 3.0f);
}
