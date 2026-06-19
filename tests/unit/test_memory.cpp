#include <catch2/catch_test_macros.hpp>

#include <ai/memory/memory.hpp>
#include <ai/session/context_strategy.hpp>
#include <ai/session/session.hpp>
#include <ai/core/generate_text.hpp>
#include <ai/model/embedding_model.hpp>
#include <ai/prompt/message.hpp>
#include <ai/tool/tool_set.hpp>
#include <ai/test/mock_model.hpp>

#include <boost/asio.hpp>

#include <atomic>
#include <cctype>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace fs = std::filesystem;

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

fs::path make_unique_temp_dir() {
    static std::atomic<unsigned> counter{0};
    auto dir = fs::temp_directory_path() / ("ai-sdk-mem-test-" + std::to_string(++counter));
    fs::create_directories(dir);
    return dir;
}
struct TempDir {
    fs::path path;
    TempDir() : path(make_unique_temp_dir()) {}
    ~TempDir() {
        std::error_code ec;
        fs::remove_all(path, ec);
    }
};

/// Bag-of-words embedding model over a fixed vocab, so semantically overlapping
/// texts get similar vectors — enough to exercise EmbeddingRetriever offline.
class FakeEmbeddingModel : public ai::EmbeddingModel {
public:
    std::string_view provider() const override { return "fake"; }
    std::string_view model_id() const override { return "fake-embed"; }

    ai::Task<ai::EmbedResult> do_embed(ai::EmbedOptions options) override {
        std::vector<std::vector<float>> out;
        out.reserve(options.values.size());
        for (auto& v : options.values) out.push_back(embed_one(v));
        ai::EmbedResult r;
        r.embeddings = std::move(out);
        co_return r;
    }

    static std::vector<float> embed_one(const std::string& s) {
        static const std::vector<std::string> vocab = {
            "auth", "login", "database", "file", "test", "api", "error", "session"};
        std::string low;
        low.reserve(s.size());
        for (char c : s) low += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        std::vector<float> v(vocab.size(), 0.0f);
        for (std::size_t i = 0; i < vocab.size(); ++i) {
            std::size_t pos = 0;
            while ((pos = low.find(vocab[i], pos)) != std::string::npos) {
                v[i] += 1.0f;
                pos += vocab[i].size();
            }
        }
        return v;
    }
};

} // namespace

// ---------------------------------------------------------------------------
// MarkdownMemoryStore
// ---------------------------------------------------------------------------

TEST_CASE("MarkdownMemoryStore round-trips records with fidelity", "[memory]") {
    TempDir tmp;
    ai::memory::MarkdownMemoryStore store(tmp.path);

    std::string id = store.add(ai::memory::MemoryRecord{
        .scope = "project", .key = "Auth decision",
        .content = "We chose JWT for auth and login.", .tags = {"auth", "security"}});

    auto got = store.get(id);
    REQUIRE(got.has_value());
    REQUIRE(got->scope == "project");
    REQUIRE(got->key == "Auth decision");
    REQUIRE(got->content == "We chose JWT for auth and login.");
    REQUIRE(got->tags.size() == 2);

    auto all = store.list();
    REQUIRE(all.size() == 1);
    auto scoped = store.list("project");
    REQUIRE(scoped.size() == 1);
    REQUIRE(store.list("checkpoint").empty());

    store.remove(id);
    REQUIRE_FALSE(store.get(id).has_value());
}

// ---------------------------------------------------------------------------
// KeywordRetriever
// ---------------------------------------------------------------------------

TEST_CASE("KeywordRetriever recalls by term overlap", "[memory]") {
    TempDir tmp;
    ai::memory::MarkdownMemoryStore store(tmp.path);
    store.add({.scope = "project", .key = "Auth", .content = "auth login token"});
    store.add({.scope = "project", .key = "DB", .content = "database connection pool"});

    ai::memory::KeywordRetriever retriever(store);
    boost::asio::io_context ioc;
    auto hits = run(retriever.query("auth login", 5), ioc);

    REQUIRE(hits.size() >= 1);
    REQUIRE(hits[0].record.key == "Auth");
    REQUIRE(hits[0].score > 0.0);
}

// ---------------------------------------------------------------------------
// EmbeddingRetriever
// ---------------------------------------------------------------------------

TEST_CASE("EmbeddingRetriever recalls semantically relevant records", "[memory]") {
    TempDir tmp;
    ai::memory::MarkdownMemoryStore store(tmp.path);
    store.add({.scope = "project", .key = "Auth", .content = "auth login token session"});
    store.add({.scope = "project", .key = "DB", .content = "database connection pool"});

    ai::memory::EmbeddingRetriever retriever(store,
        std::make_shared<FakeEmbeddingModel>());
    boost::asio::io_context ioc;
    auto hits = run(retriever.query("auth login session", 5), ioc);

    REQUIRE(hits.size() >= 1);
    REQUIRE(hits[0].record.key == "Auth");
    REQUIRE(hits[0].score > 0.5);  // strong cosine match

    // Cache hit: a second query with the same records does not re-embed-fail.
    auto hits2 = run(retriever.query("database", 5), ioc);
    REQUIRE_FALSE(hits2.empty());
    REQUIRE(hits2[0].record.key == "DB");
}

// ---------------------------------------------------------------------------
// MemoryContextStrategy
// ---------------------------------------------------------------------------

TEST_CASE("MemoryContextStrategy injects project memory and checkpoint", "[memory]") {
    TempDir tmp;
    auto store = std::make_shared<ai::memory::MarkdownMemoryStore>(tmp.path);
    store->add({.scope = "project", .key = "Auth design",
                .content = "We chose JWT for auth and login."});
    store->add({.scope = "checkpoint", .key = "checkpoint",
                .content = "Mid-refactor of the auth module."});

    auto retriever = std::make_shared<ai::memory::KeywordRetriever>(*store);
    auto inner = std::make_shared<ai::SlidingWindowStrategy>();
    ai::memory::MemoryContextStrategy mcs(store, retriever, inner, 4096, 4);

    ai::Prompt hist;
    hist.push_back(ai::SystemMessage{.content = "sys"});
    ai::UserContent uc;
    uc.push_back(ai::TextPart{.text = "how does auth work?"});
    hist.push_back(ai::UserMessage{.content = std::move(uc)});

    ai::ContextWindow window{.max_tokens = 100000, .reserved_output_tokens = 0};
    ai::ApproximateTokenCounter counter;
    ai::Summarizer noop;
    boost::asio::io_context ioc;
    auto managed = run(mcs.manage(hist, window, counter, noop), ioc);

    // The memory block must be present right after the system message.
    bool found_memory = false;
    bool found_checkpoint = false;
    for (auto& m : managed) {
        if (auto* s = std::get_if<ai::SystemMessage>(&m)) {
            if (s->content.find("## Project memory") != std::string::npos) found_memory = true;
            if (s->content.find("Mid-refactor") != std::string::npos) found_checkpoint = true;
        }
    }
    REQUIRE(found_memory);
    REQUIRE(found_checkpoint);
    REQUIRE(managed.size() == hist.size() + 1);  // one injected memory message
}

// ---------------------------------------------------------------------------
// memory_tools
// ---------------------------------------------------------------------------

TEST_CASE("memory_tools save_memory and recall_memory", "[memory]") {
    TempDir tmp;
    auto store = std::make_shared<ai::memory::MarkdownMemoryStore>(tmp.path);
    auto retriever = std::make_shared<ai::memory::KeywordRetriever>(*store);
    ai::ToolSet tools = ai::memory::memory_tools(store, retriever);

    boost::asio::io_context ioc;
    const auto* save = tools.find("save_memory");
    REQUIRE(save);
    auto out = run((*save->execute)(
        boost::json::object{{"scope", "project"}, {"key", "Decision"},
                            {"content", "we use postgres database"}},
        ai::ToolExecutionContext{}), ioc);
    REQUIRE(out.as_string().find("saved memory") != std::string::npos);
    REQUIRE(store->list("project").size() == 1);

    const auto* recall = tools.find("recall_memory");
    REQUIRE(recall);
    auto hits = run((*recall->execute)(
        boost::json::object{{"query", "database"}}, ai::ToolExecutionContext{}), ioc);
    REQUIRE(hits.is_array());
    REQUIRE(hits.as_array().size() == 1);
    REQUIRE(hits.as_array()[0].at("key").as_string() == "Decision");
}

// ---------------------------------------------------------------------------
// CheckpointWriter (Session post-turn hook)
// ---------------------------------------------------------------------------

namespace {
struct LoopAgent : ai::Agent {
    std::shared_ptr<ai::LanguageModel> model;
    explicit LoopAgent(std::shared_ptr<ai::LanguageModel> m) : model(std::move(m)) {}
    ai::Task<ai::GenerateTextResult> call(std::string p, ai::CancellationToken = {}) override {
        ai::GenerateTextOptions o; o.model = model; o.prompt = std::move(p);
        co_return co_await ai::generate_text(std::move(o));
    }
    ai::Task<ai::GenerateTextResult> call(std::vector<ai::Message> m, ai::CancellationToken = {}) override {
        ai::GenerateTextOptions o; o.model = model; o.messages = std::move(m);
        co_return co_await ai::generate_text(std::move(o));
    }
};
} // namespace

TEST_CASE("CheckpointWriter writes a checkpoint every N turns", "[memory]") {
    TempDir tmp;
    auto store = std::make_shared<ai::memory::MarkdownMemoryStore>(tmp.path);

    ai::Summarizer summarizer = [](const ai::Prompt&) -> ai::Task<std::string> {
        co_return "auto-checkpoint-summary";
    };
    auto writer = ai::memory::make_checkpoint_writer(store, summarizer, /*every_n_turns=*/1);

    boost::asio::io_context ioc;
    auto model = std::make_shared<ai::test::MockLanguageModel>();
    model->queue_text("ok");
    LoopAgent agent(model);
    ai::Session session(agent);
    session.set_on_turn_finish(writer);

    run(session.send("do something"), ioc);

    // After 1 turn (== every_n_turns), a checkpoint record should exist.
    auto cps = store->list("checkpoint");
    REQUIRE(cps.size() == 1);
    REQUIRE(cps[0].content == "auto-checkpoint-summary");
}
