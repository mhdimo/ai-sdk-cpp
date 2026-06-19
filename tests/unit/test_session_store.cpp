#include <catch2/catch_test_macros.hpp>

#include <ai/session/store.hpp>
#include <ai/prompt/message.hpp>
#include <ai/core/generate_text.hpp>
#include <ai/model/language_model.hpp>
#include <ai/agent/agent.hpp>
#include <ai/test/mock_model.hpp>

#include <boost/asio.hpp>
#include <boost/json.hpp>

#include <atomic>
#include <filesystem>
#include <memory>
#include <optional>
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
    auto dir = fs::temp_directory_path() /
               ("ai-sdk-store-test-" + std::to_string(++counter));
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

ai::SessionSnapshot make_snapshot() {
    ai::SessionSnapshot snap;
    snap.id = "test-sess-1";
    snap.model_id = "claude-x";
    snap.provider_id = "anthropic";
    snap.metadata.turns = 2;
    snap.metadata.total_input_tokens = 100;
    snap.metadata.total_output_tokens = 50;

    snap.history.push_back(ai::SystemMessage{.content = "you are helpful"});

    {
        ai::UserContent c;
        c.push_back(ai::TextPart{.text = "hello"});
        snap.history.push_back(ai::UserMessage{.content = std::move(c)});
    }
    {
        ai::AssistantContent c;
        c.push_back(ai::ToolCallPart{
            .tool_call_id = "call_1",
            .tool_name = "lookup",
            .input = boost::json::object{{"q", "x"}},
        });
        snap.history.push_back(ai::AssistantMessage{.content = std::move(c)});
    }
    {
        ai::ToolContent c;
        c.push_back(ai::ToolResultPart{
            .tool_call_id = "call_1",
            .tool_name = "lookup",
            .output = ai::JsonOutput{.value = boost::json::object{{"ans", 42}}},
        });
        snap.history.push_back(ai::ToolMessage{.content = std::move(c)});
    }
    {
        ai::AssistantContent c;
        c.push_back(ai::TextPart{.text = "the answer is 42"});
        snap.history.push_back(ai::AssistantMessage{.content = std::move(c)});
    }
    return snap;
}

} // namespace

TEST_CASE("JsonFileSessionStore round-trips a tool conversation with full fidelity",
          "[session][store]") {
    boost::asio::io_context ioc;
    TempDir tmp;
    ai::JsonFileSessionStore store(tmp.path);

    ai::SessionSnapshot snap = make_snapshot();
    run(store.save(snap), ioc);

    auto loaded = run(store.load("test-sess-1"), ioc);
    REQUIRE(loaded.has_value());

    REQUIRE(loaded->id == "test-sess-1");
    REQUIRE(loaded->model_id == "claude-x");
    REQUIRE(loaded->provider_id == "anthropic");
    REQUIRE(loaded->metadata.turns == 2);
    REQUIRE(loaded->metadata.total_input_tokens == 100);
    REQUIRE(loaded->metadata.total_output_tokens == 50);
    REQUIRE(loaded->history.size() == 5);

    REQUIRE(std::holds_alternative<ai::SystemMessage>(loaded->history[0]));
    REQUIRE(std::get<ai::SystemMessage>(loaded->history[0]).content == "you are helpful");

    // The assistant tool-call must survive intact.
    auto& asst = std::get<ai::AssistantMessage>(loaded->history[2]);
    REQUIRE_FALSE(asst.content.empty());
    auto* tc = std::get_if<ai::ToolCallPart>(&asst.content.front());
    REQUIRE(tc != nullptr);
    REQUIRE(tc->tool_call_id == "call_1");
    REQUIRE(tc->tool_name == "lookup");
    REQUIRE(tc->input.at("q").as_string() == "x");

    // The matching tool result must survive intact (same id, JSON value).
    auto& tool_msg = std::get<ai::ToolMessage>(loaded->history[3]);
    REQUIRE_FALSE(tool_msg.content.empty());
    REQUIRE(tool_msg.content.front().tool_call_id == "call_1");
    auto* json_out = std::get_if<ai::JsonOutput>(&tool_msg.content.front().output);
    REQUIRE(json_out != nullptr);
    REQUIRE(json_out->value.at("ans").as_int64() == 42);

    // Terminal assistant text preserved.
    auto& final_asst = std::get<ai::AssistantMessage>(loaded->history[4]);
    REQUIRE(std::get<ai::TextPart>(final_asst.content.front()).text == "the answer is 42");
}

TEST_CASE("JsonFileSessionStore load returns nullopt for unknown id", "[session][store]") {
    boost::asio::io_context ioc;
    TempDir tmp;
    ai::JsonFileSessionStore store(tmp.path);

    auto loaded = run(store.load("does-not-exist"), ioc);
    REQUIRE_FALSE(loaded.has_value());
}

TEST_CASE("JsonFileSessionStore list and remove", "[session][store]") {
    boost::asio::io_context ioc;
    TempDir tmp;
    ai::JsonFileSessionStore store(tmp.path);

    run(store.save(make_snapshot()), ioc);

    auto metas = run(store.list(), ioc);
    REQUIRE(metas.size() == 1);
    REQUIRE(metas[0].id == "test-sess-1");
    REQUIRE(metas[0].turns == 2);

    run(store.remove("test-sess-1"), ioc);
    auto loaded = run(store.load("test-sess-1"), ioc);
    REQUIRE_FALSE(loaded.has_value());
}

TEST_CASE("Session snapshot/restore preserves history in-memory", "[session][store]") {
    boost::asio::io_context ioc;
    auto model = std::make_shared<ai::test::MockLanguageModel>();
    model->queue_text("hi");
    model->queue_text("again");

    // Minimal agent wrapper for the Session.
    struct LoopAgent : ai::Agent {
        std::shared_ptr<ai::LanguageModel> model;
        explicit LoopAgent(std::shared_ptr<ai::LanguageModel> m) : model(std::move(m)) {}
        ai::Task<ai::GenerateTextResult> call(std::string p, ai::CancellationToken = {}) override {
            ai::GenerateTextOptions o;
            o.model = model;
            o.prompt = std::move(p);
            co_return co_await ai::generate_text(std::move(o));
        }
        ai::Task<ai::GenerateTextResult> call(std::vector<ai::Message> m, ai::CancellationToken = {}) override {
            ai::GenerateTextOptions o;
            o.model = model;
            o.messages = std::move(m);
            co_return co_await ai::generate_text(std::move(o));
        }
    };

    LoopAgent agent(model);
    ai::Session s1(agent);
    run(s1.send("first"), ioc);
    run(s1.send("second"), ioc);
    REQUIRE(s1.history().size() == 4);

    ai::SessionSnapshot snap = s1.snapshot();
    REQUIRE(snap.history.size() == 4);
    REQUIRE(snap.metadata.turns == 2);

    // Restore into a fresh session and confirm the history carries over.
    LoopAgent agent2(model);
    ai::Session s2(agent2);
    REQUIRE(s2.history().empty());
    s2.restore(std::move(snap));
    REQUIRE(s2.history().size() == 4);
    REQUIRE(s2.metadata().turns == 2);
    REQUIRE_FALSE(s2.id().empty());
}
