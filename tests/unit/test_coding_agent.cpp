#include <catch2/catch_test_macros.hpp>

#include <ai/coding_agent.hpp>
#include <ai/test/mock_model.hpp>
#include <ai/memory/memory.hpp>

#include <boost/asio.hpp>

#include <atomic>
#include <filesystem>
#include <memory>
#include <string>

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
    auto dir = fs::temp_directory_path() / ("ai-sdk-facade-test-" + std::to_string(++counter));
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

} // namespace

TEST_CASE("CodingAgent runs a turn and keeps history", "[coding_agent]") {
    boost::asio::io_context ioc;
    auto model = std::make_shared<ai::test::MockLanguageModel>();
    model->queue_text("hello there");

    ai::CodingAgentOptions opts;
    opts.model = model;
    opts.instructions = "you are helpful";

    ai::CodingAgent agent(opts);
    auto r = run(agent.send("hi"), ioc);

    REQUIRE(r.text == "hello there");
    REQUIRE(agent.session().history().size() == 2);  // user + assistant
    REQUIRE_FALSE(agent.memory_store());             // memory disabled
}

TEST_CASE("CodingAgent with memory auto-checkpoints", "[coding_agent]") {
    TempDir tmp;
    boost::asio::io_context ioc;
    auto model = std::make_shared<ai::test::MockLanguageModel>();
    model->queue_text("ok");

    ai::CodingAgentOptions opts;
    opts.model = model;
    opts.enable_memory = true;
    opts.memory_dir = tmp.path;
    opts.checkpoint_every_n_turns = 1;
    opts.checkpoint_summarizer = [](const ai::Prompt&) -> ai::Task<std::string> {
        co_return "facade-checkpoint";
    };

    ai::CodingAgent agent(opts);
    run(agent.send("do something useful"), ioc);

    REQUIRE(agent.memory_store() != nullptr);
    auto cps = agent.memory_store()->list("checkpoint");
    REQUIRE(cps.size() == 1);
    REQUIRE(cps[0].content == "facade-checkpoint");
}
