#include <catch2/catch_test_macros.hpp>

#include <ai/agent/tool_loop_agent.hpp>
#include <ai/tool/tool.hpp>
#include <ai/tool/tool_set.hpp>
#include <ai/schema/json_schema.hpp>
#include <ai/test/mock_model.hpp>

#include <boost/asio.hpp>
#include <boost/asio/steady_timer.hpp>
#include <boost/system/error_code.hpp>

#include <chrono>
#include <coroutine>
#include <memory>
#include <stdexcept>
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

/// Awaits a steady_timer on the given io_context (same idiom as batch.cpp).
struct Sleep {
    boost::asio::io_context& ioc;
    std::chrono::milliseconds duration;

    bool await_ready() const noexcept { return duration.count() <= 0; }

    void await_suspend(std::coroutine_handle<> h) const {
        auto timer = std::make_shared<boost::asio::steady_timer>(ioc, duration);
        timer->async_wait([timer, h](boost::system::error_code) { h.resume(); });
    }

    void await_resume() const noexcept {}
};

/// Opens only once `expected` coroutines have arrived. A sequential
/// implementation never opens it, so tests arm a deadline that aborts the
/// waiters and flags `timed_out` instead of hanging.
struct Rendezvous {
    std::size_t expected;
    std::size_t arrived = 0;
    bool open = false;
    bool timed_out = false;
    std::vector<std::coroutine_handle<>> waiters;

    void release() {
        open = true;
        auto pending = std::move(waiters);
        waiters.clear();
        for (auto h : pending) h.resume();
    }

    void arrive() {
        ++arrived;
        if (arrived >= expected && !timed_out) release();
    }

    void abort() {
        timed_out = true;
        release();
    }

    struct WaitOp {
        Rendezvous* rz;

        bool await_ready() const noexcept { return rz->open; }

        bool await_suspend(std::coroutine_handle<> h) {
            if (rz->open) return false;
            rz->waiters.push_back(h);
            return true;
        }

        void await_resume() const noexcept {}
    };

    WaitOp wait() { return WaitOp{this}; }
};

ai::ToolCallContent make_call(std::string id, std::string name, std::string input) {
    return ai::ToolCallContent{
        .tool_call_id = std::move(id),
        .tool_name = std::move(name),
        .input = std::move(input),
    };
}

} // namespace

TEST_CASE("execute_tools runs independent tool calls concurrently", "[tool_loop]") {
    boost::asio::io_context ioc;
    Rendezvous rz{.expected = 2};

    auto model = std::make_shared<ai::test::MockLanguageModel>();
    ai::test::MockResponse step1;
    step1.tool_calls.push_back(make_call("call_a", "gate_a", R"({"v":"a"})"));
    step1.tool_calls.push_back(make_call("call_b", "gate_b", R"({"v":"b"})"));
    step1.finish_reason = ai::FinishReason::ToolCalls;
    model->queue_response(std::move(step1));
    model->queue_text("done");

    auto gate = [&rz](boost::json::value input,
                      ai::ToolExecutionContext) -> ai::Task<boost::json::value> {
        rz.arrive();
        co_await rz.wait();
        co_return boost::json::value("result-" + std::string(input.at("v").as_string()));
    };

    ai::ToolSet tools;
    tools.add(ai::tool("gate_a",
        ai::schema::JsonSchema::object({}).additional_properties(false),
        "waits for gate_b", gate));
    tools.add(ai::tool("gate_b",
        ai::schema::JsonSchema::object({}).additional_properties(false),
        "waits for gate_a", gate));

    ai::ToolLoopAgent agent({
        .model = model,
        .tools = std::move(tools),
        .max_steps = 5,
    });

    auto deadline = std::make_shared<boost::asio::steady_timer>(
        ioc, std::chrono::seconds(2));
    deadline->async_wait([&rz](boost::system::error_code ec) {
        if (!ec) rz.abort();
    });

    auto result = run(agent.call("go"), ioc);
    deadline->cancel();
    ioc.poll();

    REQUIRE_FALSE(rz.timed_out);
    REQUIRE(result.tool_calls.size() == 2);
    REQUIRE(result.tool_calls[0].tool_call_id == "call_a");
    REQUIRE(result.tool_calls[0].output.as_string() == "result-a");
    REQUIRE_FALSE(result.tool_calls[0].is_error);
    REQUIRE(result.tool_calls[1].tool_call_id == "call_b");
    REQUIRE(result.tool_calls[1].output.as_string() == "result-b");
    REQUIRE_FALSE(result.tool_calls[1].is_error);
}

TEST_CASE("execute_tools isolates failures and preserves result order", "[tool_loop]") {
    boost::asio::io_context ioc;

    auto model = std::make_shared<ai::test::MockLanguageModel>();
    ai::test::MockResponse step1;
    step1.tool_calls.push_back(make_call("call_1", "boom", "{}"));
    step1.tool_calls.push_back(make_call("call_2", "instant", "{}"));
    step1.tool_calls.push_back(make_call("call_3", "slow", "{}"));
    step1.finish_reason = ai::FinishReason::ToolCalls;
    model->queue_response(std::move(step1));
    model->queue_text("done");

    ai::ToolSet tools;
    tools.add(ai::tool("boom",
        ai::schema::JsonSchema::object({}).additional_properties(false),
        "always throws",
        [](boost::json::value, ai::ToolExecutionContext) -> ai::Task<boost::json::value> {
            throw std::runtime_error("boom failed");
            co_return boost::json::value{};
        }));
    tools.add(ai::tool("instant",
        ai::schema::JsonSchema::object({}).additional_properties(false),
        "completes without suspending",
        [](boost::json::value, ai::ToolExecutionContext) -> ai::Task<boost::json::value> {
            co_return boost::json::value("instant-done");
        }));
    tools.add(ai::tool("slow",
        ai::schema::JsonSchema::object({}).additional_properties(false),
        "suspends on a timer",
        [&ioc](boost::json::value, ai::ToolExecutionContext) -> ai::Task<boost::json::value> {
            co_await Sleep{ioc, std::chrono::milliseconds(50)};
            co_return boost::json::value("slow-done");
        }));

    ai::ToolLoopAgent agent({
        .model = model,
        .tools = std::move(tools),
        .max_steps = 5,
    });

    auto result = run(agent.call("go"), ioc);

    REQUIRE(result.tool_calls.size() == 3);
    REQUIRE(result.tool_calls[0].tool_call_id == "call_1");
    REQUIRE(result.tool_calls[0].is_error);
    REQUIRE(result.tool_calls[0].output.as_string() == "Error: boom failed");
    REQUIRE(result.tool_calls[1].tool_call_id == "call_2");
    REQUIRE_FALSE(result.tool_calls[1].is_error);
    REQUIRE(result.tool_calls[1].output.as_string() == "instant-done");
    REQUIRE(result.tool_calls[2].tool_call_id == "call_3");
    REQUIRE_FALSE(result.tool_calls[2].is_error);
    REQUIRE(result.tool_calls[2].output.as_string() == "slow-done");
    REQUIRE(result.text == "done");
}

TEST_CASE("execute_tools reports an error for unknown tools", "[tool_loop]") {
    boost::asio::io_context ioc;

    auto model = std::make_shared<ai::test::MockLanguageModel>();
    ai::test::MockResponse step1;
    step1.tool_calls.push_back(make_call("call_x", "does_not_exist", "{}"));
    step1.finish_reason = ai::FinishReason::ToolCalls;
    model->queue_response(std::move(step1));
    model->queue_text("done");

    ai::ToolSet tools;
    ai::ToolLoopAgent agent({
        .model = model,
        .tools = std::move(tools),
        .max_steps = 5,
    });

    auto result = run(agent.call("go"), ioc);

    REQUIRE(result.tool_calls.size() == 1);
    REQUIRE(result.tool_calls[0].is_error);
    REQUIRE(result.tool_calls[0].output.as_string() ==
        "Tool not found or not executable: does_not_exist");
}
