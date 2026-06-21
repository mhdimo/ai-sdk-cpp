#include <catch2/catch_test_macros.hpp>

#include <ai/model/language_model.hpp>
#include <ai/model/wrap_language_model.hpp>
#include <ai/model/caching_middleware.hpp>
#include <ai/model/logging_middleware.hpp>
#include <ai/model/call_options.hpp>
#include <ai/test/mock_model.hpp>

#include <boost/asio.hpp>
#include <memory>

namespace {

template <typename T>
T run(ai::Task<T> task, boost::asio::io_context& ioc) {
    task.start();
    int guard = 0;
    while (!task.done() && guard++ < 100'000) ioc.run_one();
    REQUIRE(task.done());
    return task.get();
}

ai::CallOptions simple_prompt(const std::string& text) {
    ai::CallOptions co;
    ai::UserContent uc;
    uc.push_back(ai::TextPart{.text = text});
    co.prompt.push_back(ai::UserMessage{.content = std::move(uc)});
    return co;
}

} // namespace

TEST_CASE("CachingMiddleware caches identical generate calls", "[middleware]") {
    boost::asio::io_context ioc;
    auto mock = std::make_shared<ai::test::MockLanguageModel>();
    mock->queue_text("cached response");

    auto cached = ai::wrap_language_model(mock, {std::make_shared<ai::CachingMiddleware>(10)});

    auto r1 = run(cached->do_generate(simple_prompt("hello")), ioc);
    REQUIRE(r1.text() == "cached response");
    REQUIRE(mock->call_count() == 1);

    auto r2 = run(cached->do_generate(simple_prompt("hello")), ioc);
    REQUIRE(r2.text() == "cached response");
    REQUIRE(mock->call_count() == 1);  // still 1 — served from cache
}

TEST_CASE("LoggingMiddleware passes through", "[middleware]") {
    boost::asio::io_context ioc;
    auto mock = std::make_shared<ai::test::MockLanguageModel>();
    mock->queue_text("passthrough");

    auto logged = ai::wrap_language_model(mock, {ai::make_logging_middleware()});

    auto r = run(logged->do_generate(simple_prompt("test")), ioc);
    REQUIRE(r.text() == "passthrough");
    REQUIRE(mock->call_count() == 1);
}
