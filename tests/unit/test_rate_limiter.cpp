#include <catch2/catch_test_macros.hpp>

#include <ai/util/rate_limiter.hpp>
#include <ai/stream/async_generator.hpp>

#include <boost/asio.hpp>

namespace {

// Drives a Task<void> to completion.
void run(ai::Task<void> task, boost::asio::io_context& ioc) {
    task.start();
    int guard = 0;
    while (!task.done() && guard++ < 100'000) {
        ioc.run_one();
    }
    REQUIRE(task.done());
    task.get();
}

} // namespace

TEST_CASE("RateLimiter tracks the token budget", "[rate_limit]") {
    ai::RateLimiter rl(ai::RateLimitConfig{
        .max_requests_per_minute = 100, .max_tokens_per_minute = 1000});

    REQUIRE_FALSE(rl.would_exceed());
    rl.record_usage(600);
    REQUIRE_FALSE(rl.would_exceed());
    rl.record_usage(400);  // total 1000 -> at limit
    REQUIRE(rl.would_exceed());

    rl.reset();
    REQUIRE_FALSE(rl.would_exceed());
}

TEST_CASE("RateLimiter tracks the request budget via acquire", "[rate_limit]") {
    boost::asio::io_context ioc;
    ai::RateLimiter rl(ai::RateLimitConfig{
        .max_requests_per_minute = 2, .max_tokens_per_minute = 100000});

    run(rl.acquire(), ioc);
    REQUIRE_FALSE(rl.would_exceed());
    run(rl.acquire(), ioc);  // 2 requests -> at limit
    REQUIRE(rl.would_exceed());
}
