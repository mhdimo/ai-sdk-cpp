#include <catch2/catch_test_macros.hpp>

#include <ai/util/retry.hpp>
#include <ai/error/api_call_error.hpp>

#include <stdexcept>

TEST_CASE("compute_delay grows exponentially within jitter bounds", "[retry]") {
    ai::RetryOptions opts;  // initial 1000ms, backoff 2.0

    auto d0 = ai::detail::compute_delay(opts, 0);  // base 1000 -> [800,1200]
    auto d1 = ai::detail::compute_delay(opts, 1);  // base 2000 -> [1600,2400]

    REQUIRE(d0.count() >= 800);
    REQUIRE(d0.count() <= 1200);
    REQUIRE(d1.count() >= 1600);
    REQUIRE(d1.count() <= 2400);
    REQUIRE(d1 > d0);
}

TEST_CASE("should_retry classifies retryable vs non-retryable", "[retry]") {
    ai::error::Headers h;
    ai::error::RateLimitError rate("rate limited", "https://x", "{}", h);
    ai::error::BadRequestError bad("bad request", "https://x", "{}", h);
    ai::error::StreamError stream("stream broke");
    std::runtime_error other("unrelated");

    REQUIRE(ai::detail::should_retry(rate));
    REQUIRE_FALSE(ai::detail::should_retry(bad));
    REQUIRE(ai::detail::should_retry(stream));
    REQUIRE_FALSE(ai::detail::should_retry(other));
}

TEST_CASE("get_retry_delay falls back to backoff without retry-after", "[retry]") {
    ai::RetryOptions opts;
    ai::error::StreamError stream("transient");

    auto d = ai::detail::get_retry_delay(stream, opts, 0);
    REQUIRE(d.count() >= 800);
    REQUIRE(d.count() <= 1200);
}
