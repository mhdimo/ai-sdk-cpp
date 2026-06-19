#pragma once

#include <ai/stream/async_generator.hpp>
#include <ai/error/api_call_error.hpp>
#include <chrono>
#include <cmath>
#include <random>
#include <thread>
#include <future>
#include <coroutine>
#include <type_traits>

namespace ai {

struct RetryOptions {
    int max_retries = 2;
    std::chrono::milliseconds initial_delay{1000};
    double backoff_factor = 2.0;
};

namespace detail {

inline std::chrono::milliseconds compute_delay(const RetryOptions& opts, int attempt) {
    auto base = opts.initial_delay.count() * std::pow(opts.backoff_factor, attempt);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> jitter(0.8, 1.2);
    return std::chrono::milliseconds(static_cast<int64_t>(base * jitter(gen)));
}

inline bool should_retry(const std::exception& e) {
    if (auto* api_err = dynamic_cast<const error::ApiCallError*>(&e)) {
        return api_err->is_retryable();
    }
    if (dynamic_cast<const error::StreamError*>(&e)) {
        return true;
    }
    return false;
}

inline std::chrono::milliseconds get_retry_delay(const std::exception& e, const RetryOptions& opts, int attempt) {
    if (auto* api_err = dynamic_cast<const error::ApiCallError*>(&e)) {
        auto server_delay = api_err->retry_after();
        if (server_delay) {
            return std::chrono::duration_cast<std::chrono::milliseconds>(*server_delay);
        }
    }
    return compute_delay(opts, attempt);
}

// Awaitable that sleeps asynchronously by launching a detached timer thread.
// The coroutine suspends and is resumed when the sleep completes.
struct AsyncSleep {
    std::chrono::milliseconds duration;

    bool await_ready() const noexcept { return duration.count() <= 0; }

    void await_suspend(std::coroutine_handle<> h) const {
        std::thread([d = this->duration, h]() {
            std::this_thread::sleep_for(d);
            h.resume();
        }).detach();
    }

    void await_resume() const noexcept {}
};

} // namespace detail

// Async retry: works within our Task<T> coroutine framework.
// Calls fn() which must return Task<T>, and retries on retryable errors
// with exponential backoff using async sleep (non-blocking from the
// coroutine's perspective).
template <typename F>
auto retry_async(RetryOptions opts, F&& fn)
    -> Task<typename std::invoke_result_t<F>::value_type>
    requires requires {
        typename std::invoke_result_t<F>::value_type;
        requires (!std::is_void_v<typename std::invoke_result_t<F>::value_type>);
    }
{
    using ResultType = typename std::invoke_result_t<F>::value_type;

    for (int attempt = 0;; ++attempt) {
        std::chrono::milliseconds retry_delay{};
        bool do_retry = false;
        try {
            ResultType result = co_await fn();
            co_return std::move(result);
        } catch (const std::exception& e) {
            if (attempt >= opts.max_retries || !detail::should_retry(e)) {
                throw;
            }
            retry_delay = detail::get_retry_delay(e, opts, attempt);
            do_retry = true;
        }
        // co_await cannot appear in a catch handler; sleep after it.
        if (do_retry) {
            co_await detail::AsyncSleep{retry_delay};
        }
    }
}

// Async retry specialization for Task<void>.
template <typename F>
auto retry_async(RetryOptions opts, F&& fn) -> Task<void>
    requires requires {
        typename std::invoke_result_t<F>::value_type;
        requires std::is_void_v<typename std::invoke_result_t<F>::value_type>;
    }
{
    for (int attempt = 0;; ++attempt) {
        std::chrono::milliseconds retry_delay{};
        bool do_retry = false;
        try {
            co_await fn();
            co_return;
        } catch (const std::exception& e) {
            if (attempt >= opts.max_retries || !detail::should_retry(e)) {
                throw;
            }
            retry_delay = detail::get_retry_delay(e, opts, attempt);
            do_retry = true;
        }
        // co_await cannot appear in a catch handler; sleep after it.
        if (do_retry) {
            co_await detail::AsyncSleep{retry_delay};
        }
    }
}

// Synchronous retry (legacy): uses std::this_thread::sleep_for.
// Kept for non-coroutine contexts.
template <typename F>
auto retry(RetryOptions opts, F&& fn) -> decltype(fn()) {
    for (int attempt = 0;; ++attempt) {
        try {
            return fn();
        } catch (const std::exception& e) {
            if (attempt >= opts.max_retries || !detail::should_retry(e)) {
                throw;
            }
            auto delay = detail::compute_delay(opts, attempt);
            std::this_thread::sleep_for(delay);
        }
    }
}

} // namespace ai
