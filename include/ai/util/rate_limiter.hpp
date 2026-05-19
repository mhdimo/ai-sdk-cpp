#pragma once

#include <ai/stream/async_generator.hpp>
#include <chrono>
#include <mutex>

namespace ai {

struct RateLimitConfig {
    int max_requests_per_minute = 60;
    int max_tokens_per_minute = 100000;
};

class RateLimiter {
public:
    explicit RateLimiter(RateLimitConfig config = {});

    Task<void> acquire();
    void record_usage(int tokens);
    bool would_exceed() const;
    void reset();

private:
    RateLimitConfig config_;
    struct Window {
        std::chrono::steady_clock::time_point start;
        int request_count = 0;
        int token_count = 0;
    };
    Window current_window_;
    mutable std::mutex mutex_;

    void maybe_reset_window();
};

} // namespace ai
