#include <ai/util/rate_limiter.hpp>
#include <thread>

namespace ai {

RateLimiter::RateLimiter(RateLimitConfig config)
    : config_(config) {
    current_window_.start = std::chrono::steady_clock::now();
}

void RateLimiter::maybe_reset_window() {
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - current_window_.start);
    if (elapsed.count() >= 60) {
        current_window_.start = now;
        current_window_.request_count = 0;
        current_window_.token_count = 0;
    }
}

Task<void> RateLimiter::acquire() {
    while (true) {
        {
            std::lock_guard lock(mutex_);
            maybe_reset_window();
            if (current_window_.request_count < config_.max_requests_per_minute &&
                current_window_.token_count < config_.max_tokens_per_minute) {
                current_window_.request_count++;
                co_return;
            }
        }
        // Wait and retry — in production use async sleep
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

void RateLimiter::record_usage(int tokens) {
    std::lock_guard lock(mutex_);
    current_window_.token_count += tokens;
}

bool RateLimiter::would_exceed() const {
    std::lock_guard lock(mutex_);
    return current_window_.request_count >= config_.max_requests_per_minute ||
           current_window_.token_count >= config_.max_tokens_per_minute;
}

void RateLimiter::reset() {
    std::lock_guard lock(mutex_);
    current_window_.start = std::chrono::steady_clock::now();
    current_window_.request_count = 0;
    current_window_.token_count = 0;
}

} // namespace ai
