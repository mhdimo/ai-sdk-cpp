#pragma once

#include <ai/stream/async_generator.hpp>
#include <ai/stream/sse_parser.hpp>
#include <ai/error/api_call_error.hpp>
#include <ai/util/cancellation.hpp>
#include <chrono>
#include <functional>
#include <optional>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <atomic>

namespace ai::stream {

struct StreamConfig {
    size_t max_buffer_size = 256;
    std::chrono::seconds idle_timeout{60};
    int max_reconnect_attempts = 3;
    std::chrono::milliseconds initial_backoff{500};
    std::chrono::milliseconds max_backoff{30000};
    double backoff_multiplier = 2.0;
};

template <typename T>
class BoundedChannel {
public:
    explicit BoundedChannel(size_t capacity) : capacity_(capacity) {}

    bool try_push(T item) {
        std::unique_lock lock(mutex_);
        if (items_.size() >= capacity_) return false;
        items_.push(std::move(item));
        cv_consumer_.notify_one();
        return true;
    }

    void push(T item) {
        std::unique_lock lock(mutex_);
        cv_producer_.wait(lock, [this] {
            return items_.size() < capacity_ || closed_;
        });
        if (closed_) return;
        items_.push(std::move(item));
        cv_consumer_.notify_one();
    }

    std::optional<T> pop() {
        std::unique_lock lock(mutex_);
        cv_consumer_.wait(lock, [this] {
            return !items_.empty() || closed_;
        });
        if (items_.empty()) return std::nullopt;
        T item = std::move(items_.front());
        items_.pop();
        cv_producer_.notify_one();
        return item;
    }

    template <typename Duration>
    std::optional<T> pop_for(Duration timeout) {
        std::unique_lock lock(mutex_);
        if (!cv_consumer_.wait_for(lock, timeout, [this] {
            return !items_.empty() || closed_;
        })) {
            return std::nullopt;
        }
        if (items_.empty()) return std::nullopt;
        T item = std::move(items_.front());
        items_.pop();
        cv_producer_.notify_one();
        return item;
    }

    void close() {
        std::lock_guard lock(mutex_);
        closed_ = true;
        cv_consumer_.notify_all();
        cv_producer_.notify_all();
    }

    bool is_closed() const {
        std::lock_guard lock(mutex_);
        return closed_;
    }

    size_t size() const {
        std::lock_guard lock(mutex_);
        return items_.size();
    }

private:
    mutable std::mutex mutex_;
    std::condition_variable cv_consumer_;
    std::condition_variable cv_producer_;
    std::queue<T> items_;
    size_t capacity_;
    bool closed_ = false;
};

using StreamFactory = std::function<AsyncGenerator<SseEvent>()>;

inline std::chrono::milliseconds compute_backoff(
    int attempt,
    std::chrono::milliseconds initial,
    std::chrono::milliseconds max_val,
    double multiplier
) {
    auto ms = initial.count();
    for (int i = 0; i < attempt; ++i) {
        ms = static_cast<long long>(ms * multiplier);
        if (ms > max_val.count()) {
            ms = max_val.count();
            break;
        }
    }
    return std::chrono::milliseconds(ms);
}

inline AsyncGenerator<SseEvent> make_resilient_stream(
    StreamFactory factory,
    StreamConfig config = {},
    CancellationToken cancel = {}
) {
    int attempt = 0;
    while (true) {
        try {
            auto stream = factory();
            while (auto event = co_await stream.next()) {
                cancel.throw_if_cancelled();
                attempt = 0;
                co_yield std::move(*event);
            }
            break;
        } catch (const error::ApiCallError& e) {
            if (!e.is_retryable() || attempt >= config.max_reconnect_attempts) {
                throw;
            }
        } catch (const error::StreamError&) {
            if (attempt >= config.max_reconnect_attempts) throw;
        } catch (...) {
            if (attempt >= config.max_reconnect_attempts) throw;
        }

        ++attempt;
        auto delay = compute_backoff(
            attempt - 1, config.initial_backoff,
            config.max_backoff, config.backoff_multiplier);
        std::this_thread::sleep_for(delay);
        cancel.throw_if_cancelled();
    }
}

} // namespace ai::stream
