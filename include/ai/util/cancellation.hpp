#pragma once

#include <atomic>
#include <memory>
#include <functional>
#include <vector>
#include <mutex>
#include <stdexcept>

namespace ai {

class OperationCancelled : public std::runtime_error {
public:
    OperationCancelled() : std::runtime_error("Operation cancelled") {}
};

class CancellationToken {
public:
    CancellationToken() : state_(std::make_shared<State>()) {}

    bool is_cancelled() const noexcept {
        return state_->cancelled.load(std::memory_order_acquire);
    }

    void throw_if_cancelled() const {
        if (is_cancelled()) {
            throw OperationCancelled{};
        }
    }

    using Callback = std::function<void()>;

    void on_cancel(Callback cb) {
        std::lock_guard lock(state_->mutex);
        if (state_->cancelled.load(std::memory_order_acquire)) {
            cb();
        } else {
            state_->callbacks.push_back(std::move(cb));
        }
    }

private:
    friend class CancellationSource;

    struct State {
        std::atomic<bool> cancelled{false};
        std::mutex mutex;
        std::vector<Callback> callbacks;
    };

    std::shared_ptr<State> state_;
};

class CancellationSource {
public:
    CancellationSource() : state_(std::make_shared<CancellationToken::State>()) {}

    CancellationToken token() const {
        CancellationToken t;
        t.state_ = state_;
        return t;
    }

    void cancel() {
        std::vector<CancellationToken::Callback> cbs;
        {
            std::lock_guard lock(state_->mutex);
            if (state_->cancelled.exchange(true, std::memory_order_release)) {
                return;
            }
            cbs = std::move(state_->callbacks);
        }
        for (auto& cb : cbs) {
            cb();
        }
    }

    bool is_cancelled() const noexcept {
        return state_->cancelled.load(std::memory_order_acquire);
    }

private:
    std::shared_ptr<CancellationToken::State> state_;
};

} // namespace ai
