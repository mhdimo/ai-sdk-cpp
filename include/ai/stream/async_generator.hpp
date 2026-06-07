#pragma once

#include <coroutine>
#include <optional>
#include <exception>
#include <utility>
#include <cassert>

namespace ai {

template <typename T>
class AsyncGenerator {
public:
    struct promise_type {
        std::optional<T> current_value;
        std::exception_ptr exception;
        std::coroutine_handle<> consumer;

        AsyncGenerator get_return_object() {
            return AsyncGenerator{std::coroutine_handle<promise_type>::from_promise(*this)};
        }

        std::suspend_always initial_suspend() noexcept { return {}; }
        std::suspend_always final_suspend() noexcept { return {}; }

        void unhandled_exception() {
            exception = std::current_exception();
        }

        void return_void() {}

        struct YieldAwaitable {
            bool await_ready() const noexcept { return false; }
            std::coroutine_handle<> await_suspend(std::coroutine_handle<promise_type>) noexcept {
                return {}; // will be set by next()
            }
            void await_resume() noexcept {}
        };

        std::suspend_always yield_value(T value) {
            current_value = std::move(value);
            return {};
        }
    };

    struct NextAwaitable {
        std::coroutine_handle<promise_type> producer;

        bool await_ready() const noexcept {
            return !producer || producer.done();
        }

        std::coroutine_handle<> await_suspend(std::coroutine_handle<> consumer) noexcept {
            producer.promise().consumer = consumer;
            return producer;
        }

        std::optional<T> await_resume() {
            if (!producer || producer.done()) {
                return std::nullopt;
            }
            if (producer.promise().exception) {
                std::rethrow_exception(producer.promise().exception);
            }
            auto val = std::move(producer.promise().current_value);
            producer.promise().current_value.reset();
            return val;
        }
    };

    AsyncGenerator() : handle_(nullptr) {}
    explicit AsyncGenerator(std::coroutine_handle<promise_type> h) : handle_(h) {}

    ~AsyncGenerator() {
        if (handle_) handle_.destroy();
    }

    AsyncGenerator(const AsyncGenerator&) = delete;
    AsyncGenerator& operator=(const AsyncGenerator&) = delete;

    AsyncGenerator(AsyncGenerator&& other) noexcept : handle_(other.handle_) {
        other.handle_ = nullptr;
    }

    AsyncGenerator& operator=(AsyncGenerator&& other) noexcept {
        if (this != &other) {
            if (handle_) handle_.destroy();
            handle_ = other.handle_;
            other.handle_ = nullptr;
        }
        return *this;
    }

    NextAwaitable next() {
        return NextAwaitable{handle_};
    }

    bool done() const {
        return !handle_ || handle_.done();
    }

    explicit operator bool() const {
        return handle_ && !handle_.done();
    }

private:
    std::coroutine_handle<promise_type> handle_;
};

template <typename T>
class Task {
public:
    using value_type = T;

    struct promise_type {
        std::optional<T> result;
        std::exception_ptr exception;
        std::coroutine_handle<> continuation;

        Task get_return_object() {
            return Task{std::coroutine_handle<promise_type>::from_promise(*this)};
        }

        std::suspend_always initial_suspend() noexcept { return {}; }

        struct FinalAwaitable {
            bool await_ready() const noexcept { return false; }
            std::coroutine_handle<> await_suspend(std::coroutine_handle<promise_type> h) noexcept {
                if (h.promise().continuation)
                    return h.promise().continuation;
                return std::noop_coroutine();
            }
            void await_resume() noexcept {}
        };

        FinalAwaitable final_suspend() noexcept { return {}; }

        void return_value(T value) {
            result = std::move(value);
        }

        void unhandled_exception() {
            exception = std::current_exception();
        }
    };

    Task() : handle_(nullptr) {}
    explicit Task(std::coroutine_handle<promise_type> h) : handle_(h) {}

    ~Task() {
        if (handle_) handle_.destroy();
    }

    Task(const Task&) = delete;
    Task& operator=(const Task&) = delete;

    Task(Task&& other) noexcept : handle_(other.handle_) {
        other.handle_ = nullptr;
    }

    Task& operator=(Task&& other) noexcept {
        if (this != &other) {
            if (handle_) handle_.destroy();
            handle_ = other.handle_;
            other.handle_ = nullptr;
        }
        return *this;
    }

    bool await_ready() const noexcept {
        return handle_.done();
    }

    std::coroutine_handle<> await_suspend(std::coroutine_handle<> awaiting) noexcept {
        handle_.promise().continuation = awaiting;
        return handle_;
    }

    T await_resume() {
        if (handle_.promise().exception) {
            std::rethrow_exception(handle_.promise().exception);
        }
        return std::move(*handle_.promise().result);
    }

    void start() {
        if (handle_ && !handle_.done()) {
            handle_.resume();
        }
    }

    bool done() const {
        return !handle_ || handle_.done();
    }

    T get() {
        if (!handle_.done()) {
            handle_.resume();
        }
        if (handle_.promise().exception) {
            std::rethrow_exception(handle_.promise().exception);
        }
        return std::move(*handle_.promise().result);
    }

private:
    std::coroutine_handle<promise_type> handle_;
};

template <>
class Task<void> {
public:
    using value_type = void;

    struct promise_type {
        std::exception_ptr exception;
        std::coroutine_handle<> continuation;

        Task get_return_object() {
            return Task{std::coroutine_handle<promise_type>::from_promise(*this)};
        }

        std::suspend_always initial_suspend() noexcept { return {}; }

        struct FinalAwaitable {
            bool await_ready() const noexcept { return false; }
            std::coroutine_handle<> await_suspend(std::coroutine_handle<promise_type> h) noexcept {
                if (h.promise().continuation)
                    return h.promise().continuation;
                return std::noop_coroutine();
            }
            void await_resume() noexcept {}
        };

        FinalAwaitable final_suspend() noexcept { return {}; }
        void return_void() {}

        void unhandled_exception() {
            exception = std::current_exception();
        }
    };

    Task() : handle_(nullptr) {}
    explicit Task(std::coroutine_handle<promise_type> h) : handle_(h) {}

    ~Task() {
        if (handle_) handle_.destroy();
    }

    Task(const Task&) = delete;
    Task& operator=(const Task&) = delete;

    Task(Task&& other) noexcept : handle_(other.handle_) {
        other.handle_ = nullptr;
    }

    Task& operator=(Task&& other) noexcept {
        if (this != &other) {
            if (handle_) handle_.destroy();
            handle_ = other.handle_;
            other.handle_ = nullptr;
        }
        return *this;
    }

    bool await_ready() const noexcept {
        return handle_.done();
    }

    std::coroutine_handle<> await_suspend(std::coroutine_handle<> awaiting) noexcept {
        handle_.promise().continuation = awaiting;
        return handle_;
    }

    void await_resume() {
        if (handle_.promise().exception) {
            std::rethrow_exception(handle_.promise().exception);
        }
    }

    void start() {
        if (handle_ && !handle_.done()) {
            handle_.resume();
        }
    }

private:
    std::coroutine_handle<promise_type> handle_;
};

} // namespace ai
