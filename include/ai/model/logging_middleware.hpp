#pragma once

#include <ai/model/middleware.hpp>
#include <chrono>
#include <functional>
#include <iostream>
#include <string>

namespace ai {

/// A middleware that logs each model call with model name, token usage, and duration.
class LoggingMiddleware : public Middleware {
public:
    using LogFn = std::function<void(const std::string&)>;

    /// Constructs a logging middleware. If no log function is provided,
    /// logs to std::cerr by default.
    explicit LoggingMiddleware(LogFn log_fn = nullptr)
        : log_fn_(std::move(log_fn)) {}

    Task<GenerateResult> wrap_generate(
        GenerateFn do_generate,
        StreamFn do_stream,
        CallOptions& options
    ) override {
        auto start = std::chrono::steady_clock::now();
        log("[generate] Starting call...");

        auto result = co_await do_generate();

        auto end = std::chrono::steady_clock::now();
        auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        std::string msg = "[generate] Completed in " + std::to_string(duration_ms) + "ms";
        if (result.usage.input_tokens.total) {
            msg += " | input_tokens=" + std::to_string(*result.usage.input_tokens.total);
        }
        if (result.usage.output_tokens.total) {
            msg += " | output_tokens=" + std::to_string(*result.usage.output_tokens.total);
        }
        msg += " | finish_reason=" + finish_reason_str(result.finish_reason);
        log(msg);

        co_return result;
    }

    Task<StreamResult> wrap_stream(
        GenerateFn do_generate,
        StreamFn do_stream,
        CallOptions& options
    ) override {
        log("[stream] Starting call...");
        auto result = co_await do_stream();
        log("[stream] Stream opened");
        co_return result;
    }

private:
    LogFn log_fn_;

    void log(const std::string& msg) {
        if (log_fn_) {
            log_fn_(msg);
        } else {
            std::cerr << "[LoggingMiddleware] " << msg << "\n";
        }
    }

    static std::string finish_reason_str(FinishReason reason) {
        switch (reason) {
            case FinishReason::Stop: return "stop";
            case FinishReason::Length: return "length";
            case FinishReason::ContentFilter: return "content_filter";
            case FinishReason::ToolCalls: return "tool_calls";
            case FinishReason::Error: return "error";
            case FinishReason::Other: return "other";
        }
        return "unknown";
    }
};

inline MiddlewarePtr make_logging_middleware(LoggingMiddleware::LogFn log_fn = nullptr) {
    return std::make_shared<LoggingMiddleware>(std::move(log_fn));
}

} // namespace ai
