#pragma once

#include <ai/model/middleware.hpp>
#include <ai/telemetry/telemetry.hpp>
#include <chrono>

namespace ai {

class TelemetryMiddleware : public Middleware {
public:
    Task<GenerateResult> wrap_generate(
        GenerateFn do_generate,
        StreamFn do_stream,
        CallOptions& options
    ) override {
        auto tracer = telemetry::get_tracer();
        if (!tracer) co_return co_await do_generate();

        telemetry::SpanAttributes attrs;
        attrs.operation = "ai.generateText";
        tracer->on_start(attrs);

        auto start = std::chrono::steady_clock::now();
        try {
            auto result = co_await do_generate();
            auto end = std::chrono::steady_clock::now();

            attrs.duration_ms = std::chrono::duration<double, std::milli>(end - start).count();
            if (result.usage.input_tokens.total) attrs.input_tokens = *result.usage.input_tokens.total;
            if (result.usage.output_tokens.total) attrs.output_tokens = *result.usage.output_tokens.total;

            switch (result.finish_reason) {
                case FinishReason::Stop: attrs.finish_reason = "stop"; break;
                case FinishReason::Length: attrs.finish_reason = "length"; break;
                case FinishReason::ToolCalls: attrs.finish_reason = "tool_calls"; break;
                case FinishReason::ContentFilter: attrs.finish_reason = "content_filter"; break;
                default: attrs.finish_reason = "other"; break;
            }

            tracer->on_end(attrs);
            co_return result;
        } catch (const std::exception& e) {
            auto end = std::chrono::steady_clock::now();
            attrs.duration_ms = std::chrono::duration<double, std::milli>(end - start).count();
            attrs.error = e.what();
            tracer->on_error(attrs);
            throw;
        }
    }

    Task<StreamResult> wrap_stream(
        GenerateFn do_generate,
        StreamFn do_stream,
        CallOptions& options
    ) override {
        auto tracer = telemetry::get_tracer();
        if (!tracer) co_return co_await do_stream();

        telemetry::SpanAttributes attrs;
        attrs.operation = "ai.streamText";
        tracer->on_start(attrs);
        co_return co_await do_stream();
    }
};

} // namespace ai
