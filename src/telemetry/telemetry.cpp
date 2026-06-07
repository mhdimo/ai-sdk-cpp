#include <ai/telemetry/telemetry.hpp>
#include <iostream>
#include <mutex>

namespace ai::telemetry {

namespace {
    TracerPtr global_tracer;
    std::mutex tracer_mutex;
}

void set_tracer(TracerPtr tracer) {
    std::lock_guard lock(tracer_mutex);
    global_tracer = std::move(tracer);
}

TracerPtr get_tracer() {
    std::lock_guard lock(tracer_mutex);
    return global_tracer;
}

void ConsoleTracer::on_start(const SpanAttributes& attrs) {
    std::cerr << "[ai] " << attrs.operation << " start"
              << " model=" << attrs.model_id
              << " provider=" << attrs.provider << "\n";
}

void ConsoleTracer::on_end(const SpanAttributes& attrs) {
    std::cerr << "[ai] " << attrs.operation << " end";
    if (attrs.duration_ms) std::cerr << " duration=" << *attrs.duration_ms << "ms";
    if (attrs.input_tokens) std::cerr << " input_tokens=" << *attrs.input_tokens;
    if (attrs.output_tokens) std::cerr << " output_tokens=" << *attrs.output_tokens;
    if (attrs.finish_reason) std::cerr << " finish=" << *attrs.finish_reason;
    std::cerr << "\n";
}

void ConsoleTracer::on_error(const SpanAttributes& attrs) {
    std::cerr << "[ai] " << attrs.operation << " ERROR";
    if (attrs.error) std::cerr << " " << *attrs.error;
    std::cerr << "\n";
}

} // namespace ai::telemetry
