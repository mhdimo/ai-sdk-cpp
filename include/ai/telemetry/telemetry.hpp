#pragma once

#include <string>
#include <optional>
#include <unordered_map>
#include <memory>

namespace ai::telemetry {

struct SpanAttributes {
    std::string operation;
    std::string model_id;
    std::string provider;
    std::optional<int> input_tokens;
    std::optional<int> output_tokens;
    std::optional<double> duration_ms;
    std::optional<std::string> finish_reason;
    std::optional<std::string> error;
    std::unordered_map<std::string, std::string> extra;
};

class Tracer {
public:
    virtual ~Tracer() = default;
    virtual void on_start(const SpanAttributes& attrs) = 0;
    virtual void on_end(const SpanAttributes& attrs) = 0;
    virtual void on_error(const SpanAttributes& attrs) = 0;
};

using TracerPtr = std::shared_ptr<Tracer>;

void set_tracer(TracerPtr tracer);
TracerPtr get_tracer();

class ConsoleTracer : public Tracer {
public:
    void on_start(const SpanAttributes& attrs) override;
    void on_end(const SpanAttributes& attrs) override;
    void on_error(const SpanAttributes& attrs) override;
};

} // namespace ai::telemetry
