#pragma once

#include <ai/stream/async_generator.hpp>
#include <string>
#include <optional>
#include <vector>
#include <cstdint>

namespace ai::stream {

struct SseEvent {
    std::string event = "message";
    std::string data;
    std::string id;
    std::optional<int> retry;
};

class SseParser {
public:
    SseParser() = default;

    AsyncGenerator<SseEvent> parse(AsyncGenerator<std::vector<uint8_t>> byte_stream);

    void feed(std::string_view chunk);
    std::optional<SseEvent> next_event();

    const std::string& last_id() const noexcept { return last_id_; }

private:
    std::string buffer_;
    std::string event_type_ = "message";
    std::string data_buffer_;
    std::string last_id_;
    bool bom_stripped_ = false;

    void strip_bom();
    void process_line(std::string_view line);
    std::optional<SseEvent> dispatch_event();
};

} // namespace ai::stream
