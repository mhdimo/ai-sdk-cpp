#include <ai/stream/sse_parser.hpp>
#include <charconv>

namespace ai::stream {

AsyncGenerator<SseEvent> SseParser::parse(AsyncGenerator<std::vector<uint8_t>> byte_stream) {
    while (auto chunk = co_await byte_stream.next()) {
        std::string_view data(reinterpret_cast<const char*>(chunk->data()), chunk->size());
        feed(data);

        while (auto event = next_event()) {
            co_yield std::move(*event);
        }
    }

    if (!buffer_.empty()) {
        buffer_ += '\n';
        while (auto event = next_event()) {
            co_yield std::move(*event);
        }
    }
}

void SseParser::strip_bom() {
    if (bom_stripped_) return;
    bom_stripped_ = true;
    if (buffer_.size() >= 3 &&
        static_cast<unsigned char>(buffer_[0]) == 0xEF &&
        static_cast<unsigned char>(buffer_[1]) == 0xBB &&
        static_cast<unsigned char>(buffer_[2]) == 0xBF) {
        buffer_.erase(0, 3);
    }
}

void SseParser::feed(std::string_view chunk) {
    buffer_.append(chunk);
    if (!bom_stripped_) strip_bom();
}

std::optional<SseEvent> SseParser::next_event() {
    while (true) {
        auto newline_pos = buffer_.find('\n');
        if (newline_pos == std::string::npos) {
            return std::nullopt;
        }

        std::string line = buffer_.substr(0, newline_pos);
        buffer_.erase(0, newline_pos + 1);

        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }

        if (line.empty()) {
            auto event = dispatch_event();
            if (event) return event;
            continue;
        }

        process_line(line);
    }
}

void SseParser::process_line(std::string_view line) {
    if (line.starts_with(':')) {
        return;
    }

    std::string_view field;
    std::string_view value;

    auto colon_pos = line.find(':');
    if (colon_pos == std::string_view::npos) {
        field = line;
        value = "";
    } else {
        field = line.substr(0, colon_pos);
        value = line.substr(colon_pos + 1);
        if (!value.empty() && value.front() == ' ') {
            value.remove_prefix(1);
        }
    }

    if (field == "event") {
        event_type_ = std::string(value);
    } else if (field == "data") {
        if (!data_buffer_.empty()) {
            data_buffer_ += '\n';
        }
        data_buffer_ += value;
    } else if (field == "id") {
        if (value.find('\0') == std::string_view::npos) {
            last_id_ = std::string(value);
        }
    } else if (field == "retry") {
        // Parse retry value - ignored in output but preserved in event
    }
}

std::optional<SseEvent> SseParser::dispatch_event() {
    if (data_buffer_.empty()) {
        event_type_ = "message";
        return std::nullopt;
    }

    SseEvent event;
    event.event = event_type_;
    event.data = std::move(data_buffer_);
    event.id = last_id_;

    data_buffer_.clear();
    event_type_ = "message";

    return event;
}

} // namespace ai::stream
