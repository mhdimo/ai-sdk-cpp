#pragma once

#include <ai/stream/async_generator.hpp>
#include <ai/stream/sse_parser.hpp>
#include <ai/stream/stream_part.hpp>
#include <ai/util/json.hpp>
#include <boost/json.hpp>
#include <functional>
#include <optional>

namespace ai::stream {

template <typename T>
AsyncGenerator<T> transform(
    AsyncGenerator<SseEvent> sse_stream,
    std::function<std::optional<T>(const boost::json::value&)> parser
) {
    while (auto event = co_await sse_stream.next()) {
        if (event->data == "[DONE]") continue;
        auto parsed = ai::json::safe_parse(event->data);
        if (!parsed) continue;
        if (auto result = parser(*parsed)) {
            co_yield std::move(*result);
        }
    }
}

} // namespace ai::stream
