#pragma once

#include <ai/stream/stream_part.hpp>
#include <string>
#include <string_view>
#include <optional>
#include <boost/json.hpp>

namespace ai::stream {

class DataStreamEncoder {
public:
    std::string encode(const StreamPart& part);
    std::string encode_text_delta(std::string_view text);
    std::string encode_tool_call_start(std::string_view id, std::string_view name);
    std::string encode_tool_call_delta(std::string_view id, std::string_view delta);
    std::string encode_finish(FinishReason reason, const Usage& usage);
    std::string encode_error(std::string_view message);
};

class DataStreamDecoder {
public:
    std::optional<StreamPart> decode_line(std::string_view line);
};

} // namespace ai::stream
