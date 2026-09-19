#include <ai/stream/data_stream.hpp>
#include <ai/util/json.hpp>
#include <boost/json.hpp>
#include <sstream>
#include <utility>

namespace ai::stream {

namespace json = boost::json;

std::string DataStreamEncoder::encode(const StreamPart& part) {
    return std::visit([this](auto& p) -> std::string {
        using T = std::decay_t<decltype(p)>;
        if constexpr (std::is_same_v<T, TextDelta>) {
            return encode_text_delta(p.delta);
        } else if constexpr (std::is_same_v<T, ToolInputStart>) {
            return encode_tool_call_start(p.id, p.tool_name);
        } else if constexpr (std::is_same_v<T, ToolInputDelta>) {
            return encode_tool_call_delta(p.id, p.delta);
        } else if constexpr (std::is_same_v<T, ToolInputEnd>) {
            return "3:" + json::serialize(json::object{{"toolCallId", p.id}}) + "\n";
        } else if constexpr (std::is_same_v<T, FinishPart>) {
            return encode_finish(p.reason, p.usage);
        } else if constexpr (std::is_same_v<T, ErrorPart>) {
            return encode_error(p.message);
        }
        return "";
    }, part);
}

std::string DataStreamEncoder::encode_text_delta(std::string_view text) {
    return "0:" + json::serialize(json::value(std::string(text))) + "\n";
}

std::string DataStreamEncoder::encode_tool_call_start(std::string_view id, std::string_view name) {
    json::object obj;
    obj["toolCallId"] = std::string(id);
    obj["toolName"] = std::string(name);
    return "1:" + json::serialize(obj) + "\n";
}

std::string DataStreamEncoder::encode_tool_call_delta(std::string_view id, std::string_view delta) {
    json::object obj;
    obj["toolCallId"] = std::string(id);
    obj["argsTextDelta"] = std::string(delta);
    return "2:" + json::serialize(obj) + "\n";
}

std::string DataStreamEncoder::encode_finish(FinishReason reason, const Usage& usage) {
    json::object obj;
    switch (reason) {
        case FinishReason::Stop: obj["finishReason"] = "stop"; break;
        case FinishReason::Length: obj["finishReason"] = "length"; break;
        case FinishReason::ToolCalls: obj["finishReason"] = "tool-calls"; break;
        default: obj["finishReason"] = "other"; break;
    }
    json::object u;
    if (usage.input_tokens.total) u["promptTokens"] = *usage.input_tokens.total;
    if (usage.output_tokens.total) u["completionTokens"] = *usage.output_tokens.total;
    obj["usage"] = std::move(u);
    return "d:" + json::serialize(obj) + "\n";
}

std::string DataStreamEncoder::encode_error(std::string_view message) {
    return "e:" + json::serialize(json::value(std::string(message))) + "\n";
}

std::optional<StreamPart> DataStreamDecoder::decode_line(std::string_view line) {
    if (line.size() < 2 || line[1] != ':') return std::nullopt;

    char type = line[0];
    auto payload = line.substr(2);

    // These frames arrive off a socket, so a truncated or non-conforming one is
    // the expected case rather than an exceptional one. Every parse and every
    // field lookup below therefore has to fail as nullopt: an exception here
    // unwinds out of the caller's read loop, which is the one place it cannot be
    // handled. ai::json wraps boost::json's non-throwing forms for exactly this.
    auto val = ai::json::safe_parse(payload);
    if (!val) return std::nullopt;

    switch (type) {
        case '0': {
            if (!val->is_string()) return std::nullopt;
            return TextDelta{.id = "0", .delta = std::string(val->as_string())};
        }
        case '1': {
            auto id = ai::json::get_string(*val, "toolCallId");
            auto name = ai::json::get_string(*val, "toolName");
            if (!id || !name) return std::nullopt;
            return ToolInputStart{.id = std::move(*id), .tool_name = std::move(*name)};
        }
        case '2': {
            auto id = ai::json::get_string(*val, "toolCallId");
            auto delta = ai::json::get_string(*val, "argsTextDelta");
            if (!id || !delta) return std::nullopt;
            return ToolInputDelta{.id = std::move(*id), .delta = std::move(*delta)};
        }
        case '3': {
            auto id = ai::json::get_string(*val, "toolCallId");
            if (!id) return std::nullopt;
            return ToolInputEnd{.id = std::move(*id)};
        }
        case 'd': {
            if (!val->is_object()) return std::nullopt;
            FinishReason reason = FinishReason::Stop;
            if (auto fr = ai::json::get_string(*val, "finishReason")) {
                if (*fr == "length") reason = FinishReason::Length;
                else if (*fr == "tool-calls") reason = FinishReason::ToolCalls;
            }
            Usage usage;
            // `usage` is nested one level down, so it needs its own lookup
            // before the accessors apply.
            auto& obj = val->as_object();
            if (auto it = obj.find("usage"); it != obj.end() && it->value().is_object()) {
                if (auto p = ai::json::get_int(it->value(), "promptTokens")) {
                    usage.input_tokens.total = static_cast<int>(*p);
                }
                if (auto c = ai::json::get_int(it->value(), "completionTokens")) {
                    usage.output_tokens.total = static_cast<int>(*c);
                }
            }
            return FinishPart{.reason = reason, .usage = usage};
        }
        case 'e': {
            if (!val->is_string()) return std::nullopt;
            return ErrorPart{.message = std::string(val->as_string())};
        }
    }
    return std::nullopt;
}

} // namespace ai::stream
