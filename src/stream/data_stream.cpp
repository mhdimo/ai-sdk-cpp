#include <ai/stream/data_stream.hpp>
#include <boost/json.hpp>
#include <sstream>

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

    switch (type) {
        case '0': {
            auto val = json::parse(payload);
            return TextDelta{.id = "0", .delta = std::string(val.as_string())};
        }
        case '1': {
            auto val = json::parse(payload);
            auto& obj = val.as_object();
            return ToolInputStart{
                .id = std::string(obj.at("toolCallId").as_string()),
                .tool_name = std::string(obj.at("toolName").as_string()),
            };
        }
        case '2': {
            auto val = json::parse(payload);
            auto& obj = val.as_object();
            return ToolInputDelta{
                .id = std::string(obj.at("toolCallId").as_string()),
                .delta = std::string(obj.at("argsTextDelta").as_string()),
            };
        }
        case '3': {
            auto val = json::parse(payload);
            auto& obj = val.as_object();
            return ToolInputEnd{.id = std::string(obj.at("toolCallId").as_string())};
        }
        case 'd': {
            auto val = json::parse(payload);
            auto& obj = val.as_object();
            FinishReason reason = FinishReason::Stop;
            if (obj.contains("finishReason")) {
                auto fr = std::string(obj.at("finishReason").as_string());
                if (fr == "length") reason = FinishReason::Length;
                else if (fr == "tool-calls") reason = FinishReason::ToolCalls;
            }
            Usage usage;
            if (obj.contains("usage")) {
                auto& u = obj.at("usage").as_object();
                if (u.contains("promptTokens")) usage.input_tokens.total = (int)u.at("promptTokens").as_int64();
                if (u.contains("completionTokens")) usage.output_tokens.total = (int)u.at("completionTokens").as_int64();
            }
            return FinishPart{.reason = reason, .usage = usage};
        }
        case 'e': {
            auto val = json::parse(payload);
            return ErrorPart{.message = std::string(val.as_string())};
        }
    }
    return std::nullopt;
}

} // namespace ai::stream
