#include <ai/providers/openai/openai_realtime.hpp>
#include <ai/error/api_call_error.hpp>

namespace ai::providers::openai {

namespace json = boost::json;

static std::vector<uint8_t> base64_decode(std::string_view input) {
    static constexpr unsigned char table[] = {
        64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,
        64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,
        64,64,64,64,64,64,64,64,64,64,64,62,64,64,64,63,
        52,53,54,55,56,57,58,59,60,61,64,64,64,64,64,64,
        64, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,
        15,16,17,18,19,20,21,22,23,24,25,64,64,64,64,64,
        64,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,
        41,42,43,44,45,46,47,48,49,50,51,64,64,64,64,64,
        64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,
        64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,
        64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,
        64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,
        64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,
        64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,
        64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,
        64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64
    };

    std::vector<uint8_t> out;
    out.reserve(input.size() * 3 / 4);
    int val = 0, valb = -8;
    for (unsigned char c : input) {
        if (table[c] == 64) break;
        val = (val << 6) + table[c];
        valb += 6;
        if (valb >= 0) {
            out.push_back(static_cast<uint8_t>((val >> valb) & 0xFF));
            valb -= 8;
        }
    }
    return out;
}

static std::string base64_encode(const std::vector<uint8_t>& data) {
    static constexpr char table[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string result;
    result.reserve(((data.size() + 2) / 3) * 4);
    int val = 0, valb = -6;
    for (uint8_t c : data) {
        val = (val << 8) + c;
        valb += 8;
        while (valb >= 0) {
            result.push_back(table[(val >> valb) & 0x3F]);
            valb -= 6;
        }
    }
    if (valb > -6) result.push_back(table[((val << 8) >> (valb + 8)) & 0x3F]);
    while (result.size() % 4) result.push_back('=');
    return result;
}

OpenAIRealtimeSession::OpenAIRealtimeSession(
    boost::asio::io_context& ioc,
    RealtimeConfig config
)
    : config_(std::move(config))
    , ws_(ioc) {}

OpenAIRealtimeSession::~OpenAIRealtimeSession() = default;

Task<void> OpenAIRealtimeSession::connect(CancellationToken cancel) {
    std::string url = config_.base_url + "?model=" + config_.model;

    ai::http::WsHeaders headers;
    headers["Authorization"] = "Bearer " + config_.api_key;
    headers["OpenAI-Beta"] = "realtime=v1";

    co_await ws_.connect(url, headers, cancel);

    json::object session_config;
    if (config_.instructions) {
        session_config["instructions"] = *config_.instructions;
    }
    session_config["voice"] = config_.voice;
    session_config["input_audio_format"] = config_.input_audio_format;
    session_config["output_audio_format"] = config_.output_audio_format;
    session_config["temperature"] = config_.temperature;

    if (config_.max_response_output_tokens) {
        session_config["max_response_output_tokens"] = *config_.max_response_output_tokens;
    }

    if (!config_.tools.empty()) {
        json::array tools;
        for (auto& t : config_.tools) tools.push_back(t);
        session_config["tools"] = std::move(tools);
    }

    json::object turn_detection;
    turn_detection["type"] = config_.turn_detection;
    session_config["turn_detection"] = std::move(turn_detection);

    json::object update_msg;
    update_msg["type"] = "session.update";
    update_msg["session"] = std::move(session_config);
    co_await ws_.send(json::value(std::move(update_msg)));
}

AsyncGenerator<RealtimeEvent> OpenAIRealtimeSession::events() {
    auto msg_stream = ws_.messages();
    while (auto msg = co_await msg_stream.next()) {
        auto parsed = json::parse(*msg);
        if (!parsed.is_object()) continue;
        auto& obj = parsed.as_object();
        auto event = parse_server_event(obj);
        if (event) {
            co_yield std::move(*event);
        }
    }
}

std::optional<RealtimeEvent> OpenAIRealtimeSession::parse_server_event(
    const json::object& obj
) {
    auto type_it = obj.find("type");
    if (type_it == obj.end() || !type_it->value().is_string()) {
        return std::nullopt;
    }
    auto type = std::string_view(type_it->value().as_string());

    if (type == "session.created") {
        SessionCreated ev;
        if (auto it = obj.find("session"); it != obj.end() && it->value().is_object()) {
            ev.session = it->value().as_object();
        }
        return ev;
    }
    if (type == "session.updated") {
        SessionUpdated ev;
        if (auto it = obj.find("session"); it != obj.end() && it->value().is_object()) {
            ev.session = it->value().as_object();
        }
        return ev;
    }
    if (type == "response.created") {
        ResponseCreated ev;
        if (auto it = obj.find("response"); it != obj.end() && it->value().is_object()) {
            auto& resp = it->value().as_object();
            if (auto id = resp.find("id"); id != resp.end() && id->value().is_string()) {
                ev.response_id = std::string(id->value().as_string());
            }
        }
        return ev;
    }
    if (type == "response.done") {
        ResponseDone ev;
        if (auto it = obj.find("response"); it != obj.end() && it->value().is_object()) {
            auto& resp = it->value().as_object();
            if (auto id = resp.find("id"); id != resp.end() && id->value().is_string()) {
                ev.response_id = std::string(id->value().as_string());
            }
            if (auto s = resp.find("status"); s != resp.end() && s->value().is_string()) {
                ev.status = std::string(s->value().as_string());
            }
            if (auto u = resp.find("usage"); u != resp.end() && u->value().is_object()) {
                ev.usage = u->value().as_object();
            }
        }
        return ev;
    }
    if (type == "response.audio.delta") {
        AudioDelta ev;
        if (auto it = obj.find("response_id"); it != obj.end() && it->value().is_string())
            ev.response_id = std::string(it->value().as_string());
        if (auto it = obj.find("item_id"); it != obj.end() && it->value().is_string())
            ev.item_id = std::string(it->value().as_string());
        if (auto it = obj.find("output_index"); it != obj.end() && it->value().is_int64())
            ev.output_index = static_cast<int>(it->value().as_int64());
        if (auto it = obj.find("content_index"); it != obj.end() && it->value().is_int64())
            ev.content_index = static_cast<int>(it->value().as_int64());
        if (auto it = obj.find("delta"); it != obj.end() && it->value().is_string()) {
            ev.audio = base64_decode(it->value().as_string());
        }
        return ev;
    }
    if (type == "response.text.delta") {
        TextDelta ev;
        if (auto it = obj.find("response_id"); it != obj.end() && it->value().is_string())
            ev.response_id = std::string(it->value().as_string());
        if (auto it = obj.find("item_id"); it != obj.end() && it->value().is_string())
            ev.item_id = std::string(it->value().as_string());
        if (auto it = obj.find("output_index"); it != obj.end() && it->value().is_int64())
            ev.output_index = static_cast<int>(it->value().as_int64());
        if (auto it = obj.find("content_index"); it != obj.end() && it->value().is_int64())
            ev.content_index = static_cast<int>(it->value().as_int64());
        if (auto it = obj.find("delta"); it != obj.end() && it->value().is_string())
            ev.delta = std::string(it->value().as_string());
        return ev;
    }
    if (type == "response.audio_transcript.delta") {
        TranscriptDelta ev;
        if (auto it = obj.find("response_id"); it != obj.end() && it->value().is_string())
            ev.response_id = std::string(it->value().as_string());
        if (auto it = obj.find("item_id"); it != obj.end() && it->value().is_string())
            ev.item_id = std::string(it->value().as_string());
        if (auto it = obj.find("output_index"); it != obj.end() && it->value().is_int64())
            ev.output_index = static_cast<int>(it->value().as_int64());
        if (auto it = obj.find("content_index"); it != obj.end() && it->value().is_int64())
            ev.content_index = static_cast<int>(it->value().as_int64());
        if (auto it = obj.find("delta"); it != obj.end() && it->value().is_string())
            ev.delta = std::string(it->value().as_string());
        return ev;
    }
    if (type == "response.function_call_arguments.delta") {
        FunctionCallArgsDelta ev;
        if (auto it = obj.find("response_id"); it != obj.end() && it->value().is_string())
            ev.response_id = std::string(it->value().as_string());
        if (auto it = obj.find("item_id"); it != obj.end() && it->value().is_string())
            ev.item_id = std::string(it->value().as_string());
        if (auto it = obj.find("call_id"); it != obj.end() && it->value().is_string())
            ev.call_id = std::string(it->value().as_string());
        if (auto it = obj.find("output_index"); it != obj.end() && it->value().is_int64())
            ev.output_index = static_cast<int>(it->value().as_int64());
        if (auto it = obj.find("delta"); it != obj.end() && it->value().is_string())
            ev.delta = std::string(it->value().as_string());
        return ev;
    }
    if (type == "response.function_call_arguments.done") {
        FunctionCallDone ev;
        if (auto it = obj.find("response_id"); it != obj.end() && it->value().is_string())
            ev.response_id = std::string(it->value().as_string());
        if (auto it = obj.find("item_id"); it != obj.end() && it->value().is_string())
            ev.item_id = std::string(it->value().as_string());
        if (auto it = obj.find("call_id"); it != obj.end() && it->value().is_string())
            ev.call_id = std::string(it->value().as_string());
        if (auto it = obj.find("output_index"); it != obj.end() && it->value().is_int64())
            ev.output_index = static_cast<int>(it->value().as_int64());
        if (auto it = obj.find("name"); it != obj.end() && it->value().is_string())
            ev.name = std::string(it->value().as_string());
        if (auto it = obj.find("arguments"); it != obj.end() && it->value().is_string())
            ev.arguments = std::string(it->value().as_string());
        return ev;
    }
    if (type == "input_audio_buffer.committed") {
        InputAudioCommitted ev;
        if (auto it = obj.find("item_id"); it != obj.end() && it->value().is_string())
            ev.item_id = std::string(it->value().as_string());
        return ev;
    }
    if (type == "input_audio_buffer.speech_started") {
        return SpeechStarted{};
    }
    if (type == "input_audio_buffer.speech_stopped") {
        return SpeechStopped{};
    }
    if (type == "error") {
        ErrorEvent ev;
        if (auto it = obj.find("error"); it != obj.end() && it->value().is_object()) {
            auto& err = it->value().as_object();
            if (auto t = err.find("type"); t != err.end() && t->value().is_string())
                ev.type = std::string(t->value().as_string());
            if (auto c = err.find("code"); c != err.end() && c->value().is_string())
                ev.code = std::string(c->value().as_string());
            if (auto m = err.find("message"); m != err.end() && m->value().is_string())
                ev.message = std::string(m->value().as_string());
        }
        return ev;
    }

    return std::nullopt;
}

Task<void> OpenAIRealtimeSession::send_audio(const std::vector<uint8_t>& pcm_data) {
    json::object msg;
    msg["type"] = "input_audio_buffer.append";
    msg["audio"] = base64_encode(pcm_data);
    co_await ws_.send(json::value(std::move(msg)));
}

Task<void> OpenAIRealtimeSession::commit_audio() {
    json::object msg;
    msg["type"] = "input_audio_buffer.commit";
    co_await ws_.send(json::value(std::move(msg)));
}

Task<void> OpenAIRealtimeSession::create_response() {
    json::object msg;
    msg["type"] = "response.create";
    co_await ws_.send(json::value(std::move(msg)));
}

Task<void> OpenAIRealtimeSession::cancel_response() {
    json::object msg;
    msg["type"] = "response.cancel";
    co_await ws_.send(json::value(std::move(msg)));
}

Task<void> OpenAIRealtimeSession::submit_tool_output(
    std::string_view call_id,
    const json::value& output
) {
    json::object item;
    item["type"] = "function_call_output";
    item["call_id"] = std::string(call_id);
    item["output"] = json::serialize(output);

    json::object msg;
    msg["type"] = "conversation.item.create";
    msg["item"] = std::move(item);
    co_await ws_.send(json::value(std::move(msg)));
}

Task<void> OpenAIRealtimeSession::update_session(const json::object& session_update) {
    json::object msg;
    msg["type"] = "session.update";
    msg["session"] = session_update;
    co_await ws_.send(json::value(std::move(msg)));
}

Task<void> OpenAIRealtimeSession::send_text(std::string_view text) {
    json::object content;
    content["type"] = "input_text";
    content["text"] = std::string(text);

    json::object item;
    item["type"] = "message";
    item["role"] = "user";
    item["content"] = json::array{json::value(std::move(content))};

    json::object msg;
    msg["type"] = "conversation.item.create";
    msg["item"] = std::move(item);
    co_await ws_.send(json::value(std::move(msg)));
}

Task<void> OpenAIRealtimeSession::close() {
    co_await ws_.close();
}

bool OpenAIRealtimeSession::is_connected() const noexcept {
    return ws_.is_connected();
}

} // namespace ai::providers::openai
