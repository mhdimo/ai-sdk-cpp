#pragma once

#include <ai/http/websocket.hpp>
#include <ai/stream/async_generator.hpp>
#include <ai/util/cancellation.hpp>
#include <boost/json.hpp>
#include <string>
#include <string_view>
#include <vector>
#include <optional>
#include <variant>
#include <functional>

namespace ai::providers::openai {

struct RealtimeConfig {
    std::string api_key;
    std::string model = "gpt-4o-realtime-preview";
    std::string base_url = "wss://api.openai.com/v1/realtime";
    std::optional<std::string> instructions;
    std::string voice = "alloy";
    std::string input_audio_format = "pcm16";
    std::string output_audio_format = "pcm16";
    double temperature = 0.8;
    std::optional<int> max_response_output_tokens;
    std::vector<boost::json::object> tools;
    std::string turn_detection = "server_vad";
};

struct AudioDelta {
    std::string response_id;
    std::string item_id;
    int output_index;
    int content_index;
    std::vector<uint8_t> audio;
};

struct TextDelta {
    std::string response_id;
    std::string item_id;
    int output_index;
    int content_index;
    std::string delta;
};

struct TranscriptDelta {
    std::string response_id;
    std::string item_id;
    int output_index;
    int content_index;
    std::string delta;
};

struct FunctionCallArgsDelta {
    std::string response_id;
    std::string item_id;
    std::string call_id;
    int output_index;
    std::string delta;
};

struct FunctionCallDone {
    std::string response_id;
    std::string item_id;
    std::string call_id;
    int output_index;
    std::string name;
    std::string arguments;
};

struct InputAudioCommitted {
    std::string item_id;
};

struct SpeechStarted {};
struct SpeechStopped {};

struct ResponseCreated {
    std::string response_id;
};

struct ResponseDone {
    std::string response_id;
    std::string status;
    boost::json::object usage;
};

struct SessionCreated {
    boost::json::object session;
};

struct SessionUpdated {
    boost::json::object session;
};

struct ErrorEvent {
    std::string type;
    std::string code;
    std::string message;
};

using RealtimeEvent = std::variant<
    AudioDelta,
    TextDelta,
    TranscriptDelta,
    FunctionCallArgsDelta,
    FunctionCallDone,
    InputAudioCommitted,
    SpeechStarted,
    SpeechStopped,
    ResponseCreated,
    ResponseDone,
    SessionCreated,
    SessionUpdated,
    ErrorEvent
>;

class OpenAIRealtimeSession {
public:
    OpenAIRealtimeSession(boost::asio::io_context& ioc, RealtimeConfig config);
    ~OpenAIRealtimeSession();

    Task<void> connect(CancellationToken cancel = {});

    AsyncGenerator<RealtimeEvent> events();

    Task<void> send_audio(const std::vector<uint8_t>& pcm_data);
    Task<void> commit_audio();
    Task<void> create_response();
    Task<void> cancel_response();
    Task<void> submit_tool_output(
        std::string_view call_id,
        const boost::json::value& output
    );
    Task<void> update_session(const boost::json::object& session_update);
    Task<void> send_text(std::string_view text);
    Task<void> close();

    bool is_connected() const noexcept;

private:
    RealtimeConfig config_;
    ai::http::WebSocketClient ws_;
    std::optional<RealtimeEvent> parse_server_event(const boost::json::object& obj);
};

} // namespace ai::providers::openai
